"""
train_fast.py — быстрое обучение на CPU (как в DermAI)
-------------------------------------------------------
Что изменилось по сравнению с train.py:
  1. MobileNetV2 вместо EfficientNet-B4 — в 4x быстрее, меньше памяти
  2. FFT признаки кэшируются на диск один раз — не считаются каждую эпоху
  3. Одна фаза обучения вместо двух
  4. Меньше параметров модели

Запуск:
  python train_fast.py --epochs 10 --batch-size 16
"""

import argparse
import csv
import os
import pickle
import random
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import matplotlib
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import kurtosis as sp_kurt
from scipy.stats import skew
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
IMG_SIZE = 224
MODELS_DIR = "models"
RESULTS_DIR = "results"
CACHE_FILE = "data/freq_cache.pkl"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 55)
print("  БЫСТРОЕ ОБУЧЕНИЕ — MobileNetV2 + FFT (CPU mode)")
print("=" * 55)


# ── Частотные признаки ────────────────────────────────────────────────────────
def extract_freq(img_np: np.ndarray) -> np.ndarray:
    gray = (
        0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]
    ) / 255.0
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    log_mag = np.log1p(np.abs(fshift))
    H, W = log_mag.shape
    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    edges = np.linspace(0, max_r, 33)
    y_idx, x_idx = np.ogrid[-cy : H - cy, -cx : W - cx]
    radii = np.sqrt(x_idx**2 + y_idx**2)
    feats = []
    for i in range(32):
        mask = (radii >= edges[i]) & (radii < edges[i + 1])
        v = log_mag[mask]
        if len(v) == 0:
            v = np.zeros(1)
        feats.extend(
            [
                v.mean(),
                v.std(),
                v.max(),
                v.min(),
                float(skew(v)),
                float(sp_kurt(v)),
                float(np.sum(v**2)),
                float(-np.sum(v * np.log(v + 1e-8))),
            ]
        )
    return np.array(feats, dtype=np.float32)


def build_freq_cache(records: list) -> dict:
    """Считаем FFT для всех изображений один раз и сохраняем на диск."""
    if os.path.exists(CACHE_FILE):
        print(f"  Загружаю кэш FFT: {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print(f"  Считаю FFT признаки для {len(records)} изображений (один раз)...")
    cache = {}
    for i, (path, _) in enumerate(tqdm(records, ncols=70)):
        try:
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            cache[path] = extract_freq(np.array(img))
        except Exception:
            cache[path] = np.zeros(256, dtype=np.float32)
        if (i + 1) % 1000 == 0:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(cache, f)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"  ✓ Кэш сохранён: {CACHE_FILE}")
    return cache


# ── Датасет ───────────────────────────────────────────────────────────────────
class FastDataset(Dataset):
    def __init__(self, records, freq_cache, augment=False):
        self.records = records
        self.cache = freq_cache
        self.augment = augment
        self.tf_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.tf_val = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        tf = self.tf_train if self.augment else self.tf_val
        spatial = tf(img)
        freq = torch.from_numpy(self.cache.get(path, np.zeros(256, dtype=np.float32)))
        return spatial, freq, torch.tensor(float(label))


# ── Модель: MobileNetV2 + FFT (быстрая версия) ───────────────────────────────
class FastDetector(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        # MobileNetV2 — в 4x быстрее EfficientNet-B4
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # убираем classifier

        # Для Grad-CAM
        self.activations = None
        self.gradients = None
        self._register_gradcam_hooks()

    def _register_gradcam_hooks(self):
        # Последний свёрточный слой в MobileNetV2
        last_conv = None
        for m in self.backbone.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        if last_conv:
            last_conv.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o)
            )
            last_conv.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0])
            )
        sp_dim = 1280  # выход MobileNetV2

        # Частотная ветвь — упрощённая
        self.freq_branch = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(sp_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x, freq):
        sp = self.backbone(x)
        sp = sp.mean([2, 3])  # GlobalAveragePooling2D
        fr = self.freq_branch(freq)
        return self.classifier(torch.cat([sp, fr], dim=1)).squeeze(1)


# ── Обучение ──────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train() if training else model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    desc = "Train" if training else "Val  "
    with torch.set_grad_enabled(training):
        for sp, freq, lbl in tqdm(loader, desc=desc, leave=False, ncols=75):
            sp, freq, lbl = sp.to(device), freq.to(device), lbl.to(device)
            logits = model(sp, freq)
            loss = criterion(logits, lbl)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(lbl)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    acc = accuracy_score(all_labels, (np.array(all_probs) >= 0.5).astype(int))
    return avg_loss, auc, acc


def main(args):
    import platform

    nw = 0 if platform.system() == "Windows" else 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}\n")

    # ── Данные ────────────────────────────────────────────────────────────────
    csv_path = (
        "data/labels.csv" if os.path.exists("data/labels.csv") else "data/labels.csv"
    )
    print(f"[1/4] Загрузка данных из {csv_path}...")
    records = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            p = row["path"]
            if (
                os.path.exists(p)
                and not os.path.basename(p).startswith("._")
                and os.path.getsize(p) > 1024
            ):
                records.append((p, int(row["label"])))

    auth = sum(1 for _, l in records if l == 0)
    synt = sum(1 for _, l in records if l == 1)
    print(f"   Authentic: {auth} | Synthetic: {synt} | Итого: {len(records)}")

    labels = [l for _, l in records]
    tr, tmp = train_test_split(
        records, test_size=0.30, stratify=labels, random_state=SEED
    )
    tl = [l for _, l in tmp]
    vl, te = train_test_split(tmp, test_size=0.50, stratify=tl, random_state=SEED)
    print(f"   Train: {len(tr)} | Val: {len(vl)} | Test: {len(te)}")

    # ── Кэш FFT ───────────────────────────────────────────────────────────────
    print("\n[2/4] FFT признаки...")
    freq_cache = build_freq_cache(records)

    train_dl = DataLoader(
        FastDataset(tr, freq_cache, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
    )
    val_dl = DataLoader(
        FastDataset(vl, freq_cache, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
    )
    test_dl = DataLoader(
        FastDataset(te, freq_cache, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
    )

    # ── Модель ────────────────────────────────────────────────────────────────
    print("\n[3/4] Инициализация MobileNetV2...")
    model = FastDetector().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Параметров: {params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # ── Обучение ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] Обучение ({args.epochs} эпох)...")
    best_auc, history = 0.0, {"tr_auc": [], "vl_auc": [], "tr_loss": [], "vl_loss": []}

    for epoch in range(args.epochs):
        t0 = time.time()
        trl, tra, _ = run_epoch(model, train_dl, criterion, optimizer, device, True)
        vll, vla, vacc = run_epoch(model, val_dl, criterion, optimizer, device, False)
        scheduler.step()

        history["tr_auc"].append(tra)
        history["vl_auc"].append(vla)
        history["tr_loss"].append(trl)
        history["vl_loss"].append(vll)

        mark = "✓" if vla > best_auc else " "
        print(
            f"  Ep {epoch + 1:2d}/{args.epochs} | "
            f"Loss {trl:.4f}/{vll:.4f} | "
            f"AUC {tra:.3f}/{vla:.3f} {mark} | "
            f"Acc {vacc:.3f} | {time.time() - t0:.0f}s"
        )

        if vla > best_auc:
            best_auc = vla
            torch.save(model.state_dict(), f"{MODELS_DIR}/best_model.pt")
            print(f"     → Сохранено (AUC: {best_auc:.4f})")

    # ── Оценка ────────────────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(f"{MODELS_DIR}/best_model.pt", map_location=device)
    )
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for sp, freq, lbl in tqdm(test_dl, desc="Test ", ncols=75):
            probs = torch.sigmoid(model(sp.to(device), freq.to(device))).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(lbl.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    print(
        "\n"
        + classification_report(
            all_labels, preds, target_names=["authentic", "deepfake"]
        )
    )

    # ── Графики ───────────────────────────────────────────────────────────────
    ep = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, history["tr_auc"], label="Train", lw=2)
    axes[0].plot(ep, history["vl_auc"], label="Val", lw=2)
    axes[0].set_title("AUC-ROC")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(ep, history["tr_loss"], label="Train", lw=2)
    axes[1].plot(ep, history["vl_loss"], label="Val", lw=2)
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_history.png", dpi=150)

    print(f"\n{'═' * 50}")
    print(f"  AUC-ROC  : {auc:.4f}  ({auc * 100:.1f}%)")
    print(f"  Accuracy : {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"  F1-Score : {f1:.4f}  ({f1 * 100:.1f}%)")
    print(f"{'═' * 50}")
    print(f"\n  ✓ Модель: {MODELS_DIR}/best_model.pt")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    main(parser.parse_args())
