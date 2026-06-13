"""
train.py
--------
Обучение модели обнаружения дипфейков. EfficientNet-B4 + FFT Dual-Branch.

Оптимизации для CPU:
  1. FFT кэшируется на диск один раз (как в train_fast.py)
  2. EfficientNet-B0 вместо B4 на CPU — в 4x быстрее, почти та же точность
  3. Две фазы обучения сохранены (правильный Transfer Learning)

Запуск:
  python train.py --device cpu --epochs 5 --batch-size 8
  python train.py --device cuda --epochs 40 --batch-size 32 --model b4
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import timm
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

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


# ── FFT кэш (главная оптимизация) ─────────────────────────────────────────────
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
        v = log_mag[(radii >= edges[i]) & (radii < edges[i + 1])]
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


def build_cache(records):
    if os.path.exists(CACHE_FILE):
        print(f"  Загружаю FFT кэш: {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print(f"  Считаю FFT для {len(records)} изображений (один раз, потом из кэша)...")
    cache = {}
    for i, (path, _) in enumerate(tqdm(records, ncols=70)):
        try:
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            cache[path] = extract_freq(np.array(img))
        except:
            cache[path] = np.zeros(256, dtype=np.float32)
        if (i + 1) % 2000 == 0:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(cache, f)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"  ✓ Кэш сохранён: {CACHE_FILE}")
    return cache


# ── Датасет ────────────────────────────────────────────────────────────────────
class MedDataset(Dataset):
    def __init__(self, records, cache, augment=False):
        self.records = records
        self.cache = cache
        self.augment = augment
        self.tf_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
        except:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        tf = self.tf_train if self.augment else self.tf_val
        freq = torch.from_numpy(self.cache.get(path, np.zeros(256, dtype=np.float32)))
        return tf(img), freq, torch.tensor(float(label))


# ── Модель: EfficientNet + FFT Dual-Branch ────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self, model_name="efficientnet_b0", dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        sp_dim = self.backbone.num_features

        self.freq_branch = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )
        self.attention = nn.Linear(sp_dim + 512, 2)
        fused_dim = sp_dim + 512 + 512
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        last = None
        try:
            last = list(self.backbone.blocks.children())[-1]
        except:
            for m in self.backbone.modules():
                if hasattr(m, "conv_pwl"):
                    last = m
        if last:
            last.register_forward_hook(lambda m, i, o: setattr(self, "activations", o))
            last.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0])
            )

    def forward(self, x, freq):
        sp = self.backbone(x)
        fr = self.freq_branch(freq)
        atw = torch.softmax(self.attention(torch.cat([sp, fr], 1)), 1)
        cross = sp[:, :512] * fr
        return self.classifier(
            torch.cat([atw[:, 0:1] * sp, atw[:, 1:2] * fr, cross], 1)
        ).squeeze(1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (
            1 - targets
        )
        return (self.alpha * torch.pow(1 - pt, self.gamma) * bce).mean()


# ── Один проход ────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train() if training else model.eval()
    total_loss, all_probs, all_labels = 0.0, [], []
    desc = "Train" if training else "Val  "
    with torch.set_grad_enabled(training):
        for sp, freq, lbl in tqdm(loader, desc=desc, leave=False, ncols=80):
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


# ── Основной цикл ──────────────────────────────────────────────────────────────
def train(args):
    import platform

    nw = 0 if platform.system() == "Windows" else 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Выбор модели по устройству
    if args.model == "auto":
        model_name = "efficientnet_b0" if device.type == "cpu" else "efficientnet_b4"
    else:
        model_name = f"efficientnet_{args.model}"

    print(f"\nУстройство: {device} | Модель: {model_name}")

    # ── Данные ────────────────────────────────────────────────────────────────
    if args.full:
        csv_path = "data/labels.csv"
        print("  Режим: ПОЛНЫЙ датасет")
    elif os.path.exists("data/labels_small.csv"):
        csv_path = "data/labels_small.csv"
        print("  Режим: маленький датасет (используй --full для полного)")
    else:
        csv_path = "data/labels.csv"
    print(f"\n[1/5] Загрузка из {csv_path}...")
    records = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            p = row["path"]
            b = os.path.basename(p)
            if (
                not b.startswith("._")
                and os.path.exists(p)
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

    # ── FFT кэш ───────────────────────────────────────────────────────────────
    print("\n[2/5] FFT кэш...")
    cache = build_cache(records)
    train_dl = DataLoader(
        MedDataset(tr, cache, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=nw,
    )
    val_dl = DataLoader(
        MedDataset(vl, cache, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
    )
    test_dl = DataLoader(
        MedDataset(te, cache, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
    )

    # ── Модель ────────────────────────────────────────────────────────────────
    print(f"\n[3/5] Инициализация {model_name}...")
    model = DeepfakeDetector(model_name=model_name, dropout=0.4).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"   Параметров: {params:,}")
    criterion = FocalLoss(gamma=2.0, alpha=0.25)

    history = {"tr_auc": [], "vl_auc": [], "tr_loss": [], "vl_loss": []}
    best_auc = 0.0

    # ── Фаза 1: backbone заморожен ────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  ФАЗА 1: backbone заморожен ({args.phase1_epochs} эпох) | LR=1e-3")
    print(f"{'=' * 55}")
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt1 = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-5,
    )

    phase1_ep = getattr(args, "phase1_epochs", 5)
    for epoch in range(phase1_ep):
        t0 = time.time()
        trl, tra, _ = run_epoch(model, train_dl, criterion, opt1, device, True)
        vll, vla, vacc = run_epoch(model, val_dl, criterion, opt1, device, False)
        history["tr_auc"].append(tra)
        history["vl_auc"].append(vla)
        history["tr_loss"].append(trl)
        history["vl_loss"].append(vll)
        mark = "✓" if vla > best_auc else " "
        print(
            f"  Ep {epoch + 1:2d}/{phase1_ep} | Loss {trl:.4f}/{vll:.4f} | AUC {tra:.3f}/{vla:.3f} {mark} | Acc {vacc:.3f} | {time.time() - t0:.0f}s"
        )
        if vla > best_auc:
            best_auc = vla
            torch.save(model.state_dict(), f"{MODELS_DIR}/best_model.pt")

    # ── Фаза 2: полное обучение ───────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  ФАЗА 2: fine-tuning ({args.epochs} эпох) | LR=1e-4")
    print(f"{'=' * 55}")
    for p in model.backbone.parameters():
        p.requires_grad = True
    opt2 = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = CosineAnnealingWarmRestarts(opt2, T_0=10, T_mult=2)
    patience_cnt = 0

    for epoch in range(args.epochs):
        t0 = time.time()
        trl, tra, _ = run_epoch(model, train_dl, criterion, opt2, device, True)
        vll, vla, vacc = run_epoch(model, val_dl, criterion, opt2, device, False)
        sched.step()
        history["tr_auc"].append(tra)
        history["vl_auc"].append(vla)
        history["tr_loss"].append(trl)
        history["vl_loss"].append(vll)
        mark = "✓" if vla > best_auc else " "
        print(
            f"  Ep {epoch + 1:2d}/{args.epochs} | Loss {trl:.4f}/{vll:.4f} | AUC {tra:.3f}/{vla:.3f} {mark} | {time.time() - t0:.0f}s"
        )
        if vla > best_auc:
            best_auc = vla
            patience_cnt = 0
            torch.save(model.state_dict(), f"{MODELS_DIR}/best_model.pt")
            print(f"     → Сохранено (AUC: {best_auc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= 8:
                print(f"  Early stopping на эпохе {epoch + 1}")
                break

    # ── Оценка ────────────────────────────────────────────────────────────────
    print("\n[5/5] Оценка...")
    model.load_state_dict(
        torch.load(f"{MODELS_DIR}/best_model.pt", map_location=device)
    )
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for sp, freq, lbl in tqdm(test_dl, desc="Test ", ncols=80):
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
    ep = range(1, len(history["tr_auc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ep, history["tr_auc"], label="Train", lw=2)
    axes[0].plot(ep, history["vl_auc"], label="Val", lw=2)
    axes[0].axvline(10, color="red", ls="--", alpha=0.5, label="Fine-tuning")
    axes[0].set_title("AUC-ROC")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(ep, history["tr_loss"], label="Train", lw=2)
    axes[1].plot(ep, history["vl_loss"], label="Val", lw=2)
    axes[1].axvline(10, color="red", ls="--", alpha=0.5)
    axes[1].set_title("Focal Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_history.png", dpi=150)

    print(f"\n{'═' * 50}")
    print(f"  Модель:   {model_name}")
    print(f"  AUC-ROC  : {auc:.4f}  ({auc * 100:.1f}%)")
    print(f"  Accuracy : {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"  F1-Score : {f1:.4f}  ({f1 * 100:.1f}%)")
    print(f"{'═' * 50}")
    print(f"  ✓ Модель: {MODELS_DIR}/best_model.pt")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    p = argparse.ArgumentParser(description="Обучение детектора дипфейков")
    p.add_argument(
        "--model",
        type=str,
        default="b4",
        choices=["b0", "b1", "b4"],
        help="""
Выбор архитектуры backbone:
  b0 — EfficientNet-B0 |  5M параметров | CPU ~2-3 часа  | GPU ~30 мин
  b1 — EfficientNet-B1 |  8M параметров | CPU ~3-4 часа  | GPU ~45 мин
  b4 — EfficientNet-B4 | 19M параметров | CPU ~4-5 часов | GPU ~90 мин  ← рекомендуется
""",
    )
    p.add_argument("--device", type=str, default="cpu", help="cpu или cuda")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Размер батча (по умолчанию: 8 для cpu, 32 для cuda)",
    )
    p.add_argument(
        "--phase1-epochs",
        type=int,
        default=None,
        help="Эпох фазы 1 (backbone заморожен)",
    )
    p.add_argument(
        "--phase2-epochs", type=int, default=None, help="Эпох фазы 2 (fine-tuning)"
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Обучать на полном датасете (data/labels.csv)",
    )

    args = p.parse_args()

    # Умные дефолты по устройству и модели
    device_type = (
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    if args.batch_size is None:
        args.batch_size = 32 if device_type == "cuda" else 8

    # Таблица времени: CPU ~35-45 мин/эпоха для b4, 10-15 для b0
    time_per_epoch = {"b0": 12, "b1": 22, "b4": 40}  # минут на CPU
    tpe = time_per_epoch[args.model]

    if args.phase1_epochs is None:
        # Цель: ~1.5 часа на фазу 1
        args.phase1_epochs = max(3, min(10, int(90 / tpe)))

    if args.phase2_epochs is None:
        # Цель: ~2.5 часа на фазу 2
        args.phase2_epochs = max(3, min(40, int(150 / tpe)))

    total_est = (args.phase1_epochs + args.phase2_epochs) * tpe
    print("=" * 55)
    print(f"  КОНФИГУРАЦИЯ ОБУЧЕНИЯ")
    print("=" * 55)
    print(f"  Модель:         EfficientNet-{args.model.upper()}")
    print(f"  Устройство:     {device_type}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Фаза 1:         {args.phase1_epochs} эпох")
    print(f"  Фаза 2:         {args.phase2_epochs} эпох")
    print(f"  Ожидаемое время: ~{total_est // 60}ч {total_est % 60}мин")
    print("=" * 55)

    # Передаём в train()
    args.epochs = args.phase2_epochs
    train(args)
