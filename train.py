"""
train.py
--------
Обучение модели обнаружения дипфейков в медицинских изображениях.

Архитектура: EfficientNet-B4 (пространственная ветвь) + FFT признаки (частотная ветвь)
Задача: бинарная классификация — authentic (0) vs deepfake (1)

Обучение в два этапа:
  Фаза 1 (10 эпох):  backbone заморожен, обучается только классификатор
  Фаза 2 (40 эпох):  полное обучение с cosine annealing

Функция потерь: Focal Loss (gamma=2.0) — борется с дисбалансом классов

Запуск:
  python train.py [--epochs 50] [--batch-size 32] [--device cuda]
"""

import argparse
import csv
import os
import random
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model.detector import DeepfakeDetector, FocalLoss
from model.freq_features import extract_freq_features

# ── Настройки ─────────────────────────────────────────────────────────────────
SEED = 42
IMG_SIZE = 224
RESULTS_DIR = "results"
MODELS_DIR = "models"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("  ОБУЧЕНИЕ ДЕТЕКТОРА ДИПФЕЙКОВ")
print("  Архитектура: EfficientNet-B4 + FFT features (Dual-Branch)")
print("=" * 60)


# ── Датасет ───────────────────────────────────────────────────────────────────
class MedicalDeepfakeDataset(Dataset):
    """
    Датасет медицинских изображений с метками authentic/deepfake.

    Каждый элемент возвращает тройку:
      (spatial_tensor, freq_tensor, label)
    где label: 0 = authentic, 1 = deepfake
    """

    def __init__(self, records: list, augment: bool = False):
        self.records = records
        self.augment = augment

        # Аугментации для обучения — симулируем условия реальных forensic сценариев
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Для валидации/теста — только нормализация
        self.val_transform = transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]

        # Загружаем изображение (пропускаем повреждённые файлы)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
        img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))

        # Частотные признаки — до аугментации (на оригинальном изображении)
        freq = extract_freq_features(img_np)
        freq_tensor = torch.from_numpy(freq)

        # Пространственные признаки — с аугментацией при обучении
        transform = self.train_transform if self.augment else self.val_transform
        spatial_tensor = transform(img)

        return spatial_tensor, freq_tensor, torch.tensor(float(label))


def load_dataset(csv_path: str = "data/labels.csv"):
    """Загружает список путей и меток из CSV файла."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV не найден: {csv_path}\nЗапустите: python download_dataset.py"
        )

    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row["path"]
            # Пропускаем macOS метафайлы (._filename) и скрытые файлы
            basename = os.path.basename(path)
            if basename.startswith("._") or basename.startswith("."):
                continue
            if os.path.exists(path) and os.path.getsize(path) > 1024:
                records.append((path, int(row["label"])))

    print(f"   Найдено изображений: {len(records)}")
    authentic = sum(1 for _, l in records if l == 0)
    synthetic = sum(1 for _, l in records if l == 1)
    print(f"   Authentic: {authentic}")
    print(f"   Synthetic: {synthetic}")
    return records


def create_dataloaders(records, batch_size):
    """Разбивает на train/val/test и создаёт DataLoader'ы."""
    labels = [l for _, l in records]

    # 70/15/15 с сохранением пропорций классов
    train_r, temp_r = train_test_split(
        records, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_labels = [l for _, l in temp_r]
    val_r, test_r = train_test_split(
        temp_r, test_size=0.50, stratify=val_labels, random_state=SEED
    )

    print(f"   Train: {len(train_r)} | Val: {len(val_r)} | Test: {len(test_r)}")

    train_ds = MedicalDeepfakeDataset(train_r, augment=True)
    val_ds = MedicalDeepfakeDataset(val_r, augment=False)
    test_ds = MedicalDeepfakeDataset(test_r, augment=False)

    # Windows требует num_workers=0 (нет fork)
    import platform

    nw = 0 if platform.system() == "Windows" else 4
    pm = platform.system() != "Windows"

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pm
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pm
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pm
    )

    return train_dl, val_dl, test_dl


# ── Один эпох обучения / валидации ───────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()

    total_loss = 0.0
    all_probs, all_labels = [], []

    desc = "Train" if training else "Val  "
    with torch.set_grad_enabled(training):
        for spatial, freq, labels in tqdm(loader, desc=desc, leave=False, ncols=80):
            spatial = spatial.to(device)
            freq = freq.to(device)
            labels = labels.to(device)

            logits = model(spatial, freq)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — предотвращает взрыв градиентов при fine-tuning
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)

    return avg_loss, auc, acc


# ── Основной цикл обучения ────────────────────────────────────────────────────
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nУстройство: {device}")

    # Данные
    print("\n[1/5] Загрузка данных...")
    records = load_dataset()
    train_dl, val_dl, test_dl = create_dataloaders(records, args.batch_size)

    # Модель
    print("\n[2/5] Инициализация модели EfficientNet-B4...")
    model = DeepfakeDetector(freq_input_dim=256, dropout=0.5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Всего параметров: {total_params:,}")

    criterion = FocalLoss(gamma=2.0, alpha=0.25)

    # ── Фаза 1: замораживаем backbone ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ФАЗА 1: Обучение классификатора (backbone заморожен)")
    print(f"  Эпох: 10 | LR: 1e-3")
    print("=" * 60)

    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer_1 = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-5,
    )

    best_auc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}

    for epoch in range(10):
        t0 = time.time()
        tr_loss, tr_auc, tr_acc = run_epoch(
            model, train_dl, criterion, optimizer_1, device, training=True
        )
        vl_loss, vl_auc, vl_acc = run_epoch(
            model, val_dl, criterion, optimizer_1, device, training=False
        )
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(vl_auc)

        print(
            f"  Epoch {epoch + 1:2d}/10 | "
            f"Loss: {tr_loss:.4f}/{vl_loss:.4f} | "
            f"AUC: {tr_auc:.4f}/{vl_auc:.4f} | "
            f"Acc: {vl_acc:.3f} | {elapsed:.0f}s"
        )

        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), f"{MODELS_DIR}/best_model.pt")

    # ── Фаза 2: полное обучение (fine-tuning) ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  ФАЗА 2: Fine-tuning (все параметры)")
    print(f"  Эпох: {args.epochs} | LR: 1e-4 | CosineAnnealing")
    print("=" * 60)

    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer_2 = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer_2, T_0=10, T_mult=2)

    patience_counter = 0
    patience = 8  # early stopping

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_auc, tr_acc = run_epoch(
            model, train_dl, criterion, optimizer_2, device, training=True
        )
        vl_loss, vl_auc, vl_acc = run_epoch(
            model, val_dl, criterion, optimizer_2, device, training=False
        )
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(vl_auc)

        improved = "✓" if vl_auc > best_auc else " "
        print(
            f"  Epoch {epoch + 1:2d}/{args.epochs} | "
            f"Loss: {tr_loss:.4f}/{vl_loss:.4f} | "
            f"AUC: {tr_auc:.4f}/{vl_auc:.4f} {improved} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s"
        )

        if vl_auc > best_auc:
            best_auc = vl_auc
            patience_counter = 0
            torch.save(model.state_dict(), f"{MODELS_DIR}/best_model.pt")
            print(f"     → Лучшая модель сохранена (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping на эпохе {epoch + 1}")
                break

    # ── Оценка на тестовых данных ─────────────────────────────────────────────
    print("\n[5/5] Оценка на тестовых данных...")
    model.load_state_dict(
        torch.load(f"{MODELS_DIR}/best_model.pt", map_location=device)
    )
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for spatial, freq, labels in test_dl:
            logits = model(spatial.to(device), freq.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average="binary")

    report = classification_report(
        all_labels, preds, target_names=["authentic", "deepfake"]
    )
    print("\n" + report)

    # ── Графики ───────────────────────────────────────────────────────────────
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ft_start = 10  # начало фазы 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        epochs_range, history["train_auc"], label="Train", color="steelblue", lw=2
    )
    axes[0].plot(
        epochs_range, history["val_auc"], label="Val", color="darkorange", lw=2
    )
    axes[0].axvline(
        ft_start, color="red", ls="--", alpha=0.5, label="Fine-tuning start"
    )
    axes[0].set_title("AUC-ROC по эпохам")
    axes[0].set_xlabel("Эпоха")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs_range, history["train_loss"], label="Train", color="steelblue", lw=2
    )
    axes[1].plot(
        epochs_range, history["val_loss"], label="Val", color="darkorange", lw=2
    )
    axes[1].axvline(ft_start, color="red", ls="--", alpha=0.5)
    axes[1].set_title("Focal Loss по эпохам")
    axes[1].set_xlabel("Эпоха")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/training_history.png", dpi=150)
    print(f"   ✓ {RESULTS_DIR}/training_history.png")

    # ── Итог ──────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print(f"  AUC-ROC  : {auc:.4f}  ({auc * 100:.1f}%)")
    print(f"  Accuracy : {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"  F1-Score : {f1:.4f}  ({f1 * 100:.1f}%)")
    print(f"{'═' * 55}")
    print(f"\n  Модель сохранена: {MODELS_DIR}/best_model.pt")
    print(f"  Следующий шаг:   python evaluate.py")


# ── Точка входа ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение детектора дипфейков")
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Количество эпох фазы 2 (по умолчанию: 40)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Размер батча (по умолчанию: 32)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Устройство: cuda или cpu"
    )
    args = parser.parse_args()

    train(args)
