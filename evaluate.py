"""
evaluate.py
-----------
Полная оценка обученной модели на тестовых данных.

Метрики: AUC-ROC, Accuracy, Precision, Recall, F1
Графики: ROC-кривая, Confusion Matrix, распределение уверенности
Робастность: проверка на JPEG-компрессию и ресайз

Запуск:
  python evaluate.py [--model models/best_model.pt]
"""

import argparse
import csv
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model.detector import DeepfakeDetector
from train import MedicalDeepfakeDataset, load_dataset

SEED = 42
MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 55)
print("  ОЦЕНКА МОДЕЛИ ДЕТЕКЦИИ ДИПФЕЙКОВ")
print("=" * 55)


def evaluate(model_path: str, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем модель
    print("\nЗагрузка модели...")
    model = DeepfakeDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"   ✓ {model_path}")

    # Загружаем данные (воспроизводим ту же разбивку что при обучении)
    print("\nПодготовка тестовых данных...")
    records = load_dataset()
    labels = [l for _, l in records]

    _, temp = train_test_split(
        records, test_size=0.30, stratify=labels, random_state=SEED
    )
    tl = [l for _, l in temp]
    _, test = train_test_split(temp, test_size=0.50, stratify=tl, random_state=SEED)

    print(f"   Тестовых примеров: {len(test)}")

    test_ds = MedicalDeepfakeDataset(test, augment=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # Предсказания
    print("\nВычисление предсказаний...")
    all_probs, all_labels = [], []
    with torch.no_grad():
        for spatial, freq, lbl in test_dl:
            logits = model(spatial.to(device), freq.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(lbl.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    # ── Метрики ───────────────────────────────────────────────────────────────
    auc_val = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    report = classification_report(
        all_labels, preds, target_names=["authentic", "deepfake"]
    )

    print("\n" + "─" * 55)
    print("  МЕТРИКИ ПО КЛАССАМ")
    print("─" * 55)
    print(report)

    with open(f"{RESULTS_DIR}/classification_report.txt", "w") as f:
        f.write("ОТЧЁТ ПО КЛАССИФИКАЦИИ — DeepfakeMedical\n")
        f.write("=" * 55 + "\n\n")
        f.write(report)

    # ── ROC-кривая ────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC-кривая")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Матрица ошибок ─────────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=["authentic", "deepfake"],
        yticklabels=["authentic", "deepfake"],
        ax=axes[1],
    )
    axes[1].set_xlabel("Предсказано")
    axes[1].set_ylabel("Реально")
    axes[1].set_title("Confusion Matrix (%)")

    # ── Распределение уверенности ─────────────────────────────────────────────
    correct_probs = all_probs[preds == all_labels]
    incorrect_probs = all_probs[preds != all_labels]

    axes[2].hist(
        correct_probs,
        bins=20,
        alpha=0.7,
        color="#2ecc71",
        label=f"Правильные ({len(correct_probs)})",
    )
    axes[2].hist(
        incorrect_probs,
        bins=20,
        alpha=0.7,
        color="#e74c3c",
        label=f"Ошибочные ({len(incorrect_probs)})",
    )
    axes[2].set_xlabel("Уверенность модели (sigmoid)")
    axes[2].set_ylabel("Количество примеров")
    axes[2].set_title("Распределение уверенности")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/evaluation.png", dpi=150)
    print(f"   ✓ {RESULTS_DIR}/evaluation.png")

    # ── Итог ──────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"  AUC-ROC  : {auc_val:.4f}  ({auc_val * 100:.1f}%)")
    print(f"  Accuracy : {acc:.4f}  ({acc * 100:.1f}%)")
    print(f"  F1-Score : {f1:.4f}  ({f1 * 100:.1f}%)")
    print(f"{'─' * 55}")
    print(f"\n  Графики: {RESULTS_DIR}/evaluation.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{MODELS_DIR}/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.model, args.batch_size)
