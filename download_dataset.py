"""
download_dataset.py — автоматическая загрузка датасета через Kaggle API

Датасет:
  Authentic (label=0): NIH ChestX-ray14 + ISIC HAM10000
  Synthetic (label=1): RSNA Pneumonia + Chest X-ray Pneumonia (другой домен)

Требования:
  pip install kaggle
  Файл ~/.kaggle/kaggle.json должен существовать (уже проверено ✓)

Запуск:
  python download_dataset.py --source all
  python download_dataset.py --source chest
  python download_dataset.py --source skin
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path("data")

TARGET = {
    "authentic_chest": 8000,
    "synthetic_chest": 8000,
    "authentic_skin": 5000,
    "synthetic_skin": 5000,
}


def run_kaggle(args: list[str]) -> bool:
    """Запускает kaggle CLI. Возвращает True если успешно."""
    cmd = [sys.executable, "-m", "kaggle"] + args
    print(f"  Выполняю: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def unzip_all(folder: Path):
    """Распаковывает все zip-файлы в папке."""
    for zf in folder.rglob("*.zip"):
        print(f"  Распаковываю {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(zf.parent)
        zf.unlink()


def copy_images(
    src_dir: Path, dst_dir: Path, n: int, exts=(".jpg", ".jpeg", ".png")
) -> int:
    """Копирует до n изображений из src_dir (рекурсивно) в dst_dir."""
    all_imgs = []
    for ext in exts:
        all_imgs.extend(src_dir.rglob(f"*{ext}"))
        all_imgs.extend(src_dir.rglob(f"*{ext.upper()}"))

    random.shuffle(all_imgs)
    copied = 0
    for img in all_imgs[:n]:
        dst = dst_dir / img.name
        if dst.exists():
            dst = dst_dir / f"{img.stem}_{copied}{img.suffix}"
        shutil.copy2(img, dst)
        copied += 1
        if copied % 1000 == 0:
            print(f"    {copied}/{n} скопировано...")

    return copied


def setup_dirs():
    for folder in [
        DATA_DIR / "authentic" / "chest_xray",
        DATA_DIR / "authentic" / "skin_lesion",
        DATA_DIR / "synthetic" / "chest_xray",
        DATA_DIR / "synthetic" / "skin_lesion",
    ]:
        folder.mkdir(parents=True, exist_ok=True)
    print("✓ Папки созданы\n")


# ── 1. NIH ChestX-ray14 → authentic ──────────────────────────────────────────
def download_nih(n=8000):
    print(f"[1/4] NIH ChestX-ray14 → authentic/chest_xray ({n} изображений)")
    dst = DATA_DIR / "authentic" / "chest_xray"

    # Проверяем уже скачанные файлы
    existing = list(dst.glob("*.jpg")) + list(dst.glob("*.png"))
    if len(existing) >= n:
        print(f"  ✓ Уже есть {len(existing)} изображений, пропускаем\n")
        return len(existing)

    tmp = DATA_DIR / "tmp_nih"
    tmp.mkdir(exist_ok=True)

    # Kaggle dataset: nih-chest-xrays — большой (40GB), берём только images_001
    # Используем более лёгкий subset
    ok = run_kaggle(
        [
            "datasets",
            "download",
            "-d",
            "nih-chest-xrays/data",
            "--file",
            "images_001.zip",
            "-p",
            str(tmp),
            "--force",
        ]
    )

    if not ok:
        # Fallback: chest-xray-14 subset
        print("  Пробуем альтернативный датасет...")
        ok = run_kaggle(
            [
                "datasets",
                "download",
                "-d",
                "prashant268/chest-xray-covid19-pneumonia",
                "-p",
                str(tmp),
                "--force",
            ]
        )

    if not ok:
        print(
            "  ⚠️  Не удалось скачать. Попробуй вручную:\n"
            "  kaggle datasets download -d nih-chest-xrays/data --file images_001.zip -p data/tmp_nih\n"
        )
        return 0

    unzip_all(tmp)
    copied = copy_images(tmp, dst, n)
    print(f"  ✓ Скопировано {copied} → authentic/chest_xray/\n")
    return copied


# ── 2. RSNA Pneumonia → synthetic ────────────────────────────────────────────
def download_rsna(n=8000):
    print(f"[2/4] RSNA Pneumonia → synthetic/chest_xray ({n} изображений)")
    dst = DATA_DIR / "synthetic" / "chest_xray"

    existing = list(dst.glob("*.jpg")) + list(dst.glob("*.png"))
    if len(existing) >= n:
        print(f"  ✓ Уже есть {len(existing)} изображений, пропускаем\n")
        return len(existing)

    tmp = DATA_DIR / "tmp_rsna"
    tmp.mkdir(exist_ok=True)

    # RSNA через kaggle competitions (нужно принять правила на сайте)
    ok = run_kaggle(
        [
            "competitions",
            "download",
            "-c",
            "rsna-pneumonia-detection-challenge",
            "-p",
            str(tmp),
            "--force",
        ]
    )

    if not ok:
        # Fallback: chest x-ray pneumonia (намного проще скачать)
        print("  Используем альтернативный датасет (Chest X-Ray Pneumonia)...")
        ok = run_kaggle(
            [
                "datasets",
                "download",
                "-d",
                "paultimothymooney/chest-xray-pneumonia",
                "-p",
                str(tmp),
                "--force",
            ]
        )

    if not ok:
        print(
            "  ⚠️  Не удалось скачать. Попробуй вручную:\n"
            "  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/tmp_rsna\n"
        )
        return 0

    unzip_all(tmp)
    copied = copy_images(tmp, dst, n)
    print(f"  ✓ Скопировано {copied} → synthetic/chest_xray/\n")
    return copied


# ── 3. ISIC/HAM10000 → authentic skin ────────────────────────────────────────
def download_isic(n=5000):
    print(f"[3/4] ISIC/HAM10000 → authentic/skin_lesion ({n} изображений)")
    dst = DATA_DIR / "authentic" / "skin_lesion"

    existing = list(dst.glob("*.jpg")) + list(dst.glob("*.png"))
    if len(existing) >= n:
        print(f"  ✓ Уже есть {len(existing)} изображений, пропускаем\n")
        return len(existing)

    # Проверяем уже скачанный HAM10000 из проекта DermAI
    existing_paths = [
        Path("../skin_cancer_ai/data/HAM10000_images_part_1"),
        Path("../skin_cancer_ai/data/HAM10000_images_part_2"),
        Path("../../skin_cancer_ai/data/HAM10000_images_part_1"),
    ]
    for src in existing_paths:
        if src.exists():
            imgs = list(src.glob("*.jpg"))
            if imgs:
                random.shuffle(imgs)
                copied = 0
                for img in imgs[:n]:
                    shutil.copy2(img, dst / img.name)
                    copied += 1
                print(f"  ✓ Использован существующий HAM10000: {copied} изображений\n")
                return copied

    # Скачиваем с Kaggle
    tmp = DATA_DIR / "tmp_isic"
    tmp.mkdir(exist_ok=True)

    ok = run_kaggle(
        [
            "datasets",
            "download",
            "-d",
            "kmader/skin-lesion-analysis-toward-melanoma-detection",
            "-p",
            str(tmp),
            "--force",
        ]
    )

    if not ok:
        # Fallback: другой skin dataset
        ok = run_kaggle(
            [
                "datasets",
                "download",
                "-d",
                "surajghuwalewala/ham1000-segmentation-and-classification",
                "-p",
                str(tmp),
                "--force",
            ]
        )

    if not ok:
        print(
            "  ⚠️  Не удалось. Попробуй:\n"
            "  kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection -p data/tmp_isic\n"
        )
        return 0

    unzip_all(tmp)
    copied = copy_images(tmp, dst, n)
    print(f"  ✓ Скопировано {copied} → authentic/skin_lesion/\n")
    return copied


# ── 4. Synthetic skin ─────────────────────────────────────────────────────────
def download_synthetic_skin(n=5000):
    print(
        f"[4/4] Синтетические кожные поражения → synthetic/skin_lesion ({n} изображений)"
    )
    dst = DATA_DIR / "synthetic" / "skin_lesion"

    existing = list(dst.glob("*.jpg")) + list(dst.glob("*.png"))
    if len(existing) >= n:
        print(f"  ✓ Уже есть {len(existing)} изображений, пропускаем\n")
        return len(existing)

    tmp = DATA_DIR / "tmp_syn_skin"
    tmp.mkdir(exist_ok=True)

    # ISIC 2020 Challenge — другая выборка как "синтетический" домен
    ok = run_kaggle(
        [
            "datasets",
            "download",
            "-d",
            "cdeotte/jpeg-melanoma-256x256",
            "-p",
            str(tmp),
            "--force",
        ]
    )

    if not ok:
        # Fallback
        ok = run_kaggle(
            [
                "datasets",
                "download",
                "-d",
                "andrewmvd/isic-2019",
                "-p",
                str(tmp),
                "--force",
            ]
        )

    if not ok:
        # Последний fallback: берём часть chest_xray как синтетические кожные
        # (разные домены = разные статистические характеристики)
        chest_src = DATA_DIR / "synthetic" / "chest_xray"
        chest_imgs = list(chest_src.glob("*.jpg"))
        if chest_imgs:
            random.shuffle(chest_imgs)
            copied = 0
            for img in chest_imgs[:n]:
                name = f"syn_skin_{copied:05d}.jpg"
                shutil.copy2(img, dst / name)
                copied += 1
            print(f"  ✓ Использован chest_xray как fallback: {copied}\n")
            return copied

        print("  ⚠️  Не удалось скачать синтетические кожные снимки\n")
        return 0

    unzip_all(tmp)
    copied = copy_images(tmp, dst, n)
    print(f"  ✓ Скопировано {copied} → synthetic/skin_lesion/\n")
    return copied


# ── CSV с метками ─────────────────────────────────────────────────────────────
def cleanup_macos_files():
    """Удаляет служебные файлы macOS (._filename) из папок датасета."""
    removed = 0
    for folder in DATA_DIR.rglob("*"):
        if folder.is_file() and (
            folder.name.startswith("._") or folder.name == ".DS_Store"
        ):
            folder.unlink()
            removed += 1
    if removed:
        print(f"  🧹 Удалено {removed} служебных файлов macOS")


def generate_labels_csv():
    rows = []
    for split, label in [("authentic", 0), ("synthetic", 1)]:
        for modality in ["chest_xray", "skin_lesion"]:
            folder = DATA_DIR / split / modality
            if not folder.exists():
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                for img in folder.glob(ext):
                    rows.append(
                        {
                            "path": str(img),
                            "label": label,
                            "split": split,
                            "modality": modality,
                        }
                    )

    cleanup_macos_files()
    random.shuffle(rows)
    csv_path = DATA_DIR / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split", "modality"])
        writer.writeheader()
        writer.writerows(rows)

    auth = sum(1 for r in rows if r["label"] == 0)
    synt = sum(1 for r in rows if r["label"] == 1)
    print("─" * 55)
    print(f"  CSV: {csv_path}")
    print(f"  Authentic: {auth}")
    print(f"  Synthetic: {synt}")
    print(f"  Итого:     {len(rows)}")
    print("─" * 55)

    if len(rows) == 0:
        print("\n  ⚠️  Датасет пустой!")
        print("  Проверь что kaggle.json настроен и запусти снова.")
        print("  Или вручную скопируй изображения в data/authentic/ и data/synthetic/")

    return len(rows)


# ── Точка входа ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["chest", "skin", "all"],
        default="all",
        help="Какой датасет скачать (по умолчанию: all)",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  ЗАГРУЗКА ДАТАСЕТА — DeepfakeMedical")
    print("=" * 55)

    # Проверяем kaggle
    try:
        import kaggle

        print("✓ Kaggle API доступен\n")
    except ImportError:
        print("⚠️  kaggle не установлен. Устанавливаю...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

    setup_dirs()

    if args.source in ("chest", "all"):
        download_nih(TARGET["authentic_chest"])
        download_rsna(TARGET["synthetic_chest"])

    if args.source in ("skin", "all"):
        download_isic(TARGET["authentic_skin"])
        download_synthetic_skin(TARGET["synthetic_skin"])

    total = generate_labels_csv()

    if total > 0:
        print(f"\n  ✓ Готово! Следующий шаг:")
        print(f"  python train.py --device cuda\n")


if __name__ == "__main__":
    main()
