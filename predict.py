"""
predict.py
----------
Предсказание для одного медицинского изображения + Grad-CAM визуализация.

Выходные данные:
  - Метка: authentic / deepfake
  - Уверенность: вероятность [0, 1]
  - Grad-CAM тепловая карта: зоны артефактов

Запуск:
  python predict.py <путь_к_изображению.jpg>
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from model.detector import DeepfakeDetector
from model.freq_features import extract_freq_features
from model.gradcam import compute_gradcam, overlay_heatmap

MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = 224


def predict_image(image_path: str, model_path: str = f"{MODELS_DIR}/best_model.pt"):
    """
    Выполняет предсказание для одного изображения.

    Возвращает dict:
        label, confidence, probabilities, cam
    """
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        print("   Запустите обучение: python train.py")
        return None

    if not os.path.exists(image_path):
        print(f"❌ Изображение не найдено: {image_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    model = DeepfakeDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Загрузка и предобработка изображения
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))

    # Пространственный тензор (для backbone)
    spatial_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    spatial = spatial_transform(img).unsqueeze(0).to(device)

    # Частотные признаки
    freq = torch.from_numpy(extract_freq_features(img_np)).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        logit = model(spatial, freq)
    prob = torch.sigmoid(logit).item()
    label = "deepfake" if prob >= 0.5 else "authentic"

    # Grad-CAM
    cam = compute_gradcam(model, spatial, freq, str(device))

    # Визуализация
    overlay = overlay_heatmap(cam, img_np)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    risk_color = "#e74c3c" if label == "deepfake" else "#2ecc71"

    fig.suptitle(
        f"Результат: {label.upper()}  |  Уверенность: {prob:.1%}",
        fontsize=13,
        fontweight="bold",
        color=risk_color,
    )

    axes[0].imshow(img_np)
    axes[0].set_title("Исходное изображение")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM\n(красный = зоны артефактов)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Наложение на оригинал")
    axes[2].axis("off")

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(RESULTS_DIR, f"predict_{base}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    # Консольный вывод
    print(f"\n{'=' * 55}")
    print(f"  Изображение: {os.path.basename(image_path)}")
    print(f"  Результат:   {label.upper()}")
    print(f"  Уверенность: {prob:.1%}")
    print(f"{'─' * 55}")

    if label == "deepfake":
        print(f"\n  ⚠️  ВОЗМОЖНЫЙ ДИПФЕЙК")
        print(f"     Изображение может быть синтетически сгенерировано.")
        print(f"     Рекомендуется дополнительная экспертиза.")
    else:
        print(f"\n  ✓  Признаков синтетического происхождения не обнаружено.")

    print(f"\n  Grad-CAM сохранён: {out_path}")
    print(f"{'=' * 55}\n")

    return {"label": label, "confidence": prob, "cam": cam}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python predict.py <путь_к_изображению.jpg>")
        sys.exit(1)
    predict_image(sys.argv[1])
