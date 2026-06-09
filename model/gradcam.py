"""
model/gradcam.py
----------------
Grad-CAM (Gradient-weighted Class Activation Mapping) для детектора дипфейков.

Grad-CAM показывает какие области изображения повлияли на решение модели.
В контексте детекции дипфейков — это зоны где модель нашла артефакты генерации.

Алгоритм (Selvaraju et al., 2017):
  1. Делаем forward pass, сохраняем активации последнего conv блока
  2. Делаем backward pass от предсказанного класса
  3. Усредняем градиенты по пространственным измерениям (global avg pooling)
  4. Взвешиваем активации этими коэффициентами
  5. ReLU(∑ alpha_k * A_k) → upscale до размера входного изображения
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model.detector import DeepfakeDetector


def compute_gradcam(
    model: DeepfakeDetector,
    spatial_x: torch.Tensor,
    freq_x: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    Вычисляет тепловую карту Grad-CAM для одного изображения.

    Параметры:
        model     — обученная модель DeepfakeDetector
        spatial_x — предобработанный тензор изображения (1, 3, 224, 224)
        freq_x    — частотные признаки (1, 256)
        device    — устройство вычислений

    Возвращает:
        cam — numpy array (224, 224), значения [0, 1]
              Высокие значения = зоны артефактов дипфейка
    """
    model.eval()
    spatial_x = spatial_x.to(device).requires_grad_(True)
    freq_x = freq_x.to(device)

    # Forward pass — хуки сохранят активации
    logit = model(spatial_x, freq_x)

    # Backward pass — хуки сохранят градиенты
    model.zero_grad()
    logit.backward()

    # Получаем сохранённые активации и градиенты
    acts = model.activations.detach()  # (1, C, 7, 7) — активации conv блока
    grads = model.gradients.detach()  # (1, C, 7, 7) — градиенты

    # Global Average Pooling градиентов → веса важности каналов
    alpha = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

    # Взвешенная сумма активационных карт
    cam = torch.relu((alpha * acts).sum(dim=1, keepdim=True))  # (1, 1, 7, 7)

    # Апстемплинг до размера входного изображения (224×224)
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Нормализация в [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32)


def overlay_heatmap(
    cam: np.ndarray,
    img_rgb: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Накладывает тепловую карту Grad-CAM на оригинальное изображение.

    Параметры:
        cam     — тепловая карта (H, W), значения [0, 1]
        img_rgb — оригинальное изображение (H, W, 3), uint8
        alpha   — прозрачность тепловой карты (0=невидима, 1=полная)

    Возвращает:
        overlay — изображение с наложенной тепловой картой (H, W, 3), uint8
    """
    # Масштабируем cam до размера изображения
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Применяем цветовую карту (синий→зелёный→красный)
    heatmap = np.uint8(255 * cam_resized)
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Линейное наложение
    overlay = (
        img_rgb.astype(np.float32) * (1 - alpha)
        + heatmap_colored.astype(np.float32) * alpha
    ).astype(np.uint8)

    return overlay


def save_gradcam(
    cam: np.ndarray,
    img_rgb: np.ndarray,
    save_path: str,
    alpha: float = 0.4,
):
    """Сохраняет изображение с наложенной тепловой картой Grad-CAM."""
    overlay = overlay_heatmap(cam, img_rgb, alpha)
    # Сохраняем через cv2 (BGR)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
