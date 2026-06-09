"""
model/freq_features.py
----------------------
Извлечение спектральных признаков из медицинских изображений.

Идея: каждая архитектура генеративных сетей оставляет характерный след
в частотном пространстве изображения. GAN-сети с transpose convolution
создают периодические паттерны в спектре Фурье. Диффузионные модели
оставляют другие артефакты.

Мы извлекаем 256-мерный вектор признаков:
  - 32 радиальных кольца × 8 статистик каждое = 256 чисел
  - Статистики: среднее, std, max, min, асимметрия, эксцесс, энергия, энтропия

Это дополняет пространственные признаки EfficientNet-B4.
"""

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.stats import skew


def extract_freq_features(img_rgb: np.ndarray) -> np.ndarray:
    """
    Извлекает 256-мерный вектор частотных признаков из RGB изображения.

    Параметры:
        img_rgb — массив uint8 формата [H, W, 3], значения [0, 255]

    Возвращает:
        features — numpy array формата (256,), dtype float32
    """
    # Переводим в оттенки серого (взвешенная сумма RGB каналов)
    gray = 0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]
    gray = gray / 255.0  # нормализуем в [0, 1]

    # 2D быстрое преобразование Фурье
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)  # центрируем спектр

    # Логарифм амплитудного спектра (подавляем выбросы)
    log_mag = np.log1p(np.abs(fshift))

    H, W = log_mag.shape
    cy, cx = H // 2, W // 2

    # Максимальный радиус (ограничен меньшей стороной изображения)
    max_r = min(cy, cx)

    # Делим частотное пространство на 32 концентрических кольца
    band_edges = np.linspace(0, max_r, 33)

    # Сетка расстояний от центра
    y_idx, x_idx = np.ogrid[-cy : H - cy, -cx : W - cx]
    radii = np.sqrt(x_idx**2 + y_idx**2)

    features = []
    for i in range(32):
        r_min = band_edges[i]
        r_max = band_edges[i + 1]

        # Маска кольца
        mask = (radii >= r_min) & (radii < r_max)
        vals = log_mag[mask]

        if len(vals) == 0:
            vals = np.zeros(1)

        # 8 статистик для каждого кольца
        features.extend(
            [
                float(vals.mean()),  # среднее
                float(vals.std()),  # стандартное отклонение
                float(vals.max()),  # максимум
                float(vals.min()),  # минимум
                float(skew(vals)),  # асимметрия
                float(scipy_kurtosis(vals)),  # эксцесс (островерхость)
                float(np.sum(vals**2)),  # энергия
                float(-np.sum(vals * np.log(vals + 1e-8))),  # энтропия
            ]
        )

    return np.array(features, dtype=np.float32)  # форма (256,)


def batch_freq_features(images: list) -> np.ndarray:
    """
    Извлекает частотные признаки для пакета изображений.

    Параметры:
        images — список массивов uint8 [H, W, 3]

    Возвращает:
        numpy array формата (N, 256)
    """
    return np.stack([extract_freq_features(img) for img in images])


def freq_features_from_tensor(tensor_batch: "torch.Tensor") -> "torch.Tensor":
    """
    Обёртка для PyTorch: принимает батч тензоров (B, 3, H, W) в [0,1]
    и возвращает тензор частотных признаков (B, 256).
    Используется во время обучения для предвычисления признаков.
    """
    import torch

    batch_np = (tensor_batch.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    freq_np = batch_freq_features(list(batch_np))
    return torch.from_numpy(freq_np).float().to(tensor_batch.device)
