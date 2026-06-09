"""
model/detector.py
-----------------
Двухветвевая модель обнаружения дипфейков в медицинских изображениях.

Архитектура:
  Ветвь 1 (пространственная):
    EfficientNet-B4 (предобученный на ImageNet) → 1792-мерный вектор признаков

  Ветвь 2 (частотная):
    FFT 256-мерный вектор → Dense(512) → Dense(512)

  Fusion (слияние):
    Attention-weighted concatenation → классификатор

  Выход:
    Скалярное значение logit → sigmoid → вероятность того что изображение СИНТЕТИЧЕСКОЕ

Обе ветви обучаются совместно через backpropagation.
Grad-CAM применяется к последнему свёрточному блоку EfficientNet-B4.
"""

import timm
import torch
import torch.nn as nn


class DeepfakeDetector(nn.Module):
    """
    Двухветвевой детектор дипфейков:
      - Пространственная ветвь (EfficientNet-B4)
      - Частотная ветвь (MLP на FFT признаках)
    """

    def __init__(self, freq_input_dim: int = 256, dropout: float = 0.5):
        super().__init__()

        # ── Ветвь 1: EfficientNet-B4 backbone ────────────────────────────────
        # pretrained=True загружает веса ImageNet
        # num_classes=0 убирает финальный классификатор → получаем feature vector
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=True,
            num_classes=0,
        )
        spatial_dim = self.backbone.num_features  # 1792

        # ── Ветвь 2: Частотный MLP ────────────────────────────────────────────
        # Три слоя с BatchNorm и GELU активацией
        self.freq_branch = nn.Sequential(
            nn.Linear(freq_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
        )

        # ── Слияние с обучаемым вниманием ─────────────────────────────────────
        # Веса внимания определяют вклад каждой ветви
        self.attention = nn.Linear(spatial_dim + 512, 2)

        # ── Классификатор ─────────────────────────────────────────────────────
        # Входная размерность: spatial(1792) + freq(512) + поэлементное произведение(512)
        # Но мы ограничиваем поэлементное произведение до 512 (min из двух ветвей)
        fused_dim = spatial_dim + 512 + 512
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1),
        )

        # ── Хранилище для Grad-CAM (forward/backward hooks) ───────────────────
        self.gradients = None  # градиенты в последнем conv блоке
        self.activations = None  # активации в последнем conv блоке
        self._register_gradcam_hooks()

    def _register_gradcam_hooks(self):
        """
        Регистрируем хуки на последний свёрточный блок EfficientNet-B4.
        Совместимо с timm >= 0.9 и >= 1.0
        """
        # Ищем последний блок с поддержкой hooks
        last_block = None
        try:
            # timm < 1.0
            last_block = list(self.backbone.blocks.children())[-1]
        except Exception:
            # timm >= 1.0: используем named_modules
            for name, module in self.backbone.named_modules():
                if hasattr(module, "conv_pwl") or hasattr(module, "conv_dw"):
                    last_block = module
            if last_block is None:
                # fallback: последний модуль с параметрами
                for module in self.backbone.modules():
                    if hasattr(module, "weight") and len(list(module.children())) == 0:
                        last_block = module

        if last_block is None:
            raise RuntimeError("Не удалось найти блок для Grad-CAM hooks")

        # Forward hook: сохраняем активации
        last_block.register_forward_hook(
            lambda m, inp, out: setattr(self, "activations", out)
        )

        # Backward hook: сохраняем градиенты
        last_block.register_full_backward_hook(
            lambda m, grad_in, grad_out: setattr(self, "gradients", grad_out[0])
        )

    def forward(self, spatial_x: torch.Tensor, freq_x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.

        Параметры:
            spatial_x — тензор изображений (B, 3, 224, 224) нормализованный ImageNet
            freq_x    — тензор FFT признаков (B, 256)

        Возвращает:
            logits — тензор (B,), необработанные логиты (до sigmoid)
                     Для бинарной метки: 1 = deepfake, 0 = authentic
        """
        # Пространственные признаки через EfficientNet-B4
        sp_feat = self.backbone(spatial_x)  # (B, 1792)

        # Частотные признаки через MLP
        fr_feat = self.freq_branch(freq_x)  # (B, 512)

        # Вычисляем веса внимания для каждой ветви
        combined = torch.cat([sp_feat, fr_feat], dim=1)  # (B, 2304)
        attn_w = torch.softmax(self.attention(combined), dim=1)  # (B, 2)

        # Взвешенные признаки
        sp_weighted = attn_w[:, 0:1] * sp_feat  # (B, 1792)
        fr_weighted = attn_w[:, 1:2] * fr_feat  # (B, 512)

        # Поэлементное произведение (cross-modal interaction)
        # Берём первые 512 из пространственных (sp_feat может быть 1792-dim)
        cross = sp_feat[:, :512] * fr_feat  # (B, 512)

        # Конкатенируем все три представления
        fused = torch.cat([sp_weighted, fr_weighted, cross], dim=1)  # (B, 2816)

        # Финальная классификация
        return self.classifier(fused).squeeze(1)  # (B,)


class FocalLoss(nn.Module):
    """
    Focal Loss для несбалансированной классификации.

    Уменьшает штраф за лёгкие примеры (высокая уверенность модели)
    и увеличивает для сложных. Хорошо работает при дисбалансе классов.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    gamma=2.0 — стандартное значение для медицинских задач (из статьи Lin et al., 2017).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Параметры:
            logits  — необработанные логиты (B,), до sigmoid
            targets — бинарные метки (B,), float32: 0.0 = authentic, 1.0 = deepfake
        """
        # Вероятности через sigmoid
        probs = torch.sigmoid(logits)

        # Бинарная cross-entropy (поэлементно)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t — вероятность правильного класса
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: (1-p_t)^gamma — чем увереннее правильный ответ, тем меньше вес
        focal_weight = self.alpha * torch.pow(1.0 - p_t, self.gamma)

        return (focal_weight * bce).mean()
