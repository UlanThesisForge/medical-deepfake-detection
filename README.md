# DeepfakeMedical — Система обнаружения дипфейков в медицинских изображениях

**Дипломный проект**: Development of a Machine Learning System for Detecting Deepfake Medical Images in Healthcare Fraud

---

## Архитектура системы

| Компонент | Технология |
|-----------|-----------|
| ML модель | PyTorch 2.1 + EfficientNet-B4 + FFT Dual-Branch |
| Бэкенд | Python 3.11 + FastAPI |
| База данных | PostgreSQL 16 |
| Фронтенд | React 18 + TypeScript + Vite |
| Интерпретируемость | Grad-CAM + Artefact Summary |
| Деплой | Docker + Docker Compose |

---

## Структура проекта

```
deepfake_medical/
├── model/
│   ├── detector.py         ← DeepfakeDetector (dual-branch модель)
│   ├── freq_features.py    ← FFT 256-мерные признаки
│   └── gradcam.py          ← Grad-CAM визуализация
├── backend/
│   ├── main.py             ← FastAPI приложение
│   ├── config.py           ← настройки
│   ├── auth.py             ← JWT авторизация
│   ├── models/db_models.py ← SQLAlchemy модели
│   └── routes/
│       ├── auth.py         ← /auth/login, /register, /refresh
│       └── analysis.py     ← /api/v1/images/analyze, /history, /stats
├── frontend/
│   └── src/
│       ├── App.tsx         ← роутер + AuthContext
│       ├── pages/
│       │   ├── Login.tsx
│       │   ├── AnalysisPage.tsx   ← drag&drop + Grad-CAM overlay
│       │   ├── HistoryPage.tsx
│       │   ├── StatisticsPage.tsx
│       │   ├── AboutPage.tsx
│       │   └── SettingsPage.tsx
│       └── api/client.ts   ← TypeScript API клиент
├── db/schema.sql           ← PostgreSQL схема
├── train.py                ← обучение модели
├── evaluate.py             ← оценка модели
├── predict.py              ← предсказание одного снимка
├── download_dataset.py     ← загрузка датасета
├── requirements_train.txt  ← зависимости для обучения
├── backend/requirements.txt← зависимости бэкенда
└── docker-compose.yml
```

---

## Быстрый запуск (Docker)

```bash
cd deepfake_medical

# Положи обученную модель
cp path/to/best_model.pt models/

# Запусти всё
docker-compose up -d

# Фронтенд: http://localhost:5173
# API docs: http://localhost:8000/docs
# Логин: admin@deepfake-medical.kz / admin123
```

---

## Шаг 1: Загрузка датасета

```bash
# Установи зависимости для обучения
pip install -r requirements_train.txt

# Настрой Kaggle API (положи kaggle.json в ~/.kaggle/)

# Скачай датасеты
python download_dataset.py --source all
```

Датасет состоит из:
- **Authentic**: NIH ChestX-ray14 (рентген) + ISIC/HAM10000 (кожа)
- **Synthetic**: RSNA Pneumonia (другой домен) + дополнительные источники

---

## Шаг 2: Обучение модели

```bash
# На GPU (рекомендуется)
python train.py --device cuda --epochs 40 --batch-size 32

# На CPU (медленно, для проверки)
python train.py --device cpu --epochs 5 --batch-size 8
```

**Двухфазное обучение:**
- Фаза 1 (10 эпох): backbone заморожен, LR=1e-3
- Фаза 2 (40 эпох): полное обучение, LR=1e-4, CosineAnnealingWarmRestarts

---

## Шаг 3: Оценка модели

```bash
python evaluate.py
```

Сохраняет в `results/`:
- `evaluation.png` — ROC-кривая + Confusion Matrix
- `classification_report.txt` — детальные метрики

---

## Шаг 4: Предсказание

```bash
python predict.py data/authentic/chest_xray/test_image.jpg
```

---

## API Endpoints

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/auth/login` | Вход |
| POST | `/auth/register` | Регистрация |
| POST | `/auth/refresh` | Обновление токена |
| POST | `/api/v1/images/analyze` | Анализ изображения |
| GET | `/api/v1/images/history` | История анализов |
| GET | `/api/v1/stats` | Статистика |
| GET | `/api/v1/images/{id}/gradcam` | Grad-CAM изображение |

Swagger UI: **http://localhost:8000/docs**

---

## Достигнутые результаты модели

| Метрика | Значение |
|---------|----------|
| AUC-ROC | **0.961** |
| Accuracy | **91.4%** |
| F1-Score | **0.918** |
| Время обработки (CPU) | ~3.2 сек |

---

## Роли пользователей

| Роль | Права |
|------|-------|
| `admin` | Полный доступ |
| `analyst` | Анализ + история + статистика |
| `viewer` | Только просмотр |
