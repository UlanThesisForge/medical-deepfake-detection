"""
backend/routes/analysis.py — эндпоинты для анализа изображений
"""

import os
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from torchvision import transforms

from backend.auth import get_current_user
from backend.config import settings
from backend.database import get_db
from backend.models.db_models import (
    AnalysisResult,
    Investigator,
    ModelVersion,
    SubmittedImage,
)

# Добавляем корневую папку проекта в path для импорта model/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(prefix="/api/v1", tags=["Анализ"])

# Модель загружается один раз
_model = None

IMG_SIZE = 224
IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_model():
    global _model
    if _model is None and os.path.exists(settings.MODEL_PATH):
        from model.detector import DeepfakeDetector

        _model = DeepfakeDetector()
        _model.load_state_dict(torch.load(settings.MODEL_PATH, map_location="cpu"))
        _model.eval()
        print(f"✓ Модель загружена: {settings.MODEL_PATH}")
    return _model


def run_inference(img_bytes: bytes):
    """Запускает полный inference pipeline для одного изображения."""
    from model.freq_features import extract_freq_features
    from model.gradcam import compute_gradcam, overlay_heatmap

    model = get_model()
    if model is None:
        raise HTTPException(503, "Модель не загружена. Запустите python train.py")

    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))

    # Пространственный тензор
    spatial = IMG_TRANSFORM(img).unsqueeze(0)

    # Частотные признаки
    freq_np = extract_freq_features(img_np)
    freq = torch.from_numpy(freq_np).unsqueeze(0)

    # Forward pass
    start_ms = time.time()
    with torch.no_grad():
        logit = model(spatial, freq)
    prob = torch.sigmoid(logit).item()
    ms = int((time.time() - start_ms) * 1000)

    label = "deepfake" if prob >= settings.DECISION_THRESHOLD else "authentic"

    # Grad-CAM
    cam = compute_gradcam(model, spatial, freq)
    overlay = overlay_heatmap(cam, img_np)

    return label, prob, cam, overlay, img_np, ms


def generate_artefact_summary(cam: np.ndarray) -> dict:
    """Анализирует тепловую карту Grad-CAM и описывает артефакты."""
    h, w = cam.shape
    zones = {
        "top_left": cam[: h // 2, : w // 2].mean(),
        "top_right": cam[: h // 2, w // 2 :].mean(),
        "bottom_left": cam[h // 2 :, : w // 2].mean(),
        "bottom_right": cam[h // 2 :, w // 2 :].mean(),
        "center": cam[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean(),
    }
    primary = [k for k, v in zones.items() if v > 0.5]
    level = "high" if cam.max() > 0.7 else "medium" if cam.max() > 0.4 else "low"

    return {
        "primary_regions": primary or ["distributed"],
        "activation_level": level,
        "max_activation": float(cam.max()),
        "mean_activation": float(cam.mean()),
        "frequency_signature": "periodic_artefacts"
        if cam.std() > 0.3
        else "diffuse_noise",
    }


# ── Анализ одного изображения ──────────────────────────────────────────────
@router.post("/images/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: Investigator = Depends(get_current_user),
):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Поддерживаются только JPEG и PNG")

    img_bytes = await file.read()
    if len(img_bytes) > 50 * 1024 * 1024:
        raise HTTPException(413, "Файл превышает 50 МБ")

    # Сохраняем оригинал
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.GRADCAM_DIR, exist_ok=True)

    image_id = str(uuid.uuid4())
    ext = Path(file.filename or "img.jpg").suffix or ".jpg"
    file_path = os.path.join(settings.UPLOAD_DIR, f"{image_id}{ext}")
    with open(file_path, "wb") as f:
        f.write(img_bytes)

    # Запись в БД
    img_record = SubmittedImage(
        image_id=image_id,
        investigator_id=user.investigator_id,
        original_filename=file.filename,
        file_path=file_path,
        file_format=ext.lstrip("."),
        file_size_kb=len(img_bytes) // 1024,
        processing_status="processing",
    )
    db.add(img_record)
    await db.flush()

    # Inference
    label, prob, cam, overlay, img_np, ms = run_inference(img_bytes)

    # Сохраняем Grad-CAM
    gradcam_path = os.path.join(settings.GRADCAM_DIR, f"gradcam_{image_id}.jpg")
    cv2.imwrite(gradcam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Артефакт-саммари
    summary = generate_artefact_summary(cam)

    # Получаем активную версию модели
    mv_res = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True)
    )
    mv = mv_res.scalar_one_or_none()

    result = AnalysisResult(
        image_id=image_id,
        model_version_id=mv.model_version_id if mv else None,
        label=label,
        confidence_score=float(prob),
        heatmap_path=gradcam_path,
        artefact_summary=summary,
        processing_time_ms=ms,
    )
    db.add(result)
    img_record.processing_status = "done"

    return {
        "job_id": str(result.result_id),
        "image_id": image_id,
        "label": label,
        "confidence": float(prob),
        "artefact_summary": summary,
        "heatmap_url": f"/api/v1/images/{image_id}/gradcam",
        "image_url": f"/api/v1/images/{image_id}/file",
        "processing_ms": ms,
        "model_version": mv.version_number if mv else "N/A",
    }


# ── Получить изображение и Grad-CAM ───────────────────────────────────────
@router.get("/images/{image_id}/gradcam")
async def get_gradcam(image_id: str):
    path = os.path.join(settings.GRADCAM_DIR, f"gradcam_{image_id}.jpg")
    if not os.path.exists(path):
        raise HTTPException(404, "Grad-CAM не найден")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/images/{image_id}/file")
async def get_image_file(image_id: str):
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(settings.UPLOAD_DIR, f"{image_id}{ext}")
        if os.path.exists(path):
            return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(404, "Изображение не найдено")


# ── История анализов ────────────────────────────────────────────────────────
@router.get("/images/history")
async def get_history(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user: Investigator = Depends(get_current_user),
):
    result = await db.execute(
        select(AnalysisResult, SubmittedImage)
        .join(SubmittedImage)
        .where(SubmittedImage.investigator_id == user.investigator_id)
        .order_by(desc(AnalysisResult.created_at))
        .limit(limit)
        .offset(offset)
    )
    rows = result.all()
    return [
        {
            "result_id": str(r.result_id),
            "image_id": str(r.image_id),
            "label": r.label,
            "confidence": float(r.confidence_score),
            "heatmap_url": f"/api/v1/images/{str(r.image_id)}/gradcam",
            "image_url": f"/api/v1/images/{str(r.image_id)}/file",
            "filename": img.original_filename,
            "created_at": r.created_at.isoformat(),
            "processing_ms": r.processing_time_ms,
        }
        for r, img in rows
    ]


# ── Статистика ──────────────────────────────────────────────────────────────
@router.get("/stats")
async def get_stats(
    db: AsyncSession = Depends(get_db),
    user: Investigator = Depends(get_current_user),
):
    from sqlalchemy import func

    total = await db.execute(
        select(func.count(AnalysisResult.result_id))
        .join(SubmittedImage)
        .where(SubmittedImage.investigator_id == user.investigator_id)
    )
    by_label = await db.execute(
        select(AnalysisResult.label, func.count())
        .join(SubmittedImage)
        .where(SubmittedImage.investigator_id == user.investigator_id)
        .group_by(AnalysisResult.label)
    )
    avg_ms = await db.execute(
        select(func.avg(AnalysisResult.processing_time_ms))
        .join(SubmittedImage)
        .where(SubmittedImage.investigator_id == user.investigator_id)
    )
    mv_res = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True)
    )
    mv = mv_res.scalar_one_or_none()

    return {
        "total": total.scalar() or 0,
        "by_label": {row[0]: row[1] for row in by_label.all()},
        "avg_processing_ms": int(avg_ms.scalar() or 0),
        "model": {
            "architecture": mv.architecture if mv else "N/A",
            "version": mv.version_number if mv else "N/A",
            "auc_roc": float(mv.auc_roc) if mv else 0,
            "accuracy": float(mv.accuracy) if mv else 0,
        }
        if mv
        else None,
    }
