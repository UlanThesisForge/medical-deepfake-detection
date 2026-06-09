"""
backend/main.py — точка входа FastAPI приложения
Запуск: uvicorn backend.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.database import engine
from backend.models.db_models import Base
from backend.routes import analysis, auth


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.GRADCAM_DIR, exist_ok=True)
    print("✓ База данных инициализирована")
    yield


app = FastAPI(
    title="DeepfakeMedical Detection API",
    description="API для обнаружения дипфейков в медицинских изображениях. EfficientNet-B4 + FFT Dual-Branch.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(analysis.router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "model": "EfficientNet-B4 + FFT Dual-Branch",
    }
