"""
backend/routes/auth.py — авторизация
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from backend.config import settings
from backend.database import get_db
from backend.models.db_models import Investigator, UserSession

router = APIRouter(prefix="/auth", tags=["Авторизация"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    organization: str = ""
    role: str = "analyst"


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/login")
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Investigator).where(Investigator.email == data.email)
    )
    user = result.scalar_one_or_none()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(401, "Неверный email или пароль")
    if not user.is_active:
        raise HTTPException(403, "Аккаунт деактивирован")

    user.last_login = datetime.utcnow()
    access = create_access_token(str(user.investigator_id), user.role)
    refresh = create_refresh_token(str(user.investigator_id))

    db.add(
        UserSession(
            investigator_id=user.investigator_id,
            refresh_token=refresh,
            expires_at=datetime.utcnow()
            + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
    )

    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "user_id": str(user.investigator_id),
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role,
        "organization": user.organization,
    }


@router.post("/register")
async def register(data: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(
        select(Investigator).where(Investigator.email == data.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Email уже используется")

    user = Investigator(
        full_name=data.full_name,
        email=data.email,
        password_hash=hash_password(data.password),
        organization=data.organization,
        role=data.role if data.role in ("analyst", "viewer") else "analyst",
    )
    db.add(user)
    await db.flush()

    access = create_access_token(str(user.investigator_id), user.role)
    refresh = create_refresh_token(str(user.investigator_id))
    db.add(
        UserSession(
            investigator_id=user.investigator_id,
            refresh_token=refresh,
            expires_at=datetime.utcnow()
            + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
    )
    return {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "user_id": str(user.investigator_id),
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role,
        "organization": user.organization,
    }


@router.post("/refresh")
async def refresh(data: RefreshRequest, db: AsyncSession = Depends(get_db)):
    payload = decode_token(data.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(401, "Неверный тип токена")

    sess_res = await db.execute(
        select(UserSession).where(UserSession.refresh_token == data.refresh_token)
    )
    session = sess_res.scalar_one_or_none()
    if not session:
        raise HTTPException(401, "Токен не найден или отозван")
    await db.delete(session)

    user_res = await db.execute(
        select(Investigator).where(Investigator.investigator_id == payload["sub"])
    )
    user = user_res.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(401, "Пользователь не найден")

    access = create_access_token(str(user.investigator_id), user.role)
    refresh_new = create_refresh_token(str(user.investigator_id))
    db.add(
        UserSession(
            investigator_id=user.investigator_id,
            refresh_token=refresh_new,
            expires_at=datetime.utcnow()
            + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
    )
    return {
        "access_token": access,
        "refresh_token": refresh_new,
        "token_type": "bearer",
        "user_id": str(user.investigator_id),
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role,
    }


@router.post("/logout")
async def logout(data: RefreshRequest, db: AsyncSession = Depends(get_db)):
    res = await db.execute(
        select(UserSession).where(UserSession.refresh_token == data.refresh_token)
    )
    s = res.scalar_one_or_none()
    if s:
        await db.delete(s)
    return {"message": "Выход выполнен"}


@router.get("/me")
async def me(user: Investigator = Depends(get_current_user)):
    return {
        "user_id": str(user.investigator_id),
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role,
        "organization": user.organization,
    }
