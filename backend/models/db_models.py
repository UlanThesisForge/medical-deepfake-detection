"""
backend/models/db_models.py — SQLAlchemy ORM модели
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ModelVersion(Base):
    __tablename__ = "model_version"
    model_version_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    architecture = Column(String(100), nullable=False)
    version_number = Column(String(20), nullable=False)
    auc_roc = Column(Numeric(5, 4))
    accuracy = Column(Numeric(5, 4))
    description = Column(Text)
    is_active = Column(Boolean, default=False)
    deployed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    results = relationship("AnalysisResult", back_populates="model_version")


class Investigator(Base):
    __tablename__ = "investigator"
    investigator_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String(200), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="analyst")
    organization = Column(String(200))
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    images = relationship("SubmittedImage", back_populates="investigator")
    sessions = relationship("UserSession", back_populates="investigator")


class SubmittedImage(Base):
    __tablename__ = "submitted_image"
    image_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    investigator_id = Column(
        UUID(as_uuid=True), ForeignKey("investigator.investigator_id"), nullable=True
    )
    original_filename = Column(String(255))
    file_path = Column(String(500), nullable=False)
    file_format = Column(String(10), default="jpg")
    file_size_kb = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processing_status = Column(String(20), default="pending")
    investigator = relationship("Investigator", back_populates="images")
    result = relationship("AnalysisResult", back_populates="image", uselist=False)


class AnalysisResult(Base):
    __tablename__ = "analysis_result"
    result_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey("submitted_image.image_id"))
    model_version_id = Column(
        UUID(as_uuid=True), ForeignKey("model_version.model_version_id"), nullable=True
    )
    label = Column(String(10), nullable=False)
    confidence_score = Column(Numeric(5, 4), nullable=False)
    heatmap_path = Column(String(500))
    artefact_summary = Column(JSON)
    processing_time_ms = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    image = relationship("SubmittedImage", back_populates="result")
    model_version = relationship("ModelVersion", back_populates="results")


class UserSession(Base):
    __tablename__ = "user_session"
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    investigator_id = Column(
        UUID(as_uuid=True), ForeignKey("investigator.investigator_id")
    )
    refresh_token = Column(String(500), unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    investigator = relationship("Investigator", back_populates="sessions")
