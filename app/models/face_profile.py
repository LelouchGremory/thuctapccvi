import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.db.base import Base

class FaceProfile(Base):
    __tablename__ = "face_profiles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String(36), ForeignKey("employees.id"), nullable=False)
    embedding = Column(Vector(512), nullable=False)
    image_path = Column(String(255), nullable=True)
    model_name = Column(String(50), default="sface")
    model_version = Column(String(50), default="1.0")
    active_ai_combo = Column(String(50), default="YUNET_SFACE")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    employee = relationship("Employee", back_populates="face_profiles")
