import uuid
from sqlalchemy import Column, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base

class RecognitionEvent(Base):
    __tablename__ = "recognition_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String(36), ForeignKey("cameras.id"), nullable=True)
    employee_id = Column(String(36), ForeignKey("employees.id"), nullable=True)
    track_id = Column(String(50), nullable=True)
    similarity = Column(Float, nullable=True)
    status = Column(String(20), nullable=False) # RECOGNIZED, UNCERTAIN, UNKNOWN, REJECTED
    crop_image_path = Column(String(255), nullable=True)
    model_name = Column(String(50), nullable=True)
    active_ai_combo = Column(String(50), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    camera = relationship("Camera", back_populates="recognition_events")
    employee = relationship("Employee", back_populates="recognition_events")
