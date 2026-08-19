import uuid
from sqlalchemy import Column, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    source = Column(String(255), nullable=False)
    location = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    recognition_events = relationship("RecognitionEvent", back_populates="camera")
    checkin_events = relationship("CheckinEvent", back_populates="camera")
