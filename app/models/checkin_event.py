import uuid
from sqlalchemy import Column, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base

class CheckinEvent(Base):
    __tablename__ = "checkin_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String(36), ForeignKey("employees.id"), nullable=False)
    camera_id = Column(String(36), ForeignKey("cameras.id"), nullable=True)
    track_id = Column(String(50), nullable=True)
    checkin_time = Column(DateTime(timezone=True), server_default=func.now())
    confidence = Column(Float, nullable=True)
    image_path = Column(String(255), nullable=True)

    employee = relationship("Employee", back_populates="checkin_events")
    camera = relationship("Camera", back_populates="checkin_events")
