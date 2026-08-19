import uuid
from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base

class Employee(Base):
    __tablename__ = "employees"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    department = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    face_profiles = relationship("FaceProfile", back_populates="employee", cascade="all, delete-orphan")
    recognition_events = relationship("RecognitionEvent", back_populates="employee")
    checkin_events = relationship("CheckinEvent", back_populates="employee")
