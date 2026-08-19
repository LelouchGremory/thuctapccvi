import uuid
from sqlalchemy import Column, String, Float, DateTime
from sqlalchemy.sql import func
from app.db.base import Base

class SystemBacklog(Base):
    """
    Strict Error Tracking Backlog Table:
    Ghi nhận đầy đủ các trường hợp bị từ chối (REJECTED), nghi ngờ (UNCERTAIN),
    hoặc lỗi ở từng giai đoạn của Pipeline.
    """
    __tablename__ = "system_backlog"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    track_id = Column(String(50), nullable=True)
    failure_stage = Column(String(50), nullable=False) # camera, person_detection, tracking, face_detection, quality_gate, embedding, matching, checkin, database
    failure_reason = Column(String(255), nullable=False)
    model_name = Column(String(50), nullable=True)
    model_version = Column(String(50), nullable=True)
    active_ai_combo = Column(String(50), nullable=True)
    camera_id = Column(String(50), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    image_path = Column(String(255), nullable=True)
    confidence_score = Column(Float, nullable=True)
