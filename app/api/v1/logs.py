from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.session import get_db
from app.models.recognition_event import RecognitionEvent
from app.models.checkin_event import CheckinEvent
from app.models.system_backlog import SystemBacklog

router = APIRouter()

class RecognitionLogSchema(BaseModel):
    id: str
    track_id: Optional[str]
    similarity: Optional[float]
    status: str
    crop_image_path: Optional[str]
    active_ai_combo: Optional[str]

class CheckinLogSchema(BaseModel):
    id: str
    employee_id: str
    track_id: Optional[str]
    confidence: Optional[float]

class BacklogLogSchema(BaseModel):
    id: str
    track_id: Optional[str]
    failure_stage: str
    failure_reason: str
    model_name: Optional[str]
    active_ai_combo: Optional[str]
    image_path: Optional[str]
    confidence_score: Optional[float]

@router.get("/logs/recognition", response_model=List[RecognitionLogSchema], summary="GET recognition events log")
def get_recognition_logs(limit: int = Query(50, le=200), db: Session = Depends(get_db)):
    events = db.query(RecognitionEvent).order_by(RecognitionEvent.timestamp.desc()).limit(limit).all()
    return [
        RecognitionLogSchema(
            id=e.id,
            track_id=e.track_id,
            similarity=e.similarity,
            status=e.status,
            crop_image_path=e.crop_image_path,
            active_ai_combo=e.active_ai_combo
        ) for e in events
    ]

@router.get("/logs/checkin", response_model=List[CheckinLogSchema], summary="GET checkin events log")
def get_checkin_logs(limit: int = Query(50, le=200), db: Session = Depends(get_db)):
    checkins = db.query(CheckinEvent).order_by(CheckinEvent.checkin_time.desc()).limit(limit).all()
    return [
        CheckinLogSchema(
            id=c.id,
            employee_id=c.employee_id,
            track_id=c.track_id,
            confidence=c.confidence
        ) for c in checkins
    ]

@router.get("/logs/backlog", response_model=List[BacklogLogSchema], summary="GET SystemBacklog strict error tracking log")
def get_system_backlog_logs(limit: int = Query(50, le=200), db: Session = Depends(get_db)):
    backlogs = db.query(SystemBacklog).order_by(SystemBacklog.timestamp.desc()).limit(limit).all()
    return [
        BacklogLogSchema(
            id=b.id,
            track_id=b.track_id,
            failure_stage=b.failure_stage,
            failure_reason=b.failure_reason,
            model_name=b.model_name,
            active_ai_combo=b.active_ai_combo,
            image_path=b.image_path,
            confidence_score=b.confidence_score
        ) for b in backlogs
    ]
