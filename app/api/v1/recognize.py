import os
import cv2
import logging
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.config import settings
from app.db.session import get_db
from app.models.recognition_event import RecognitionEvent
from app.models.checkin_event import CheckinEvent
from app.services.detector import FaceDetectorCascade
from app.services.quality_gate import QualityGate
from app.services.embedding import FeatureEmbedder
from app.services.matcher import MatchingEngine
from app.services.strict_tracker import strict_tracker
from app.services.anti_duplicate import AntiDuplicateManager

logger = logging.getLogger(__name__)
router = APIRouter()

detector = FaceDetectorCascade()
quality_gate = QualityGate()
embedder = FeatureEmbedder()
matcher = MatchingEngine()
anti_duplicate = AntiDuplicateManager()

class EmployeeMatchData(BaseModel):
    employee_id: str
    employee_code: str
    name: str
    department: Optional[str] = None

class MatchResponseItem(BaseModel):
    face_index: int
    bbox: List[int]
    confidence: float
    status: str # RECOGNIZED, UNCERTAIN, UNKNOWN, REJECTED
    similarity: float
    employee: Optional[EmployeeMatchData] = None
    reason: str
    multi_frame_confirmed: Optional[bool] = False

class RecognizeResponse(BaseModel):
    face_count: int
    results: List[MatchResponseItem]

@router.post("/recognize", response_model=RecognizeResponse, summary="Recognize face & 4-State Classification (Week 5)")
async def recognize_face(
    image: UploadFile = File(..., description="Camera captured frame"),
    track_id: Optional[str] = Form(None, description="ByteTrack persistent Track ID"),
    camera_id: Optional[str] = Form("CAM01", description="Camera identifier"),
    db: Session = Depends(get_db)
):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        logger.error(f"Image read error: {e}")
        raise HTTPException(status_code=400, detail="Could not read image file.")

    faces = detector.detect(img)
    if not faces:
        strict_tracker.log_failure(
            db=db,
            failure_stage="face_detection",
            failure_reason="No face detected in frame",
            image_crop=img,
            track_id=track_id,
            camera_id=camera_id
        )
        return RecognizeResponse(face_count=0, results=[])

    results = []
    for i, face_info in enumerate(faces):
        x, y, w, h = face_info["bbox"]
        h_img, w_img = img.shape[:2]
        crop = img[max(0, y):min(h_img, y+h), max(0, x):min(w_img, x+w)]

        # 1. Quality Gate Check
        passed, reason, _ = quality_gate.evaluate(img, face_info)
        if not passed:
            strict_tracker.log_failure(
                db=db,
                failure_stage="quality_gate",
                failure_reason=reason,
                image_crop=crop,
                track_id=track_id,
                camera_id=camera_id,
                confidence_score=face_info["confidence"]
            )
            results.append(MatchResponseItem(
                face_index=i + 1,
                bbox=face_info["bbox"],
                confidence=face_info["confidence"],
                status="REJECTED",
                similarity=0.0,
                employee=None,
                reason=reason
            ))
            continue

        # 2. Alignment & Feature Extraction
        aligned = quality_gate.align_and_crop(img, face_info)
        vector = embedder.extract(aligned)

        # 3. Matching Engine (RAM Cache)
        match_info = matcher.match(query_vector=vector, db=db, track_id=track_id, face_crop=crop)

        # 4. Multi-frame Voting
        if track_id:
            match_info = matcher.multi_frame_voting(track_id=track_id, current_result=match_info)

        status = match_info["status"]
        similarity = match_info["similarity"]
        emp_info = match_info.get("employee")

        # Save crop image for event logging
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        crop_path = os.path.join(settings.CAPTURES_DIR, f"rec_{status}_{timestamp_str}.jpg")
        try:
            cv2.imwrite(crop_path, crop)
        except Exception:
            crop_path = None

        # Log RecognitionEvent
        emp_id = emp_info["employee_id"] if emp_info else None
        event = RecognitionEvent(
            camera_id=None,
            employee_id=emp_id,
            track_id=track_id,
            similarity=similarity,
            status=status,
            crop_image_path=crop_path,
            model_name="YuNet/SFace",
            active_ai_combo=settings.ACTIVE_AI_COMBO
        )
        db.add(event)

        # Checkin Event logging if RECOGNIZED and passed anti-duplicate cooldown
        if status == "RECOGNIZED" and emp_id:
            if anti_duplicate.should_allow_checkin(emp_id):
                checkin = CheckinEvent(
                    employee_id=emp_id,
                    track_id=track_id,
                    confidence=similarity,
                    image_path=crop_path
                )
                db.add(checkin)

        db.commit()

        emp_model = EmployeeMatchData(
            employee_id=emp_info["employee_id"],
            employee_code=emp_info["employee_code"],
            name=emp_info["name"],
            department=emp_info.get("department")
        ) if emp_info else None

        results.append(MatchResponseItem(
            face_index=i + 1,
            bbox=face_info["bbox"],
            confidence=face_info["confidence"],
            status=status,
            similarity=similarity,
            employee=emp_model,
            reason=match_info["reason"],
            multi_frame_confirmed=match_info.get("multi_frame_confirmed", False)
        ))

    return RecognizeResponse(face_count=len(faces), results=results)
