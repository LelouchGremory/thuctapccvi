import os
import cv2
import logging
from datetime import datetime
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.system_backlog import SystemBacklog

logger = logging.getLogger(__name__)

class StrictErrorTracker:
    """
    Zero-Silent-Drop Strict Error Tracker:
    Saves failed crop images to storage/failed/ and records detailed failure logs in SystemBacklog.
    """

    def __init__(self):
        settings.setup_directories()

    def log_failure(
        self,
        db: Session,
        failure_stage: str,
        failure_reason: str,
        image_crop: cv2.typing.MatLike = None,
        track_id: str = None,
        model_name: str = "YuNet/SFace",
        model_version: str = "1.0",
        active_ai_combo: str = settings.ACTIVE_AI_COMBO,
        camera_id: str = "CAM01",
        confidence_score: float = None,
        sub_folder: str = "recognition"
    ) -> SystemBacklog:
        image_path = None
        if image_crop is not None and image_crop.size > 0:
            try:
                target_dir = os.path.join(settings.FAILED_DIR, sub_folder)
                os.makedirs(target_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"fail_{failure_stage}_{timestamp}.jpg"
                image_path = os.path.join(target_dir, filename)
                cv2.imwrite(image_path, image_crop)
            except Exception as e:
                logger.error(f"StrictErrorTracker: Error saving failed crop: {e}")

        try:
            backlog_entry = SystemBacklog(
                track_id=track_id,
                failure_stage=failure_stage,
                failure_reason=failure_reason,
                model_name=model_name,
                model_version=model_version,
                active_ai_combo=active_ai_combo,
                camera_id=camera_id,
                image_path=image_path,
                confidence_score=confidence_score
            )
            db.add(backlog_entry)
            db.commit()
            db.refresh(backlog_entry)
            logger.info(f"StrictErrorTracker Logged: [{failure_stage}] {failure_reason} (track_id={track_id})")
            return backlog_entry
        except Exception as e:
            db.rollback()
            logger.error(f"StrictErrorTracker: Error inserting into SystemBacklog DB: {e}")
            return None

strict_tracker = StrictErrorTracker()
