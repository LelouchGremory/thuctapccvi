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
from app.models.employee import Employee
from app.models.face_profile import FaceProfile
from app.services.detector import FaceDetectorCascade
from app.services.quality_gate import QualityGate
from app.services.embedding import FeatureEmbedder
from app.services.strict_tracker import strict_tracker

logger = logging.getLogger(__name__)
router = APIRouter()

detector = FaceDetectorCascade()
quality_gate = QualityGate()
embedder = FeatureEmbedder()

class ErrorDetail(BaseModel):
    failure_stage: str
    failure_reason: str
    image_path: Optional[str] = None

class EnrollResponse(BaseModel):
    success: bool
    message: str
    employee_code: str
    name: str
    valid_images_count: int
    failed_images_count: int
    error_details: List[ErrorDetail] = []

@router.post("/enroll", response_model=EnrollResponse, summary="Enroll new employee face (Week 4)")
async def enroll_employee(
    employee_code: str = Form(..., description="Unique Employee Code (e.g. NV001)"),
    name: str = Form(..., description="Employee Full Name"),
    department: str = Form("General", description="Department Name"),
    images: List[UploadFile] = File(..., description="Tải lên từ 1 đến nhiều ảnh khuôn mặt (khuyên dùng 3-10 ảnh)"),
    db: Session = Depends(get_db)
):
    upload_list = [img for img in images if img is not None and img.filename]

    if not upload_list:
        raise HTTPException(
            status_code=400,
            detail="No image file provided. Please select image file(s) to upload."
        )

    valid_embeddings = []
    error_details = []

    for index, upload_file in enumerate(upload_list):
        try:
            contents = await upload_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                err = strict_tracker.log_failure(
                    db=db,
                    failure_stage="enrollment",
                    failure_reason="Invalid or corrupted image file",
                    sub_folder="enrollment"
                )
                error_details.append(ErrorDetail(failure_stage="read_file", failure_reason="Corrupted image file", image_path=err.image_path if err else None))
                continue
        except Exception as e:
            logger.error(f"Enrollment read error: {e}")
            continue

        faces = detector.detect(img)
        if not faces:
            err = strict_tracker.log_failure(
                db=db,
                failure_stage="face_detection",
                failure_reason="No face detected in enrollment image",
                image_crop=img,
                sub_folder="enrollment"
            )
            error_details.append(ErrorDetail(failure_stage="face_detection", failure_reason="No face detected", image_path=err.image_path if err else None))
            continue

        face_info = faces[0]
        passed, reason, _ = quality_gate.evaluate(img, face_info)
        if not passed:
            x, y, w, h = face_info["bbox"]
            crop = img[max(0, y):y+h, max(0, x):x+w]
            err = strict_tracker.log_failure(
                db=db,
                failure_stage="quality_gate",
                failure_reason=reason,
                image_crop=crop,
                sub_folder="enrollment"
            )
            error_details.append(ErrorDetail(failure_stage="quality_gate", failure_reason=reason, image_path=err.image_path if err else None))
            continue

        try:
            aligned = quality_gate.align_and_crop(img, face_info)
            vector = embedder.extract(aligned)
            if vector is not None and len(vector) > 0:
                valid_embeddings.append((vector, aligned))
            else:
                err = strict_tracker.log_failure(
                    db=db,
                    failure_stage="embedding",
                    failure_reason="Feature extraction returned empty vector",
                    image_crop=aligned,
                    sub_folder="enrollment"
                )
                error_details.append(ErrorDetail(failure_stage="embedding", failure_reason="Empty vector extracted", image_path=err.image_path if err else None))
        except Exception as e:
            logger.error(f"Pipeline error during enrollment: {e}")

    if not valid_embeddings:
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "message": "Enrollment failed! None of the uploaded images passed Quality Gate.",
                "valid_images_count": 0,
                "failed_images_count": len(upload_list),
                "error_details": [e.model_dump() for e in error_details]
            }
        )

    # Average L2-normalized embedding calculation
    vectors = [v[0] for v in valid_embeddings]
    mean_vec = np.mean(vectors, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm

    try:
        employee = db.query(Employee).filter(Employee.employee_code == employee_code).first()
        if not employee:
            employee = Employee(
                employee_code=employee_code,
                name=name,
                department=department
            )
            db.add(employee)
            db.flush()
        else:
            employee.name = name
            employee.department = department

        # Save successful crop image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        crop_save_path = os.path.join(settings.CAPTURES_DIR, f"enroll_{employee_code}_{timestamp}.jpg")
        cv2.imwrite(crop_save_path, valid_embeddings[0][1])

        profile = FaceProfile(
            employee_id=employee.id,
            embedding=mean_vec.tolist(),
            image_path=crop_save_path,
            model_name="sface",
            active_ai_combo=settings.ACTIVE_AI_COMBO
        )
        db.add(profile)
        db.commit()

        # Import global matcher to reload cache
        from app.api.v1.recognize import matcher
        matcher.load_cache(db)

        return EnrollResponse(
            success=True,
            message=f"Successfully enrolled employee {name} ({employee_code})",
            employee_code=employee_code,
            name=name,
            valid_images_count=len(valid_embeddings),
            failed_images_count=len(upload_list) - len(valid_embeddings),
            error_details=error_details
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Database enrollment error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
