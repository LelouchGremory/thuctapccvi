import base64
import asyncio
import cv2
import logging
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
from app.core.config import settings
from app.db.session import SessionLocal
from app.services.camera_agent import CameraAgent
from app.services.detector import FaceDetectorCascade
from app.services.quality_gate import QualityGate
from app.services.embedding import FeatureEmbedder
from app.services.strict_tracker import strict_tracker
from app.models.recognition_event import RecognitionEvent
from app.api.v1.recognize import matcher

logger = logging.getLogger(__name__)
router = APIRouter()

camera_agent = CameraAgent()
detector = FaceDetectorCascade()
quality_gate = QualityGate()
embedder = FeatureEmbedder()

@router.websocket("/ws/stream")
async def websocket_video_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/stream")

    if not camera_agent.is_running:
        camera_agent.start()

    last_client_frame = None
    last_log_time = 0.0

    try:
        while True:
            frame = None
            try:
                client_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                if client_msg.startswith("data:image"):
                    base64_data = client_msg.split(",", 1)[1] if "," in client_msg else client_msg
                    img_bytes = base64.b64decode(base64_data)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        frame = decoded
                        last_client_frame = decoded
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.debug(f"Client frame processing error: {e}")

            if frame is None:
                if last_client_frame is not None:
                    frame = last_client_frame
                else:
                    frame = camera_agent.get_frame()

            if frame is None:
                await asyncio.sleep(0.03)
                continue

            # Run detection & 4-state classification
            faces = detector.detect(frame)
            face_results = []
            now_time = time.time()

            for i, face_info in enumerate(faces):
                passed, reason, _ = quality_gate.evaluate(frame, face_info)
                if not passed:
                    face_results.append({
                        "bbox": face_info["bbox"],
                        "confidence": face_info["confidence"],
                        "status": "REJECTED",
                        "label": f"REJECTED ({reason[:15]}...)",
                        "similarity": 0.0
                    })

                    # Log to SystemBacklog (throttled to once every 1.5s)
                    if now_time - last_log_time >= 1.5:
                        last_log_time = now_time
                        try:
                            db = SessionLocal()
                            x, y, w, h = face_info["bbox"]
                            h_img, w_img = frame.shape[:2]
                            crop = frame[max(0, y):min(h_img, y+h), max(0, x):min(w_img, x+w)]
                            strict_tracker.log_failure(
                                db=db,
                                failure_stage="quality_gate",
                                failure_reason=reason,
                                image_crop=crop,
                                confidence_score=face_info["confidence"]
                            )
                            db.close()
                        except Exception as log_err:
                            logger.error(f"Error logging backlog in WS stream: {log_err}")
                    continue

                aligned = quality_gate.align_and_crop(frame, face_info)
                vector = embedder.extract(aligned)
                match_info = matcher.match(query_vector=vector)

                status = match_info["status"]
                similarity = match_info["similarity"]
                emp = match_info.get("employee")
                label = f"{emp['name']} ({similarity:.2f})" if emp else f"{status} ({similarity:.2f})"

                face_results.append({
                    "bbox": face_info["bbox"],
                    "confidence": face_info["confidence"],
                    "status": status,
                    "label": label,
                    "similarity": similarity
                })

                # Log to RecognitionEvent (throttled to once every 1.5s)
                if now_time - last_log_time >= 1.5:
                    last_log_time = now_time
                    try:
                        db = SessionLocal()
                        emp_id = emp["employee_id"] if emp else None
                        event = RecognitionEvent(
                            camera_id=None,
                            employee_id=emp_id,
                            similarity=similarity,
                            status=status,
                            model_name="YuNet/SFace",
                            active_ai_combo=settings.ACTIVE_AI_COMBO
                        )
                        db.add(event)
                        db.commit()
                        db.close()
                    except Exception as log_err:
                        logger.error(f"Error logging recognition in WS stream: {log_err}")

            # Encode frame to JPEG Base64
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            base64_frame = base64.b64encode(buffer).decode('utf-8')

            payload = {
                "image": f"data:image/jpeg;base64,{base64_frame}",
                "faces": face_results,
                "fps": settings.CAMERA_FPS,
                "combo": settings.ACTIVE_AI_COMBO
            }

            await websocket.send_json(payload)
            await asyncio.sleep(1.0 / settings.CAMERA_FPS)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
