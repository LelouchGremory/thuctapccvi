from fastapi import APIRouter
from pydantic import BaseModel
from app.core.config import settings

router = APIRouter()

class CameraStatusResponse(BaseModel):
    active_ai_combo: str
    camera_source: str
    camera_fps: int
    reconnect_interval: float

@router.get("/camera/status", response_model=CameraStatusResponse, summary="GET camera feed status")
def get_camera_status():
    return CameraStatusResponse(
        active_ai_combo=settings.ACTIVE_AI_COMBO,
        camera_source=settings.CAMERA_SOURCE,
        camera_fps=settings.CAMERA_FPS,
        reconnect_interval=settings.CAMERA_RECONNECT_INTERVAL
    )
