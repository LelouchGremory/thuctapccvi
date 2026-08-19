from fastapi import APIRouter
from app.api.v1.enroll import router as enroll_router
from app.api.v1.recognize import router as recognize_router
from app.api.v1.camera import router as camera_router
from app.api.v1.logs import router as logs_router
from app.api.v1.websocket import router as websocket_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(enroll_router, tags=["Enrollment"])
api_router.include_router(recognize_router, tags=["Recognition"])
api_router.include_router(camera_router, tags=["Camera"])
api_router.include_router(logs_router, tags=["Logs & Backlog"])

root_router = APIRouter()
root_router.include_router(api_router)
root_router.include_router(websocket_router)
