import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "face_rec_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/face_rec_db"

    # AI Combo Strategy
    ACTIVE_AI_COMBO: Literal["YUNET_SFACE", "RETINAFACE_ARCFACE", "HYBRID_CASCADE"] = "YUNET_SFACE"

    # Camera Agent
    CAMERA_SOURCE: str = "0"
    CAMERA_FPS: int = 30
    CAMERA_RECONNECT_INTERVAL: float = 5.0

    # Face Quality Gate Thresholds
    QUALITY_MIN_FACE_SIZE: int = 40
    QUALITY_BLUR_THRESHOLD: float = 80.0
    QUALITY_MIN_BRIGHTNESS: float = 40.0
    QUALITY_MAX_BRIGHTNESS: float = 220.0
    QUALITY_MAX_POSE_ANGLE: float = 30.0

    # Matching Thresholds
    SIMILARITY_RECOGNIZED_THRESHOLD: float = 0.65
    SIMILARITY_UNCERTAIN_THRESHOLD: float = 0.45

    # Anti-Duplicate Check-in
    CHECKIN_COOLDOWN_SECONDS: int = 300

    # Storage Paths
    STORAGE_DIR: str = "storage"
    CAPTURES_DIR: str = "storage/captures"
    FAILED_DIR: str = "storage/failed"
    FAILED_ENROLLMENT_DIR: str = "storage/failed/enrollment"
    FAILED_RECOGNITION_DIR: str = "storage/failed/recognition"
    MODELS_DIR: str = "storage/models"

    def setup_directories(self):
        for path in [
            self.STORAGE_DIR,
            self.CAPTURES_DIR,
            self.FAILED_DIR,
            self.FAILED_ENROLLMENT_DIR,
            self.FAILED_RECOGNITION_DIR,
            self.MODELS_DIR
        ]:
            os.makedirs(path, exist_ok=True)

settings = Settings()
settings.setup_directories()
