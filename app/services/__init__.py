from app.services.camera_agent import CameraAgent
from app.services.person_tracker import PersonTracker
from app.services.best_frame import BestFrameSelector
from app.services.detector import FaceDetectorCascade
from app.services.quality_gate import QualityGate
from app.services.embedding import FeatureEmbedder
from app.services.matcher import MatchingEngine
from app.services.strict_tracker import strict_tracker
from app.services.anti_duplicate import AntiDuplicateManager

__all__ = [
    "CameraAgent",
    "PersonTracker",
    "BestFrameSelector",
    "FaceDetectorCascade",
    "QualityGate",
    "FeatureEmbedder",
    "MatchingEngine",
    "strict_tracker",
    "AntiDuplicateManager"
]
