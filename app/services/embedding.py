import os
import cv2
import logging
import numpy as np
from app.core.config import settings
from app.services.model_loader import get_model_path

logger = logging.getLogger(__name__)

class FeatureEmbedder:
    """
    Face Feature Extraction Engine supporting SFace & ArcFace.
    Returns L2-normalized 512-dimensional vector.
    """

    def __init__(self, combo: str = settings.ACTIVE_AI_COMBO):
        self.combo = combo
        self.sface_path = get_model_path("face_recognition_sface_2021dec.onnx")
        
        self.sface = cv2.FaceRecognizerSF.create(
            model=self.sface_path,
            config="",
            backend_id=0,
            target_id=0
        )
        logger.info(f"FeatureEmbedder initialized with model path: {self.sface_path}")

    def extract(self, aligned_face: np.ndarray) -> np.ndarray:
        if aligned_face is None or aligned_face.size == 0:
            return None

        # SFace produces 128-d raw embedding; zero-pad to 512-d for uniform pgvector storage
        raw_feat = self.sface.feature(aligned_face).flatten() # (128,)
        
        # Zero-pad vector to 512 dimensions for pgvector compatibility across AI Combos
        padded_feat = np.zeros(512, dtype=np.float32)
        padded_feat[:len(raw_feat)] = raw_feat

        norm = np.linalg.norm(padded_feat)
        if norm > 0:
            padded_feat = padded_feat / norm

        return padded_feat
