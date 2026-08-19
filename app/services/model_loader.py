import os
import urllib.request
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

MODELS_URLS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
}

def ensure_models_downloaded():
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    for model_name, url in MODELS_URLS.items():
        # Check both root and storage/models/
        root_path = model_name
        target_path = os.path.join(settings.MODELS_DIR, model_name)
        
        if not os.path.exists(root_path) and not os.path.exists(target_path):
            logger.info(f"Model {model_name} not found. Downloading from {url}...")
            try:
                urllib.request.urlretrieve(url, target_path)
                logger.info(f"Successfully downloaded {model_name}")
            except Exception as e:
                logger.error(f"Failed downloading {model_name}: {e}")

def get_model_path(model_name: str) -> str:
    root_path = model_name
    if os.path.exists(root_path):
        return root_path
    target_path = os.path.join(settings.MODELS_DIR, model_name)
    if os.path.exists(target_path):
        return target_path
    ensure_models_downloaded()
    return target_path if os.path.exists(target_path) else root_path
