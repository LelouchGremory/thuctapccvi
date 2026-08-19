import cv2
import numpy as np
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class QualityGate:
    """
    Rigorous Face Quality Gate:
    Checks Blur (Laplacian variance), Brightness bounds, Min Size, Pose Angle, and Occlusion/Mask.
    """

    def __init__(self):
        self.min_size = settings.QUALITY_MIN_FACE_SIZE
        self.blur_threshold = settings.QUALITY_BLUR_THRESHOLD
        self.min_brightness = settings.QUALITY_MIN_BRIGHTNESS
        self.max_brightness = settings.QUALITY_MAX_BRIGHTNESS
        self.max_pose_angle = settings.QUALITY_MAX_POSE_ANGLE

    def check_blur(self, face_crop: np.ndarray) -> tuple[bool, float]:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return (blur_score >= self.blur_threshold), blur_score

    def check_brightness(self, face_crop: np.ndarray) -> tuple[bool, float]:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        is_valid = (self.min_brightness <= brightness <= self.max_brightness)
        return is_valid, brightness

    def check_pose_angle(self, landmarks: list[int]) -> tuple[bool, float]:
        if not landmarks or len(landmarks) < 10:
            return True, 0.0
        right_eye = (landmarks[0], landmarks[1])
        left_eye = (landmarks[2], landmarks[3])
        dY = left_eye[1] - right_eye[1]
        dX = left_eye[0] - right_eye[0]
        angle = float(np.abs(np.degrees(np.arctan2(dY, dX))))
        return (angle <= self.max_pose_angle), angle

    def check_occlusion(self, landmarks: list[int]) -> bool:
        if not landmarks or len(landmarks) < 10:
            return False
        # If mouth landmarks are missing or zeroed out, face is occluded/masked
        r_mouth = landmarks[6:8]
        l_mouth = landmarks[8:10]
        if r_mouth == [0, 0] or l_mouth == [0, 0]:
            return True
        return False

    def evaluate(self, frame: np.ndarray, face_info: dict) -> tuple[bool, str, dict]:
        """
        Evaluates full Quality Gate for a detected face.
        Returns: (passed: bool, reason: str, metrics: dict)
        """
        x, y, w, h = face_info["bbox"]
        
        # 1. Size check
        if w < self.min_size or h < self.min_size:
            return False, f"Face size too small ({w}x{h} < {self.min_size}px)", {"size": [w, h]}

        # Crop face
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return False, "Empty face crop area", {}

        # 2. Occlusion / Mask check
        landmarks = face_info.get("landmarks", [])
        if self.check_occlusion(landmarks):
            return False, "Face occluded or masked (landmark failed)", {}

        # 3. Blur check
        blur_passed, blur_score = self.check_blur(crop)
        if not blur_passed:
            return False, f"Face is blurry (Laplacian var {blur_score:.1f} < {self.blur_threshold})", {"blur": blur_score}

        # 4. Brightness check
        bright_passed, bright_score = self.check_brightness(crop)
        if not bright_passed:
            return False, f"Invalid brightness ({bright_score:.1f} outside range [{self.min_brightness}, {self.max_brightness}])", {"brightness": bright_score}

        # 5. Pose Angle check
        pose_passed, pose_angle = self.check_pose_angle(landmarks)
        if not pose_passed:
            return False, f"Extreme pose angle ({pose_angle:.1f}° > {self.max_pose_angle}°)", {"pose_angle": pose_angle}

        return True, "Passed Quality Gate", {
            "blur": blur_score,
            "brightness": bright_score,
            "pose_angle": pose_angle,
            "size": [w, h]
        }

    def align_and_crop(self, frame: np.ndarray, face_info: dict, target_size=(112, 112)) -> np.ndarray:
        landmarks = face_info.get("landmarks")
        if landmarks and len(landmarks) == 10:
            right_eye = (landmarks[0], landmarks[1])
            left_eye = (landmarks[2], landmarks[3])
            dY = left_eye[1] - right_eye[1]
            dX = left_eye[0] - right_eye[0]
            angle = np.degrees(np.arctan2(dY, dX))

            eye_center = (int((right_eye[0] + left_eye[0]) // 2), int((right_eye[1] + left_eye[1]) // 2))
            M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
            h, w = frame.shape[:2]
            aligned = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

            x, y, fw, fh = face_info["bbox"]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + fw), min(h, y + fh)
            crop = aligned[y1:y2, x1:x2]
        else:
            x, y, fw, fh = face_info["bbox"]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + fw), min(h, y + fh)
            crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return cv2.resize(frame, target_size)

        return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
