import cv2
import numpy as np

class BestFrameSelector:
    """
    Best Frame Selection Manager per Track ID.
    Maintains best quality score frame for each track_id to avoid redundant face inference.
    """

    def __init__(self, max_track_age: int = 30):
        self.track_buffers = {}

    def calculate_frame_score(self, face_crop: np.ndarray) -> float:
        if face_crop is None or face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        size_score = face_crop.shape[0] * face_crop.shape[1]
        return float(blur_score * 0.7 + size_score * 0.3)

    def is_best_frame(self, track_id: str, frame: np.ndarray, face_info: dict) -> bool:
        if not track_id:
            return True

        x, y, w, h = face_info["bbox"]
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        crop = frame[y1:y2, x1:x2]

        score = self.calculate_frame_score(crop)

        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = {"best_score": score, "count": 1}
            return True

        current_best = self.track_buffers[track_id]["best_score"]
        if score > current_best * 1.15: # 15% quality improvement threshold
            self.track_buffers[track_id]["best_score"] = score
            return True

        return False
