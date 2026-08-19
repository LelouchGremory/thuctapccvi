import logging
from collections import deque
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.face_profile import FaceProfile
from app.models.employee import Employee
from app.services.strict_tracker import strict_tracker

logger = logging.getLogger(__name__)

class MatchingEngine:
    """
    Tuần 5 Matching Engine:
    - Preloads all face vectors from PostgreSQL/pgvector into RAM Cache on startup.
    - Performs batch Cosine Similarity matrix multiplication.
    - Classifies results into 4 States: RECOGNIZED, UNCERTAIN, UNKNOWN, REJECTED.
    - Automatically records SystemBacklog for REJECTED and UNCERTAIN states.
    - Multi-frame voting across 2-3 consecutive frames per track_id.
    """

    def __init__(self):
        self.recognized_threshold = settings.SIMILARITY_RECOGNIZED_THRESHOLD
        self.uncertain_threshold = settings.SIMILARITY_UNCERTAIN_THRESHOLD
        self.cache_profiles = []
        self.embedding_matrix = None
        self.voting_buffers = {}

    def load_cache(self, db: Session) -> int:
        try:
            profiles = db.query(FaceProfile).join(Employee).all()
            self.cache_profiles = []
            embeddings_list = []

            for p in profiles:
                if p.embedding is not None and p.employee:
                    vec = np.array(p.embedding, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    self.cache_profiles.append({
                        "profile_id": p.id,
                        "employee_id": p.employee.id,
                        "employee_code": p.employee.employee_code,
                        "name": p.employee.name,
                        "department": p.employee.department,
                        "embedding": vec
                    })
                    embeddings_list.append(vec)

            if embeddings_list:
                self.embedding_matrix = np.vstack(embeddings_list) # (N, 512)
            else:
                self.embedding_matrix = None

            logger.info(f"MatchingEngine: Loaded {len(self.cache_profiles)} embeddings into RAM Cache.")
            return len(self.cache_profiles)
        except Exception as e:
            logger.error(f"MatchingEngine: Error loading RAM cache from DB: {e}")
            self.cache_profiles = []
            self.embedding_matrix = None
            return 0

    def match(self, query_vector: np.ndarray, db: Session = None, track_id: str = None, face_crop: np.ndarray = None) -> dict:
        if query_vector is None or len(query_vector) == 0:
            if db:
                strict_tracker.log_failure(
                    db=db,
                    failure_stage="matching",
                    failure_reason="Invalid or empty query vector",
                    image_crop=face_crop,
                    track_id=track_id,
                    confidence_score=0.0
                )
            return {
                "status": "REJECTED",
                "similarity": 0.0,
                "employee": None,
                "reason": "Invalid or empty feature vector"
            }

        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        if self.embedding_matrix is None or len(self.cache_profiles) == 0:
            return {
                "status": "UNKNOWN",
                "similarity": 0.0,
                "employee": None,
                "reason": "RAM Cache is empty (no enrolled employees)"
            }

        # Batch Cosine Similarity (Dot Product on L2 normalized vectors)
        scores = np.dot(self.embedding_matrix, query_vector) # (N,)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        matched = self.cache_profiles[best_idx]

        if best_score >= self.recognized_threshold:
            status = "RECOGNIZED"
            reason = f"Recognized employee {matched['name']} ({matched['employee_code']})"
            emp_data = matched
        elif best_score >= self.uncertain_threshold:
            status = "UNCERTAIN"
            reason = f"Uncertain match for {matched['name']} ({matched['employee_code']})"
            emp_data = matched
            # Strict Error Tracking for UNCERTAIN state
            if db:
                strict_tracker.log_failure(
                    db=db,
                    failure_stage="matching",
                    failure_reason=f"Uncertain matching similarity score ({best_score:.4f} in [{self.uncertain_threshold}, {self.recognized_threshold}])",
                    image_crop=face_crop,
                    track_id=track_id,
                    confidence_score=best_score
                )
        else:
            status = "UNKNOWN"
            reason = f"Unknown person (Best similarity {best_score:.4f} < {self.uncertain_threshold})"
            emp_data = None

        return {
            "status": status,
            "similarity": round(best_score, 4),
            "employee": emp_data,
            "reason": reason
        }

    def multi_frame_voting(self, track_id: str, current_result: dict, window_size: int = 3) -> dict:
        if not track_id:
            return current_result

        if track_id not in self.voting_buffers:
            self.voting_buffers[track_id] = deque(maxlen=window_size)

        buffer = self.voting_buffers[track_id]
        buffer.append(current_result)

        if len(buffer) < window_size:
            return current_result

        employee_counts = {}
        for item in buffer:
            if item.get("employee") and item["employee"].get("employee_id"):
                emp_id = item["employee"]["employee_id"]
                employee_counts[emp_id] = employee_counts.get(emp_id, 0) + 1

        majority = (window_size // 2) + 1
        for emp_id, count in employee_counts.items():
            if count >= majority:
                last_match = next((item for item in reversed(buffer) if item.get("employee") and item["employee"].get("employee_id") == emp_id), current_result)
                voted = dict(last_match)
                voted["status"] = "RECOGNIZED"
                voted["multi_frame_confirmed"] = True
                return voted

        return current_result
