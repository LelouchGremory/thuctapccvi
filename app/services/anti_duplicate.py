import time
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class AntiDuplicateManager:
    """
    Anti-Duplicate Check-in Manager:
    Applies cooldown seconds per track_id / employee_id to prevent multiple duplicate CheckinEvents.
    """

    def __init__(self, cooldown_seconds: int = settings.CHECKIN_COOLDOWN_SECONDS):
        self.cooldown_seconds = cooldown_seconds
        self.last_checkins = {} # {employee_id: timestamp}

    def should_allow_checkin(self, employee_id: str) -> bool:
        if not employee_id:
            return False

        now = time.time()
        if employee_id in self.last_checkins:
            elapsed = now - self.last_checkins[employee_id]
            if elapsed < self.cooldown_seconds:
                logger.info(f"AntiDuplicate: Checkin blocked for employee {employee_id} (Cooldown {elapsed:.1f}s < {self.cooldown_seconds}s)")
                return False

        self.last_checkins[employee_id] = now
        return True
