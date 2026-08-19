import cv2
import time
import logging
import threading
from queue import Queue
from datetime import datetime
from app.core.config import settings

logger = logging.getLogger(__name__)

class CameraAgent:
    """Multi-source Camera Agent with auto-reconnect, FPS management, timestamping & ROI."""

    def __init__(self, source: str = settings.CAMERA_SOURCE, fps: int = settings.CAMERA_FPS):
        self.source_str = source
        self.fps = fps
        self.source = int(source) if source.isdigit() else source

        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)
        self.thread = None
        self.roi = None
        self.last_frame = None

    def set_roi(self, x: int, y: int, w: int, h: int):
        self.roi = (x, y, w, h)
        logger.info(f"CameraAgent ROI set to: ({x}, {y}, {w}, {h})")

    def _add_timestamp(self, frame):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame, 
            now_str, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        return frame

    def _apply_roi(self, frame):
        if self.roi is None:
            return frame
        x, y, w, h = self.roi
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        return frame[y1:y2, x1:x2]

    def _generate_fallback_frame(self):
        import numpy as np
        return np.full((480, 640, 3), (30, 30, 35), dtype=np.uint8)

    def connect(self):
        logger.info(f"CameraAgent connecting to source: {self.source}")
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.warning(f"Failed to open camera source '{self.source}'. Stream will use Docker/Fallback mode.")
                return False
            logger.info("Camera connected successfully.")
            return True
        except Exception as e:
            logger.warning(f"Exception opening camera source '{self.source}': {e}")
            return False

    def _capture_loop(self):
        interval = 1.0 / max(1, self.fps)
        last_time = time.time()
        last_reconnect_attempt = 0

        while self.is_running:
            current_time = time.time()
            if current_time - last_time < interval:
                time.sleep(0.005)
                continue

            frame = None
            if self.cap and self.cap.isOpened():
                ret, read_frame = self.cap.read()
                if ret and read_frame is not None:
                    frame = self._add_timestamp(read_frame)
                else:
                    logger.warning("Empty frame received. Releasing camera.")
                    if self.cap:
                        self.cap.release()

            if frame is None:
                if current_time - last_reconnect_attempt > settings.CAMERA_RECONNECT_INTERVAL:
                    last_reconnect_attempt = current_time
                    self.connect()
                frame = self._generate_fallback_frame()

            last_time = current_time
            frame = self._apply_roi(frame)
            self.last_frame = frame.copy()

            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except Exception:
                    pass

    def start(self):
        if self.is_running:
            return
        self.connect()
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("CameraAgent thread started.")

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        logger.info("CameraAgent stopped.")

    def get_frame(self):
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return self.last_frame
