import threading
import time
from collections.abc import Callable

import cv2
import numpy as np

from app.screens.utils.custom_logging import get_logger
from app.screens.utils.utils import get_system_type

logger = get_logger(__name__)


class CameraStream:
    """Capture frames off the UI thread while applying bounded back-pressure."""

    def __init__(self, on_frame: Callable[[np.ndarray, threading.Event], None]):
        self.on_frame = on_frame
        self.capture = None
        self.frame_time = 1 / 30
        self.running = threading.Event()
        self.thread = None
        self.video_source = "webcam"

    def open(self, video_source: str, video_path: str | None = None) -> bool:
        self.stop()
        self.video_source = video_source

        if video_source == "file":
            self.capture = cv2.VideoCapture(video_path or "")
        elif get_system_type() == "Linux":
            self.capture = cv2.VideoCapture(-1)
        else:
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.capture.isOpened():
            logger.error("Unable to open camera or video source")
            self.capture.release()
            self.capture = None
            return False

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if video_source == "file":
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.frame_time = 1 / fps if fps and fps > 0 else 1 / 30
        else:
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            self.frame_time = 1 / 30
        return True

    def start(self) -> bool:
        if self.capture is None or self.running.is_set():
            return self.running.is_set()
        self.running.set()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        self.running.clear()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=1)
        self.thread = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        cv2.destroyAllWindows()

    def _run(self) -> None:
        try:
            while self.running.is_set() and self.capture is not None:
                loop_start = time.perf_counter()
                ret, frame = self.capture.read()
                if not ret:
                    if self.video_source == "file":
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.warning("Unable to read frame from camera")
                    break

                frame_done = threading.Event()
                self.on_frame(frame, frame_done)
                frame_done.wait(timeout=max(self.frame_time * 2, 1.0))

                delay = self.frame_time - (time.perf_counter() - loop_start)
                if delay > 0:
                    time.sleep(delay)
        finally:
            self.running.clear()
            if self.capture is not None:
                self.capture.release()
                self.capture = None
