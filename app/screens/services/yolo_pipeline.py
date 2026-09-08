import gc
import os
from typing import Any

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from app.screens.services.model_export import get_best_model_paths
from app.screens.utils.custom_logging import get_logger

logger = get_logger(__name__)


class YoloInferencePipeline:
    """Domain service managing YOLO model loading, warm-up, and frame inferencing."""

    def __init__(self, confidence: float = 0.5):
        self.model: Any | None = None
        self.model_name: str | None = None
        self.confidence = confidence

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def is_available(self) -> bool:
        return YOLO is not None

    def ensure_model(self, model_path: str) -> bool:
        """Download or validate a model file before loading it."""
        if YOLO is None:
            logger.error("Ultralytics YOLO is not installed.")
            return False
        if os.path.exists(model_path):
            return True
        try:
            YOLO(model_path, task="detect")
            return True
        except Exception as exc:
            logger.error(f"Failed to download YOLO model {model_path}: {exc}")
            return False

    def load_model(self, project_folder: str, model_name: str) -> bool:
        """Load, fuse, and warm up the best available YOLO model variant."""
        if YOLO is None:
            logger.error("Ultralytics YOLO is not installed.")
            return False

        base_stem = model_name.split(".")[0]
        available_models = get_best_model_paths(project_folder, base_stem)

        for model_path, model_type in available_models:
            logger.info(f"Trying to load YOLO model variant: {model_path} ({model_type})")
            try:
                model = YOLO(model_path, task="detect")
                model.overrides["verbose"] = False

                if model_type == "PyTorch":
                    try:
                        model.fuse()
                        logger.debug("YOLO PyTorch layer fuse successful")
                    except Exception as e:
                        logger.error(f"Failed to fuse layers for {model_path}: {e}")

                warmup_img = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
                model(warmup_img)
                logger.debug("YOLO model warmup successful")

                self.model = model
                self.model_name = model_name
                logger.info(f"YOLO model {model_path} initialized successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to load YOLO model variant {model_path}: {e}")

        return False

    def infer_frame(self, cv2_frame: np.ndarray) -> np.ndarray:
        """Execute YOLO inference on BGR opencv frame and return plotted results."""
        if self.model is None:
            return cv2_frame

        rgb_frame = cv2_frame[:, :, ::-1]
        results = self.model(rgb_frame, conf=self.confidence)

        if len(results) > 1:
            logger.debug("Multiple YOLO detection results returned for frame")

        res_plotted = results[0].plot()
        return res_plotted[:, :, ::-1]

    def unload(self) -> None:
        """Release YOLO model from memory and run GC."""
        self.model = None
        self.model_name = None
        gc.collect()
        logger.info("YOLO model unloaded")
