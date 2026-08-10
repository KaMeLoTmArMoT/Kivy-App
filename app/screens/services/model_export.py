import os
import platform

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    torch = None
    YOLO = None

from app.screens.utils.custom_logging import get_logger

logger = get_logger(__name__)


def _require_runtime() -> None:
    if torch is None or YOLO is None:
        raise RuntimeError("PyTorch and Ultralytics are required for model export.")


def log_gpu(tag: str, summary: bool = False) -> None:
    if torch is None or not torch.cuda.is_available():
        return
    logger.debug(f"{tag}[Used]     {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    logger.debug(f"{tag}[Reserved] {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    if summary:
        logger.debug(f"{torch.cuda.memory_summary()}")


def get_hardware_acceleration_type() -> str:
    if torch is not None and torch.cuda.is_available():
        logger.debug("NVIDIA CUDA device found. Best acceleration: TensorRT")
        return "TensorRT"

    if "intel" in platform.processor().lower():
        logger.debug("Intel CPU detected. Best acceleration: OpenVINO")
        return "OpenVINO"

    logger.warning("No specific hardware acceleration detected.")
    return "None"


def export_to_best_available(pt_model_path: str, force_export: list[str] | None = None):
    _require_runtime()
    force_export = force_export or []
    if not os.path.exists(pt_model_path):
        logger.error(f"Cannot export. Model not found at {pt_model_path}")
        return False

    accel_type = get_hardware_acceleration_type()
    model = YOLO(pt_model_path)
    name = os.path.basename(pt_model_path)

    logger.info(f"Exporting '{name}' to ONNX format for general acceleration...")
    model.export(format="onnx", half=True, simplify=True)
    logger.info("Export to ONNX complete.")

    if accel_type == "TensorRT" or "TensorRT" in force_export:
        logger.info(f"Exporting '{name}' to TensorRT format...")
        model.export(format="tensorrt", half=True, simplify=True)
        logger.info("Export to TensorRT complete.")

    if accel_type == "OpenVINO" or "OpenVINO" in force_export:
        logger.info(f"Exporting '{name}' to OpenVINO format...")
        model.export(format="openvino", half=True)
        logger.info("Export to OpenVINO complete.")
    return True


def get_best_model_paths(base_dir: str, model_name: str) -> list[tuple[str, str]]:
    available_models = []
    logger.info("Searching for best available model to load...")

    candidates = [
        (os.path.join(base_dir, f"{model_name}.engine"), "TensorRT", os.path.isfile),
        (os.path.join(base_dir, f"{model_name}_openvino_model"), "OpenVINO", os.path.isdir),
        (os.path.join(base_dir, f"{model_name}.onnx"), "ONNX", os.path.isfile),
        (os.path.join(base_dir, f"{model_name}.pt"), "PyTorch", os.path.isfile),
    ]
    for path, model_type, exists in candidates:
        if exists(path):
            logger.info(f"Found {model_type} model: {path}")
            available_models.append((path, model_type))

    if not available_models:
        logger.error(f"No model file found for '{model_name}' in '{base_dir}'")
    return available_models
