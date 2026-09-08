import os
import platform
from enum import StrEnum

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    torch = None
    YOLO = None

from app.screens.utils.custom_logging import get_logger

logger = get_logger(__name__)


class HardwareAcceleration(StrEnum):
    TENSORRT = "TensorRT"
    OPENVINO = "OpenVINO"
    NONE = "None"


class ModelFormat(StrEnum):
    ENGINE = "TensorRT"
    OPENVINO = "OpenVINO"
    ONNX = "ONNX"
    PYTORCH = "PyTorch"


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
        return str(HardwareAcceleration.TENSORRT)

    if "intel" in platform.processor().lower():
        logger.debug("Intel CPU detected. Best acceleration: OpenVINO")
        return str(HardwareAcceleration.OPENVINO)

    logger.warning("No specific hardware acceleration detected.")
    return str(HardwareAcceleration.NONE)


def export_to_best_available(pt_model_path: str, force_export: list[str] | None = None):
    _require_runtime()
    force_export = force_export or []
    if not os.path.exists(pt_model_path):
        logger.error(f"Cannot export. Model not found at {pt_model_path}")
        return False

    accel_type = get_hardware_acceleration_type()
    model = YOLO(pt_model_path)
    name = os.path.basename(pt_model_path)

    try:
        logger.info(f"Exporting '{name}' to ONNX format for general acceleration...")
        model.export(format="onnx", quantize=16, simplify=True)
        logger.info("Export to ONNX complete.")
    except Exception as e:
        e.add_note(f"Context: export '{name}' ({pt_model_path}) to ONNX")
        logger.error(f"ONNX export error: {e}", exc_info=True)

    if accel_type == str(HardwareAcceleration.TENSORRT) or "TensorRT" in force_export:
        try:
            logger.info(f"Exporting '{name}' to TensorRT format...")
            model.export(format="tensorrt", quantize=16, simplify=True)
            logger.info("Export to TensorRT complete.")
        except Exception as e:
            e.add_note(f"Context: export '{name}' ({pt_model_path}) to TensorRT")
            logger.error(f"TensorRT export error: {e}", exc_info=True)

    if accel_type == str(HardwareAcceleration.OPENVINO) or "OpenVINO" in force_export:
        try:
            logger.info(f"Exporting '{name}' to OpenVINO format...")
            model.export(format="openvino", quantize=16)
            logger.info("Export to OpenVINO complete.")
        except Exception as e:
            e.add_note(f"Context: export '{name}' ({pt_model_path}) to OpenVINO")
            logger.error(f"OpenVINO export error: {e}", exc_info=True)
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
