import configparser
import gc
import os
import platform
import sys
from collections import deque

import numpy as np
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torch import optim
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from ultralytics import YOLO
except ImportError:
    torch = None
    nn = models = optim = DataLoader = SummaryWriter = transforms = ImageFolder = None
    YOLO = None

from app.screens.utils.custom_logging import get_logger
from app.screens.utils.db import DB

logger = get_logger(__name__)


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed. Install torch/torchvision to use ML features."
        )


def _require_ultralytics():
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not installed. Install ultralytics to use YOLO export."
        )


class KModel:
    def __init__(self):
        self.device = None

        self.model = None

        self.classes = None

        self.loss = None
        self.acc = None
        self.processed_steps = None
        self.data_iter = None
        self.criterion = None
        self.writer = None
        self.optimizer = None
        self.terminate_training = None

        self.transform = None

    def get_device(self):
        _require_torch()
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device

    def load_model(self, save_path, config_path):
        _require_torch()
        logger.info(f"load model: {save_path}")

        logger.info(f"Load model config path {config_path}")
        if not os.path.exists(config_path):
            logger.error("No config")
            return

        self.unload_model()

        (
            model_type,
            num_classes,
            img_shape,
            self.classes,
        ) = read_config_file(config_path)
        logger.info(
            f"Load done:\n"
            f"- type: {model_type}\n"
            f"- shape: {img_shape}\n"
            f"- num classes: {num_classes}\n"
            f"- classes: {self.classes}\n"
        )

        self.model = get_base_model(
            model_type,
            num_classes=num_classes,
            no_weights=True,
        )
        state_dict = torch.load(save_path, map_location=self.get_device())
        self.model.load_state_dict(state_dict)
        self.model.to(self.get_device())

    def unload_model(self):
        if self.model is None:
            return
        _require_torch()

        log_gpu("--- Before")

        self.model.to("cpu")
        del self.model
        self.model = None

        torch.cuda.empty_cache()
        gc.collect()

        logger.debug("Unload complete. GPU memory should be freed.")
        log_gpu("--- After unload")

    def save_model(self, model_dir, save_path):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.debug("save complete")

    def evaluate_model(self, data):
        self.model.eval()

        self.loss = None
        self.acc = None

        self.processed_steps = 0

        self.data_iter = iter(data)  # Convert DataLoader into list of batches
        self.criterion = torch.nn.CrossEntropyLoss()

    def async_eval_cycle(self):
        try:
            images, labels = next(self.data_iter)
            self.processed_steps += 1

        except StopIteration:
            logger.info(f"Final Eval loss: {self.loss:.4f}, acc: {self.acc:.4f}")
            return False, 0, 0, 0

        images = images.to(self.get_device())
        labels = labels.to(self.get_device())

        with torch.no_grad():
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)

        # Sliding window averaging
        if self.loss is None or self.acc is None:
            self.loss, self.acc = loss.item(), accuracy
        else:
            self.loss = self.loss * 0.9 + loss.item() * 0.1
            self.acc = self.acc * 0.9 + accuracy * 0.1

        return True, self.processed_steps, self.loss, accuracy

    def create_model(
        self,
        model_name,
        classes,
        model_type,
        model_dir,
        save_path,
    ):
        _require_torch()
        self.unload_model()
        self.classes = classes

        logger.debug(f"creating {model_name}")

        self.model = get_base_model(
            model_type,
            num_classes=len(self.classes),
        )
        self.model.to(device=self.get_device())

        for param in self.model.features.parameters():
            param.requires_grad = False

        logger.debug("create complete")
        # logger.info(f"{self.model}")

        self.save_model(model_dir, save_path)

    def model_predict(self, image):
        self.model.eval()

        image = (
            self.transform(image).unsqueeze(0).to(self.get_device())
        )  # Add batch dim and move to device

        with torch.no_grad():
            output = self.model(image)
            pred = torch.argmax(output, dim=1).item()

        cls_name = self.classes[pred].replace("train/", "")
        logger.info(f"CLS: |{cls_name}| ID: |{pred}|")
        return cls_name

    def update_params(self):
        _require_torch()
        img_shape = DB().get_config_typed("IMG_SHAPE")
        mean = DB().get_config_typed("MEAN")
        std = DB().get_config_typed("STD")

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_shape[0], img_shape[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def train_model(self, data, log_dir):
        epochs_s1 = DB().get_config_typed("epochs_s1")
        epochs_s2 = DB().get_config_typed("epochs_s2")

        lr_s1 = DB().get_config_typed("lr_s1")
        lr_s2 = DB().get_config_typed("lr_s2")

        self.model.train()

        self.writer = SummaryWriter(log_dir=log_dir)
        self.criterion = nn.CrossEntropyLoss()

        try:
            logger.info("\n--- Stage 1: Fine-tuning the classifier ---")
            for p in self.model.features.parameters():
                p.requires_grad = False
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr_s1)
            self.train_cycle(epochs_s1, data, start_epoch=0)

            logger.info("\n--- Stage 2: Unfreeze all layers ---")
            for p in self.model.parameters():
                p.requires_grad = True
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr_s2)
            self.train_cycle(epochs_s2, data, start_epoch=epochs_s1)

        finally:
            self.terminate_training = False
            if self.writer:
                self.writer.close()

    def train_cycle(self, epochs, dataset, start_epoch=0):
        smooth_window = DB().get_config_typed("smooth_window")
        total_epochs = start_epoch + epochs

        for epoch_idx in range(epochs):
            if self.terminate_training:
                logger.warning("Training terminated by user")
                return

            epoch = start_epoch + epoch_idx
            running_loss = 0.0
            correct, total = 0, 0
            loss_window = deque(maxlen=smooth_window)

            progress_bar = tqdm(
                dataset,
                desc=f"Epoch {epoch + 1}/{total_epochs}",
                file=sys.stdout,
            )

            for images, labels in progress_bar:
                if self.terminate_training:
                    logger.warning("Early termination inside batch")
                    return

                images, labels = images.to(self.get_device()), labels.to(
                    self.get_device()
                )
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                loss_value = loss.item()
                running_loss += loss_value

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss_window.append(loss_value)
                smoothed_loss = np.mean(loss_window)
                progress_bar.set_postfix(loss=f"{smoothed_loss:.4f}")

            avg_loss = running_loss / len(dataset)
            accuracy = 100 * correct / total

            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/train", accuracy, epoch)

            logger.info(
                f"Epoch [{epoch + 1}/{total_epochs}], "
                f"Loss: {avg_loss:.4f}, "
                f"Accuracy: {accuracy:.2f}%\n"
            )


def get_base_model(model_type: str, num_classes: int, no_weights=False):
    _require_torch()
    # Weights handling
    pretrained = not no_weights
    weights = None  # For newer versions, if needed

    model_type = model_type.lower()

    if model_type == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.DEFAULT if not no_weights else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_type == "mobilenetv3":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if not no_weights else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_type == "resnet":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "resnext":
        model = models.resnext50_32x4d(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_type == "efficientnet":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_type == "efficientnetv2":
        model = models.efficientnet_v2_s(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_type == "alexnet":
        model = models.alexnet(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_type == "vgg":
        model = models.vgg11(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def create_config_file(model_name, model_type, num_classes, classes, config_dir):
    img_shape = DB().get_config_typed("IMG_SHAPE")

    config = configparser.ConfigParser()
    config["Model"] = {
        "model_name": model_name,
        "model_type": model_type,
        "num_classes": num_classes,
        "classes": "-".join(sorted(classes)),
        "width": img_shape[0],
        "height": img_shape[1],
        "channels": img_shape[2],
    }
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, model_name + ".conf")
    with open(config_path, "w") as configfile:
        config.write(configfile)


def read_config_file(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    model_section = config["Model"]

    model_type = model_section["model_type"]
    num_classes = int(model_section["num_classes"])
    classes = model_section["classes"].split("-")
    width = int(model_section["width"])
    height = int(model_section["height"])
    channels = int(model_section["channels"])
    img_shape = (width, height, channels)

    return model_type, num_classes, img_shape, classes


def log_gpu(tag, summary=False):
    logger.debug(f"{tag}[Used]     {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    logger.debug(f"{tag}[Reserved] {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    if summary:
        logger.debug(f"{torch.cuda.memory_summary()}")


def prepare_dataset(
    ml_train_folder,
    transform,
    batch_size=8,
    shuffle=True,
):
    test_dataset = ImageFolder(
        root=ml_train_folder,
        transform=transform,
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # TODO: set to >0 if needed and supported
    )

    return testloader


def get_hardware_acceleration_type() -> str:
    if torch.cuda.is_available():
        logger.debug("NVIDIA CUDA device found. Best acceleration: TensorRT")
        return "TensorRT"

    try:
        if "intel" in platform.processor().lower():
            logger.debug("Intel CPU detected. Best acceleration: OpenVINO")
            return "OpenVINO"
    except Exception as e:
        logger.debug(f"{e}")

    logger.warning("No specific hardware acceleration detected.")
    return "None"


def export_to_best_available(pt_model_path, force_export=None):
    _require_torch()
    _require_ultralytics()
    if not os.path.exists(pt_model_path):
        logger.error(f"Cannot export. Model not found at {pt_model_path}")
        return

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


def get_best_model_paths(base_dir, model_name):
    available_models = []
    logger.info("Searching for best available model to load...")

    tensorrt_path = os.path.join(base_dir, f"{model_name}.engine")
    if os.path.exists(tensorrt_path):
        logger.info("Found high-performance TensorRT model.")
        available_models.append([tensorrt_path, "TensorRT"])

    openvino_path = os.path.join(base_dir, f"{model_name}_openvino_model")
    if os.path.isdir(openvino_path):
        logger.info("Found optimized OpenVINO model.")
        available_models.append([openvino_path, "OpenVINO"])

    onnx_path = os.path.join(base_dir, f"{model_name}.onnx")
    if os.path.exists(onnx_path):
        logger.info("Found general-purpose ONNX model.")
        available_models.append([onnx_path, "ONNX"])

    pt_path = os.path.join(base_dir, f"{model_name}.pt")
    if os.path.exists(pt_path):
        logger.info("Found baseline PyTorch model.")
        available_models.append([pt_path, "PyTorch"])

    if available_models:
        return available_models

    else:
        logger.error(f"No model file found for '{model_name}' in '{base_dir}'")
        return [None, None]
