import configparser
import gc
import os

import torch
import torch.nn as nn
import torchvision.models as models

from screens.custom_logging import get_logger
from screens.db import DB

logger = get_logger(__name__)


class KModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_type = None
        self.num_classes = None
        self.img_shape = None
        self.classes = None

        self.loss = None
        self.acc = None

        self.total_steps = None
        self.processed_steps = None

        self.data_iter = None
        self.criterion = None

        self.model = None

    def load_model(self, save_path, config_path):
        logger.info(f"load model: {save_path}")

        logger.info(f"Load model config path {config_path}")
        if os.path.exists(config_path):
            (
                self.model_type,
                self.num_classes,
                self.img_shape,
                self.classes,
            ) = read_config_file(config_path)
            logger.info(
                f"Load done:\n"
                f"- type: {self.model_type}\n"
                f"- shape: {self.img_shape}\n"
                f"- num classes: {self.num_classes}\n"
                f"- classes: {self.classes}\n"
            )

        else:
            logger.warning("No config")

        self.model = get_base_model(
            self.model_type,
            num_classes=self.num_classes,
            no_weights=True,
        )
        state_dict = torch.load(save_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def unload_model(self):
        if self.model is None:
            return

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

    def evaluate_model(self, data=None):
        self.model.eval()
        if data is None:
            data = self.prepare_dataset(batch_size=32, shuffle=False)  # TODO fix

        self.loss = None
        self.acc = None

        self.total_steps = len(data)
        self.processed_steps = 0

        self.data_iter = iter(data)  # Convert DataLoader into list of batches
        self.criterion = torch.nn.CrossEntropyLoss()

    def async_eval_cycle(self):
        try:
            images, labels = next(self.data_iter)
            self.processed_steps += 1

        except StopIteration:
            logger.info(f"Final Eval loss: {self.loss:.4f}, acc: {self.acc:.4f}")
            return False

        images = images.to(self.device)
        labels = labels.to(self.device)

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

        return True

    def create_model(
        self,
        model_name,
        classes,
        model_type,
        model_dir,
        save_path,
    ):
        self.unload_model()

        logger.debug(f"creating {model_name}")

        self.model = get_base_model(
            model_type,
            num_classes=len(classes),
        )
        self.model.to(device=self.device)

        for param in self.model.features.parameters():
            param.requires_grad = False

        logger.debug("create complete")
        # logger.info(f"{self.model}")

        self.save_model(model_dir, save_path)

    def update_params(self):
        pass

    def predict(self):
        pass

    def train(self):
        pass


def get_base_model(model_type: str, num_classes: int, no_weights=False):
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
