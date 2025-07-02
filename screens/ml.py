import configparser
import os

import torch.nn as nn
import torchvision.models as models

from screens.db import DB


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
