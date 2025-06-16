import configparser
import os

import torch.nn as nn
import torchvision.models as models

from screens.configs import IMG_SHAPE


def get_base_model(model_type, num_classes, device, no_weights=False):
    if model_type == "MobileNetV2":
        if not no_weights:
            weights = None
        else:
            weights = models.MobileNet_V2_Weights.DEFAULT

        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model.to(device=device)

    return model


def get_model_preprocess(model_type):
    if model_type == "MobileNetV2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        preprocess = weights.transforms()
        # TODO: check custom

    return preprocess


def create_config_file(model_name, model_type, num_classes, classes, config_dir):
    config = configparser.ConfigParser()
    config["Model"] = {
        "model_name": model_name,
        "model_type": model_type,
        "num_classes": num_classes,
        "classes": "-".join(sorted(classes)),
        "width": IMG_SHAPE[0],
        "height": IMG_SHAPE[1],
        "channels": IMG_SHAPE[2],
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
