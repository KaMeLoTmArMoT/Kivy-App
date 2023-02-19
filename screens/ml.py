import configparser
import os

import tensorflow as tf

from screens.configs import IMG_SHAPE


def get_base_model(model_type):
    if model_type == "MobileNet":
        model = tf.keras.applications.MobileNet(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )
    elif model_type == "DenseNet121":
        model = tf.keras.applications.DenseNet121(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )
    elif model_type == "NASNetMobile":
        model = tf.keras.applications.NASNetMobile(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )
    elif model_type == "EfficientNetB0":
        model = tf.keras.applications.EfficientNetB0(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )
    elif model_type == "EfficientNetB1":
        model = tf.keras.applications.EfficientNetB1(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )
    elif model_type == "EfficientNetV2B0":
        model = tf.keras.applications.EfficientNetV2B0(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )
    elif model_type == "EfficientNetV2B1":
        model = tf.keras.applications.EfficientNetV2B1(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )
    else:  # "MobileNetV2"
        model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )

    return model


def get_model_preprocess(model_type):
    if model_type in ["DenseNet121"]:
        preprocess = tf.keras.layers.Rescaling(1.0 / 255)
    elif model_type in [
        "EfficientNetB0",
        "EfficientNetB1",
        "EfficientNetV2B0",
        "EfficientNetV2B1",
    ]:
        preprocess = tf.keras.layers.Rescaling(1.0)
    else:  # "MobileNet" "MobileNetV2" "NASNetMobile"
        preprocess = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)

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
