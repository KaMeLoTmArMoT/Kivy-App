import keras
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
        preprocess = keras.layers.Rescaling(1.0 / 255)
    elif model_type in [
        "EfficientNetB0",
        "EfficientNetB1",
        "EfficientNetV2B0",
        "EfficientNetV2B1",
    ]:
        preprocess = keras.layers.Rescaling(1.0)
    else:  # "MobileNet" "MobileNetV2" "NASNetMobile"
        preprocess = keras.layers.Rescaling(1.0 / 127.5, offset=-1)

    return preprocess
