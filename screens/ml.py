import keras
import tensorflow as tf

from screens.configs import IMG_SHAPE


def get_base_model(model_type):
    if model_type == "MobileNetV2":
        model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
        )

        return model

    # TODO: add other models


def get_model_preprocess(model_type):
    if model_type == "MobileNetV2":
        preprocess = keras.layers.Rescaling(1.0 / 127.5, offset=-1)

        return preprocess

    # TODO: add other models
