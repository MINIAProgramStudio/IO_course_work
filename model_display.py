import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging

from tensorflow.python.ops.numpy_ops import zeros_like

import ImageContainer as IC
import fitness

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomPreprocessingLayerHSV(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomPreprocessingLayerHSV, self).__init__(**kwargs)

    def call(self, inputs):
        hsv_inputs = tf.image.rgb_to_hsv(inputs)
        condition = tf.logical_and(
            tf.logical_and(
                tf.abs(hsv_inputs[..., 1] - 0.382) < 0.2,
                tf.abs(hsv_inputs[..., 0] - 0.05) < 0.1
            ),
            hsv_inputs[..., 2] < tf.reduce_mean(hsv_inputs[..., 2])
        )
        output = tf.where(
            condition[..., tf.newaxis],
            tf.ones_like(inputs),
            tf.zeros_like(inputs)
        )
        return output

    def get_config(self):
        config = super(CustomPreprocessingLayerHSV, self).get_config()
        return config

# Параметри
INPUT_SIZE = (480, 720, 3)
MODEL_PATH = "models/CNN_rule_BASE_1.keras"

def prepare_image(input_IC):
    output_IC = IC.resize_to(input_IC, INPUT_SIZE[0], INPUT_SIZE[1])
    return output_IC

if __name__ == "__main__":
    model = tf.keras.models.load_model("models/CNN_rule_BASE_2.keras", custom_objects={"CustomPreprocessingLayerHSV": CustomPreprocessingLayerHSV})
    image = cv2.imread("dataset/manual/20250612_160817.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_ic = IC.ImageContainer(image)
    resized = prepare_image(input_ic)
    array = np.expand_dims(resized.array, axis=0)  # shape (1, H, W, 3)

    print(array.shape)  # має бути (1, 480, 720, 3)

    result = model.predict(array)
    input_ic.show_with_polygon(fitness.order_quad_vertices(result))