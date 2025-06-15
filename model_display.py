import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import ImageContainer as IC
from legacy import fitness

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Параметри
INPUT_SIZE = (224, 224, 3)
MODEL_PATH = "models/CNN_224_4.keras"

def prepare_image(input_IC):
    output_IC = IC.resize_to(input_IC, INPUT_SIZE[0], INPUT_SIZE[1])
    return output_IC

if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)
    result = model.predict(np.array([prepare_image(IC.ImageContainer("dataset/manual/20250612_160817.jpg")).array]))
    result = np.array(result, dtype = np.int32)[0]
    print(result)
    prepare_image(IC.ImageContainer("dataset/manual/20250612_160817.jpg")).show_with_polygon(fitness.order_quad_vertices(result))
