import matplotlib.pyplot as plt
import numpy as np

import ImageContainer as IC

image_0_path = "dataset/manual/20250612_160817.jpg"
image_0 = IC.ImageContiner(image_0_path)
image_0.show("Оригінальне зображення")

image_0 = IC.average_channels(image_0)
image_0.show("Середнє каналів зображення")

kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

image_0_borders = IC.apply_kernel(image_0, kernel)
image_0_borders.show("Зображення після фільтрування границь")

image_0_binarized = IC.binarize(image_0_borders, 0.2)
image_0_binarized.show("Порогова бінаризація (0.2)")

image_0_binarized = IC.dynamic_binarize(image_0_borders, 0.005, 0.001, 0.001)
image_0_binarized.show("Динамічна бінаризація (0.005)")


