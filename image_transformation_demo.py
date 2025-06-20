import matplotlib.pyplot as plt
import numpy as np

import ImageContainer as IC


"""
image_0.show("Оригінальне зображення")

image_0 = IC.to_hsv(image_0)
mean_value = np.mean(image_0.array[:, :, 2])
for i in range(image_0.array.shape[0]):
    for j in range(image_0.array.shape[1]):
        if abs(image_0.array[i][j][1]-0.382) < 0.2 and abs(image_0.array[i][j][0] - 0.05) < 0.1 and image_0.array[i][j][2] < mean_value:
            image_0.array[i][j] = np.array([0, 1, 1])

        if image_0.array[i][j][1] < 0.2 and image_0.array[i][j][2] > mean_value:
            image_0.array[i][j] = np.array([0.33, 1, 1])
image_0.show("Зображення з третім набором правил")

image_0 = IC.ImageContainer(image_0_path)"""

image_0_path = "img/DSExample1.jpg"
image_0 = IC.ImageContainer(image_0_path)
image_0 = IC.resize(image_0, 0.125)
image_0.show("Оригінальне зображення")

kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

image_0 = IC.apply_kernel(image_0, kernel)
image_0.show()

image_0 = IC.average_channels(image_0)
image_0.show("Границі HSV зображення")


image_0_binarized = IC.binarize(image_0, 0.2)
image_0_binarized.show("Порогова бінаризація (0.2)")
exit()
image_0_binarized = IC.dynamic_binarize(image_0, 0.1, 0.001, 0.00001)
image_0_binarized.show("Динамічна бінаризація (0.1)")


