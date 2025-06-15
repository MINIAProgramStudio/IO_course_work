import ImageContainer as IC
import numpy as np
import fitness
import FullCalc

image_0_path = "img/DSExample1.jpg"
image_0 = IC.ImageContainer(image_0_path)
image_0 = IC.resize(image_0, 2**(0))
image_0.show("Оригінальне зображення")

kernel = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

image_0 = IC.apply_kernel(image_0, kernel)
image_0.show()

image_0 = IC.average_channels(image_0)
image_0 = IC.binarize(image_0, 0.2)
image_0.show("Границі зображення")

print(FullCalc.fullcalc(image_0))