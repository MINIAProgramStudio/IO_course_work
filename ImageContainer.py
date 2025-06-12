import numpy as np
from PIL import Image
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from copy import deepcopy

class ImageContiner:
    def __init__(self, source, HSV = False):
        # Якщо на вхід рядок -- відкрити як шлях, якщо масив -- відкрити як масив пікселів
        if isinstance(source, str):
            im_frame = Image.open(source)
            self.array = np.array(im_frame, dtype = np.float64)
        elif isinstance(source, np.ndarray):
            self.array = np.array(source, dtype = np.float64)

        # нормалізація зображення
        if np.max(self.array) > 1:
            self.array /= 255
        self.HSV = HSV

    def show(self, title = ""):
        if self.HSV:
            plt.imshow(mcolors.hsv_to_rgb(self.array))
        else:
            plt.imshow(self.array)
        plt.axis("off")
        plt.title(title)
        plt.show()

def to_hsv(input_IC: ImageContiner):
    return ImageContiner(mcolors.rgb_to_hsv(input_IC.array), HSV = True)

def average_channels(input_IC: ImageContiner):
    img_array = deepcopy(input_IC.array)
    mean = np.mean(img_array, axis = -1)
    img_array[:,:,0] = mean
    img_array[:, :, 1] = mean
    img_array[:, :, 2] = mean
    return ImageContiner(img_array, HSV = input_IC.HSV)

def apply_kernel(input_IC: ImageContiner, kernel: np.array):
    img_array = deepcopy(input_IC.array)
    return ImageContiner(np.clip(np.stack([
        convolve(img_array[..., c], kernel, mode='reflect')
        for c in range(3)
    ], axis=-1),0, 254*(np.max(img_array)>1) + 1), HSV = input_IC.HSV)

def binarize(input_IC: ImageContiner, threshold: float = 0.5):
    img_array = np.zeros_like(input_IC.array)
    img_array[input_IC.array > threshold] = 1
    return ImageContiner(img_array, HSV = input_IC.HSV)

def dynamic_binarize(input_IC: ImageContiner, mask: float = 0.1, mask_epsilon: float = 0.005, step_epsilon: float = 0.05):
    threshold = 0.5
    step = 0.25
    img_array = np.zeros_like(input_IC.array)
    img_array[input_IC.array > threshold] = 1
    mean = np.mean(img_array)
    while abs(mean-mask) > mask_epsilon and step > step_epsilon:
        if mean > mask:
            threshold += step # WE NEED TO BUILD LESS WALLS, DONALD
        else:
            threshold -= step # WE NEED TO BUILD MORE WALLS, DONALD
        step /= 2
        img_array = np.zeros_like(input_IC.array)
        img_array[input_IC.array > threshold] = 1
        mean = np.mean(img_array)
    return ImageContiner(img_array, HSV=input_IC.HSV)


