import matplotlib.pyplot as plt
import numpy as np
import ImageContainer as IC
import BedFinder


if __name__ == "__main__":
    image_0_path = "dataset/manual/20250612_160820.jpg"
    image_0 = IC.ImageContainer(image_0_path)
    image_0 = IC.resize(image_0, 1/16)
    #image_0.show()
    image_hsv_0 = IC.to_hsv(image_0)
    mean_value = np.mean(image_0.array[:, :, 2])
    for i in range(image_0.array.shape[0]):
        for j in range(image_0.array.shape[1]):
            if abs(image_hsv_0.array[i][j][1] - 0.382) < 0.2 and abs(image_hsv_0.array[i][j][0] - 0.05) < 0.1 and \
                    image_0.array[i][j][2] < mean_value:
                image_0.array[i][j] = np.array([1, 1, 1])
            else:
                image_0.array[i][j] = np.array([0, 0, 0])
    #image_0.show("Тертій набор правил, стіл")
    temp_IC = IC.average_channels(image_0)
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    temp_IC = IC.apply_kernel(temp_IC, kernel)
    #temp_IC.show("Границі")
    temp_IC = IC.dynamic_binarize(temp_IC, 0.075, 0.001, 0.001)
    #temp_IC.show("Маска")
    pos = BedFinder.Genetic_bed_finder(image_0, progressbar=True)
    print(pos[1])
    plt.plot(pos[2])
    plt.yscale("log")
    plt.show()
    temp_IC.show_with_polygon(pos[1], title = "Стіл знайдений за допомогою генетичного алгоритму")
