if __name__ == "__main__":
    import ImageContainer as IC
    import numpy as np
    import fitness
    import FullCalc
    import RandomCalc
    import PSOCalc
    import GeneticCalc
    import matplotlib.pyplot as plt

    image_0_path = "img/DSExample1.jpg"
    image_0 = IC.ImageContainer(image_0_path)

    factor = 2**(-5)
    image_0 = IC.resize(image_0, factor)
    #image_0.show("Оригінальне зображення")
    image_0 = IC.to_hsv(image_0)
    mean_value = np.mean(image_0.array[:, :, 2])
    for i in range(image_0.array.shape[0]):
        for j in range(image_0.array.shape[1]):
            if abs(image_0.array[i][j][1] - 0.382) < 0.2 and abs(image_0.array[i][j][0] - 0.05) < 0.1 and \
                    image_0.array[i][j][2] < mean_value:
                image_0.array[i][j] = np.array([1, 1, 1])
            else:
                image_0.array[i][j] = np.array([0, 0, 0])

    #image_0.show()
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    #image_0 = IC.apply_kernel(image_0, kernel)
    #image_0.show()

    image_0 = IC.average_channels(image_0)
    image_0.HSV = False
    image_0 = IC.binarize(image_0, 0.2)
    #image_0.show("Границі")
    width = image_0.array.shape[1]-1
    height = image_0.array.shape[0]-1
    """
    result = PSOCalc.pso_calc(image_0, {
        "a1": 0.5,  # self acceleration number
        "a2": 0.75,  # population acceleration number
        "pop_size": 500,  # population size
        "dim": 8,  # dimensions
        "pos_min": np.zeros(8),  # vector of minimum positions
        "pos_max": np.array([width, height,width, height,width, height,width, height]),  # vector of maximum positions
        "speed_min": np.ones(8)*(-50),  # vector of min speed
        "speed_max": np.ones(8)*(50),  # vector of max speed
        "braking": 0.7,  # speed depletion
    }, 500, True, True, point = [55, 45])
    """

    result = GeneticCalc.genetic_calc(image_0, 10, 20, 100, 8, [[0, width],[0, height]]*4,0.2,0.2,
                                      25, True, True, point = [1760*factor, 1440*factor])

    print(result[0:2])
    plt.plot(result[2])
    plt.title("Графік фітнес-функції")
    plt.xlabel("Ітерація")
    plt.ylabel("Значення фітнес-функції")
    #plt.yscale("log")
    plt.show()
    polygon = fitness.order_quad_vertices(result[1])
    for y in range(image_0.array.shape[0]):
        for x in range(image_0.array.shape[1]):
            if fitness.is_point_in_quad([x, y], fitness.Quad(polygon)):
                image_0.array[y][x][0] = 0
    image_0.show_with_polygon(polygon)
    image_0 = IC.ImageContainer(image_0_path)
    image_0.show_with_polygon(np.array(polygon)/factor)