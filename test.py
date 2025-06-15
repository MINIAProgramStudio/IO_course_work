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
    factor = 2**(-7.5)
    image_0 = IC.resize(image_0, factor)
    #image_0.show("Оригінальне зображення")

    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])

    image_0 = IC.apply_kernel(image_0, kernel)
    #image_0.show()

    image_0 = IC.average_channels(image_0)
    image_0 = IC.binarize(image_0, 0.2)
    #image_0.show("Границі")
    width = image_0.array.shape[1]-1
    height = image_0.array.shape[0]-1
    """
    result = PSOCalc.pso_calc(image_0, {
        "a1": 1,  # self acceleration number
        "a2":1.5,  # population acceleration number
        "pop_size": 1000,  # population size
        "dim": 8,  # dimensions
        "pos_min": np.zeros(8),  # vector of minimum positions
        "pos_max": np.array([width, height,width, height,width, height,width, height]),  # vector of maximum positions
        "speed_min": np.ones(8)*(-100),  # vector of min speed
        "speed_max": np.ones(8)*(100),  # vector of max speed
        "braking": 0.9,  # speed depletion
    }, 100, True, True)

    result = GeneticCalc.genetic_calc(image_0, 2500, 2500, 0, 8, [[0, width],[0, height]]*4,0.05,0.2,
                                      50, True, True, point = [110, 90])
    """

    print(result[0:2])
    plt.plot(result[2])
    plt.yscale("log")
    plt.title("Графік фітнес-функції")
    plt.xlabel("Ітерація")
    plt.ylabel("Значення фітнес-функції")
    plt.show()
    for y in range(image_0.array.shape[0]):
        for x in range(image_0.array.shape[1]):
            if fitness.is_point_in_quad([x, y], fitness.Quad(result[1])):
                image_0.array[y][x][0] = 0
    image_0.show_with_polygon(fitness.order_quad_vertices(result[1]))
    image_0 = IC.ImageContainer(image_0_path)
    image_0.show_with_polygon(np.array(fitness.order_quad_vertices(result[1]))/factor)