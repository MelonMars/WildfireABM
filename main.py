import numpy as np
import matplotlib.pyplot as plt
import vnoise


class GridPoint:
    def __init__(self, x, y, vegType, moisture, elev, fire: bool):
        self.x = x
        self.y = y
        self.vegType = vegType
        self.moisture = moisture
        self.elev = elev
        self.fire = False


# Make gradient
def linearGradient(a: float, b: float, x: float, y: float) -> (float, float):
    return a * x + b * y


a = 1
b = 1
"""
arr = np.zeros((10, 10))
for i in range(0, 10):
    for ii in range(0, 10):
        arr[i][ii] = linearGradient(a, b, i, ii) * noise.z_noise(i, ii, 0.1, 2)


plt.imshow(arr, cmap='hot', interpolation='nearest')
plt.show()
"""
# Gonna use perlin noise tho cause better and stuff
arr = np.zeros((10, 10))
noise = vnoise.Noise()
width, height = arr.shape
for i in range(width):
    for j in range(height):
        arr[i][j] = noise.noise2(i*width, j*height, 0.1, False)

plt.imshow(arr, cmap='terrain', interpolation='nearest')
plt.colorbar()
plt.show()