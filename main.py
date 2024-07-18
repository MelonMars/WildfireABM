import numpy as np
import matplotlib.pyplot as plt
import vnoise
from scipy.ndimage import gaussian_filter

# Setup:


class GridPoint:
    def __init__(self, x, y, species, moisture, elev, fuel, density, state):
        self.x = x
        self.y = y
        self.species = species
        self.moisture = moisture
        self.elev = elev
        self.fuel = fuel
        self.density = density
        self.state = state


def linearGradient(a: float, b: float, x: float, y: float) -> (float, float):
    return a * x + b * y


width, height = 10, 10
arr = np.ones((width, height))
noise = vnoise.Noise()
noise.seed(np.random.randint(0, 10000000))
for i in range(width):
    for j in range(height):
        arr[i, j] = noise.noise2(i * 0.1, j * 0.1, grid_mode=True, lacunarity=0.3, octaves=1, persistence=5)

arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
arr = arr * 100 - 50
arr = arr ** 2
arr = arr * 0.5 + np.mean(arr) * 0.5

hill_radius = 300
hill_height = 700
center_x, center_y = width // 2, height // 2
for i in range(width):
    for j in range(height):
        dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
        if dist < hill_radius:
            arr[i, j] += hill_height * (1 - dist / hill_radius)

arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
arr = arr * 500
plt.imshow(arr, cmap='terrain', interpolation='nearest')
plt.colorbar()
plt.show()

grid = []
npGrid = np.zeros((width, height))
moisture, species = 3, 1
windDir = 0 # 0 > 360
windSpeed = 15 # mph
for i in range(width * height):
    fuel = np.random.randint(0, 100)
    density = np.random.randint(5, 20)
    gp = GridPoint(i % width, i // width, species, moisture, arr[i % width, i // width], fuel, density, 1)
    grid.append(gp)
    npGrid[i % width, i // width] = gp


# Fire starts

fire = np.zeros((width, height))
fireTile = np.random.choice(grid)
print(fireTile.x, fireTile.y)