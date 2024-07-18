import numpy as np
import matplotlib.pyplot as plt
import vnoise
import math


# Setup:


class GridPoint:
    def __init__(self, x, y, veg, moisture, elev, fuel, density, state, length=10):
        """
        :param x:
        :param y:
        :param species:
        :param moisture:
        :param elev:
        :param fuel:
        :param density:
        :param state:
            State 0: Nothing flammable ever. Won't burn at all (e.g. roads, water)
            State 1: Flammable, but not currently burning
            State 2: Started burning just now
            State 3: Currently burning
            State 4: Burned out
        """
        self.x = x
        self.y = y
        self.veg = veg
        self.moisture = moisture
        self.elev = elev
        self.fuel = fuel
        self.density = density
        self.state = state
        self.length = length
        self.propFac = None

    def step(self, gpGrid: np.ndarray, windDir: float, windSpeed):
        if self.state == 0:
            self.state = 0
        elif self.state == 1:
            self.state = 1
        elif self.state == 2:
            self.state = 3
            self.spreadFire(gpGrid, windDir, windSpeed)
        elif self.state == 3:
            self.state = 4

    def spreadFire(self, gpGrid: np.ndarray, windDir: float, windSpeed: float):
        """
        Spread fire to neighboring cells
        Vegetation factor is from 1-3, 1 agricultural, 2 shrubs, 3 trees
        Density factor is from 1-3, 1 sparse, 2 medium, 3 dense
        Wind factor:
            c1 is a constant
            c2 is a constant
            V is the wind speed
            theta is the angle between fire direction and wind direction
        Slope factor:
            a is a constant
            thetaS is the slope angle of the path
        :return:
        """

        directions_horizontal_vertical = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        directions_diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        neighbors_hor_ver = [gpGrid[self.x + dx, self.y + dy] for dx, dy in directions_horizontal_vertical if 0 <= self.x + dx < width and 0 <= self.y + dy < height]
        neighbors_dia = [gpGrid[self.x + dx, self.y + dy] for dx, dy in directions_diagonal if 0 <= self.x + dx < width and 0 <= self.y + dy < height]
        c1 = 0
        c2 = 0
        V = windSpeed
        theta = self.propFac - windDir
        a = 0
        pH = 0  # Constant factor
        pW = np.exp(np.dot(c1, V)) * np.exp(np.dot(V, np.dot(c2, (math.cos(theta) - 1))))
        for tile in neighbors_hor_ver:
            thetaS = math.atan((tile.elev - self.elev) / self.length)
            pVeg = tile.veg # Vegetation factor
            pDen = tile.density # Density factor
            pS = np.exp(np.dot(a,thetaS)) # Slope factor
            pBurn = pH * (1 + pVeg) * (1 + pDen) * pW * pS
            if (np.random.randint(0, 1) > pBurn):
                tile.state = 2
                # Set tile propFac to the propagation factor from here (i.e. tell the tile the angle the fire came at it from)

        for tile in neighbors_dia:
            thetaS = math.atan((tile.elev - self.elev) / self.length*math.sqrt(2))
            pVeg = tile.veg  # Vegetation factor
            pDen = tile.density  # Density factor
            pS = np.exp(np.dot(a, thetaS))  # Slope factor
            pBurn = pH * (1 + pVeg) * (1 + pDen) * pW * pS
            if (np.random.randint(0, 1) > pBurn):
                tile.state = 2
                # Set tile propFac to the propagation factor from here (i.e. tell the tile the angle the fire came at it from)


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