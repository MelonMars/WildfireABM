import time

import numpy as np
import matplotlib.pyplot as plt
import vnoise
import math
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector


# Setup:


class GridPoint(Agent):
    def __init__(self, unique_id, model, x, y, veg, moisture, elev, fuel, density, state, length=10):
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
        super().__init__(unique_id, model)
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
        self.windDir = 0
        self.windSpeed = 0
        self.gpGrid = None

    def step(self):
        if self.state == 1:
            return  # Nothing happens if flammable but not currently burning
        elif self.state == 2:
            self.state = 3  # Transition from started burning to currently burning
            self.spreadFire(self.gpGrid, self.windDir, self.windSpeed)
        elif self.state == 3:
            self.state = 4

    def spreadFire(self, gpGrid: np.ndarray, windDir: float, windSpeed: float):
        """
        Spread fire to neighboring cells.
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
        width, height = gpGrid.shape
        directions_horizontal_vertical = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        directions_diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        neighbors_hor_ver = [
            gpGrid[self.x + dx, self.y + dy]
            for dx, dy in directions_horizontal_vertical
            if 0 <= self.x + dx < width and 0 <= self.y + dy < height
        ]
        neighbors_dia = [
            gpGrid[self.x + dx, self.y + dy]
            for dx, dy in directions_diagonal
            if 0 <= self.x + dx < width and 0 <= self.y + dy < height
        ]

        c1 = 0
        c2 = 0
        V = windSpeed
        theta = self.propFac - windDir
        a = 0
        pH = 0  # Constant factor
        pW = np.exp(np.dot(c1, V)) * np.exp(np.dot(V, np.dot(c2, (math.cos(theta) - 1))))

        for tile in neighbors_hor_ver + neighbors_dia:
            if tile.state != 1:  # Only spread to flammable cells
                continue
            thetaS = math.atan(
                (tile.elev - self.elev) / (self.length if tile in neighbors_hor_ver else self.length * math.sqrt(2)))
            pVeg = tile.veg  # Vegetation factor
            pDen = tile.density  # Density factor
            pS = np.exp(np.dot(a, thetaS))  # Slope factor
            pBurn = pH * (1 + pVeg) * (1 + pDen) * pW * pS
            if np.random.random() > pBurn:
                tile.state = 2  # Set the neighbor to start burning
                tile.propFac = math.degrees(math.atan2(tile.y - self.y, tile.x - self.x))  # Set the propagation factor


class WildfireModel(Model):
    def __init__(self, width, height, hill_radius, hill_height, moisture, species, wind_dir, wind_speed):
        super().__init__()
        self.width = width
        self.height = height
        self.hill_radius = hill_radius
        self.hill_height = hill_height
        self.moisture = moisture
        self.species = species
        self.wind_dir = wind_dir
        self.wind_speed = wind_speed
        self.grid = []
        self.npGrid = np.empty((width, height), dtype=object)
        self.fire_started = False

        # Initialize terrain using vnoise
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

        for i in range(width):
            for j in range(height):
                fuel = np.random.randint(0, 100)
                density = np.random.randint(5, 20)
                gp = GridPoint(
                    unique_id=i * width + j,  # Example unique_id calculation, adjust as needed
                    model=self,
                    x=i,
                    y=j,
                    veg=self.species,
                    moisture=self.moisture,
                    elev=arr[i, j],
                    fuel=fuel,
                    density=density,
                    state=1  # Initial state, adjust as needed
                )
                self.grid.append(gp)
                self.npGrid[i, j] = gp

        # Start fire
        fireTile = np.random.choice(self.grid)
        fireTile.state = 2
        fireTile.propFac = 0
        self.fire_started = True

        self.schedule = self.create_schedule()

        self.datacollector = DataCollector(
            agent_reporters={"State": "state"}
        )

    def create_schedule(self):
        return RandomActivation(self)

    def step(self):
        for gp in self.grid:
            gp.gpGrid = self.npGrid
            gp.windDir = self.wind_dir
            gp.windSpeed = self.wind_speed
            gp.step()

        if self.schedule.steps % 4 == 0:
            self.visualize_grid()

        self.datacollector.collect(self)

    def visualize_grid(self):
        plt.imshow(np.array([[instance.state for instance in row] for row in self.npGrid]), cmap='terrain',
                   interpolation='nearest')
        plt.colorbar()
        plt.show()
        time.sleep(0.1)


if __name__ == "__main__":
    model = WildfireModel(width=100, height=100, hill_radius=30, hill_height=700, moisture=3, species=1, wind_dir=0,
                          wind_speed=15)
    for i in range(100):
        model.step()

        time.sleep(0.1)
