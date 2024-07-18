import numpy as np
import vnoise
import math
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace, MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider
from mesa.visualization.modules import ChartModule


# Setup:


class GridPoint(Agent):
    def __init__(self, unique_id, model, x, y, veg, moisture, elev, fuel, density, state, c1, c2, a, pH, length=10):
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
        # Next values are defined in the spreadFire method
        self.c1 = 0
        self.c2 = 0
        self.a = 0
        self.pH = 0
        self.pBurn = None

    def step(self):
        self.pBurn = None
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

        neighbors = self.model.grid.get_neighbors((self.x, self.y), moore=True, include_center=False)

        V = windSpeed
        theta = self.propFac - windDir
        pW = np.exp(self.c1 * V) * np.exp(self.c2 * (math.cos(math.radians(theta)) - 1))

        for tile in neighbors:
            if tile.state != 1:  # Only spread to flammable cells
                continue

            dx = tile.x - self.x
            dy = tile.y - self.y
            is_diagonal = (dx, dy) in directions_diagonal

            distance = self.length if not is_diagonal else self.length * math.sqrt(2)
            thetaS = math.atan((tile.elev - self.elev) / distance)

            pVeg = tile.veg  # Vegetation factor
            pDen = tile.density  # Density factor
            pS = np.exp(self.a * thetaS)  # Slope factor

            self.pBurn = self.pH * (1 + pVeg) * (1 + pDen) * pW * pS
            print(self.pBurn)
            if np.random.random() < self.pBurn:
                tile.state = 2  # Set the neighbor to start burning
                tile.propFac = math.degrees(math.atan2(tile.y - self.y, tile.x - self.x))


class WildfireModel(Model):
    def __init__(self, width, height, hill_radius, hill_height, moisture, species, wind_dir, wind_speed, c1, c2, a, pH):
        super().__init__()
        self.width = width
        self.height = height
        self.hill_radius = hill_radius
        self.hill_height = hill_height
        self.moisture = moisture
        self.species = species
        self.wind_dir = wind_dir
        self.wind_speed = wind_speed
        self.grid = MultiGrid(width, height, torus=True)
        self.npGrid = np.empty((width, height), dtype=object)
        self.fire_started = False
        self.schedule = RandomActivation(self)
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
                    state=1,  # Initial state, adjust as needed
                    c1=c1,
                    c2=c2,
                    a=a,
                    pH=pH
                )
                self.grid.place_agent(gp, (i, j))
                self.schedule.add(gp)
                self.npGrid[i, j] = gp

        # Start fire
        fireTile = np.random.choice(self.schedule.agents)
        fireTile.state = 2
        fireTile.propFac = 0
        self.fire_started = True

        self.datacollector = DataCollector(
            agent_reporters={"State": "state",
                             "PBurn": "pBurn"},
            model_reporters={"Average_pBurn": self.compute_average_pBurn,
                             "Burners": self.compute_burners},
        )

    def compute_average_pBurn(self):
        agent_pBurns = [agent.pBurn for agent in self.schedule.agents if agent.pBurn is not None]
        if agent_pBurns:
            return sum(agent_pBurns) / len(agent_pBurns)
        else:
            return 100

    def compute_burners(self):
        return len([agent for agent in self.schedule.agents if agent.pBurn is not None])

    def step(self):
        # Possibly change wind dir and speed here for more complex simulations
        for agent in self.schedule.agents:
            agent.windDir = self.wind_dir
            agent.windSpeed = self.wind_speed
            agent.gpGrid = self.npGrid
            agent.step()

        self.datacollector.collect(self)

    def grid(self):
        portrayal = {}
        for gp in self.grid:
            portrayal[gp.unique_id] = {
                "x": gp.x,
                "y": gp.y,
                "Shape": "rect",
                "w": 1,
                "h": 1,
                "Filled": "true",
                "Color": "red" if gp.state == 2 else "green"
            }
        return portrayal


def agent_portrayal(agent):
    if agent.state == 2:
        color = "red"
    elif agent.state == 1:
        color = "green"
    else:
        color = "black"

    return {"Shape": "rect", "Color": color, "Filled": "true", "Layer": 0, "w": 1, "h": 1}


grid = CanvasGrid(agent_portrayal, 100, 100, 500, 500)
chart = ChartModule(
    [
        {"Label": "Average_pBurn", "Color": "Black"},
        {"Label": "Burners", "Color": "Red"}
     ],
    data_collector_name='datacollector'
)


if __name__ == "__main__":

    server = ModularServer(
        WildfireModel,
        [grid, chart],
        "Wildfire Model",
        {"width": 100,
            "height": 100,
            "hill_radius": 30,
            "hill_height": 700,
            "moisture": Slider("Moisture", 1, 1, 10),
            "species": Slider("Species", 1, 1, 3),
            "wind_dir": Slider("Wind Direction", 0, 0, 360),
            "wind_speed": Slider("Wind Speed", 15, 0, 30),
            "c1": Slider("c1", 1, -100, 100),
            "c2": Slider("c2", 1, -100, 100),
            "a": Slider("a", 1, -100, 100),
            "pH": Slider("pH", 1, -100, 100),
         }
    )

    server.port = 8521
    server.launch()
