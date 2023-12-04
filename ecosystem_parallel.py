import random
import cv2
import os
import math
from copy import copy
import functools

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec

def env(render_mode=None):
    env = raw_env(render_mode=render_mode)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

# ObsType = TypeVar("ObsType")
# ActionType = TypeVar("ActionType")

def raw_env(render_mode=None):
     env = parallel_env(render_mode=render_mode)
     env = parallel_to_aec(env)
     return env

class parallel_env(ParallelEnv):
    metadata = {
        "name": "ecosystem_v1",
        "render_modes": ["human", "ansi", "rgb_array"],
        "is_parallelizable": False,
        "render_fps": 2
    }

    def __init__(self, render_mode, fish_amount=3, shark_amount=3):
        super().__init__()
        self.render_mode = render_mode

        self.gridsize = (32, 32)
        self.min_depth = 32
        self.max_depth = 64
        print("Loading map")
        self.filename = os.path.join(
            os.path.dirname(__file__), 'depth_map.png')
        print("Map loaded")
        self.depth_map = cv2.imread(self.filename)
        self.depth_map = cv2.cvtColor(self.depth_map, cv2.COLOR_BGR2GRAY)
        self.depth_map = cv2.resize(
            self.depth_map, self.gridsize)
        self.cross = math.sqrt(
            self.gridsize[0] ** 2 + self.gridsize[1] ** 2 + self.max_depth ** 2)

        # Environment properties

        self.fish_amount = fish_amount
        self.shark_amount = shark_amount
        self.max_timesteps = 200000
        self.current_fish = 0
        self.current_shark = 0

        # Agent properties

        self.foodAmount = 4
        self.food_value = 200
        self.distance_factor = 10

        self.fish_max_food = 400
        self.max_hp = 2000
        self.fish_cost_s = 1  # Cost of doing nothing
        self.fish_cost_m = 2  # Cost of basic movement
        self.fishEatRange = 3

        self.shark_max_food = 400
        self.shark_damage = 140
        self.shark_cost_s = 1  # Cost of doing nothing
        self.shark_cost_m = 2  # Cost of basic movement
        self.sharkEatRange = 3

        self.death_penalty = -200
        self.damage_penalty_multiplier = 0.5
        self.required_egg_age = self.fish_max_food * 2
        self.possible_agents = [f"fish_{i}" for i in range(
            self.fish_amount)] + [f"shark_{i}" for i in range(self.shark_amount)]  # Creates agents
        self.foods = []

        self.win_reward = 500

        self.timestep = 0  # Resets the timesteps
        self.terrain_shape = (3, 3)
        self.r_x = int((self.terrain_shape[0] - 1) / 2)
        self.r_y = int((self.terrain_shape[0] - 1) / 2)
        
        metadata = {"render_modes": ["human"], "name": "ecosystem_v1"}

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # return spaces.Discrete(43)
        return spaces.Box(-1.0, 1.0, (23,), np.float32)
        # return spaces.Dict()

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(7)
    
    def action_mask():
        return spaces.Box(0, 1, (7,), int)

    def render(self):  # Simple OpenCV display of the environment
        image = self.toImage((400, 400))
        scale_x = 400 / 32
        scale_y = 400 / 32
        fish_color = (0, 0, 255)
        shark_color = (255, 255, 0)
        for agent in self.agents:
            if "fish" in agent:
                color = fish_color
            else:
                color = shark_color
            org = (int(self.agentData[agent]["y"] * scale_x),
                   int(self.agentData[agent]["x"] * scale_y - 10))
            image = cv2.putText(image, str(self.agentData[agent]["depth"]), org, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
        cv2.imshow("map", image)
        cv2.waitKey(1)

    def toImage(self, window_size):  # Converts the map to a ready to display image
        fish_color = [0, 0, 255]
        shark_color = [255, 255, 0]
        food_color = [0, 255, 0]

        img = cv2.cvtColor(self.depth_map, cv2.COLOR_GRAY2BGR)

        for food in self.foods:
            img[food[0]][food[1]][0] = food_color[0]
            img[food[0]][food[1]][1] = food_color[1]
            img[food[0]][food[1]][2] = food_color[2]

        for agent in self.agents:
            a_x = self.agentData[agent]["x"]
            a_y = self.agentData[agent]["y"]
            a_d = self.agentData[agent]["depth"]
            color_factor = 0.25 + (1 - (a_d / self.max_depth)) * 0.75
            if "fish" in agent:
                img[a_x][a_y][0] = fish_color[0] * color_factor
                img[a_x][a_y][1] = fish_color[1] * color_factor
                img[a_x][a_y][2] = fish_color[2] * color_factor
            else:
                img[a_x][a_y][0] = shark_color[0] * color_factor
                img[a_x][a_y][1] = shark_color[1] * color_factor
                img[a_x][a_y][2] = shark_color[2] * color_factor

        return cv2.resize(img, window_size, interpolation=cv2.INTER_NEAREST)

    def close(self):
        cv2.destroyAllWindows()

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = copy(self.possible_agents)
        self.agentData = {}
        self.timestep = 0
        self.map = np.zeros(self.gridsize)
        self.foods = []
        self.generateMap()  # This function starts the world - loads the map from depth_map.png,
        # spawns agents and generates food
        print("Reset start")
        self.terminated = []

        self.current_fish = 0
        self.current_shark = 0

        for agent in self.agents:
            if "fish" in agent:
                self.current_fish += 1
            else:
                self.current_shark += 1

        self.rewards = {
            name: 0.0 for name in self.agents
        }

        observations = {agent: self.observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        print("Reset done")
        self.state = observations

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}

        The action order should be as follows:
        1. Attacks -> Check all damaged agents if they will live
        2. Rest of the agent actions
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        # rewards for all agents are placed in the rewards dictionary to be returned
        self.rewards = {name: 0.0 for name in self.agents}
        terminations = {agent: False for agent in self.agents}
        attack = []
        rest = {}
        for agent, action in actions.items():
            if agent in self.agents:
                if action == 7 and "shark" in agent:
                    attack.append(agent)
                else:
                    rest[agent] = action

        for agent in attack:
            food = self.performAction(agent, 6)
            if food > 0:
                self.rewards[agent] += food * 20            
            else:
                self.rewards[agent] += food
        for agent, action in rest.items():
            food = self.performAction(agent, action)
            if food > 0:
                self.rewards[agent] += food * 20            
            else:
                self.rewards[agent] += food

        self.timestep += 1
        env_truncation = self.timestep >= self.max_timesteps
        if env_truncation == False:
            current_fish = 0
            current_shark = 0
            for agent in self.agents:
                if "fish" in agent:
                    current_fish += 1
                else:
                    current_shark += 1
            if current_shark == 0 or current_fish == 0:
                for agent in self.rewards:
                    self.rewards[agent] += self.win_reward
                env_truncation = True
        truncations = {agent: env_truncation for agent in self.agents}

        for agent in self.agents:
            if self.agentData[agent]["hp"] <= 0 or self.agentData[agent]["food"] <= 0:
                self.agents.pop(self.agents.index(agent))
                terminations[agent] = True
                self.rewards[agent] += self.death_penalty

        observations = {
            agent: self.observation(agent) for agent in self.agents
        }
        self.state = observations

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, self.rewards, terminations, truncations, infos
    
    def observation(self, agentName):
        a_x = self.agentData[agentName]["x"]
        a_y = self.agentData[agentName]["y"]
        a_d = self.agentData[agentName]["depth"]
        agentType = self.agentData[agentName]["agentType"]

        closest_shark_1 = [self.cross, 0]
        closest_shark_2 = [self.cross, 0]
        closest_fish_1 = [self.cross, 0]
        closest_fish_2 = [self.cross, 0]
        closest_food_1 = [self.cross, 0]
        closest_food_2 = [self.cross, 0]
        self_data = []

        terrain = np.full(self.terrain_shape, 1.0)
        surrounding = [1.0, 1.0, 1.0, 1.0, 1.0]

        for x in range(-self.r_x, self.r_x + 1):
            for y in range(-self.r_y, self.r_y + 1):
                c_x = a_x + x
                c_y = a_y + y
                if 0 <= c_x < self.gridsize[0] and 0 <= c_y < self.gridsize[1]:
                    terrain[x][y] = (self.map[c_x][c_y] - a_d) / self.max_depth * 2 - 1

        surrounding[0] = terrain[2][1]
        surrounding[1] = terrain[0][1]
        surrounding[2] = terrain[1][1]
        surrounding[3] = terrain[1][2]
        surrounding[4] = terrain[1][0]
        for agent in self.agents:
            if agentName != agent:
                b_x = self.agentData[agent]["x"]
                b_y = self.agentData[agent]["y"]
                b_d = self.agentData[agent]["depth"]
                if self.rayCast(a_x, a_y, a_d, b_x, b_y, b_d):
                    x, y, d, dist, best_action = self.calcVector(
                        agentName, agent)
                    if "fish" in agent:
                        if dist < closest_fish_1[0]:
                            closest_fish_2[0] = closest_fish_1[0]
                            closest_fish_2[1] = closest_fish_1[1]
                            closest_fish_1[0] = dist
                            closest_fish_1[1] = best_action
                        elif dist < closest_fish_2[0]:
                            closest_fish_2[0] = dist
                            closest_fish_2[1] = best_action
                    else:
                        if dist < closest_shark_1[0]:
                            closest_shark_2[0] = closest_shark_1[0]
                            closest_shark_2[1] = closest_shark_1[1]
                            closest_shark_1[0] = dist
                            closest_shark_1[1] = best_action
                        elif dist < closest_shark_2[0]:
                            closest_shark_2[0] = dist
                            closest_shark_2[1] = best_action

        for food in self.foods:
                x, y, d, dist, best_action = self.calcVectorCoord(
                    agentName, food[0], food[1], food[2])
                if dist < closest_food_1[0]:
                    closest_food_2[0] = closest_food_1[0]
                    closest_food_2[1] = closest_food_1[1]
                    closest_food_1[0] = dist
                    closest_food_1[1] = best_action
                elif dist < closest_food_2[0]:
                    closest_food_2[0] = dist
                    closest_food_2[1] = best_action

        closest_shark_1[0] = closest_shark_1[0] / self.cross * 2 - 1
        closest_shark_2[0] = closest_shark_2[0] / self.cross * 2 - 1
        closest_fish_1[0] = closest_fish_1[0] / self.cross * 2 - 1
        closest_fish_2[0] = closest_fish_2[0] / self.cross * 2 - 1
        closest_food_1[0] = closest_food_1[0] / self.cross * 2 - 1
        closest_food_2[0] = closest_food_2[0] / self.cross * 2 - 1
        closest_shark_1[1] = closest_shark_1[1] / 3.5 - 1
        closest_shark_2[1] = closest_shark_2[1] / 3.5 - 1
        closest_fish_1[1] = closest_fish_1[1] / 3.5 - 1
        closest_fish_2[1] = closest_fish_2[1] / 3.5 - 1
        closest_food_1[1] = closest_food_1[1] / 3.5 - 1
        closest_food_2[1] = closest_food_2[1] / 3.5 - 1
        b_x = self.agentData[agentName]["x"] / self.cross * 2 - 1
        b_y = self.agentData[agentName]["y"] / self.cross * 2 - 1
        b_d = self.agentData[agentName]["depth"] / self.cross * 2 - 1
        hp = self.agentData[agentName]["hp"] / self.max_hp * 2 - 1
        
        if "fish" in agentName:
            food = self.agentData[agentName]["food"] / self.fish_max_food * 2 - 1
        else:
            food = self.agentData[agentName]["food"] / self.shark_max_food * 2 - 1
        self_data = [agentType, b_x, b_y, b_d, hp, food]

        if "fish" in agentName:
            distanceReward = (closest_shark_1[0] * 2 - closest_food_1[0]) / 3 * self.distance_factor
        else:
            distanceReward = -closest_fish_1[0] * self.distance_factor

        self.rewards[agentName] += distanceReward

        
        observation = self_data + surrounding + closest_fish_1 + closest_fish_2 + closest_shark_1 + closest_shark_2 + closest_food_1 + closest_food_2
        return np.array(observation)

    
    # This function either performs or calculates the effect of performing a certain action
    def performAction(self, agentName, action):
        cost_s = self.shark_cost_s
        cost_m = self.shark_cost_m
        if "fish" in agentName:
            cost_s = self.fish_cost_s
            cost_m = self.fish_cost_m
        wrong_penalty = cost_m * 3
        x = self.agentData[agentName]["x"]
        y = self.agentData[agentName]["y"]
        d = self.agentData[agentName]["depth"]
        f = 0

        movement_x = 0
        movement_y = 0
        movement_d = 0
        match action:
            case 0:  # y+
                if y < self.gridsize[1] - 1:
                    movement_y = 1
                    f -= cost_m
                else:
                    self.rewards[agentName] -= wrong_penalty
            case 1:  # y-
                if y > 0:
                    movement_y = -1
                    f -= cost_m
                else:
                    self.rewards[agentName] -= wrong_penalty
            case 2:  # x+
                if x < self.gridsize[0] - 1:
                    movement_x = 1
                    f -= cost_m
                else:
                    self.rewards[agentName] -= wrong_penalty
            case 3:  # x-
                if x > 0:
                    movement_x = -1
                    f -= cost_m
                else:
                    self.rewards[agentName] -= wrong_penalty
            case 4:  # depth+
                if d < self.map[x][y] - 1:
                    movement_d = 1
                    f -= cost_m
                else:
                    self.rewards[agentName] -= wrong_penalty
            case 5:  # depth-
                if d > 0:
                    movement_d = -1
                    f -= cost_m
                else:
                    self.rewards[agentName] -= wrong_penalty
            case 6:
                eaten = False
                if "fish" in agentName:
                    for food in self.foods:
                        if self.getDistance(x, y, d, food[0], food[1], food[2]) <= self.fishEatRange and eaten == False:
                            f += self.food_value
                            self.foods.pop(self.foods.index(food))
                            eaten = True
                            self.generateNewFood()
                            break                            
                else:
                    for agent in self.agents:
                        if "fish" in agent:
                            if self.getDistance(x, y, d, self.agentData[agent]["x"], self.agentData[agent]["y"], self.agentData[agent]["depth"]) <= self.sharkEatRange and eaten == False:
                                self.damage(agent)
                                f += self.shark_damage
                                eaten = True
                                break
                if eaten == False:
                    self.rewards[agentName] -= wrong_penalty

        if f == 0:
            f -= cost_s
            if action == 6:
                f -= cost_m

        self.agentData[agentName]["x"] += movement_x
        self.agentData[agentName]["y"] += movement_y
        self.agentData[agentName]["depth"] += movement_d
        self.agentData[agentName]["food"] += f
        return f
    
    def damage(self, agentID):
        self.agentData[agentID]["hp"] -= self.shark_damage
        self.rewards[agentID] -= self.shark_damage * \
            self.damage_penalty_multiplier

    def calcVector(self, agent1, agent2):
        x = self.agentData[agent2]["x"] - self.agentData[agent1]["x"]
        y = self.agentData[agent2]["y"] - self.agentData[agent1]["y"]
        d = self.agentData[agent2]["depth"] - self.agentData[agent1]["depth"]
        distance = self.getDistance(self.agentData[agent1]["x"],
                                    self.agentData[agent1]["y"],
                                    self.agentData[agent1]["depth"],
                                    self.agentData[agent2]["x"],
                                    self.agentData[agent2]["y"],
                                    self.agentData[agent2]["depth"])
        return x, y, d, distance, self.bestAction(x, y, d, distance)
    
    def calcVectorCoord(self, agent1, b_x, b_y, b_d):
        x = b_x - self.agentData[agent1]["x"]
        y = b_y - self.agentData[agent1]["y"]
        d = b_d - self.agentData[agent1]["depth"]
        distance = self.getDistance(self.agentData[agent1]["x"],
                                    self.agentData[agent1]["y"],
                                    self.agentData[agent1]["depth"],
                                    b_x,
                                    b_y,
                                    b_d)
        return x, y, d, distance, self.bestAction(x, y, d, distance)
    
    # Function to select what action should be taken to go towards coordinates
    def bestAction(self, x, y, d, dist):
        if dist < 2:
            return 7
        if abs(y) >= abs(x) and abs(y) >= abs(d):
            if y > 0:
                return 0
            else:
                return 1
        if abs(x) >= abs(y) and abs(x) >= abs(d):
            if x > 0:
                return 2
            else:
                return 3
        if abs(d) >= abs(y) and abs(d) >= abs(x):
            if x > 0:
                return 4
            else:
                return 5
            
    def getDistance(self, a_x, a_y, a_d, b_x, b_y, b_d):
        return math.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2 + (b_d - a_d) ** 2)

    def rewardDistance(self, agentName):
        # This is the cross distance of the map, which is the maixmum distance there can be
        nearestFish = self.cross
        nearestShark = self.cross
        nearestFood = self.cross
        c_x = self.agentData[agentName]["x"]
        c_y = self.agentData[agentName]["y"]
        c_d = self.agentData[agentName]["depth"]

        for agent in self.agents:
            a_x = self.agentData[agent]["x"]
            a_y = self.agentData[agent]["y"]
            a_d = self.agentData[agent]["depth"]
            dist = self.getDistance(c_x, c_y, c_d, a_x, a_y, a_d)
            if "fish" in agent:
                if dist < nearestFish:
                    nearestFish = dist
            else:
                if dist < nearestShark:
                    nearestShark = dist

        if "fish" in agentName:
            for food in self.foods:
                dist = self.getDistance(
                    c_x, c_y, c_d, food[0], food[1], food[2])
                if dist < nearestFood:
                    nearestFood = dist

            distanceReward = ((nearestShark / self.cross) +
                              (0.5 - (nearestFood / self.cross))) * self.distance_factor
        else:
            distanceReward = (0.5 - nearestFish / self.cross) * \
                self.distance_factor

        self.rewards[agentName] += distanceReward

    def cstCoord(self, x, y):  # Constrain the passed coordinates so they don't exceed the map
        if x > self.gridsize[0] - 1:
            x = self.gridsize[0] - 1
        elif x < 0:
            x = 0
        if y > self.gridsize[1] - 1:
            y = self.gridsize[1] - 1
        elif y < 0:
            y = 0
        return x, y

    def checkForFood(self, agentName):  # Check food in 1 grid radius
        foods = []
        for x in range(3):
            for y in range(3):
                c_x, c_y = self.cstCoord(
                    self.agentData[agentName]["x"] + x, self.agentData[agentName]["x"] + y)
                if (
                    self.map[c_x][c_y] == 2 and "fish" in agentName
                ):
                    foods.append([x, y])
        return foods

    def gridRInt(self, xy):  # Returns random int in the map axis limit
        if xy == "x":
            return random.randint(0, self.gridsize[0] - 1)
        else:
            return random.randint(0, self.gridsize[1] - 1)
        
    def generateMap(self):
        # Loads map file and converts it to obstacle map
        self.map = ((self.max_depth - self.min_depth)*(self.depth_map -
                                                       np.min(self.depth_map))/np.ptp(self.depth_map)).astype(int) + self.min_depth

        # Generate starting food
        foodGenerated = 0
        while foodGenerated < self.foodAmount:
            food_x = self.gridRInt("x")
            food_y = self.gridRInt("y")
            depth_point = self.map[food_x][food_y]
            self.foods.append([food_x, food_y, depth_point])
            foodGenerated += 1

        # Generate agents
        for agentName in self.agents:
            x = self.gridRInt("x")
            y = self.gridRInt("y")
            depth_point = self.map[x][y]
            depth = random.randint(0, depth_point - 1)
            self.createNewAgent(x, y, depth, agentName, False)

    # Check how much food is present and generate what is missing
    def generateNewFood(self):
        currentFood = len(self.foods)
        while currentFood < self.foodAmount:
            x = self.gridRInt("x")
            y = self.gridRInt("y")
            depth_point = self.map[x][y]
            self.foods.append([x, y, depth_point])
            currentFood += len(self.foods)

    # Check if 2 coordinates have line of sight
    def rayCast(self, a_x, a_y, a_d, b_x, b_y, b_d):
        vec = [b_x - a_x, b_y - a_y, b_d - a_d]
        dist = self.getDistance(a_x, a_y, a_d, b_x, b_y, b_d)
        if abs(dist) < 0.001:
            return True
        mini = [vec[0] / dist, vec[1] / dist, vec[2] / dist]
        c_x = a_x
        c_y = a_y
        c_d = a_d

        for i in range(round(dist) + 1):
            if self.map[round(c_x)][round(c_y)] < c_d:
                if round(c_x) == b_x and round(c_y) == b_y:
                    return True
                else:
                    return False
            c_x += mini[0]
            c_y += mini[1]
            c_d += mini[2]

            if b_x < 0:
                if c_x <= b_x:
                    return True
            else:
                if c_x >= b_x:
                    return True
            if b_y < 0:
                if c_y <= b_y:
                    return True
            else:
                if c_y >= b_y:
                    return True

        return True
    
    def createNewAgent(self, x, y, d, agentName, is_new):
        if is_new:
            self.agents.append(agentName)
            self.rewards[agentName] = 0.0
        if "fish" in agentName:
            hp = self.max_hp
            food = self.fish_max_food
            self.current_fish += 1
            agentType = 1
        else:
            hp = 1000
            food = self.shark_max_food
            self.current_shark += 1
            agentType = -1

        self.agentData[agentName] = {
            "x": x, "y": y, "depth": d, "hp": hp, "food": food, "egg": self.timestep, "agentType": agentType}