import numpy as np
import gymnasium as gym

class RoombaEnv(gym.Env):
    #Partially observable 2d grid-world environment where the agents acts as a roomba trying to clean a room
    #Roomba moves one grid cell at a time in one of the directions.
    #It does not observe the grid or its own position, the observation is only the result of the last action.

    #actions: up, right, down, left = 0,1,2,3
    #objects: roomba, empty, cleared, wall, table, sofa, couch, chair = -1,0,1,2,3,4,5,6

    def __init__(self, gridSize, observationMode, randomStart, obstacleMode, maxSteps, renderMode, noiseLevel):
        super().__init__()

        ###Lav nogle checks om input passer

        self.width = gridSize[0]
        self.height = gridSize[1]
        self.observationMode = observationMode
        self.randomStart = randomStart
        self.obstacleMode = obstacleMode
        self.maxSteps = maxSteps
        self.noiseLevel = noiseLevel

        self.action_space = gym.spaces.Discrete(4)

        if observationMode == "binary":
            self.observation_space = gym.spaces.Discrete(2)
        elif observationMode == "object":
            self.observation_space = gym.spaces.Discrete(6) ###Skift når jeg adder flere ting
        else:
            raise ValueError("Please give valid observationMode, either \"binary\" or \"object\" ")

        #Internal state
        self.grid = None
        self.roombaPosition = None
        self.lastObservation = 0
        self.stepCount = 0

        # For rendering
        self.window = None
        self.clock = None
        self.renderMode = renderMode


    def reset(self, *, seed=None):
        super().reset(seed=seed)
        self.stepCount = 0
        self._createGrid()

        if self.renderMode == "human":
            self.render()

        self.lastObservation = 0
        info = self.grid, self.roombaPosition
        return self.lastObservation, info

    def step(self, action):
        #Execute one movement
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        self.stepCount += 1
        x,y = self.roombaPosition
        if action == 0:
            dx, dy = 0, -1
        elif action == 1:
            dx, dy = 1, 0
        elif action == 2:
            dx, dy = 0, 1
        elif action == 3:
            dx, dy = -1, 0
        else:
            raise ValueError(f"Error converting action to position, action: {action}")

        newPos = (x + dx, y + dy)
        #Check collision
        noiseVal = np.random.rand(1)
        obj = self._getObj(newPos)
        if obj == 0 or obj == 1:
            x,y = self.roombaPosition
            self.grid[y,x] = 1
            self.roombaPosition = newPos
            if noiseVal >= self.noiseLevel:
                self.lastObservation = 0
            else:
                self.lastObservation = 1
            x,y = newPos
            self.grid[y,x] = -1
        elif self.observationMode == "binary":
            if noiseVal >= self.noiseLevel:
                self.lastObservation = 1
            else:
                self.lastObservation = 0
        else:
            if noiseVal >= self.noiseLevel:
                self.lastObservation = obj
            else:
                self.lastObservation = 0

        ###Lige nu er det ikke environmentet der giver en reward
        reward = 0.0

        if np.count_nonzero(self.grid) == self.width*self.height:
            terminated = True
        else:
            terminated = False
        truncated = self.stepCount >= self.maxSteps

        info = self.grid, self.roombaPosition

        #Render
        if self.renderMode == "human":
            self.render()

        return self.lastObservation, reward, terminated, truncated, info

    def _createGrid(self):
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.grid[0,:] = 2
        self.grid[-1,:] = 2
        self.grid[:,0] = 2
        self.grid[:,-1] = 2

        if self.obstacleMode == "preset":
            if self.width != 10 or self.height != 10:
                raise ValueError(f"Grid size should be (10,10) for preset grid")
            self.grid[1,5:9] = 4
            self.grid[2,5] = 4
            self.grid[2,8] = 4
            self.grid[6:8,2] = 5
            self.grid[7,3] = 5
            self.grid[3,3] = 6
            self.grid[6,5] = 6
            self.grid[6,6] = 3
            self.grid[6,8] = 3
            self.grid[8,6] = 3
            self.grid[8,8] = 3
        else:
            ###Implementer funktionalitet for tilfældigt at genererede grids
            return

        if self.randomStart:
            while True:
                x,y = self._randPos()
                if self.grid[y, x] == 0:
                    self.grid[y, x] = -1
                    break
            self.roombaPosition = (x, y)
        else:
            self.grid[1,1] = -1
            self.roombaPosition = (1,1)
        return

    def _randPos(self):
        x = np.random.randint(1, self.width - 1)
        y = np.random.randint(1, self.height - 1)
        return x,y

    def _getObj(self, position):
        x, y = position
        return self.grid[y,x]


    #Chatgpt helped with coding the render part
    def render(self):
        import pygame

        #Create window or array
        cell_size = 50
        window_width = self.width * cell_size
        window_height = self.height * cell_size
        if self.window is None:
            pygame.init()

            if self.renderMode == "human":
                self.window = pygame.display.set_mode(
                    (window_width, window_height)
                )
                pygame.display.set_caption("Roomba Environment")

            else:
                self.window = pygame.Surface(
                    (window_width, window_height)
                )

            self.clock = pygame.time.Clock()

        self.drawPanel(self.window)
        if self.renderMode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(5)
            return None

        elif self.renderMode == "rgb_array":
            array = pygame.surfarray.array3d(self.window)
            # pygame uses (width, height, channels)
            # Gymnasium convention is (height, width, channels)
            return np.transpose(array, (1, 0, 2))

    def drawPanel(self, surface, cellSize=50, showRoomba=True, showBlocks=True):
        import pygame
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x*cellSize, y*cellSize, cellSize, cellSize)
                objectType = self.grid[y, x]

                ###Aflys denne her hvis det er showblocks
                # Cleared cell
                if objectType == 1:
                    color = (0, 128, 0) #green

                elif objectType == -1:
                    if showRoomba:
                        #Roomba
                        color = (128, 128, 128) #grey
                    else:
                        # Cleared cell (because the roomba is on top)
                        color = (0, 128, 0) #green

                elif showBlocks:
                    # Wall
                    if objectType == 2:
                        color = (0, 0, 0) #black
                    
                    # Table
                    elif objectType == 3:
                        color = (101, 67, 33) #dark brown
                    
                    # Sofa
                    elif objectType == 4:
                        color = (0, 0, 128) #blue
                    
                    # Couch
                    elif objectType == 5:
                        color = (173, 216, 230) #light blue
                    
                    # Chair
                    elif objectType == 6:
                        color = (150, 75, 0) #brown

                    # Empty cell
                    else:
                        color = (128,0,0) #red

                # Wall
                elif objectType == 2:
                    color = (0, 0, 0) #black
                
                # Empty cell
                else:
                    color = (128,0,0) #red
        
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (180, 180, 180), rect, width=1)

        #draw roomba again as circle
        if showRoomba:
            x, y = self.roombaPosition
            center = (x*cellSize + cellSize//2, y * cellSize + cellSize//2)
            radius = cellSize//3
            pygame.draw.circle(surface, (50, 100, 220), center, radius)

    def close(self):
        #Close rendering window
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

            self.window = None
            self.clock = None

if __name__ == "__main__":
    env = RoombaEnv((10,10), "binary", False, "preset", 10000, "human", 0)