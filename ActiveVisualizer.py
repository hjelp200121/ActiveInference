import numpy as np
import pygame

# Chatgpt helped with coding the render part
class RoombaVisualizer:
    def __init__(self, env, agent, cellSize=23, headerHeight=30, displayFps=10):
        pygame.font.init()

        self.pygame = pygame
        self.env = env
        self.agent = agent

        self.cellSize = cellSize
        self.headerHeight = headerHeight
        self.displayFps = displayFps

        self.panelWidth = env.width * cellSize
        self.panelHeight = env.height * cellSize
        self.rowHeight = self.panelHeight + headerHeight

        # Two columns and three rows.
        self.canvasWidth = 2 * self.panelWidth
        self.canvasHeight = 3 * self.rowHeight

        # This is just an off-screen drawing surface.
        self.canvas = pygame.Surface((self.canvasWidth, self.canvasHeight))

        self.screen = None
        self.clock = None
        self.quitRequested = False

        self.titleFont = pygame.font.SysFont(None, 22)

        self.panelTitles = [
            ["real: Roomba position", "belief: Roomba position"],
            ["real: Obstacles",       "belief: Obstacles"],
            ["real: Combined",        "belief: Combined"],
        ]

    def render(self, showWindow=False):
        pygame = self.pygame
        self.canvas.fill((250, 250, 250))

        for row in range(3):
            for col in range(2):
                x = col * self.panelWidth
                y = row * self.rowHeight

                # Draw the panel header.
                title = self.panelTitles[row][col]
                text = self.titleFont.render(title, True, (30, 30, 30))

                self.canvas.blit(text, (x + 6, y + (self.headerHeight - text.get_height()) // 2))

                # Create a view into the relevant part of the canvas.
                panelRect = pygame.Rect(x, y + self.headerHeight, self.panelWidth, self.panelHeight)

                panelSurface = self.canvas.subsurface(panelRect)

                #Column 1: real
                if col == 0:
                    if row == 0:
                        self.env.drawPanel(panelSurface, self.cellSize, True, False)

                    elif row == 1:
                        self.env.drawPanel(panelSurface, self.cellSize, False, True)

                    else:
                        self.env.drawPanel(panelSurface, self.cellSize, True, True)

                #Column 2: belief
                else:
                    mode = ["position", "blocks", "combined"][row]
                    self.agent.drawBeliefPanel(panelSurface, self.cellSize, mode)

                # Draw a frame around each panel.
                pygame.draw.rect(self.canvas, (100, 100, 100), pygame.Rect(x, y, self.panelWidth, self.rowHeight), width=1)

        # Display the composed image only when requested.
        if showWindow:
            if self.screen is None:
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.canvasWidth, self.canvasHeight))
                pygame.display.set_caption("Real vs belief of active inference")
                self.clock = pygame.time.Clock()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.quitRequested = True

            if not self.quitRequested:
                self.screen.blit(self.canvas, (0, 0))
                pygame.display.flip()
                self.clock.tick(self.displayFps)

        # Return an image regardless of whether a window was shown.
        # Pygame arrays are (width, height, channels); convert to
        # the standard image layout (height, width, channels).
        return np.transpose(pygame.surfarray.array3d(self.canvas), (1, 0, 2)).copy()

    def close(self):
        if self.screen is not None:
            self.pygame.display.quit()
            self.screen = None
            self.clock = None

if __name__ == "__main__":
    from RoombaEnv import RoombaEnv
    from ActiveInference import ActiveAgent
    from ActiveVisualizer import RoombaVisualizer
    import imageio.v2 as imageio

    show = True
    filename = "ActiveInference/Roomba/gifs/roombaActiveInference.gif"

    env = RoombaEnv((10,10), "binary", True, "preset", 10000, None, 0.0) ###sæt tilbage til true
    envVariables = (env.width, env.height, env.observationMode, env.noiseLevel)
    agent = ActiveAgent(envVariables)
    visualizer = RoombaVisualizer(env, agent)

    with imageio.get_writer(filename, mode="I", fps=10, loop=0) as writer:
        obs, info = env.reset()
        writer.append_data(visualizer.render(show))
        while True:
            action = agent.chooseAction()
            obs, reward, terminated, truncated, info = env.step(action)
            agent.update(action, obs)
            writer.append_data(visualizer.render(show))

            if terminated or truncated:
                obs, info = env.reset()
                agent.resetEpisode()

            if visualizer.quitRequested:
                break

    visualizer.close()
    env.close()