import numpy as np
import pygame

# Chatgpt helped with coding the render part
class RoombaVisualizer:
    def __init__(self, env, agent, cellSize=40, headerHeight=30, statusHeight=40, displayFps=10):
        pygame.font.init()

        self.pygame = pygame
        self.env = env
        self.agent = agent

        self.cellSize = cellSize
        self.headerHeight = headerHeight
        self.statusHeight = statusHeight
        self.displayFps = displayFps

        self.panelWidth = env.width * cellSize
        self.panelHeight = env.height * cellSize
        self.rowHeight = self.panelHeight + headerHeight

        # Two columns and three rows.
        self.canvasWidth = 3 * self.panelWidth
        self.canvasHeight = statusHeight + 2 * self.rowHeight

        # This is just an off-screen drawing surface.
        self.canvas = pygame.Surface((self.canvasWidth, self.canvasHeight))

        self.screen = None
        self.clock = None
        self.quitRequested = False

        self.titleFont = pygame.font.SysFont(None, 22)
        self.status_font = pygame.font.SysFont(None, 25)

        self.panelTitles = [
            ["real: Roomba position", "real: Obstacles", "real: Combined"],
            ["belief: Roomba position", "belief: Obstacles", "belief: Combined"],
        ]

    def render(self, showWindow=False, optimalMoves=None, currentMoves=0, lastMoves=None):
        pygame = self.pygame
        self.canvas.fill((250, 250, 250))

        optText = (str(optimalMoves) if optimalMoves is not None else "N/A")

        lastText = (str(lastMoves) if lastMoves is not None else "N/A")

        text = [f"Optimal moves: {optText}", f"Current moves: {currentMoves}", f"Last environment: {lastText}"]

        for col, text in enumerate(text):
            showText = self.status_font.render(text, True,(30, 30, 30))
            center = (col * self.panelWidth + self.panelWidth // 2)
            textBox = showText.get_rect(center=(center, self.statusHeight // 2))
            self.canvas.blit(showText, textBox)

        pygame.draw.line(self.canvas, (100, 100, 100), (0, self.statusHeight - 1), (self.canvasWidth, self.statusHeight - 1), width=1)

        for row in range(2):
            for col in range(3):
                x = col * self.panelWidth
                y = self.statusHeight + row * self.rowHeight

                # Draw the panel header.
                title = self.panelTitles[row][col]
                text = self.titleFont.render(title, True, (30, 30, 30))

                self.canvas.blit(text, (x + 6, y + (self.headerHeight - text.get_height()) // 2))

                # Create a view into the relevant part of the canvas.
                panelRect = pygame.Rect(x, y + self.headerHeight, self.panelWidth, self.panelHeight)

                panelSurface = self.canvas.subsurface(panelRect)

                #Column 1: real
                if row == 0:
                    if col == 0:
                        self.env.drawPanel(panelSurface, self.cellSize, True, False)

                    elif col == 1:
                        self.env.drawPanel(panelSurface, self.cellSize, False, True)

                    else:
                        self.env.drawPanel(panelSurface, self.cellSize, True, True)

                #Column 2: belief
                else:
                    self.agent.drawBeliefPanel(panelSurface, self.cellSize, ["position", "blocks", "combined"][col])

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
    from OptimalMoves import optimalPath
    import imageio.v2 as imageio

    randomStart = False
    show = True
    filename = "gifs/roombaActiveInference.gif"

    env = RoombaEnv((10,10), "binary", randomStart, "preset", 10000, None, 0.0) ###sæt tilbage til true
    envVariables = (env.width, env.height, env.observationMode, env.noiseLevel)
    agent = ActiveAgent(envVariables)
    visualizer = RoombaVisualizer(env, agent)

    with imageio.get_writer(filename, mode="I", fps=10, loop=0) as writer:
        obs, info = env.reset()
        agent.resetEpisode(fullReset=True, randomStart=randomStart)
        optimalMoves = optimalPath(env.grid, 1000, 1000)
        currentMoves = 0
        lastMoves = None
        writer.append_data(visualizer.render(show, optimalMoves, currentMoves, lastMoves))
        while True:
            action = agent.chooseAction()
            obs, reward, terminated, truncated, info = env.step(action)
            currentMoves += 1
            agent.update(action, obs)
            writer.append_data(visualizer.render(show, optimalMoves, currentMoves, lastMoves))

            if terminated or truncated:
                obs, info = env.reset()
                agent.resetEpisode(randomStart=randomStart)
                lastMoves = currentMoves
                currentMoves = 0

            if visualizer.quitRequested:
                break

    visualizer.close()
    env.close()