import pygame
from RoombaEnv import RoombaEnv

env = RoombaEnv((10,10), "binary", False, "preset", 10000, "human", 0)
obs, info = env.reset()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            pygame.quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                action = 0
            elif event.key == pygame.K_d:
                action = 1
            elif event.key == pygame.K_s:
                action = 2
            elif event.key == pygame.K_a:
                action = 3
            else:
                continue
            # Perform the action
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                obs, info = env.reset()