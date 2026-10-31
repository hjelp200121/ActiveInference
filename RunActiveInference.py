from RoombaEnv import RoombaEnv
from ActiveInference import ActiveAgent

show = "human"
player = "random"

env = RoombaEnv((10,10), "binary", True, "preset", 10000, show, 0)
envVariables = (env.width, env.height, env.observationMode, env.noiseLevel)
agent = ActiveAgent(envVariables)

obs, info = env.reset()
while True:
    if player == "ai":
        action = agent.chooseAction()
    else: 
        action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    agent.update(action, obs)

    if terminated or truncated:
        obs, info = env.reset()
        agent.resetEpisode()