from RoombaEnv import RoombaEnv
from ActiveInference import ActiveAgent
from FullyObservableAI import FullyObservableAgent

show = "human"
player = "ai"

randomStart = True

env = RoombaEnv((10,10), "binary", randomStart, "preset", 10000, show, 0)
envVariables = (env.width, env.height, env.observationMode, env.noiseLevel)

if player == "ai":
    agent = ActiveAgent(envVariables)
elif player == "fullObs":
    agent = FullyObservableAgent(envVariables)

obs, info = env.reset()

if player == "ai":
    agent.resetEpisode(fullReset=True, randomStart=randomStart)
elif player == "fullObs":
    agent.resetEpisode(fullReset=True, randomStart=randomStart)
    agent.setFullGrid(info[0], info[1])

while True:
    if player == "ai":
        action = agent.chooseAction()
    elif player == "random":
        action = env.action_space.sample()
    elif player == "fullObs":
        action = agent.chooseAction()
    obs, reward, terminated, truncated, info = env.step(action)
    if player == "ai":
        agent.update(action, obs)
    elif player == "fullobs":
        agent.setFullGrid(info[0], info[1])

    if terminated or truncated:
        obs, info = env.reset()
        agent.resetEpisode(randomStart=randomStart)