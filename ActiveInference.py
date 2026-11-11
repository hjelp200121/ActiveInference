import numpy as np
from itertools import product

#A-B-C-D Active inference algorithm
#Based on the book Fundamentals of active inference
#Uses a factorised binary map for its beliefs
class ActiveAgent:
    def __init__(self, envVariables, initialObstacleProbability=0.2, horizon=3, temperature=4.0, preference=0.5, policyDiscount=1.0, seed=None):

        self.rng = np.random.default_rng(seed) #Used for picking random actions

        #Hyper parameters for the agent
        self.initialObstacleProbability = initialObstacleProbability #What is the initial chance of blocks, slightly useless
        self.horizon = horizon #How many moves ahead should the agent look
        self.temperature = temperature #How much should lower EFE actions be prioritized if not deterministic
        self.preference = preference #What is the preference for not colliding ###Denne her er dårlig obv og skal måske ændres?
        self.policyDiscount = policyDiscount #Discount to future policies
    
        ###Lav nogle input checks her

        #Get variables from the environment
        self.width, self.height, self.observationMode, self.noiseLevel = envVariables
        #Currently the agent only works for observationMode "binary"
        if self.observationMode != "binary":
            raise ValueError("Need ObservationMode to be \"binary \"")

        #Create possible positions
        positions = [(x,y) for y in range(1, self.height - 1) for x in range(1, self.width - 1)]
        self.positionIndex = {p: i for i, p in enumerate(positions)} #Faster lookup (and easier to code) for positions
        self.numCells = len(positions)

        #Create possible actions
        self.actions = [(0,-1), (1,0), (0,1), (-1,0)] #up, right, down, left
        self.numActions = len(self.actions)

        #Create array where, given current position and planned action, outputs endpoint
        self.actionOutcome = np.full((self.numCells, self.numActions), -1, dtype=np.int32)
        for i, (x, y) in enumerate(positions):
            for j, (dx, dy) in enumerate(self.actions):
                target = (x + dx, y + dy)
                if target in self.positionIndex:
                    self.actionOutcome[i, j] = self.positionIndex[target]

        #C model, currently just the preference for not colliding
        self.C = np.array([self.preference, 1 - self.preference], dtype=np.float64)

    def resetEpisode(self, fullReset=False, randomStart=True):
        #Call after environment reset
        #Gives starting B model from D model (not saved explicit)

        if fullReset:
            self.blockBelief = np.full(self.numCells, self.initialObstacleProbability, dtype=np.float64) #Initial block belief at creation of agent, uses obstacleProb hyperparameter

        ###Skal måske også resette noget af dens belief om blocke? 

        if randomStart == True:
            #Changes position prior to be weighted towards cells the agent think is free based on earlier environments
            startWeights = np.clip(1.0 - self.blockBelief, 1e-9, 1.0)
            self.positionBelief = startWeights / startWeights.sum()
        else:
            self.positionBelief = np.zeros_like(self.blockBelief, dtype=np.float64)
            self.positionBelief[0] = 1.0

        if self.positionBelief.max() > 1.0:
            raise ValueError("Position belief > 1 after reseting episode")

    def _collisionProbabilities(self, blockBelief, action):
        #Calculates P(collision| position, action, map belief)
        targets = self.actionOutcome[:, action]
        collisionProb = np.ones(self.numCells, dtype=np.float64)
        valid = targets >= 0
        collisionProb[valid] = blockBelief[targets[valid]]
        if collisionProb.max() > 1.0:
            raise ValueError("Collision prob > 1 in collisionprobabilities")
        if collisionProb.min() < 0.0:
            raise ValueError("collision prob < 0 in collisionprobabilities")
        return collisionProb

    def _observationProbabilities(self, probCol):
        #A model: P(observation| map belief, position belief, action)
        probColObs = probCol * (1.0 - self.noiseLevel) + (1.0 - probCol) * self.noiseLevel
        return np.array([1.0 - probColObs, probColObs], dtype=np.float64)

    def _inferObservation(self, positionBelief, blockBelief, action, observation, test=False):
        #Calculates the updated beliefs after action and observation

        collisionProb = self._collisionProbabilities(blockBelief, action)
        if observation == 1:
            likelihoodCol = 1.0 - self.noiseLevel
            likelihoodMove = self.noiseLevel
        else:
            likelihoodCol = self.noiseLevel
            likelihoodMove = 1.0 - self.noiseLevel

        #Calculate the probability, given current beliefs, that the observation is true
        likelihoodTruth = collisionProb * likelihoodCol + (1 - collisionProb) * likelihoodMove #Given position, how much do we believe the observation
        probTruth = positionBelief @ likelihoodTruth
        probTruth = np.clip(probTruth, 1e-15, 1) #The prob sometimes becomes 1.0000000000000002 due error in the normalisation of the position belief

        #Update position belief
        nextPos = np.zeros(self.numCells, dtype=np.float64)
        for source in range(self.numCells):
            weight = positionBelief[source]
            probBlock = collisionProb[source]
            target = self.actionOutcome[source, action]

            nextPos[source] += weight * probBlock * likelihoodCol
            if target >= 0:
                nextPos[target] += weight * (1 - probBlock) * likelihoodMove

        if not nextPos.sum() == 0:
            nextPos = nextPos / nextPos.sum()

        #Update map belief
        ###Skal lige checke denne her igen
        nextMap = blockBelief.copy()
        for cell in range(self.numCells):
            source = self.actionOutcome[cell, action-2] #Equivalent to actionIncome[cell, action] deprecated
            if source < 0: #it is a wall
                continue

            likelihoodSource = likelihoodTruth[source]
            posEvidence = probTruth - positionBelief[source] * likelihoodSource
            mapBelief = blockBelief[cell] * (posEvidence + positionBelief[source] * likelihoodCol)
            nextMap[cell] = np.clip(mapBelief / probTruth, 0.0, 1.0)

        # if nextPos.max() > 1.0:
        #     raise ValueError("nextPos in inferobs is > 1")
        # if nextMap.max() > 1.0:
        #     raise ValueError("nextMap in inferobs is > 1")
        # if nextPos.min() < 0.0:
        #     raise ValueError("nextPos in inferobs is < 0")
        # if nextPos.min() < 0.0:
        #     raise ValueError("nextMap in inferobs is > 1")

        # if test:
        #     print("hej")

        return nextPos, nextMap

    def update(self, action, observation):
        #Updates the agent
        self.positionBelief, self.blockBelief = self._inferObservation(self.positionBelief, self.blockBelief, action, observation, True)

    @staticmethod
    def _binaryEntropy(probability):
        if probability <= 0.0 or probability >= 1.0:
            return 0.0
        return -probability * np.log(probability) - (1.0 - probability) * np.log(1.0 - probability)

    def _oneStepEFE(self, positionBelief, blockBelief, action):
        collisionProb = self._collisionProbabilities(blockBelief, action)
        probCol = positionBelief @ collisionProb
        probCol = np.clip(probCol, 0, 1) #The prob sometimes becomes 1.0000000000000002 due error in the normalisation of the position belief
        probObs = self._observationProbabilities(probCol)

        risk = np.sum(probObs * (np.log(np.clip(probObs, 1e-15, 1.0)) - np.log(self.C)))

        ambiguity = self._binaryEntropy(probObs[1]) - self._binaryEntropy(self.noiseLevel)

        if ambiguity < 0.0:
            print(f"Negative ambiguity {ambiguity}")
            ambiguity = 0.0

        EFE = risk + ambiguity ###Måske tilføj mere vægt til ambiguity
        return EFE, probObs

    def _evaluatePolicy(self, policy, positionBelief, blockBelief, depth=0):
        #Recursive function for evaluating policy

        action = policy[depth]
        immediateEFE, probObs = self._oneStepEFE(positionBelief, blockBelief, action)
        if depth >= len(policy) - 1:
            return immediateEFE
        
        expectedEFE = 0.0
        for obs in (0,1): #act like we have seen observation (no collision) and then after act like we have seen observation collision
            prob = probObs[obs]
            if not prob == 0.0:
                nextPos, nextMap = self._inferObservation(positionBelief, blockBelief, action, obs)
                if nextPos.sum() == 0.0:
                    continue
                futureVal = self._evaluatePolicy(policy, nextPos, nextMap, depth + 1)
                expectedEFE += prob * futureVal

        return immediateEFE + self.policyDiscount * expectedEFE

    def chooseAction(self, deterministic = False):
        #Chose action based on horizon 

        ###Vil gerne lave en mere sofistikeret valg af action, så horizon kan være større
        policies = list(product(range(self.numActions), repeat=self.horizon))
        EFE = np.array([self._evaluatePolicy(policy, self.positionBelief, self.blockBelief) for policy in policies], dtype=np.float64)
        shiftedWeights = -self.temperature * (EFE - EFE.min())
        weights = np.exp(shiftedWeights)
        policyProbs = weights / weights.sum()

        if deterministic:
            policyIndex = np.argmin(EFE)
        else:
            policyIndex = self.rng.choice(len(policies), p=policyProbs)

        action = policies[policyIndex][0]
        return action

    #Chatgpt helped with coding the render part
    def drawBeliefPanel(self, surface, cellSize=50, mode="position", blockThreshold=0.8, posThreshold=0.5):
        import pygame

        if mode not in {"position", "blocks", "combined"}:
            raise ValueError("mode in drawBeliefPanel must be 'position', 'blocks' or 'combined'")

        def blend(a, b, amount):
            return tuple(int(a * (1.0 - amount) + b * amount) for a, b in zip(a, b))

        red = (128,0,0)
        black = (0,0,0)
        blue = (50, 100, 220)
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x*cellSize, y*cellSize, cellSize, cellSize)
                idx = self.positionIndex.get((x,y))

                #Wall
                if idx is None:
                    color = black

                elif mode == "blocks":
                    #The more black the higher prob
                    probBlock = self.blockBelief[idx]
                    color = blend(red, black, probBlock)

                elif mode == "combined":
                    probBlock = self.blockBelief[idx]
                    #shows blocks if they are above 80% prob
                    if probBlock >= blockThreshold:
                        color = black
                    else:
                        color = red

                else: #position
                    probPos = self.positionBelief[idx]
                    color = blend(red, blue, probPos)

                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (180, 180, 180), rect, width=1)

                #Shows where it thinks there is a high likelihood the agent is
                if idx is not None and mode == "combined":
                    probPos = self.positionBelief[idx]
                    if probPos >= posThreshold:
                        center = (x*cellSize + cellSize//2, y * cellSize + cellSize//2)
                        radius = cellSize//3
                        pygame.draw.circle(surface, (50, 100, 220), center, radius)
                    

if __name__ == "__main__":
    agent = ActiveAgent((10,10,"binary",0))