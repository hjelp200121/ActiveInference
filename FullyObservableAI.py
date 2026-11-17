from ActiveInference import  ActiveAgent
import numpy as np

#Active inference agent that gets the full grid as input 
class FullyObservableAgent(ActiveAgent):
    def setFullGrid(self, grid, position=None):
        grid = np.asarray(grid)

        # Find Roomba position if it is represented in the grid
        if position is None:
            locations = np.argwhere(grid == -1)
            y, x = locations[0]
            position = (x, y)

        x, y = position

        # Set exact beliefs about obstacles and explored cells
        self.blockBelief = np.zeros(self.numCells, dtype=np.float64)
        self.exploredBelief = np.zeros(self.numCells, dtype=np.float64)

        for (cx, cy), idx in self.positionIndex.items():

            cell = grid[cy, cx]

            self.blockBelief[idx] = cell in {2, 3, 4, 5, 6}
            self.exploredBelief[idx] = cell == 1

        posIdx = self.positionIndex[(x, y)]
        self.positionBelief = np.zeros(self.numCells, dtype=np.float64)
        self.positionBelief[posIdx] = 1.0
        self.exploredBelief[posIdx] = 1.0

    def _advanceFullState(self, positionBelief, blockBelief, exploredBelief, action):
        current = np.argmax(positionBelief)
        target = self.actionOutcome[current, action]

        nextPosition = current

        if target >= 0 and blockBelief[target] < 1:
            nextPosition = target

        nextPos = np.zeros(self.numCells, dtype=np.float64)
        nextPos[nextPosition] = 1.0

        nextExplore = exploredBelief
        nextExplore[nextPosition] = 1.0

        return nextPos, blockBelief, nextExplore

    def _evaluatePolicy(self, policy, positionBelief, blockBelief, exploredBelief, depth=0):
        action = policy[depth]

        immediateEFE, _ = self._oneStepEFE(positionBelief, blockBelief, exploredBelief, action)

        if depth >= len(policy) - 1:
            return immediateEFE

        nextPos, nextMap, nextExplore = self._advanceFullState(positionBelief, blockBelief, exploredBelief, action)

        futureEFE = self._evaluatePolicy(policy, nextPos, nextMap, nextExplore, depth + 1)

        return immediateEFE + self.policyDiscount * futureEFE

if __name__ == "__main__":
    agent = FullyObservableAgent((10,10,"binary",0))