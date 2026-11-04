import numpy as np
from collections import deque
import random

from RoombaEnv import RoombaEnv

#Calculates the optimal path
#Uses BFS to find shortest path between all cells, which gives us a TSP using the shortest paths as edge weights
#Then uses NN + 2-opt heuristics to find the optimal path non-exact
#I know that it can be done exact using branc-and-cut and other algorithms but they were too annoying to implement

def _prepare(grid):
    #Has been made without numpy as i think normal arrays are faster since we use no matrix mult
    rows = [list(row) for row in grid]
    width = grid.shape[0]
    height = grid.shape[1]

    walkable = [False] * (height * width)
    startIdx = None
    emptyIdx = []

    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            idx = r * width + c
            if val == 0:
                walkable[idx] = True
                emptyIdx.append(idx)
            elif val == -1:
                walkable[idx] = True
                startIdx = idx

    neighbors = [[] for _ in range(height * width)]
    actions = [(0,-1), (1,0), (0,1), (-1,0)]
    for r in range(height):
        for c in range(width):
            idx = r * width + c 
            if not walkable[idx]:
                continue
            for (dr, dc) in actions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    nidx = nr * width + nc
                    if walkable[nidx]:
                        neighbors[idx].append(nidx)

    return neighbors, startIdx, emptyIdx

def _bfs(sourceId, neighbors, numCells):
    dists = [-1] * numCells
    parents = [-1] * numCells
    dists[sourceId] = 0

    queue = deque([sourceId])
    while queue:
        curr = queue.popleft()
        nextDist = dists[curr] + 1

        for neighbor in neighbors[curr]:
            if dists[neighbor] == -1:
                dists[neighbor] = nextDist
                parents[neighbor] = curr
                queue.append(neighbor)

    return dists, parents

def _nn(pairDist, rng):
    numPoints = len(pairDist)
    unvisited = set(range(1, numPoints))
    route = [0]

    while unvisited:
        curr = route[-1]
        nearestDist = min(pairDist[curr][target] for target in unvisited)
        tied = [target for target in unvisited if pairDist[curr][target] == nearestDist]
        choice = rng.choice(tied)
        route.append(choice)
        unvisited.remove(choice)

    return route

def _twoOpt(route, pairDist, maxPasses=1000):
    route = route.copy()
    numMoves = len(route)

    for _ in range(maxPasses):
        bestDelta = 0
        bestI = None
        bestJ = None
        for i in range(1, numMoves-1):
            a = route[i-1]
            b = route[i]

            for j in range(i + 1, numMoves):
                c = route[j]
                delta = pairDist[a][c] - pairDist[a][b]

                if j + 1 < numMoves:
                    d = route[j+1]
                    delta += pairDist[b][d] - pairDist[c][d]

                if delta < bestDelta:
                    bestDelta = delta
                    bestI = i
                    bestJ = j

        if bestI is None:
            break

        route[bestI:bestJ + 1] = reversed(route[bestI:bestJ + 1])

    return route


def optimalPath(grid, tries=1000, max2opt=1000, seed=0):
    grid = grid[1:-1, 1:-1]
    width = grid.shape[0]
    height = grid.shape[1]
    numCells = width * height

    neighbors, startPos, empty = _prepare(grid)
    points = [startPos] + empty
    numPoints = len(points)
    pairDist = [[-1] * numPoints for _ in range(numPoints)]
    parentsSource = []

    for sourceIdx, sourceId in enumerate(points):
        dist, parents = _bfs(sourceId, neighbors, numCells)
        parentsSource.append(parents)

        for destIdx, destId in enumerate(points):
            pairDist[sourceIdx][destIdx] = dist[destId]

    rng = random.Random(seed)
    bestLen = float("inf")

    for _ in range(tries):
        nnRoute = _nn(pairDist, rng)
        optRoute = _twoOpt(nnRoute, pairDist, max2opt)
        optLen = sum(pairDist[optRoute[i]][optRoute[i + 1]] for i in range(len(optRoute) - 1))

        if optLen < bestLen:
            bestLen = optLen

    return bestLen

if __name__ == "__main__":
    env = RoombaEnv((10,10), "binary", False, "preset", 10000, None, 0)
    env.reset()
    bestLen = optimalPath(env.grid, 1000, 1000)
    print(bestLen)