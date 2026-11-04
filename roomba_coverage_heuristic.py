"""Heuristic coverage planner for RoombaEnv.

Grid convention:
    -1: Roomba's starting position
     0: empty cell that must be visited
     1: empty/cleared cell that must be visited (as specified by the caller)
    2-6: obstacles (not traversable)

The planner builds shortest-path distances between the start and every target
using BFS, then solves the resulting open-route TSP approximately with:
    1. Nearest neighbour (NN) to create an initial target order.
    2. 2-opt to improve that order by reversing route segments.

The Roomba may revisit cells and does not need to return to its start.

Gymnasium action mapping:
    0 = up, 1 = right, 2 = down, 3 = left

No third-party dependencies are required. The input may be a Python nested
list or a NumPy array (NumPy is not required by this module).
"""

from collections import deque
import random


ACTIONS = {
    0: (-1, 0),  # up
    1: (0, 1),   # right
    2: (1, 0),   # down
    3: (0, -1),  # left
}

WALKABLE_VALUES = {-1, 0, 1}


def _prepare_grid(grid):
    """Validate the grid and return a flat walkability map and key positions."""
    rows = [list(row) for row in grid]
    if not rows or not rows[0]:
        raise ValueError("Grid cannot be empty.")

    height = len(rows)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Grid must be rectangular.")

    walkable = [False] * (height * width)
    start_id = None
    target_ids = []

    for r, row in enumerate(rows):
        for c, raw_value in enumerate(row):
            value = int(raw_value)
            cell_id = r * width + c

            if value in WALKABLE_VALUES:
                walkable[cell_id] = True

            if value == -1:
                if start_id is not None:
                    raise ValueError("Grid contains more than one Roomba (-1).")
                start_id = cell_id
            elif value in (0, 1):
                target_ids.append(cell_id)

    if start_id is None:
        raise ValueError("Grid must contain exactly one Roomba position (-1).")

    return rows, height, width, walkable, start_id, target_ids


def _build_neighbors(height, width, walkable):
    """Precompute traversable neighbours for every cell, using flat IDs."""
    neighbors = [[] for _ in range(height * width)]

    for r in range(height):
        for c in range(width):
            cell_id = r * width + c
            if not walkable[cell_id]:
                continue

            # Keep the same deterministic order as the environment actions.
            for dr, dc in ACTIONS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    next_id = nr * width + nc
                    if walkable[next_id]:
                        neighbors[cell_id].append(next_id)

    return neighbors


def _bfs(source_id, neighbors, number_of_cells):
    """Return shortest distances and predecessor IDs from one source."""
    distances = [-1] * number_of_cells
    parents = [-1] * number_of_cells
    distances[source_id] = 0

    queue = deque([source_id])
    while queue:
        current = queue.popleft()
        next_distance = distances[current] + 1

        for neighbor in neighbors[current]:
            if distances[neighbor] == -1:
                distances[neighbor] = next_distance
                parents[neighbor] = current
                queue.append(neighbor)

    return distances, parents


def _nearest_neighbor_order(distance, rng):
    """Build a route starting at node 0 by repeatedly selecting the nearest
    unvisited target. Equal-distance choices are randomized by ``rng``.
    """
    number_of_points = len(distance)
    unvisited = set(range(1, number_of_points))
    route = [0]  # Point 0 is the Roomba's starting position.

    while unvisited:
        current = route[-1]
        nearest_distance = min(distance[current][candidate] for candidate in unvisited)
        tied_candidates = [
            candidate
            for candidate in unvisited
            if distance[current][candidate] == nearest_distance
        ]
        chosen = rng.choice(tied_candidates)
        route.append(chosen)
        unvisited.remove(chosen)

    return route


def _route_length(route, distance):
    """Length of an open route; there is no edge back to the start."""
    return sum(distance[route[i]][route[i + 1]] for i in range(len(route) - 1))


def _two_opt(route, distance, max_passes=100):
    """Improve an open route using best-improvement 2-opt.

    The start (route[0]) is fixed. A move reverses a consecutive section of
    the target sequence. Because grid distances are symmetric, the costs of
    the edges inside the reversed section do not change; only the two boundary
    edges need to be considered.

    Returns the improved route and the number of passes performed.
    """
    route = route.copy()
    n = len(route)
    passes = 0

    if n < 3:
        return route, passes

    for _ in range(max_passes):
        best_delta = 0
        best_i = None
        best_j = None

        # i starts at 1 so the starting point is never moved.
        for i in range(1, n - 1):
            a = route[i - 1]
            b = route[i]

            for j in range(i + 1, n):
                c = route[j]

                # Replace edge (a,b) with (a,c).
                delta = distance[a][c] - distance[a][b]

                # If j is not the final point, also replace (c,d) with (b,d).
                if j + 1 < n:
                    d = route[j + 1]
                    delta += distance[b][d] - distance[c][d]

                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_i is None:
            break  # No improving 2-opt reversal remains.

        route[best_i:best_j + 1] = reversed(route[best_i:best_j + 1])
        passes += 1

    return route, passes


def _reconstruct_path(source_id, destination_id, parents, width):
    """Reconstruct one BFS path as (row, column) coordinates."""
    path_ids = []
    current = destination_id

    while current != -1:
        path_ids.append(current)
        if current == source_id:
            break
        current = parents[current]

    if not path_ids or path_ids[-1] != source_id:
        raise RuntimeError("Could not reconstruct a path between two targets.")

    path_ids.reverse()
    return [divmod(cell_id, width) for cell_id in path_ids]


def _action_between(a, b):
    """Convert adjacent coordinates into a RoombaEnv action index."""
    delta = (b[0] - a[0], b[1] - a[1])
    for action, direction in ACTIONS.items():
        if direction == delta:
            return action
    raise ValueError(f"Coordinates {a} and {b} are not adjacent.")


def heuristic_coverage_plan(grid, restarts=10, max_2opt_passes=100, seed=0):
    """Plan a heuristic route visiting every cell with value 0 or 1.

    Args:
        grid: Rectangular nested list or NumPy array. Values -1, 0, and 1 are
            traversable; values 2-6 are treated as obstacles.
        restarts: Number of NN constructions to try. Equal-distance nearest
            choices are randomized, then each route is improved with 2-opt.
            Set to 1 for one deterministic-seed NN+2-opt run.
        max_2opt_passes: Maximum best-improvement 2-opt passes per restart.
        seed: Random seed for reproducible tie-breaking.

    Returns:
        Dictionary with:
            moves:       Heuristic route length in individual grid moves.
            path:        Full list of visited coordinates, including detours.
            actions:     Gymnasium action indices (0=up, 1=right, 2=down, 3=left).
            target_order: Waypoint order selected by NN+2-opt.
            reachable:   False if at least one required cell cannot be reached.
            initial_moves: NN route length before 2-opt, for the best restart.
            two_opt_passes: Number of improving passes for the chosen route.
            restarts:    Number of NN constructions attempted.

    This is a heuristic, not an optimality guarantee. The route is an open
    path: it does not need to return to the starting position.
    """
    if restarts < 1:
        raise ValueError("restarts must be at least 1.")
    if max_2opt_passes < 0:
        raise ValueError("max_2opt_passes cannot be negative.")

    _, height, width, walkable, start_id, target_ids = _prepare_grid(grid)
    start = divmod(start_id, width)
    number_of_targets = len(target_ids)

    if number_of_targets == 0:
        return {
            "moves": 0,
            "path": [start],
            "actions": [],
            "target_order": [],
            "reachable": True,
            "initial_moves": 0,
            "two_opt_passes": 0,
            "restarts": restarts,
        }

    neighbors = _build_neighbors(height, width, walkable)
    points = [start_id] + target_ids
    number_of_points = len(points)
    number_of_cells = height * width

    # Compute distances between all important points. Keep BFS predecessor
    # arrays so we can later expand the selected waypoint route into grid moves.
    pair_dist = [[-1] * number_of_points for _ in range(number_of_points)]
    parents_by_source = []

    for source_index, source_id in enumerate(points):
        distances, parents = _bfs(source_id, neighbors, number_of_cells)
        parents_by_source.append(parents)

        for destination_index, destination_id in enumerate(points):
            pair_dist[source_index][destination_index] = distances[destination_id]

    # Since this is an undirected grid, all targets must be in the start's
    # connected component. If any is disconnected, full coverage is impossible.
    if any(pair_dist[0][j] < 0 for j in range(1, number_of_points)):
        return {
            "moves": float("inf"),
            "path": [],
            "actions": [],
            "target_order": [],
            "reachable": False,
            "initial_moves": float("inf"),
            "two_opt_passes": 0,
            "restarts": restarts,
        }

    rng = random.Random(seed)
    best_route = None
    best_initial_length = None
    best_length = float("inf")
    best_passes = 0

    for _ in range(restarts):
        initial_route = _nearest_neighbor_order(pair_dist, rng)
        initial_length = _route_length(initial_route, pair_dist)
        improved_route, passes = _two_opt(
            initial_route, pair_dist, max_passes=max_2opt_passes
        )
        improved_length = _route_length(improved_route, pair_dist)

        if improved_length < best_length:
            best_route = improved_route
            best_length = improved_length
            best_initial_length = initial_length
            best_passes = passes

    # Expand the selected waypoint sequence into the actual cell-by-cell path.
    full_path = [start]
    for route_position in range(1, len(best_route)):
        source_point_index = best_route[route_position - 1]
        destination_point_index = best_route[route_position]
        source_id = points[source_point_index]
        destination_id = points[destination_point_index]

        segment = _reconstruct_path(
            source_id,
            destination_id,
            parents_by_source[source_point_index],
            width,
        )
        full_path.extend(segment[1:])  # Avoid duplicating segment start.

    actions = [
        _action_between(full_path[i], full_path[i + 1])
        for i in range(len(full_path) - 1)
    ]

    # This invariant should hold: each pairwise distance is expanded into
    # exactly that many single-cell moves.
    if len(actions) != best_length:
        raise RuntimeError(
            "Route reconstruction length does not match the waypoint distance."
        )

    target_order = [divmod(points[index], width) for index in best_route[1:]]

    return {
        "moves": int(best_length),
        "path": full_path,
        "actions": actions,
        "target_order": target_order,
        "reachable": True,
        "initial_moves": int(best_initial_length),
        "two_opt_passes": best_passes,
        "restarts": restarts,
    }


if __name__ == "__main__":
    example_grid = [
        [-1, 0, 0, 2, 0],
        [ 0, 2, 0, 2, 0],
        [ 0, 0, 0, 0, 0],
        [ 2, 0, 2, 0, 0],
    ]

    result = heuristic_coverage_plan(example_grid, restarts=10, seed=0)
    if result["reachable"]:
        print("Heuristic moves:", result["moves"])
        print("NN moves before 2-opt:", result["initial_moves"])
        print("2-opt improving passes:", result["two_opt_passes"])
        print("Waypoint order:", result["target_order"])
        print("Full coordinate path:", result["path"])
        print("Gymnasium actions:", result["actions"])
    else:
        print("Not all required cells are reachable from the Roomba.")
