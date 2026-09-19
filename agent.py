# agent.py
import random
import math
from collections import deque
import heapq


class SimpleReflexAgent:
    """Practical 02 - Simple Reflex Agent.

    Uses strictly IF-THEN condition-action rules on the CURRENT percept only.
    There is deliberately NO __init__ and NO memory / percept history, so in a
    partially observable world it can get trapped repeating the same cycle
    (for example in a corner) - the failure the lab asks you to observe.
    """

    def sense_and_act(self, percept: dict) -> str:
        # Rule 1: IF food_here THEN suck (the environment collects the pellet).
        if percept.get('food_here', False):
            return 'Suck'

        # Rule 2: IF wall_ahead THEN turn (pick a different direction).
        if percept.get('wall_ahead', False):
            return 'Right'

        # Rule 3: ELSE move forward.
        return 'Up'


class ModelBasedAgent:
    """Practical 02 - Model-Based Reflex Agent (internal memory state).

    The agent never sees its coordinates, so it keeps its own model of the
    world in RELATIVE coordinates (its start cell is (0, 0)) and updates it
    from the last action and the current percept before choosing a rule.
    """

    ACTIONS = ['Up', 'Right', 'Down', 'Left']
    DELTAS = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}

    def __init__(self):
        self.actions_pool = list(self.ACTIONS)
        # ---- internal memory state ----
        self.position = (0, 0)          # dead-reckoned (relative) position
        self.visited_cells = {(0, 0)}   # cells I have been to
        self.visit_counts = {}          # how often I have stood on each cell
        self.blocked_cells = set()      # walls / edges I have discovered
        self.toxic_cells = set()        # cells where I smelled a toxin
        self.facing = 'Up'              # the environment starts facing Up
        self.last_action = None         # last action taken
        self.last_percept = None        # last percept recorded

    def _target(self, action):
        dx, dy = self.DELTAS[action]
        return (self.position[0] + dx, self.position[1] + dy)

    def _update_state(self, percept: dict):
        """Update memory BEFORE choosing an action (transition + sensor model)."""
        # Transition model: how did my last action change my position?
        if self.last_action in self.DELTAS:
            target = self._target(self.last_action)
            if percept.get('bumped', False):
                self.blocked_cells.add(target)   # the move failed: obstacle there
            else:
                self.position = target           # the move succeeded
            self.facing = self.last_action       # I now face the way I tried to go

        # Sensor model: what do my sensors say about the cell ahead / here?
        if percept.get('wall_ahead', False):
            self.blocked_cells.add(self._target(self.facing))
        if percept.get('smells_toxin', False):
            self.toxic_cells.add(self.position)

        self.visited_cells.add(self.position)
        self.visit_counts[self.position] = self.visit_counts.get(self.position, 0) + 1
        self.last_percept = percept              # record the current percept

    def _commit(self, action: str) -> str:
        self.last_action = action                # record the action taken
        return action

    def sense_and_act(self, percept: dict) -> str:
        # Step 1: update the internal state from the last action + new percept.
        self._update_state(percept)

        # Step 2: IF-THEN rules that query the memory.
        # Rule 1: IF food_ahead AND that cell is not a known obstacle THEN go for it.
        if percept.get('food_ahead', False) and \
                self._target(self.facing) not in self.blocked_cells:
            return self._commit(self.facing)

        # Rule 2: ELSE explore - among the moves not known to be blocked, avoid
        # remembered toxic cells, then prefer the LEAST visited cell (so the agent
        # breaks out of loops), then keep going straight, then the fixed order.
        options = [a for a in self.ACTIONS if self._target(a) not in self.blocked_cells]
        if not options:                          # boxed in by remembered obstacles
            self.blocked_cells.clear()
            options = list(self.ACTIONS)

        def preference(action):
            cell = self._target(action)
            return (cell in self.toxic_cells,
                    self.visit_counts.get(cell, 0),
                    action != self.facing,
                    self.ACTIONS.index(action))

        return self._commit(min(options, key=preference))


class SearchAgent:
    """Goal-based agent using BFS, DFS, UCS, or A* to build an offline plan."""

    MOVES = [
        ('Up', (0, 1)),
        ('Right', (1, 0)),
        ('Down', (0, -1)),
        ('Left', (-1, 0)),
    ]

    def __init__(self, active_algo='BFS'):
        self.plan = []
        # BFS / DFS / UCS are stored upper-case; A* is stored as 'AStar' so that
        # the lab's literal check `self.active_algo == 'AStar'` works.
        algo = str(active_algo).strip()
        if algo.upper() in ('ASTAR', 'A*', 'A_STAR'):
            self.active_algo = 'AStar'
        else:
            self.active_algo = algo.upper()
        # Number of nodes expanded by the most recent search (used to show how
        # much A* reduces the search effort compared with the uninformed searches).
        self.nodes_expanded = 0

    @staticmethod
    def _valid_position(pos, walls, grid_size):
        x, y = pos
        width, height = grid_size
        return 0 <= x < width and 0 <= y < height and pos not in walls

    def _successors(self, state, walls, grid_size):
        x, y = state
        for action, (dx, dy) in self.MOVES:
            next_state = (x + dx, y + dy)
            if self._valid_position(next_state, walls, grid_size):
                yield next_state, action

    def bfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Breadth-First Graph Search: FIFO frontier."""
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)
        walls = set(map(tuple, walls))

        self.nodes_expanded = 0

        if start_pos == goal_pos:
            return []

        frontier = deque([(start_pos, [])])
        reached = {start_pos}

        while frontier:
            state, path = frontier.popleft()
            self.nodes_expanded += 1

            for next_state, action in self._successors(state, walls, grid_size):
                if next_state in reached:
                    continue

                new_path = path + [action]
                if next_state == goal_pos:
                    return new_path

                reached.add(next_state)
                frontier.append((next_state, new_path))

        return None

    def dfs_search(self, start_pos, goal_pos, walls, grid_size):
        """Depth-First Graph Search: LIFO frontier."""
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)
        walls = set(map(tuple, walls))

        self.nodes_expanded = 0

        if start_pos == goal_pos:
            return []

        frontier = [(start_pos, [])]
        reached = {start_pos}

        while frontier:
            state, path = frontier.pop()
            self.nodes_expanded += 1

            for next_state, action in self._successors(state, walls, grid_size):
                if next_state in reached:
                    continue

                new_path = path + [action]
                if next_state == goal_pos:
                    return new_path

                reached.add(next_state)
                frontier.append((next_state, new_path))

        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size):
        """Uniform-Cost Graph Search: lowest total path cost g(n) first."""
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)
        walls = set(map(tuple, walls))

        self.nodes_expanded = 0

        if start_pos == goal_pos:
            return []

        # (cost, insertion_order, state, path)
        frontier = []
        counter = 0
        heapq.heappush(frontier, (0, counter, start_pos, []))
        best_cost = {start_pos: 0}
        reached = set()

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)

            if state in reached:
                continue
            reached.add(state)
            self.nodes_expanded += 1

            if state == goal_pos:
                return path

            for next_state, action in self._successors(state, walls, grid_size):
                if next_state in reached:
                    continue

                new_cost = cost + 1  # every grid move costs 1
                if new_cost < best_cost.get(next_state, float('inf')):
                    best_cost[next_state] = new_cost
                    counter += 1
                    heapq.heappush(
                        frontier,
                        (new_cost, counter, next_state, path + [action])
                    )

        return None

    # ------------------------------------------------------------------
    # Practical 04 - Step 1.1: heuristic functions h(n)
    # ------------------------------------------------------------------
    def manhattan_distance(self, pos, goal):
        """h(n) = |x1 - x2| + |y1 - y2|  (returns an int for integer inputs)."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal):
        """h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)  (straight-line distance)."""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    def _heuristic(self, pos, goal, heuristic_type):
        """Dispatch to the chosen heuristic by name."""
        kind = str(heuristic_type).lower()
        if kind == 'manhattan':
            return self.manhattan_distance(pos, goal)
        if kind == 'euclidean':
            return self.euclidean_distance(pos, goal)
        raise ValueError("heuristic_type must be 'manhattan' or 'euclidean'")

    # ------------------------------------------------------------------
    # Practical 04 - Step 1.2: A* search
    # ------------------------------------------------------------------
    def astar_search(self, start_pos, goal_pos, walls, grid_size,
                     heuristic_type='manhattan'):
        """A* Graph Search: expand the node with the lowest f(n) = g(n) + h(n).

        Frontier entries are (f_cost, g_cost, current_pos, path_taken).
        Returns a list of actions, [] if start == goal, or None if unreachable.
        """
        start_pos = tuple(start_pos)
        goal_pos = tuple(goal_pos)
        walls = set(map(tuple, walls))
        self.nodes_expanded = 0

        if start_pos == goal_pos:
            return []

        # Priority queue (min-heap) ordered by f_cost, then g_cost, then position.
        frontier = []
        h_start = self._heuristic(start_pos, goal_pos, heuristic_type)
        heapq.heappush(frontier, (0 + h_start, 0, start_pos, []))
        reached_states = set()          # closed set: states already expanded

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            # A cheaper copy of this state may already have been expanded
            # (the heap can hold stale duplicates), so skip it.
            if current_pos in reached_states:
                continue

            if current_pos == goal_pos:
                return path_taken

            reached_states.add(current_pos)
            self.nodes_expanded += 1

            for next_state, action in self._successors(current_pos, walls, grid_size):
                if next_state in reached_states:
                    continue

                g_new = g_cost + 1      # every grid move costs 1
                h_new = self._heuristic(next_state, goal_pos, heuristic_type)
                f_new = g_new + h_new
                heapq.heappush(frontier, (f_new, g_new, next_state, path_taken + [action]))

        return None

    def sense_and_act(self, percept: dict) -> str:
        """Create a complete plan when needed, then execute one action at a time."""
        if self.plan:
            return self.plan.pop(0)

        # Extract the global state from the percept dictionary.
        remaining_food = percept.get('remaining_food', None)
        foods = [tuple(food) for food in percept.get('all_food', [])]
        if remaining_food == 0 or not foods:
            return 'Stop'

        start = tuple(percept['agent_pos'])
        walls = percept['walls']
        grid_size = tuple(percept['grid_size'])

        # Closest food first (Manhattan distance from the current state). If the
        # closest one cannot be reached, fall back to the next closest.
        foods.sort(key=lambda food: self.manhattan_distance(start, food))

        result = None
        for goal in foods:
            if self.active_algo == 'BFS':
                result = self.bfs_search(start, goal, walls, grid_size)
            elif self.active_algo == 'DFS':
                result = self.dfs_search(start, goal, walls, grid_size)
            elif self.active_algo == 'UCS':
                result = self.ucs_search(start, goal, walls, grid_size)
            elif self.active_algo == 'AStar':
                # Practical 04 - Step 1.3: informed search toward the closest food.
                result = self.astar_search(start, goal, walls, grid_size,
                                           heuristic_type='manhattan')
            else:
                raise ValueError("active_algo must be 'BFS', 'DFS', 'UCS', or 'AStar'")

            if result:          # found a plan to this food
                break

        self.plan = result or []
        return self.plan.pop(0) if self.plan else 'Stop'


class GreedyGridAgent:
    """Compatibility with the older text simulator."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)


if __name__ == "__main__":
    # Practical 04 - Step 1.1 Testing Checkpoint: start (0, 0), goal (3, 4)
    checkpoint_agent = SearchAgent()
    print("Manhattan distance:", checkpoint_agent.manhattan_distance((0, 0), (3, 4)))  # 7
    print("Euclidean distance:", checkpoint_agent.euclidean_distance((0, 0), (3, 4)))  # 5.0