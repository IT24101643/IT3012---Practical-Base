# agent.py
import random
from collections import deque
import heapq


class SimpleReflexAgent:
    """Practical 01 agent: reacts only to the current percept."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('wall_ahead', False):
            return random.choice(['Left', 'Right', 'Down'])
        return random.choice(self.actions_pool)


class ModelBasedAgent:
    """Practical 02 agent: uses a tiny amount of memory to avoid repetition."""

    def __init__(self):
        self.actions_pool = ['Up', 'Right', 'Down', 'Left']
        self.last_action = None
        self.action_index = 0

    def sense_and_act(self, percept: dict) -> str:
        # When blocked, deliberately choose a different action from last time.
        if percept.get('wall_ahead', False) or percept.get('bumped', False):
            for _ in range(len(self.actions_pool)):
                action = self.actions_pool[self.action_index % len(self.actions_pool)]
                self.action_index += 1
                if action != self.last_action:
                    self.last_action = action
                    return action

        action = self.actions_pool[self.action_index % len(self.actions_pool)]
        self.action_index += 1
        self.last_action = action
        return action


class SearchAgent:
    """Goal-based agent using BFS, DFS, or UCS to build an offline plan."""

    MOVES = [
        ('Up', (0, 1)),
        ('Right', (1, 0)),
        ('Down', (0, -1)),
        ('Left', (-1, 0)),
    ]

    def __init__(self, active_algo='BFS'):
        self.plan = []
        self.active_algo = active_algo.upper()

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

        if start_pos == goal_pos:
            return []

        frontier = deque([(start_pos, [])])
        reached = {start_pos}

        while frontier:
            state, path = frontier.popleft()

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

        if start_pos == goal_pos:
            return []

        frontier = [(start_pos, [])]
        reached = {start_pos}

        while frontier:
            state, path = frontier.pop()

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

    def sense_and_act(self, percept: dict) -> str:
        """Create a complete plan when needed, then execute one action at a time."""
        if self.plan:
            return self.plan.pop(0)

        foods = [tuple(food) for food in percept.get('all_food', [])]
        if not foods:
            return 'Stop'

        start = tuple(percept['agent_pos'])
        walls = percept['walls']
        grid_size = tuple(percept['grid_size'])

        # "Closest" food by Manhattan distance from the current state.
        goal = min(foods, key=lambda food: abs(food[0] - start[0]) + abs(food[1] - start[1]))

        if self.active_algo == 'BFS':
            result = self.bfs_search(start, goal, walls, grid_size)
        elif self.active_algo == 'DFS':
            result = self.dfs_search(start, goal, walls, grid_size)
        elif self.active_algo == 'UCS':
            result = self.ucs_search(start, goal, walls, grid_size)
        else:
            raise ValueError("active_algo must be 'BFS', 'DFS', or 'UCS'")

        self.plan = result or []
        return self.plan.pop(0) if self.plan else 'Stop'


class GreedyGridAgent:
    """Compatibility with the older text simulator."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        return random.choice(self.actions_pool)