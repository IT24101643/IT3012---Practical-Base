# visual_grid_game.py
import random
import sys
from agent import SearchAgent, SimpleReflexAgent, ModelBasedAgent
try:
    import tkinter as tk
except ImportError:
    tk = None


class VisualGridHuntGame:
    """Grid Hunt environment used by Practicals 01-04."""

    DIR_VECTORS = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}

    def __init__(self, width=10, height=10, num_food=10, num_traps=3, custom_walls=None,
                 expose_world_model=True):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]
        self.facing = 'Up'              # which way the agent last tried to move
        self.last_move_blocked = False  # simple bump sensor

        # Practical 01 (Task 1.1, Q3): no opponent agents -> Single-Agent environment.
        self.opponents = []

        # Practical 02 vs 03/04: False = strictly local (partially observable) percepts;
        # True = also expose the world model (agent_pos, grid_size, walls, all_food)
        # that the planning agents of Practicals 03/04 need.
        self.expose_world_model = expose_world_model

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        # Never ask for more traps/food than there are free cells (avoids an endless loop).
        free_cells = self.width * self.height - 1 - len(self.walls - {(0, 0)})
        num_traps = min(num_traps, max(0, free_cells))

        # Step 2.1: toxic traps 
        # Populated randomly, safely avoiding (0, 0) and existing walls.
        self.toxic_traps = set()
        while len(self.toxic_traps) < num_traps:
            tx = random.randint(0, self.width - 1)
            ty = random.randint(0, self.height - 1)
            pos = (tx, ty)
            if pos != (0, 0) and pos not in self.walls:
                self.toxic_traps.add(pos)
       

        num_food = min(num_food, max(0, free_cells - len(self.toxic_traps)))
        self.food_positions = set()
        while len(self.food_positions) < num_food:
            fx = random.randint(0, self.width - 1)
            fy = random.randint(0, self.height - 1)
            pos_tuple = (fx, fy)
            if (pos_tuple != (0, 0)
                    and pos_tuple not in self.walls
                    and pos_tuple not in self.toxic_traps):
                self.food_positions.add(pos_tuple)

        self.score = 0
        self.steps = 0

    def get_percept(self) -> dict:
        dx, dy = self.DIR_VECTORS[self.facing]
        ahead = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        in_bounds = 0 <= ahead[0] < self.width and 0 <= ahead[1] < self.height
        wall_ahead = (not in_bounds) or (ahead in self.walls)

        here = tuple(self.agent_pos)

        percept = {
            # Practical 02: local booleans only (what the agent's sensors can see)
            'wall_ahead': wall_ahead,
            'food_here': here in self.food_positions,
            'food_ahead': ahead in self.food_positions,
            'bumped': self.last_move_blocked,
            # Practical 01 Step 2.2: toxin sensor
            'smells_toxin': tuple(self.agent_pos) in self.toxic_traps,
            'score': self.score,
            'remaining_food': len(self.food_positions),
        }

        if self.expose_world_model:
            # Practical 03: expose the world model to the planning agent
            percept.update({
                'agent_pos': tuple(self.agent_pos),
                'grid_size': (self.width, self.height),
                'walls': list(self.walls),
                'all_food': list(self.food_positions),
            })
        return percept

    def execute_action(self, action: str):
        self.steps += 1

        if action in self.DIR_VECTORS:
            self.facing = action
            self._attempt_move(action)

            tuple_pos = tuple(self.agent_pos)

            # Practical 01 Step 2.3: toxic trap collision penalty on the updated position
            if tuple_pos in self.toxic_traps:
                self.score -= 15

            if tuple_pos in self.food_positions:
                self.food_positions.remove(tuple_pos)
                self.score += 20
        # any other/unrecognised action string: agent wastes a turn

    def _attempt_move(self, direction):
        old_pos = list(self.agent_pos)
        dx, dy = self.DIR_VECTORS[direction]
        new_pos = [self.agent_pos[0] + dx, self.agent_pos[1] + dy]
        new_pos[0] = max(0, min(self.width - 1, new_pos[0]))
        new_pos[1] = max(0, min(self.height - 1, new_pos[1]))

        if tuple(new_pos) in self.walls:
            self.score -= 5
            self.last_move_blocked = True
        else:
            self.agent_pos = new_pos
            self.last_move_blocked = (new_pos == old_pos)  # clipped at the grid edge

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 60


def build_agent(name):
    """Create an agent from a short name (used by the command line)."""
    key = name.strip().lower()
    if key in ('reflex', 'simplereflex'):
        return SimpleReflexAgent()
    if key in ('model', 'modelbased'):
        return ModelBasedAgent()
    if key in ('bfs', 'dfs', 'ucs', 'astar', 'a*'):
        return SearchAgent(active_algo=key)
    raise ValueError("Agent must be one of: reflex, model, bfs, dfs, ucs, astar")


class GridGameGUI:
    def __init__(self, root, width=10, height=10, num_food=12, num_traps=3, walls=None,
                 agent=None):
        self.root = root
        self.root.title(" Grid Hunt with Toxic Traps")

        # Practical 03/04: choose 'BFS', 'DFS', 'UCS', or 'AStar' here.
        # Practical 04: inject the informed-search agent (A* + Manhattan heuristic).
        self.agent = agent if agent is not None else SearchAgent(active_algo='AStar')

        # Planning agents (Practical 03/04) need the world model in the percept;
        # the reflex / model-based agents (Practical 02) only get local percepts.
        self.env = VisualGridHuntGame(width=width, height=height, num_food=num_food,
                                      num_traps=num_traps, custom_walls=walls,
                                      expose_world_model=isinstance(self.agent, SearchAgent))

        max_canvas_dim = 600
        self.cell_size = max(20, min(max_canvas_dim // self.env.width, max_canvas_dim // self.env.height))

        canvas_w = self.env.width * self.cell_size
        canvas_h = self.env.height * self.cell_size

        self.canvas = tk.Canvas(root, width=canvas_w, height=canvas_h, bg="white")
        self.canvas.pack()

        self.label = tk.Label(root, text="Score: 0 | Steps: 0", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Start Simulation", command=self.run_loop, font=("Arial", 12), bg="#000066", fg="white")
        self.btn.pack(pady=5)

        self.draw_grid()

    def draw_grid(self):
        self.canvas.delete("all")

        for x in range(self.env.width):
            for y in range(self.env.height):
                x1 = x * self.cell_size
                y1 = (self.env.height - 1 - y) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                color = "#f1f5f9" if (x, y) not in self.env.walls else "#64748b"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#cbd5e1")

                if self.cell_size >= 40 and (x, y) in self.env.walls:
                    self.canvas.create_text(x1 + self.cell_size / 2, y1 + self.cell_size / 2, text="W", fill="white", font=("Arial", 8, "bold"))

        # Step 2.3: draw toxic traps as purple shapes 
        for tx, ty in self.env.toxic_traps:
            offset = self.cell_size * 0.2
            x1 = tx * self.cell_size + offset
            y1 = (self.env.height - 1 - ty) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.6, y1 + self.cell_size * 0.6, fill="#8b5cf6", outline="#6d28d9")

        for fx, fy in self.env.food_positions:
            offset = self.cell_size * 0.25
            x1 = fx * self.cell_size + offset
            y1 = (self.env.height - 1 - fy) * self.cell_size + offset
            self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.5, y1 + self.cell_size * 0.5, fill="#f59e0b", outline="#d97706")

        ax, ay = self.env.agent_pos
        offset = self.cell_size * 0.15
        x1 = ax * self.cell_size + offset
        y1 = (self.env.height - 1 - ay) * self.cell_size + offset
        self.canvas.create_oval(x1, y1, x1 + self.cell_size * 0.7, y1 + self.cell_size * 0.7, fill="#000066", outline="#1e3a8a")

    def run_loop(self):
        self.btn.config(state="disabled")

        def step():
            if not self.env.is_done():
                percept = self.env.get_percept()
                action = self.agent.sense_and_act(percept)

                if action == 'Stop':
                    self.label.config(text=f"No reachable food found. Score: {self.env.score}")
                    self.btn.config(state="normal")
                    return

                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(text=f"Score: {self.env.score} | Steps: {self.env.steps} | Action: {action}")
                self.root.after(250, step)
            else:
                self.label.config(text=f"Finished! Final Score: {self.env.score}")
                self.btn.config(state="normal")

        step()


if __name__ == "__main__":
    # Usage: python visual_grid_game.py [reflex | model | bfs | dfs | ucs | astar]
    # Default (Practical 04): A* search agent.
    if tk is None:
        raise SystemExit("tkinter is not available in this Python installation.")

    algo_name = sys.argv[1] if len(sys.argv) > 1 else 'astar'

    root = tk.Tk()
    # num_traps=3 -> three purple toxic traps are drawn (Practical 01, Step 2.3).
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_traps=3,
                      agent=build_agent(algo_name))
    root.mainloop()