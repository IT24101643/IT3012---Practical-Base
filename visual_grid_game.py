# visual_grid_game.py
import random
try:
    import tkinter as tk
except ImportError:
    tk = None  # GUI unavailable (e.g. headless environment) -- VisualGridHuntGame still works


class VisualGridHuntGame:
    """A flexible Pacman-style grid environment.

    Lab 01: partial observability (wall_ahead / food_here / bumped) +
            SimpleReflexAgent / ModelBasedAgent test bed.
    Lab 02: adds toxic traps -- a hidden hazard the agent can sense
            locally but never sees the location of in advance.
    """

    DIR_VECTORS = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}

    def __init__(self, width=10, height=10, num_food=10, num_traps=3, custom_walls=None):
        self.width = width
        self.height = height
        self.agent_pos = [0, 0]
        self.facing = 'Up'              # which way the agent last tried to move
        self.last_move_blocked = False  # simple bump sensor

        if custom_walls is not None:
            self.walls = set(custom_walls)
        else:
            self.walls = {(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)}

        # --- Step 2.1: toxic traps -----------------------------------
        # Populated randomly, safely avoiding (0, 0) and existing walls.
        self.toxic_traps = set()
        while len(self.toxic_traps) < num_traps:
            tx = random.randint(0, self.width - 1)
            ty = random.randint(0, self.height - 1)
            pos = (tx, ty)
            if pos != (0, 0) and pos not in self.walls:
                self.toxic_traps.add(pos)
        # ---------------------------------------------------------------

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
        """Partial observability: the agent only senses the cell ahead
        (wall_ahead), the cell it's standing on (food_here /
        smells_toxin), and whether its last move actually happened
        (bumped). It has no idea where it is on the map.
        """
        dx, dy = self.DIR_VECTORS[self.facing]
        ahead = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        in_bounds = 0 <= ahead[0] < self.width and 0 <= ahead[1] < self.height
        wall_ahead = (not in_bounds) or (ahead in self.walls)

        here = tuple(self.agent_pos)

        return {
            'wall_ahead': wall_ahead,
            'food_here': here in self.food_positions,
            'smells_toxin': here in self.toxic_traps,   # --- Step 2.2 ---
            'bumped': self.last_move_blocked,
            'score': self.score,
            'remaining_food': len(self.food_positions),
        }

    def execute_action(self, action: str):
        self.steps += 1

        if action in self.DIR_VECTORS:
            self.facing = action
            self._attempt_move(action)
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

        tuple_pos = tuple(self.agent_pos)

        # --- Step 2.3: toxic trap collision penalty ---
        if tuple_pos in self.toxic_traps:
            self.score -= 15

        if tuple_pos in self.food_positions:
            self.food_positions.remove(tuple_pos)
            self.score += 20

    def is_done(self) -> bool:
        return len(self.food_positions) == 0 or self.steps >= 60


class GridGameGUI:
    def __init__(self, root, width=10, height=10, num_food=12, num_traps=3, walls=None):
        self.root = root
        self.root.title("IT3012 - Grid Hunt with Toxic Traps")

        self.env = VisualGridHuntGame(width=width, height=height, num_food=num_food,
                                       num_traps=num_traps, custom_walls=walls)

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

        # --- Step 2.3: draw toxic traps as purple shapes ---
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
                action = random.choice(['Up', 'Down', 'Left', 'Right'])
                self.env.execute_action(action)

                self.draw_grid()
                self.label.config(text=f"Score: {self.env.score} | Steps: {self.env.steps} | Action: {action}")
                self.root.after(250, step)
            else:
                self.label.config(text=f"Finished! Final Score: {self.env.score}")
                self.btn.config(state="normal")

        step()


if __name__ == "__main__":
    root = tk.Tk()
    app = GridGameGUI(root, width=12, height=12, num_food=15, num_traps=4)
    root.mainloop()