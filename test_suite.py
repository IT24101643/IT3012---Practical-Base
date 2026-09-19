import random
import unittest
from agent import SimpleReflexAgent, ModelBasedAgent, SearchAgent
from visual_grid_game import VisualGridHuntGame


class TestPractical1And2_ReflexAgents(unittest.TestCase):
    """
    Tests for Practicals 1 & 2: Simple Reflex and Model-Based Agents.
    Focuses on Condition-Action rules, partial observability, and memory.
    """

    def setUp(self):
        # Instantiate agents (assuming students have created these classes)
        try:
            self.simple_agent = SimpleReflexAgent()
            self.model_agent = ModelBasedAgent()
        except NameError:
            self.fail("Agent classes not found. Ensure SimpleReflexAgent and ModelBasedAgent are defined.")

    def test_simple_reflex_logic(self):
        """Test 1: Simple Reflex Agent should react purely to immediate percepts."""
        # Scenario A: Food is present -> Agent should want to collect/stay/move appropriately
        percept_food = {'wall_ahead': False, 'food_here': True}
        action = self.simple_agent.sense_and_act(percept_food)
        self.assertIsNotNone(action, "SimpleReflexAgent returned None instead of an action.")

        # Scenario B: Wall is ahead -> Agent must turn or change direction
        percept_wall = {'wall_ahead': True, 'food_here': False}
        action_wall = self.simple_agent.sense_and_act(percept_wall)
        self.assertIn(action_wall, ['Left', 'Right', 'Down', 'Up'],
                      "Agent did not output a valid movement action when facing a wall.")

    def test_model_based_memory(self):
        """Test 2: Model-Based Agent should maintain internal state to escape loops."""
        # Feed the exact same percept twice to simulate being stuck in a corner
        percept = {'wall_ahead': True, 'food_here': False}

        action_1 = self.model_agent.sense_and_act(percept)
        action_2 = self.model_agent.sense_and_act(percept)

        # A simple reflex agent would return the exact same action twice.
        # A model-based agent should remember the previous failure and try a DIFFERENT action.
        self.assertNotEqual(
            action_1,
            action_2,
            "ModelBasedAgent returned the exact same action twice in a row for the same percept. Internal state/memory is not working correctly."
        )


class TestPractical3_SearchAgent(unittest.TestCase):
    """
    Tests for Practical 3: Problem-Solving Agents.
    Focuses on offline planning and Breadth-First Search (BFS) implementation.
    """

    def setUp(self):
        try:
            self.search_agent = SearchAgent()
        except NameError:
            self.fail("SearchAgent class not found.")

    def test_bfs_shortest_path(self):
        """Test 3: BFS must find the optimal (shortest) path in a static maze."""
        # Mock Environment Data
        grid_size = (4, 4)
        start_pos = (0, 0)
        goal_pos = (3, 3)

        # Create a U-shaped wall trap that the agent must navigate around
        # Grid layout (S=Start, G=Goal, W=Wall):
        # 3 | . . . G
        # 2 | W W W .
        # 1 | . . . .
        # 0 | S W W .
        #   ---------
        #     0 1 2 3
        walls = [(1, 0), (2, 0), (0, 2), (1, 2), (2, 2)]

        # Run student's BFS algorithm
        try:
            path = self.search_agent.bfs_search(start_pos, goal_pos, walls, grid_size)
        except AttributeError:
            self.fail("bfs_search method not implemented in SearchAgent.")

        # Verify the path is valid and optimal
        self.assertIsNotNone(path, "BFS returned None. No path found.")
        self.assertIsInstance(path, list, "BFS should return a list of actions (strings).")

        # The shortest path taking Manhattan distance around these specific walls is exactly 6 steps.
        # Path: Up -> Right -> Right -> Right -> Up -> Up
        self.assertEqual(len(path), 6, f"BFS did not find the optimal path. Expected 6 steps, got {len(path)}.")

    def test_bfs_unreachable_goal(self):
        """Test 4: BFS must correctly return failure (None/Empty) if goal is blocked."""
        grid_size = (3, 3)
        start_pos = (0, 0)
        goal_pos = (2, 2)

        # Box the goal in completely
        walls = [(1, 2), (2, 1), (1, 1)]

        path = self.search_agent.bfs_search(start_pos, goal_pos, walls, grid_size)

        # The agent should realize it's impossible and return None or an empty list
        is_empty_or_none = (path is None) or (len(path) == 0)
        self.assertTrue(is_empty_or_none, "BFS should return None or [] when the goal is unreachable.")


def play_headless(agent, env, max_steps=500):
    """Run an agent in an environment (no GUI) until all food is eaten or it stops."""
    for _ in range(max_steps):
        if not env.food_positions:
            break
        action = agent.sense_and_act(env.get_percept())
        if action == 'Stop':
            break
        env.execute_action(action)
    return env


class TestPractical1_Environment(unittest.TestCase):
    """
    Practical 1: environment state, perception and action execution
    (toxic traps, 'smells_toxin' sensor, penalties/rewards, single-agent world).
    """

    def make_env(self, walls=()):
        env = VisualGridHuntGame(width=5, height=5, num_food=1, num_traps=0, custom_walls=set(walls))
        env.food_positions = {(4, 4)}
        return env

    def test_toxic_traps_placement(self):
        """Step 2.1: traps avoid (0,0), walls and food."""
        for seed in range(20):
            random.seed(seed)
            env = VisualGridHuntGame(width=8, height=8, num_food=10, num_traps=6)
            self.assertEqual(len(env.toxic_traps), 6)
            self.assertNotIn((0, 0), env.toxic_traps)
            self.assertFalse(env.toxic_traps & env.walls)
            self.assertFalse(env.toxic_traps & env.food_positions)

    def test_environment_is_single_agent(self):
        """Task 1.1 Q3: no opponents -> single-agent environment."""
        self.assertEqual(VisualGridHuntGame().opponents, [])

    def test_smells_toxin_sensor_and_penalty(self):
        """Steps 2.2 / 2.3: sensor key exists and stepping on a trap costs 15 points."""
        env = self.make_env()
        env.toxic_traps = {(0, 1)}
        self.assertFalse(env.get_percept()['smells_toxin'])
        env.execute_action('Up')
        self.assertTrue(env.get_percept()['smells_toxin'])
        self.assertEqual(env.score, -15)

    def test_wall_penalty_and_bump(self):
        """Task 1.3: hitting a wall costs 5 points and the agent does not move."""
        env = self.make_env(walls=[(0, 1)])
        env.execute_action('Up')
        self.assertEqual(env.score, -5)
        self.assertEqual(tuple(env.agent_pos), (0, 0))
        self.assertTrue(env.get_percept()['bumped'])

    def test_food_reward(self):
        """Eating a pellet gives +20 and removes it."""
        env = self.make_env()
        env.food_positions = {(0, 1)}
        env.execute_action('Up')
        self.assertEqual(env.score, 20)
        self.assertEqual(env.food_positions, set())


class TestPractical2_PartialObservability(unittest.TestCase):
    """
    Practical 2: local (partially observable) percepts, a memoryless
    SimpleReflexAgent and a ModelBasedAgent with an internal state.
    """

    def local_env(self):
        random.seed(7)
        return VisualGridHuntGame(width=12, height=12, num_food=15, num_traps=0,
                                  expose_world_model=False)

    def test_percept_is_local_only(self):
        """Step 1.1: no global coordinates, only local booleans."""
        percept = self.local_env().get_percept()
        for key in ('wall_ahead', 'food_here'):
            self.assertIn(key, percept)
            self.assertIsInstance(percept[key], bool)
        for hidden in ('agent_pos', 'grid_size', 'walls', 'all_food'):
            self.assertNotIn(hidden, percept)

    def test_wall_ahead_uses_facing_direction(self):
        """wall_ahead checks the adjacent cell in the facing direction (map edge counts)."""
        env = self.local_env()
        env.facing = 'Down'          # agent is at (0, 0): the cell below is off the map
        self.assertTrue(env.get_percept()['wall_ahead'])
        env.facing = 'Up'
        self.assertFalse(env.get_percept()['wall_ahead'])

    def test_simple_reflex_has_no_init_or_memory(self):
        """Step 1.2: no __init__ and no stored history."""
        self.assertNotIn('__init__', SimpleReflexAgent.__dict__)
        self.assertEqual(vars(SimpleReflexAgent()), {})

    def test_simple_reflex_is_purely_reactive(self):
        """Same percept -> same action every time."""
        agent = SimpleReflexAgent()
        percept = {'wall_ahead': True, 'food_here': False}
        self.assertEqual(len({agent.sense_and_act(percept) for _ in range(10)}), 1)

    def test_simple_reflex_gets_trapped(self):
        """Step 1.2 observation: the memoryless agent ends up looping on the same cell."""
        env, agent, trail = self.local_env(), SimpleReflexAgent(), []
        for _ in range(150):
            env.execute_action(agent.sense_and_act(env.get_percept()))
            trail.append(tuple(env.agent_pos))
        self.assertLessEqual(len(set(trail[-50:])), 2)

    def test_model_based_records_state(self):
        """Step 1.3: internal state records percepts, actions and visited cells."""
        agent = ModelBasedAgent()
        self.assertTrue(hasattr(agent, 'visited_cells'))
        percept = {'wall_ahead': False, 'food_here': False}
        first = agent.sense_and_act(percept)
        self.assertEqual(agent.last_action, first)
        self.assertEqual(agent.last_percept, percept)
        agent.sense_and_act(percept)
        self.assertGreaterEqual(len(agent.visited_cells), 2)

    def test_model_based_remembers_walls(self):
        """A bump is remembered so the same blocked move is not repeated."""
        agent = ModelBasedAgent()
        agent.sense_and_act({'wall_ahead': False})            # first move
        blocked_dir = agent.last_action
        agent.sense_and_act({'wall_ahead': True, 'bumped': True})
        self.assertNotEqual(agent.last_action, blocked_dir)
        self.assertTrue(agent.blocked_cells)

    def test_model_based_escapes_and_explores(self):
        """The model-based agent explores far more than the reflex agent and scores higher."""
        def walk(agent):
            env, seen = self.local_env(), set()
            for _ in range(200):
                env.execute_action(agent.sense_and_act(env.get_percept()))
                seen.add(tuple(env.agent_pos))
            return env, seen

        reflex_env, reflex_seen = walk(SimpleReflexAgent())
        model_env, model_seen = walk(ModelBasedAgent())
        self.assertGreater(len(model_seen), 5 * len(reflex_seen))
        self.assertGreater(model_env.score, reflex_env.score)


class TestPractical3_UninformedSearch(unittest.TestCase):
    """
    Practical 3: world model in the percept, BFS / DFS / UCS with a reached set,
    offline plan stored in self.plan.
    """

    WALLS = [(1, 0), (2, 0), (0, 2), (1, 2), (2, 2)]

    def setUp(self):
        self.agent = SearchAgent()

    def test_world_model_keys_in_percept(self):
        """Step 1.1: grid_size, walls and all_food are exposed (plus agent_pos for planning)."""
        percept = VisualGridHuntGame(num_traps=0).get_percept()
        for key in ('grid_size', 'walls', 'all_food', 'agent_pos'):
            self.assertIn(key, percept)

    def test_init_defaults(self):
        """Step 1.3: empty plan and default active_algo 'BFS'."""
        self.assertEqual(self.agent.plan, [])
        self.assertEqual(self.agent.active_algo, 'BFS')

    def follow(self, path, walls, size, start=(0, 0)):
        deltas = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}
        x, y = start
        for action in path:
            x, y = x + deltas[action][0], y + deltas[action][1]
            self.assertNotIn((x, y), walls)
            self.assertTrue(0 <= x < size[0] and 0 <= y < size[1])
        return (x, y)

    def test_dfs_finds_valid_path(self):
        """DFS returns a valid (not necessarily shortest) path to the goal."""
        path = self.agent.dfs_search((0, 0), (3, 3), self.WALLS, (4, 4))
        self.assertIsNotNone(path)
        self.assertEqual(self.follow(path, set(self.WALLS), (4, 4)), (3, 3))

    def test_ucs_is_optimal(self):
        """UCS matches BFS on the unit-cost grid."""
        bfs = self.agent.bfs_search((0, 0), (3, 3), self.WALLS, (4, 4))
        ucs = self.agent.ucs_search((0, 0), (3, 3), self.WALLS, (4, 4))
        self.assertEqual(len(ucs), len(bfs))

    def test_dfs_and_ucs_unreachable(self):
        """DFS and UCS report failure for a walled-in goal."""
        walls = [(1, 2), (2, 1), (1, 1)]
        self.assertFalse(self.agent.dfs_search((0, 0), (2, 2), walls, (3, 3)))
        self.assertFalse(self.agent.ucs_search((0, 0), (2, 2), walls, (3, 3)))

    def test_plan_is_stored_and_consumed(self):
        """sense_and_act plans once, stores the actions in self.plan and pops them one by one."""
        agent = SearchAgent(active_algo='BFS')
        percept = {'agent_pos': (0, 0), 'grid_size': (5, 5), 'walls': [],
                   'all_food': [(2, 0)], 'remaining_food': 1}
        self.assertEqual(agent.sense_and_act(percept), 'Right')
        self.assertEqual(agent.plan, ['Right'])
        self.assertEqual(agent.sense_and_act(percept), 'Right')
        self.assertEqual(agent.plan, [])

    def test_every_algorithm_clears_the_board(self):
        """Integration: BFS, DFS, UCS and A* each eat all the food in the visual environment."""
        for algo in ('BFS', 'DFS', 'UCS', 'AStar'):
            random.seed(3)
            env = VisualGridHuntGame(width=8, height=8, num_food=6, num_traps=0)
            play_headless(SearchAgent(active_algo=algo), env)
            self.assertEqual(len(env.food_positions), 0, f"{algo} left food uneaten.")
            self.assertEqual(env.score, 6 * 20)   # no walls hit, no traps: only food rewards


class TestPractical4_AStar(unittest.TestCase):
    """
    Tests for Practical 4: Informed Search (heuristics + A*).
    """

    def setUp(self):
        self.agent = SearchAgent()

    def test_manhattan_distance(self):
        """Test 5: Manhattan distance for (0,0) -> (3,4) must be 7."""
        self.assertEqual(self.agent.manhattan_distance((0, 0), (3, 4)), 7)

    def test_euclidean_distance(self):
        """Test 6: Euclidean distance for (0,0) -> (3,4) must be 5.0."""
        self.assertAlmostEqual(self.agent.euclidean_distance((0, 0), (3, 4)), 5.0)

    def test_astar_matches_bfs_optimal_length(self):
        """Test 7: A* must return an optimal path (same length as BFS) for both heuristics."""
        walls = [(1, 0), (2, 0), (0, 2), (1, 2), (2, 2)]
        bfs_path = self.agent.bfs_search((0, 0), (3, 3), walls, (4, 4))
        for h in ('manhattan', 'euclidean'):
            path = self.agent.astar_search((0, 0), (3, 3), walls, (4, 4), heuristic_type=h)
            self.assertIsNotNone(path)
            self.assertEqual(len(path), len(bfs_path), f"A* ({h}) was not optimal.")

    def test_astar_path_is_valid(self):
        """Test 8: Following A*'s actions from start must land exactly on the goal without hitting walls."""
        walls = {(1, 0), (2, 0), (0, 2), (1, 2), (2, 2)}
        deltas = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}
        x, y = 0, 0
        for action in self.agent.astar_search((0, 0), (3, 3), walls, (4, 4)):
            dx, dy = deltas[action]
            x, y = x + dx, y + dy
            self.assertNotIn((x, y), walls)
            self.assertTrue(0 <= x < 4 and 0 <= y < 4)
        self.assertEqual((x, y), (3, 3))

    def test_astar_unreachable_goal(self):
        """Test 9: A* must return None/[] when the goal is walled in."""
        path = self.agent.astar_search((0, 0), (2, 2), [(1, 2), (2, 1), (1, 1)], (3, 3))
        self.assertTrue(path is None or len(path) == 0)

    def test_astar_start_equals_goal(self):
        """Test 10: A* returns an empty plan if already at the goal."""
        self.assertEqual(self.agent.astar_search((1, 1), (1, 1), [], (4, 4)), [])

    def test_astar_expands_fewer_nodes_than_ucs(self):
        """Test 11: With a good heuristic A* should expand fewer nodes than UCS."""
        walls = [(2, 2), (2, 3), (5, 5), (6, 5), (3, 7)]
        self.agent.ucs_search((0, 0), (9, 4), walls, (12, 12))
        ucs_nodes = self.agent.nodes_expanded
        self.agent.astar_search((0, 0), (9, 4), walls, (12, 12))
        self.assertLess(self.agent.nodes_expanded, ucs_nodes)

    def test_sense_and_act_astar(self):
        """Test 12: sense_and_act with 'AStar' must build a plan and return a valid first action."""
        agent = SearchAgent(active_algo='AStar')
        percept = {'agent_pos': (0, 0), 'grid_size': (5, 5), 'walls': [(1, 0)],
                   'all_food': [(2, 0)], 'remaining_food': 1}
        action = agent.sense_and_act(percept)
        self.assertIn(action, ['Up', 'Down', 'Left', 'Right'])
        self.assertTrue(len(agent.plan) > 0 or action != 'Stop')

    def test_invalid_heuristic_raises(self):
        """Test 13: An unknown heuristic name is rejected."""
        with self.assertRaises(ValueError):
            self.agent.astar_search((0, 0), (1, 1), [], (3, 3), heuristic_type='chebyshev')

    def test_active_algo_astar_literal(self):
        """Step 1.3: 'AStar' (any spelling) selects the A* branch."""
        for name in ('AStar', 'astar', 'ASTAR'):
            self.assertEqual(SearchAgent(active_algo=name).active_algo, 'AStar')

    def test_stops_when_no_food_remains(self):
        """Step 1.3: remaining_food == 0 -> Stop."""
        agent = SearchAgent(active_algo='AStar')
        percept = {'agent_pos': (0, 0), 'grid_size': (3, 3), 'walls': [],
                   'all_food': [], 'remaining_food': 0}
        self.assertEqual(agent.sense_and_act(percept), 'Stop')

    def test_unreachable_closest_food_falls_back_to_next(self):
        """If the closest food is walled in, the agent heads for the next closest one."""
        agent = SearchAgent(active_algo='AStar')
        percept = {'agent_pos': (0, 0), 'grid_size': (5, 5),
                   'walls': [(0, 1), (1, 2), (0, 3)],
                   'all_food': [(0, 2), (3, 0)], 'remaining_food': 2}
        self.assertEqual(agent.sense_and_act(percept), 'Right')


if __name__ == '__main__':
    # Run the test suite
    print("=== IT3012: Intelligent Agents - Autograder Test Suite ===\n")
    unittest.main(verbosity=2)