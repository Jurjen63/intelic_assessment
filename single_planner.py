"""
General Planner Module
"""

from typing import List, Tuple, Optional, Dict
import time
import random
import numpy as np

from grid import Grid, Coord
from visualize import visualize_results


# --- Core Base Planner
class BasePlanner:
    """
    General base planner skeleton for path planning. 
    """
    DIRECTIONS: Tuple[Coord, ...] = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1)
    )

    def __init__(self, grid: Grid, start: Coord, t:int, T_ms:int, *, seed:Optional[int]=42):
        """
        Initialize the planner with the grid, start position, time step and time budget.

        Args:
            grid (Grid): The given grid to plan on.
            start (Coord): The starting coordinate. (TODO: make multi agent)
            t (int): Maximum number of steps allowed.
            T_ms (int): Time budget in milliseconds.
            seed (Optional[int], optional): Random seed for reproducibility. Defaults to 42.
        """

        self.grid = grid
        self.start = start
        self.t = t
        self.T_ms = T_ms

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def run(self) -> Tuple[List[Coord], int, int]:
        """
        Run the planner. 

        Returns:
            Tuple[List[Coord], int, int]: The planned path, the score and the time taken in miliseconds.
        """

        # Start algorithm
        path: List[Coord] = [self.start]
        x, y = self.start
        score = self.grid.traverse_plane((x, y))

        t0 = time.time()
        t_max = self.T_ms / 1000.0

        while True:
            elapsed = time.time() - t0
            remaining_time = t_max - elapsed
            remaining_steps = self.t - (len(path)-1)    # -1 because start is included

            # Check termination conditions
            if remaining_time <= 0 or remaining_steps <= 0:
                if remaining_time <= 0:
                    print("Warning: Time budget exceeded, stopping planning.")
                
                greedy_path = self._greedy_travel(path[-1], remaining_steps)
                print(f"Filling remaining {len(greedy_path)} steps with greedy travel.")
                path.extend(greedy_path) # fill remaining steps with greedy travel

                # Execute greedy path
                visited = set()
                for nx, ny in greedy_path:
                    score += self.grid.traverse_plane((nx, ny))
                    self.grid.regenerate()
                    visited.add((nx, ny))
                break
            
            segment = self._plan_segment(path[-1], remaining_steps, remaining_time)
            if not segment:
                break

            # Execute the segment
            for nx, ny in segment:
                # Check step budget
                if (len(path)-1) >= self.t:
                    break

                score += self.grid.traverse_plane((nx, ny))
                self.grid.regenerate()
                path.append((nx, ny))
            else:
                continue    # Only executed if the inner loop did NOT break
            break           # Inner loop did break, so we break the outer loop as well

        return path, int(score), int((time.time() - t0) * 1000)
    
    def _plan_segment(self, current:Coord, remaining_steps:int, remaining_time:float) -> List[Coord]:
        """
        Return the next path segment (list of coordinates), excluding the current position.
        MUST BE OVERRIDEN BY CHILD CLASSES.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    # --- Shared utilities
    @staticmethod
    def _cheb_dist(a:Coord, b:Coord) -> int:
        """
        Calculate the Chebyshev distance between two coordinates.

        Args:
            a (Coord): The first coordinate (x, y).
            b (Coord): The second coordinate (x, y).

        Returns:
            int: The Chebyshev distance.
        """
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    
    @staticmethod
    def _find_shortest_path(start:Coord, goal:Coord) -> List[Coord]:
        """
        Find the shortest path between two coordinates using 8-directional movement.

        Args:
            start (Coord): The starting coordinate (x, y).
            goal (Coord): The goal coordinate (x, y).

        Returns:
            List[Coord]: The list of coordinates representing the shortest path.
        """
        if start == goal:
            return []
        
        path = []
        x, y = start
        gx, gy = goal

        while (x, y) != (gx, gy):
            dx = int(np.sign(gx - x))
            dy = int(np.sign(gy - y))
            x += dx
            y += dy
            path.append((x, y))

        return path
    
    @staticmethod
    def _calculate_path_score(grid:Grid, path:List[Coord]) -> int:
        """
        Calculate the total score of a given path on the grid without modifying the grid.

        Assumes each plane can only be counted once, as regeneration takes longer than the path (probably).

        Args:
            grid (Grid): The grid to evaluate the path on.
            path (List[Coord]): The path as a list of coordinates.

        Returns:
            int: The total expected score of the path.
        """
        score = 0
        current: Dict[Coord, int] = {}
        for (x, y) in path:
            if (x, y) not in current:
                current[(x, y)] = grid.get_value((x, y))
            score += current[(x, y)]
            current[(x, y)] = 0     # simulate traversal

        return score
    
    def _greedy_travel(self, start:Coord, steps:int) -> List[Coord]:
        """
        Perform a greedy travel from the start position for a given number of steps.

        Simulates depletion: once a cell is visited, its value is set to 0 for the remainder of the path.

        Args:
            start (Coord): The starting coordinate (x, y).
            steps (int): The number of steps to take.

        Returns:
            List[Coord]: The list of coordinates representing the greedy path.
        """
        path = []
        current = start

        depleted = {start}
        cache: Dict[Coord, int] = {}

        def local_value(c: Coord) -> int:
            if c in depleted:
                return 0
            if c not in cache:
                cache[c] = self.grid.get_value(c)
            return cache[c]

        for _ in range(steps):
            neighbours = self.grid.get_neighbours(current)
            if not neighbours:
                break

            # Select the neighbour with the highest value
            values = [local_value(n) for n in neighbours]
            max_value = max(values)
            candidates = [n for n, v in zip(neighbours, values) if v == max_value]

            next_ = random.choice(candidates)  # break ties randomly
            path.append(next_)
            depleted.add(next_)
            current = next_

        return path

        
# --- Greedy Receding Horizon Planner
class GreedyRHPlanner(BasePlanner):
    """
    Greedy Receding Horizon Planner (RHP) implementation.
    """
    def __init__(self, grid: Grid, start: Coord, t:int, T_ms:int, *, horizon:int=5, seed:Optional[int]=42):
        """
        Initialize the Greedy RHP with the grid, start position, time step, time budget and horizon.

        Args:
            horizon (int): The planning horizon. Defaults to 5.
        """
        super().__init__(grid, start, t, T_ms, seed=seed)
        self.horizon = horizon

    def _plan_segment(self, current:Coord, remaining_steps:int, remaining_time:float) -> List[Coord]:
        """
        Plan the next segment using a greedy approach within the specified horizon.

        Args:
            current (Coord): The current coordinate (x, y).
            remaining_steps (int): The number of remaining steps.
            remaining_time (float): The remaining time in seconds.

        Returns:
            List[Coord]: The planned segment as a list of coordinates.
        """
        action, _ = self._plan_recursive(self.grid, current, min(self.horizon, remaining_steps))
        return [] if action is None else [action]
    
    def _plan_recursive(self, grid:Grid, current:Coord, horizon:int) -> Tuple[Optional[Coord], int]:
        """
        Recursively plan the next move by evaluating all possible actions up to the specified depth.

        Args:
            grid (Grid): The current (simulated) state of the grid.
            current (Coord): The current coordinate (x, y).
            horizon (int): The remaining depth to explore.
        
        Returns:
            Tuple[Optional[Coord], int]: The best action (next coordinate) and its associated score.
        """

        # Base case
        if horizon == 0:
            return None, 0
        
        candidates = grid.get_neighbours(current)
        if candidates == []:
            return None, 0
        
        # shuffle candidates to introduce some randomness in case of ties
        random.shuffle(candidates)

        # If horizon = 1, return the best immediate action
        if horizon == 1:
            max_value = max([grid.get_value(c) for c in candidates])
            best_candidates = [c for c in candidates if grid.get_value(c) == max_value]
            action = random.choice(best_candidates)

            return action, max_value

        # If horizon > 1, recursively search
        best_action = None
        best_value = -1

        for candidate in candidates:
            grid_branch = grid.copy()

            immediate_value = grid_branch.traverse_plane(candidate)
            grid_branch.regenerate()

            _, future_value = self._plan_recursive(grid_branch, candidate, horizon - 1)
            total_value = immediate_value + future_value

            if total_value > best_value:
                best_value = total_value
                best_action = candidate

        return best_action, best_value

            
# --- Top-N Planner
class TopNPlanner(BasePlanner):
    """
    Top-N planner that identified high-value planes and finds valuable paths between them.

    The planner:
    1. Identifies the top N% highest value planes on the given grid. 
    2. Orders them by proximity using a greedy nearest-neighbour approach.
    3. Uses shortest path planning with optional a Beam-Rollout search to improve score.
    4. Falls back to a greedy exploration when the goals are unreachable given the remaining steps. 
    """

    def __init__(self, grid:Grid, start:Coord, t:int, T_ms:int, *, 
                 N:int, t_extra:float, seed:Optional[int]=42, beam:int=4, rollout_d:int=8, eps:float=0.15):
        """
        Initialize the Top-N planner with the grid, start position, time step, time budget and parameters.

        Args:
            N (int): The percentage of top planes to consider (1-100).
            t_extra (float): Extra steps factor for planning (e.g., 0.15 for 15% extra steps than the shortest path).
            beam (int): The beam width for Beam-Rollout search. Defaults to 4.
            rollout (int): The rollout depth for Beam-Rollout search. Defaults to 8.
            eps (float): The epsilon value for exploration in Beam-Rollout search. Defaults to 0.15.
        """
        super().__init__(grid, start, t, T_ms, seed=seed)
        self.N = N
        self.t_extra = t_extra
        self.beam = beam
        self.rollout_d = rollout_d
        self.eps = eps

        self._goals_cache: Optional[List[Coord]] = None # Cache for the top-N goals -> will be initialized on first use
        self._debug_cache: List[Coord] = []  # Cache for debugging purposes

    def _plan_segment(self, current:Coord, remaining_steps:int, remaining_time:float) -> List[Coord]:
        """
        Plan the next segment by moving towards the next goal in the ordered top-N list.

        If the goals_cache is not yet initialized, it will be created by identifying and ordering the top-N planes.

        Args:
            current (Coord): The current coordinate (x, y).
            remaining_steps (int): The number of remaining steps.
            remaining_time (float): The remaining time in seconds.

        Returns:
            List[Coord]: The planned segment as a list of coordinates.
        """
        # Initialize goals cache if not yet done
        if self._goals_cache is None:
            top = self._identify_top_n_planes()
            radius = min(self.t, self.grid.N // 5, 250)     # Limit the radius to avoid long computations
            self._goals_cache = self._order_by_proximity(top, current, radius)
            # print(self._goals_cache)

        # If no goals left, fallback to greedy travel
        if self._goals_cache == []:
            return self._greedy_travel(current, remaining_steps)
        
        goal = self._goals_cache[0]
        shortest_path = self._find_shortest_path(current, goal)
        shortest_length = len(shortest_path)

        if remaining_steps < shortest_length:
            # Goal is unreachable, fallback to greedy travel
            return self._greedy_travel(current, remaining_steps)
        
        # Determine budgets
        step_budget = min(remaining_steps, int(shortest_length * (1 + self.t_extra)))

        estimated_targets_left = min(max(1, len(self._goals_cache)), max(1, remaining_steps // 3)) # at least 1 target, at most 1/3 of remaining steps (assuming avg 3 steps per target)
        time_budget = max(0.01,  remaining_time / estimated_targets_left)  # at least 10ms, at most evenly divide remaining time

        # Plan path using Beam-Rollout search
        br_path = self._extend_path(current, goal, shortest_path, step_budget, time_budget)

        # If the path reaches the goal, pop it
        if br_path and br_path[-1] == goal:
            self._goals_cache.pop(0)    

        # Collect debug info if it is a different path than the shortest path
        if br_path != shortest_path:
            self._debug_cache.append(goal)

        return br_path[:remaining_steps] # Ensure we do not exceed remaining steps
    
    # --- Beam-Rollout Functionalities
    def _extend_path(self, start:Coord, goal:Coord, shortest_path:List[Coord], steps_budget:int, time_budget:float) -> List[Coord]:
        """
        Extend the path from start to goal using Beam-Rollout search within the given steps and time budgets.

        Args:
            start (Coord): The starting coordinates (x, y)
            goal (Coord): The goal coordinates (x, y)
            shortest_path (List[Coord]): The shortest path from start to goal as a list of coordinates (x, y)
            steps_budget (int): The maximum number of steps allowed in the path
            time_budget (float): The maximum time in seconds to run the algorithm

        Returns:
            List[Coord]: The planned path as a list of coordinates (x, y) (excluding start position)
        """

        shortest_length = len(shortest_path)

        # Determine whether it is beneficial to use the Beam-Rollout search
        if shortest_length <= 1 or steps_budget <= shortest_length:
            return shortest_path
        
        # Run Beam-Rollout search
        deadline = time.time() + time_budget

        # Set baseline
        best_path = shortest_path
        best_score = self._calculate_path_score(self.grid, best_path)
        best_score_ratio = best_score / len(best_path) if best_path else 0

        while time.time() < deadline:
            candidate_paths = self._generate_beam_paths(best_path, goal, steps_budget)

            for p in candidate_paths:
                score = self._calculate_path_score(self.grid, p)
                score_ratio = score / len(p) if p else 0
                if score_ratio > best_score_ratio:
                    best_score = score
                    best_path = p
                    best_score_ratio = score_ratio

        return best_path[:steps_budget]  # Ensure we do not exceed steps budget
    
    def _generate_beam_paths(self, base_path:List[Coord], goal:Coord, steps_budget:int) -> List[List[Coord]]:
        """
        Generate beam search candidates by splicing and rolling out from the base path.

        Args:
            base_path (List[Coord]): The base path to modify as a list of coordinates (x, y)
            goal (Coord): The goal coordinates (x, y)
            steps_budget (int): The maximum number of steps allowed in the path

        Returns:
            List[List[Coord]]: A list of candidate paths as lists of coordinates (x, y)
        """
        candidates: List[List[Coord]] = []

        if len(base_path) <= 1:
            return candidates

        for _ in range(self.beam):
            i = np.random.randint(1, len(base_path))  # avoid 0 to ensure splice
            head = base_path[:i]
            current = head[-1]

            # steps already used by head = len(head) (start is NOT in base_path)
            steps_left = steps_budget - len(head)

            d_to_goal = self._cheb_dist(current, goal)
            if steps_left < d_to_goal:
                continue

            # leave room for tail; rollout capped to remaining free steps
            rollout = self._generate_rollout(current, goal, steps_left - d_to_goal)

            current_end = rollout[-1] if rollout else current
            # tail_budget = total left after head and rollout
            tail_budget = steps_left - len(rollout)
            tail = self._find_shortest_path(current_end, goal)[:max(0, tail_budget)]

            candidates.append(head + rollout + tail)

        return candidates
    
    def _generate_rollout(self, start:Coord, goal:Coord, steps_budget:int) -> List[Coord]:
        """
        Generate a rollout path from start towards goal using epsilon-greedy strategy.

        Args:
            start (Coord): The starting coordinates (x, y)
            goal (Coord): The goal coordinates (x, y)
            steps_budget (int): The maximum number of steps allowed in the rollout

        Returns:
            List[Coord]: The rollout path as a list of coordinates (x, y)
        """
        rollout: List[Coord] = []
        current = start

        for _ in range(min(self.rollout_d, steps_budget)):
            feasible = [n for n in self.grid.get_neighbours(current)
                        if self._cheb_dist(n, goal) <= steps_budget - len(rollout) - 1]
            if not feasible:
                break

            if np.random.rand() < self.eps:
                next_ = random.choice(feasible)  # explore
            else:
                vals = [self.grid.get_value(n) for n in feasible]
                m = max(vals)
                best_candidates = [n for n, v in zip(feasible, vals) if v == m]
                next_ = random.choice(best_candidates)  # exploit

            rollout.append(next_)
            current = next_

        return rollout
    
    # --- Top-N Functionalities
    def _identify_top_n_planes(self) -> List[Coord]:
        """
        Identify the top N% highest value planes on the grid.

        Returns:
            List[Coord]: A list of coordinates (x, y) of the top N% planes
        """

        cutoff_value = np.percentile(self.grid.grid, 100 - self.N)
        coords = np.argwhere(self.grid.grid >= cutoff_value)
        return [(coord[1], coord[0]) for coord in coords]   # (x, y) format
    
    def _order_by_proximity(self, planes:List[Coord], start:Coord, radius:int) -> List[Coord]:
        """
        Order the given coordinates by proximity starting from the start position, then 
        from the next point etc.
        
        Args:
            planes (List[Coord]): The list of coordinates (x, y) to order
            start (Coord): The starting coordinates (x, y)
            radius (int): The maximum distance to consider a point reachable

        Returns:
            List[Coord]: The ordered list of coordinates (x, y) by proximity
        """
        if radius is not None:
            planes = [p for p in planes if self._cheb_dist(p, start) <= radius]

        ordered: List[Coord] = []
        current = start
        remaining = planes.copy()

        # Warn if we think this will be slow
        if radius is None and len(planes) > 2500:
            print(f"Warning: Ordering {len(planes)} planes without radius limit may be slow.")

        if len(planes) > 2500:
            print(f"Warning: Ordering {len(planes)} planes may be slow. Consider using a radius limit.")

        while remaining:
            dist = [self._cheb_dist(current, p) for p in remaining]
            i_min = int(np.argmin(dist))
            next_ = remaining.pop(i_min)
            ordered.append(next_)
            current = next_

        return ordered   

        


if __name__ == "__main__":
    t=50
    T_ms=500
    start=(20,20)
    horizon=1
    patience=50
    grid=100


    grid = Grid(grid, f"grids/{grid}.txt", patience=patience)

    planner = TopNPlanner(grid, t=t, T_ms=T_ms, start=start, N=1, t_extra=1.0, seed=42)
    # planner = GreedyRHPlanner(grid, t=t, T_ms=T_ms, start=start, horizon=horizon, seed=42)
    path, score, T_rrt = planner.run()
    print(f"Score: {score}, Steps: {len(path)-1}/{t}, Time: {T_rrt} ms")

    visualize_results(grid, [path], 
                    planner_name="TopN",
                    time_ms=T_rrt,
                    max_time_ms=T_ms,
                    max_steps=t,
                    scores=[score],
                    patience=patience,
    )