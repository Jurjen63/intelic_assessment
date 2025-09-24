"""
General Planner Module
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Iterable
import time
import random
import numpy as np
from dataclasses import dataclass, field
from itertools import product

from grid import Grid, Coord
from visualize import visualize_results


class DroneState:
    """
    Represents the state of a single drone
    """
    def __init__(self, id:int, position:Coord, path: List[Coord]=None,
                 score:int=0, remaining_steps:int=0):
        self.id = id
        self.position = position
        self.path = path if path is not None else [position]
        self.score = score
        self.remaining_steps = remaining_steps


# --- Core Multi Class Planner
class MultiBasePlanner:
    """
    General Multi-Drone skeleton for path planning
    """
    DIRECTIONS: Tuple[Coord, ...] = (
        (-1, 1), (-1, 0), (-1, -1),
        (0, 1),          (0, -1),
        (1, 1),  (1, 0),  (1, -1)
    )   

    def __init__(self, grid:Grid, starts: List[Coord], t:int, T_ms:int, seed:Optional[int]=42):
        """
        Initialize the multi-drone planner.

        Args:
            grid (Grid): The given grid to plan on
            starts (List[Coord]): List of starting coordinates for each drone
            t (int): Maximum number of steps allowed per drone
            T_ms (int): Time budget in milliseconds
            seed (Optional[int]): Random seed for reproducibility
        """

        self.grid = grid
        self.starts = starts
        self.t = t
        self.T_ms = T_ms
        self.num_drones = len(starts)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Initialize drone states
        self.drones = [
            DroneState(id=i, position=starts[i], path=[starts[i]], score=0, remaining_steps=t)
            for i in range(self.num_drones)
        ]

        # Create subgrids if multi drone scenario:
        if self.num_drones > 1:
            self.grid.create_balanced_subgrids(starts, self.t)

    def run(self):
        """
        Run the multi-drone planner.

        Returns:
            Tuple[List[List[Coord]], List[int], int]: 
                - List of paths (one per drone)
                - List of scores (one per drone) 
                - Total time taken in milliseconds
                - Number of greedy moves made at the end to fill remaining steps (if any)
        """
        t0 = time.time()
        t_max = (self.T_ms) / 1000.0# Convert to seconds and add a reserve to complete the run with a greedy search

        # Start algorithm
        for drone in self.drones:
            x, y = drone.position
            drone.score += self.grid.traverse_plane((x, y))

        step = 0 
        greedy_moves = None # number of greedy moves made at the end to fill remaining steps
        while step < self.t:
            step += 1

            # Check time budget
            elapsed = time.time() - t0
            remaining_time = t_max - elapsed
            if remaining_time <= 0:
                print(f"Warning: Time budget exceeded after {step} steps. Filling remaining steps greedily.")
                break

            # Check if any drone can still move
            active_drones = [d for d in self.drones if d.remaining_steps > 0]
            if not active_drones:
                print("Warning: All drones have completed their steps.")
                break

            # Plan next moves
            next_moves_all_drones = self._plan_simultaneous_moves(active_drones, remaining_time)

            # Execute moves
            any_moved = False
            moves = [0] * len(self.drones)
            for drone, next_moves_per_drone in zip(active_drones, next_moves_all_drones):
                if next_moves_per_drone is not None:
                    for next_move in next_moves_per_drone:
                        if drone.remaining_steps <= 0:
                            continue

                        # Execute move
                        drone.position = next_move
                        drone.path.append(next_move)
                        drone.remaining_steps -= 1

                        x, y = next_move
                        drone.score += self.grid.traverse_plane((x, y))
                        moves[drone.id] += 1

                    any_moved = True
            
            if any_moved:
                for _ in range(max(moves)):
                    # Regenerate grid after all drones have moved once (next_moves could be a path of multiple steps)
                    self.grid.regenerate()

        # Final greedy step to use up remaining time
        if step < self.t:
            greedy_moves = self.t - step    # number of greedy moves to be made per drone
            self._fill_remaining_steps_greedy()

        total_time = int((time.time() - t0) * 1000)  # in milliseconds  

        paths = [drone.path for drone in self.drones]
        scores = [drone.score for drone in self.drones]
        return paths, scores, total_time, greedy_moves

    def _plan_simultaneous_moves(self, active_drones: List[DroneState], remaining_time: float) -> List[Optional[List[Coord]]]:
        """
        Plan the next moves for all active drones simultaneously.
        MUST BE OVERRIDEN BY SUBCLASSES.

        Args:
            active_drones (List[DroneState]): List of drones that can still move
            remaining_time (float): Remaining time in seconds
        
        Returns:
            List[Optional[List[Coord]]]: List of next coordinates (could be multiple) for each drone (None if no move)
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def _fill_remaining_steps_greedy(self) -> None:
        """
        Fill the remaining steps for each drone using a greedy approach.
        """
        for drone in self.drones:
            if drone.remaining_steps > 0:
                greedy_path = self._greedy_travel(drone.position, drone.remaining_steps, drone)

                # Execute greedy path
                for (nx, ny) in greedy_path:
                    drone.position = (nx, ny)
                    drone.path.append((nx, ny))
                    drone.score += self.grid.traverse_plane((nx, ny))
                    drone.remaining_steps -= 1

                    # Do not regenerate during greedy filling (we would generate per drone move instead of per one time step so to say)

    def _greedy_travel(self, start:Coord, steps:int, drone:DroneState) -> List[Coord]:
        """
        Perform a greedy travel from the start position for a given number of steps.

        Simulates depletion: once a cell is visited, its value is set to 0 for the remainder of the path.

        Args:
            start (Coord): Starting coordinate (x, y)
            steps (int): Number of steps to take
            drone (DroneState): The drone for which the path is being planned
        
        Returns:
            List[Coord]: The greedy path taken as a list of coordinates
        """

        path = []
        current = start
        depleted = {start}
        cache: Dict[Coord, int] = {}

        def local_value(c: Coord) -> int:
            if c in depleted:
                return 0
            if c not in cache:
                cache[c] = self.grid.get_value((c[0], c[1]))
            return cache[c]

        
        for _ in range(steps):
            neighbours = [n for n in self.grid.get_neighbours(current) if self.can_drone_visit(n, drone)]
            # print(neighbours)
            # exit()

            if not neighbours:
                break

            values = [local_value(n) for n in neighbours]
            max_value = max(values)
            candidates = [n for n, v in zip(neighbours, values) if v == max_value]

            next_ = random.choice(candidates)
            path.append(next_)
            depleted.add(next_)
            current = next_

        return path
        
    # --- Shared utility helpers
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
    
    def can_drone_visit(self, c:Coord, d:DroneState) -> bool:
        """
        Check if a drone can visit a given coordinate based on subgrid assignment.

        Args:
            c (Coord): The coordinate to check (x, y).
            d (DroneState): The drone state.
        """

        if self.num_drones == 1:
            return True
        
        return self.grid.is_plane_assigned_to_drone((c[0], c[1]), d.id)
    
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
    
# --- Greedy Receding Horizon Planner
class MultiDroneGreedyRHPlanner(MultiBasePlanner):
    """
    Multi-Drone Greedy PLanner MD_GRHP
    """
    def __init__(self, grid:Grid, starts: List[Coord], t:int, T_ms:int, seed:Optional[int]=42, *,
                 horizon:int=3):
        """
        Initialize the Multi-Drone Greedy Receding Horizon Planner.

        Args:
            horizon (int): The planning horizon (number of steps to look ahead) 
        """
        super().__init__(grid, starts, t, T_ms, seed)
        self.horizon = horizon

    def _clone_drone(self, d: DroneState, new_pos: Coord = None, step_delta: int = 0) -> DroneState:
        """
        Lightweight clone for planning in branches
        """
        return DroneState(
            id=d.id,
            position=(d.position if new_pos is None else new_pos),
            path=d.path[:],
            score=d.score,
            remaining_steps=d.remaining_steps + step_delta
        )

    def _candidate_moves(self, grid_state: Grid, d: DroneState, k: int = 3) -> List[Coord]:
        """
        Top-k neighbour moves by local value; include stay-in-place if no valid moves. 
        """
        neighbours = [n for n in grid_state.get_neighbours(d.position) if self.can_drone_visit(n, d)]
        if not neighbours:
            return [d.position]  # stay in place
        neighbours_vals = sorted(
            ((n, grid_state.get_value(n)) for n in neighbours),
            key=lambda x: x[1],
            reverse=True
        )
        return [n for n, _ in neighbours_vals[:k]]

    def _plan_simultaneous_moves(self, active_drones: List[DroneState], remaining_time: float) -> List[Optional[List[Coord]]]:
        max_steps_any = max(d.remaining_steps for d in active_drones)
        horizon_steps = min(self.horizon, max_steps_any)
        actions, _ = self._plan_recursive(active_drones, horizon_steps, self.grid)

        # Ensure shape: per-drone list of moves for this (length 1 each) or None
        return actions if actions else [None] * len(active_drones)

    def _plan_recursive(self, drones: List[DroneState], horizon: int, grid_state: Grid) -> Tuple[List[Optional[List[Coord]]], int]:
        """
        Recursive planning with full joint action enumeration up to the given horizon.

        Args:
            drones (List[DroneState]): List of drones to plan for
            horizon (int): Remaining planning horizon
            grid_state (Grid): Current state of the grid (to simulate traversals and regenerations)

        Returns:
            List[Optional[List[Coord]]]: List of next coordinates (could be multiple) for each drone (None if no move)
            int: The best total expected score from this state onward
        """
        if horizon == 0 or all(d.remaining_steps <= 0 for d in drones):
            return [[] for _ in drones], 0

        # Build top-k move sets per drone (to control the branch sizes)
        per_drone_moves = []
        for d in drones:
            if d.remaining_steps <= 0:
                per_drone_moves.append([d.position])  # forced stay
            else:
                per_drone_moves.append(self._candidate_moves(grid_state, d, k=3))

        best_total = -np.inf
        best_joint: Optional[Tuple[Coord, ...]] = None

        # Enumerate joint actions 
        for joint in product(*per_drone_moves):
            # Branch copies
            gs = grid_state.copy()
            score = 0
            next_drones: List[DroneState] = []

            # Apply all moves, then regenerate once
            for d, move in zip(drones, joint):
                # Score gain for visiting 'move'
                score += gs.traverse_plane(move)

                # Advance drone copy one step in planning
                next_drones.append(self._clone_drone(d, move, step_delta=-1))
            gs.regenerate()

            # Recurse to future steps
            _, future_score = self._plan_recursive(next_drones, horizon - 1, gs)
            total = score + future_score

            if total > best_total:
                best_total = total
                best_joint = joint

        if best_joint is None:
            # Ensure shape: per-drone list of moves for this (length 1 each) or None
            return [None] * len(drones), 0

        # Ensure shape: per-drone list of moves for this (length 1 each) or None
        actions = [[m] for m in best_joint]  # one move per drone for the next tick
        return actions, best_total

class MultiDroneTopNPlanner(MultiBasePlanner):
    """
    Multi-Drone Top-N Beam-Rollout Planner MD_TNP 

    The planner:
    1. Identifies the top-N% planes in the grid (globally) that are assigned to each drone
       (based on subgrid assignment if multiple drones).
    2. Orders these planes by proximity to the drone's start position (with a radius limit).
    3. Uses shortest path planning with optional a Beam-Rollout search to improve score.
    4. Falls back to a greedy exploration when the goals are unreachable given the remaining steps. 
    """
    def __init__(self, grid:Grid, starts:List[Coord], t:int, T_ms:int, *,
                 N:int, t_extra:float, seed:Optional[int]=42,
                 beam:int=4, rollout_d:int=8, eps:float=0.15):
        """
        Initialize the Top-N planner with the grid, start position, time step, time budget and parameters.

        Args:
            N (int): The percentage of top planes to consider (1-100).
            t_extra (float): Extra steps factor for planning (e.g., 0.15 for 15% extra steps than the shortest path).
            beam (int): The beam width for Beam-Rollout search. Defaults to 4.
            rollout (int): The rollout depth for Beam-Rollout search. Defaults to 8.
            eps (float): The epsilon value for exploration in Beam-Rollout search. Defaults to 0.15.
        """
        super().__init__(grid, starts, t, T_ms, seed=seed)
        self.N = N
        self.t_extra = t_extra
        self.beam = beam
        self.rollout_d = rollout_d
        self.eps = eps
        self._goals_cache: Dict[int, List[Coord]] = {}

    def _plan_simultaneous_moves(self, active_drones: List[DroneState], remaining_time: float) -> List[Optional[List[Coord]]]:
        """
        Plan the next moves for all active drones
        """
        if not active_drones:
            return []

        # split time per drone and keep margin
        # time_per_drone = max(0.02, 0.9 * remaining_time / len(active_drones))
        time_per_drone = remaining_time / len(active_drones)
        actions: List[Optional[List[Coord]]] = []

        for d in active_drones:
            seg = self._plan_segment_for_drone(d, time_per_drone)
            actions.append(seg if seg else None)

        return actions
    
    def _plan_segment_for_drone(self, d:DroneState, remaining_time: float) -> List[Coord]:
        """
        Plan the next path segment by moving towards the next goal in the ordered top-N list.

        First initializes the top-N list for the drone if not already done.

        Args:
            d (DroneState): The drone to plan for
            time_budget (float): Time budget in seconds for this planning call

        Returns:
            List[Coord]: The planned path segment as a list of coordinates for the given drone
        """
        if d.remaining_steps <= 0:
            return []

        # initialize drone goals if not done yet
        if d.id not in self._goals_cache:
            top = self._identify_top_n_planes_for_drone(d.id)
            radius = min(self.t, self.grid.N // 5, 250)
            self._goals_cache[d.id] = self._order_by_proximity(top, d.position, radius)
            # print(self._goals_cache[d.id])

        goals = self._goals_cache[d.id]
        if not goals:
            print(f"Warning: No goals for drone {d.id}. Falling back to greedy.")
            return self._greedy_travel(d.position, d.remaining_steps, d)

        goal = goals[0]
        shortest = self._find_shortest_path(d.position, goal)
        shortest_len = len(shortest)

        if shortest_len == 0:
            # already at goal
            goals.pop(0)
            return []
        
        if d.remaining_steps < shortest_len:
            # cannot reach goal, fallback to greedy
            return self._greedy_travel(d.position, d.remaining_steps, d)
        
        # Step budget
        steps_budget = min(d.remaining_steps, int(shortest_len * (1 + self.t_extra)))

        # Time budget
        est_targets_left = max(1, min(len(goals), d.remaining_steps // 3))    # assume 3 steps per target on average
        time_budget_segment = max(0.01, 0.9 * remaining_time / est_targets_left)
        # time_budget_segment = 0.9 * time_budget / est_targets_left

        segment = self._extend_path(d.position, goal, shortest, steps_budget, time_budget_segment)

        # if segment == shortest:
        #     print("  No improvement over shortest path")
        if not segment:
            return shortest[:steps_budget]
        
        # Pop if the segment reached the goal
        if segment and segment[-1] == goal:
            goals.pop(0)

        return segment[:d.remaining_steps]



    # --- Beam-Rollout Functionalities (same as single_planner.py)
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
    def _identify_top_n_planes_for_drone(self, drone_id:int) -> List[Coord]:
        """
        Identify the top N% highest value planes on the grid for given drone ID.

        Returns:
            List[Coord]: A list of coordinates (x, y) of the top N% planes
        """

        cutoff = np.percentile(self.grid.grid, 100 - self.N)
        ys, xs = np.where(self.grid.grid >= cutoff)

        result: List[Coord] = []
        dref = self.drones[drone_id]

        for y, x in zip(ys, xs):
            c = (int(x), int(y))
            if self.can_drone_visit(c, dref):
                result.append(c)
        return result

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

        # Warn if it might become slow
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
    t = 150
    T_ms = 500
    starts = [(20, 20), (80, 20), (50, 50), (90, 90)]  # 4 drones
    starts = [starts[0], starts[1]]  # 2 drones
    patience = 50
    grid_size = 100

    grid = Grid(grid_size, f"grids/{grid_size}.txt", patience=patience)

    planner = MultiDroneTopNPlanner(grid, starts, t, T_ms, seed=42, N=1, t_extra=1.0, beam=2, rollout_d=4, eps=0.15)
    # planner = MultiDroneGreedyRHPlanner(grid, starts, t, T_ms, seed=42, horizon=3)
    paths, scores, total_time, greedy_moves = planner.run()

    


    print(f"Total time: {total_time} ms")

    extra = f"N={planner.N} | t_extra={1+planner.t_extra} | beam={planner.beam} | rollout={planner.rollout_d} | eps={planner.eps}"
    if greedy_moves is not None:
        extra += f"\n$\\mathbf{{Incomplete\ algorithm:}}$ time budget exceeded after {t - greedy_moves} steps, filled remaining {greedy_moves} steps greedily."

    visualize_results(grid, 
                      paths, 
                      planner_name=planner.__class__.__name__,
                        time_ms=total_time,
                        max_time_ms=T_ms,
                        max_steps=t,
                        scores=scores,
                        patience=patience,
                        extra=extra,
                        show=True,
                        save_path=None
                    )


