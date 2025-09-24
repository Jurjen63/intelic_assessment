"""
Supportive Python file to load and build the grid.

"""

from pathlib import Path
import numpy as np
from typing import Tuple, List
import time
import matplotlib.pyplot as plt
from collections import deque

# Not used anymore; kept for reference. Discarded because it would probably not be as efficient as using numpy arrays directly.
class Plane:
    def __init__(self, value:int, patience:int=10):
        """
        Initialize one plane within the grid. 

        Args:
            value (int): The initial value of the plane as loaded from the grid. 
            patience (int, optional): The patience up until the the value is increased by one. Defaults to 10.

        
        Attributes:
            value (int): The current value of the plane.
            original_value (int): The original value of the plane as loaded from the grid.
            patience (int): The patience up until the the value is increased by one.
            counter (int): The counter to keep track of the patience.
        """
        self.value = value
        self.original_value = value

        self.patience = patience
        self.counter = 0


    def traverse(self):
        """
        Traverse the plane, resets value to zero and returns the current value for value cumulation. 
        """
        current_val = self.value
        self.value = 0

        return current_val
    
    def regenerate(self):
        """
        Regenerate the plane (linearly) given the patience up untill the original value.
        """
        if self.value < self.original_value:
            self.counter += 1

            if self.counter >= self.patience:
                self.value += 1
                self.counter = 0



# Helper types and constants
Coord = Tuple[int, int]
NEIGH8 = [(-1, -1), (0, -1), (1, -1),
            (-1, 0),          (1, 0),
            (-1, 1), (0, 1), (1, 1)]

class Grid:
    def __init__(self, N:int, grid_file:Path, patience:int=10):
        """
        Initialize the grid from a given .txt file and stores it as a 2D np.array. 

        Args:
            N (int): The size of the grid (N x N). (for checking correct grid size)
            grid_file (Path): The path to the .txt file containing the grid.
            patience (int, optional): The patience up until the the value is increased by one. Defaults to 10.
                
        """

        self.grid = np.zeros((N,N), dtype=np.float32)
        self.N = N

        with open(grid_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == N, f"Grid file {grid_file} does not have {N} lines."

            for y, line in enumerate(lines):
                values = list(map(int, line.strip().split()))
                assert len(values) == N, f"Line {y+1} in grid file {grid_file} does not have {N} values."
                
                for x, val in enumerate(values):
                    self.grid[y, x] = val


        # Keep a copy of the original grid for regeneration purposes
        self.original_grid = self.grid.copy()
        self.patience = patience

        self.subgrids = np.zeros((N, N), dtype=np.int32)  # Placeholder for subgrid values for multi-drone scenarios

    def traverse_plane(self, goal_pos:Coord) -> int:
        """
        Traverse the plane at the given coordinates (x, y) and return the value of the plane.

        Args:
            goal_pos (Tuple[int, int]): The (x, y) coordinates of the plane to traverse

        Returns:
            int: The value of the plane before traversal
        """

        assert 0 <= goal_pos[0] < self.grid.shape[1], f"x-coordinate {goal_pos[0]} out of bounds."
        assert 0 <= goal_pos[1] < self.grid.shape[0], f"y-coordinate {goal_pos[1]} out of bounds."

        value = self.grid[goal_pos[1], goal_pos[0]]
        self.grid[goal_pos[1], goal_pos[0]] = 0

        return int(value)   # Return as int to remove the regeneration float
    
    def regenerate(self):
        """
        Regenerate the entire grid (linearly) given the patience up untill the original value.
        """
        t0 = time.time()
        step = 1 / self.patience

        regeneration_mask = self.grid < self.original_grid
        np.add(self.grid, step, out=self.grid, where=regeneration_mask)  # only add step where regeneration is needed
        self.grid = np.minimum(self.grid, self.original_grid, out=self.grid)  # clamp to original values

        # print(f"GRID: time to regenerate {time.time() - t0}")

    def get_neighbours(self, current_pos:Coord) -> List[Coord]:
        """
        Get the valid neighbouring coordinates (x, y) of the current position.

        Args:
            current_pos (Coord): The (x, y) coordinates of the current position

        Returns:
            List[Coord]: A list of valid neighbouring coordinates (x, y)
        """

        neighbours = []

        for dx, dy in NEIGH8:
            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy

            if 0 <= new_x < self.N and 0 <= new_y < self.N:
                neighbours.append((new_x, new_y))

        return neighbours

    def get_value(self, pos:Coord) -> int:
        """
        Get the value of the plane at the given coordinates (x, y).

        Args:
            pos (Coord): The (x, y) coordinates of the plane

        Returns:
            int: The value of the plane
        """

        assert 0 <= pos[0] < self.grid.shape[1], f"x-coordinate {pos[0]} out of bounds."
        assert 0 <= pos[1] < self.grid.shape[0], f"y-coordinate {pos[1]} out of bounds."

        return int(self.grid[pos[1], pos[0]])  # Return as int to remove the regeneration float

    def copy(self) -> 'Grid':
        """
        Create a deep copy of the grid.

        Returns:
            Grid: A deep copy of the grid.
        """
        t0 = time.time()
        copy = Grid.__new__(Grid)
        copy.grid = self.grid.copy()
        copy.original_grid = self.original_grid.copy()
        copy.N = self.N
        copy.patience = self.patience

        # print(f"GRID: time to copy: {time.time() - t0}")

        return copy
    
    def create_balanced_subgrids(self, centers: List[Coord], t: int):
        """
        Create balanced subgrids for multiple drones using a balanced Breadth-First Approach. 

        The method grows each drone's subgrid simultaneously one layer at a time,
        until each drone has been assigned an equal share of the total reachable cells.

        Args:
            centers (List[Coord]): The list of (x, y) coordinates representing the centers of the subgrids for each drone.
            t (int): The maximum amount of steps a drone can take (used to limit the subgrid size).
        
        """
        # Initialize queues and tracking sets
        queues = [deque([center]) for center in centers]
        visited = set(centers)
        assigned_counts = [0] * len(centers)

        # Calculate the total number of reachable cells across all drones
        total_reachable_cells = 0
        for cx, cy in centers:
            for y in range(self.N):
                for x in range(self.N):
                    if max(abs(x - cx), abs(y - cy)) <= t:
                        total_reachable_cells += 1
        
        # Determine the target number of cells to assign per drone for balanced distribution
        target_per_drone = total_reachable_cells // len(centers)

        # Prepare the subgrids numpy array
        self.subgrids = np.zeros((self.N, self.N), dtype=np.int32)

        # Assign initial centers and update counts
        for i, (cx, cy) in enumerate(centers):
            self.subgrids[cy, cx] = i  # Assign subgrid ID starting from 0
            assigned_counts[i] = 1

        # Round-robin BFS expansion
        while any(queues) and any(count < target_per_drone for count in assigned_counts):
            for drone_id, queue in enumerate(queues):
                # Only expand if the drone's queue isn't empty and its target hasn't been met
                if queue and assigned_counts[drone_id] < target_per_drone:
                    current_x, current_y = queue.popleft()

                    # Add unvisited neighbors
                    for dx, dy in NEIGH8:
                        new_x, new_y = current_x + dx, current_y + dy

                        # Check bounds and if not visited
                        if (0 <= new_x < self.N and 0 <= new_y < self.N and 
                            (new_x, new_y) not in visited):

                            # Check if reachable from this drone's center
                            cx, cy = centers[drone_id]
                            if max(abs(new_x - cx), abs(new_y - cy)) <= t:
                                visited.add((new_x, new_y))
                                queue.append((new_x, new_y))
                                self.subgrids[new_y, new_x] = drone_id  # Assign subgrid ID
                                assigned_counts[drone_id] += 1

                                # Stop expanding for this drone once its target is reached
                                if assigned_counts[drone_id] >= target_per_drone:
                                    break
    
    def is_plane_assigned_to_drone(self, pos:Coord, drone_id:int) -> bool:
        """
        Check if the plane at the given coordinates (x, y) is assigned to the given drone ID.

        Args:
            pos (Coord): The (x, y) coordinates of the plane
            drone_id (int): The ID of the drone to check against

        Returns:
            bool: True if the plane is assigned to the given drone ID, False otherwise
        """

        assert 0 <= pos[0] < self.subgrids.shape[1], f"x-coordinate {pos[0]} out of bounds."
        assert 0 <= pos[1] < self.subgrids.shape[0], f"y-coordinate {pos[1]} out of bounds."

        return self.subgrids[pos[1], pos[0]] == drone_id



    # --- visualization methods for debugging/checking purposes
    def visualize_grid(self, centers:List[Coord]=None) -> None:
        """
        Visualize the grid using matplotlib.

        Args:
            centers (List[Coord], optional): The list of (x, y) coordinates representing the centers of the subgrids for each drone. Defaults to None.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(self.original_grid, cmap='viridis', origin='upper')

        unique_subgrids = np.unique(self.subgrids)
        if -1 in unique_subgrids:
            unique_subgrids = unique_subgrids[unique_subgrids != -1]  # Exclude -1 for untraversable areas

        levels = np.arange(unique_subgrids.max() + 1) - 0.5  # +1 to include the last level
        ax.contour(self.subgrids, levels=levels, colors='red', linewidths=2, linestyles='solid')

        center_x, center_y = zip(*centers) if centers else ([], [])
        ax.plot(center_x, center_y, 'ro', markersize=10, label='Subgrid Centers')

        ax.legend()
        plt.tight_layout()
        plt.show()

    def visualize_subgrids(self, centers: List[Coord]):
        """
        Create a visualization of the subgrid assignments using a color plot,
        including unreachable areas (-1).
        
        Args:
            centers (List[Coord]): The list of subgrid centers to plot.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_ids = np.unique(self.subgrids)
        num_colors = unique_ids.size
        
        cmap = plt.get_cmap('Paired', num_colors)
        
        mapping = {id_val: i for i, id_val in enumerate(unique_ids)}
        mapped_subgrids = np.vectorize(mapping.get)(self.subgrids)

        im = ax.imshow(mapped_subgrids, cmap=cmap, origin='upper', interpolation='nearest')
        
        cbar = fig.colorbar(im, ax=ax, ticks=np.arange(num_colors), label='Subgrid ID')
        cbar.ax.set_yticklabels(unique_ids)

        center_x, center_y = zip(*centers)
        ax.plot(center_x, center_y, 'ko', markersize=10, label='Subgrid starting points')

        ax.legend()
        plt.tight_layout()
        plt.show()

    


# For testing and visualization purposes
if __name__ == "__main__":
    grid = Grid(100, Path("grids/100.txt"), patience=50)
    drones = [(20, 20), (80, 20), (50, 50), (90, 90)]
    t = time.time()
    grid.create_balanced_bfs_subgrids(drones, t=150)
    print(f"Time to create balanced BFS subgrids: {time.time() - t:.2f} seconds")
    
    grid.visualize_grid(drones)