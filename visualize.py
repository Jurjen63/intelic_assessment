"""
Python file with functions to visualize the grid and planned path.
"""

from grid import Grid, Coord
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow
import numpy as np
from typing import List, Tuple

# Colors if n_drones <= 10
COLORS_DEFINED = [
    "#e41a1c",  
    "#377eb8",  
    "#ff00aa",  
    "#ff7f00",  
    "#984ea3",  
    "#bddbb9",  
    "#a65628",  
    "#f781bf",  
    "#999999",  
    "#66c2a5"   
]

def visualize_results(grid: Grid, 
                      paths: List[List[Coord]],
                      *,
                      planner_name:str,
                      time_ms:int,
                      max_time_ms:int,
                      max_steps:int,
                      scores:List[int],
                      patience:int,
                      extra:str="",
                      show:bool=True,
                      save_path:str=None,
                      ) -> None:
    """
    Visualize the grid and the planned path in a plot. Shows the original grid 
    values and overlays the path with arrows. 
    
    Args:
        grid (Grid): The grid object representing the environment
        paths (List[List[Coord]]): The planned paths per drone as a list of coordinates (x, y)

        planner_name (str): Name of the planner used
        time_ms (int): The total time taken to compute the path (in milliseconds)
        max_time_ms (int): The maximum allowed time for planning (in milliseconds)
        scores (List[int]): The total value collected along the path for each drone
        extra (str, optional): Any extra information to display. Defaults to "".
    """

    grid_array = grid.original_grid
    assert grid_array.ndim == 2, "Grid array must be 2D"

    fix, ax = plt.subplots(figsize=(24, 20))
    im = ax.imshow(grid_array, cmap='viridis', origin='upper')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_aspect('equal')

    # If there are subgrids, make contours
    if hasattr(grid, 'subgrids') and grid.subgrids is not None:
        sub = grid.subgrids
        unique = np.unique(sub)
        unique = unique[unique != -1]  # ignore -1 (no subgrid)

        levels = np.arange(unique.max() +1) - 0.5

        ax.contour(sub, levels=levels, colors='white', linewidths=4)
        ax.contour(sub, levels=levels, colors='black', linewidths=2)


    # Plot paths if available
    if paths:
        # use random colors if more paths than the 10 defined colors
        if len(paths) <= len(COLORS_DEFINED):
            colors = COLORS_DEFINED
        else:
            colors = np.random.shuffle(list(mcolors.CSS4_COLORS.keys()))

        for i, path in enumerate(paths):
            if len(path) > 1:
                xs, ys = zip(*path)
                # start marker
                ax.plot(xs[0], ys[0], marker='o', markersize=20, fillstyle='none',
                        markeredgewidth=3, color=colors[i], linestyle='None')
                
                # end marker
                ax.plot(xs[-1], ys[-1], marker='s', markersize=20, markeredgewidth=4,
                        markeredgecolor=colors[i], markerfacecolor='none', linestyle='None')

                # connecting arrows
                for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
                    dx, dy = x1 - x0, y1 - y0
                    # white arrow below
                    ax.arrow(x0, y0, dx, dy,
                        head_width=0.25, head_length=0.35,
                        fc='white', ec='white',
                        length_includes_head=True, linewidth=3, zorder=2)

                    # colored arrow on top
                    ax.arrow(x0, y0, dx, dy,
                            head_width=0.2, head_length=0.3,
                            fc=colors[i], ec=colors[i],
                            length_includes_head=True, linewidth=2, zorder=3)
        # legend handles
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markeredgecolor='black',
                markersize=15, markeredgewidth=2, fillstyle='none', linestyle='None', label="Start"),
            Line2D([0], [0], marker='s', color='w', markeredgecolor='black',
                markersize=15, markeredgewidth=2, fillstyle='none', linestyle='None', label="End"),
            Line2D([0], [0], color='black', linewidth=2, label="Path"),
            *[Line2D([0], [0], color=colors[i], linewidth=4, label=f"Drone {i+1} (score: {scores[i]})") for i in range(len(paths))]
        ]
        ax.legend(handles=legend_elements, fontsize=18, loc='best')
            
    avg_path_length = int(np.mean([len(p) for p in paths]) if paths else 0) - 1 # -1 because start is included
    title = f"Planner: {planner_name} |  Score: {sum(scores)} | Time: {time_ms} ms / {max_time_ms} ms\n" \
            f"               Steps: {avg_path_length}/{max_steps} | Regen. rate: {1/patience} per step"
    if extra:
        title += f" \nHyperparams: {extra}"    
    

    ax.set_title(title, fontsize=18)
    plt.tight_layout()

    if show:
        plt.show()

    else:
        if save_path is not None:
            plt.savefig(save_path, dpi=400, format='png')
        plt.close(fix)
