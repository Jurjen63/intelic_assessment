# Intelic Assignment 

This private GitHub repository contains my solutions for the Intelic assignment (see [here](./SE%20-%20assessment%201%20-%20pathplanner%20-%20English%20-%20short.md) for the original description.)

![Single-drone 20 grid example](results/20/1/51_SD_GRHP_d2_t25_T500_p50_h3.png)

## Requirements
- Python ≥ 3.8
- numpy 1.26.4
- matplotlib 3.5.1

## Project Structure
```
├── grid.py             # Grid class and helper functions
├── grids               # Provided .txt grid files
├── main.py             # Runs experiment and generates results
├── multi_planner.py    # Multi-drone planners (experimental)
├── README.md           
├── results             # Experiment results
│   ├── 100             # 100x100 grid (1, 2, 4 drones, various scenarios)
│   ├── 1000            # 1000x1000 grid (1 drone)
│   └── 20              # 20x20 grid (1, 2, 4 drones, various scenarios)
├── SE - assessment 1 - pathplanner - English - short.md
├── single_planner.py   # Single-drone planners
└── visualize.py        # Plotting utilities
```

## Implemented Planners
The objective was to maximize the total score collected by the drone(s) by moving through the grid (which was able to regenerate). For this, I implemented two approaches:

### 1. Greedy Receding Horizon Planner
- Parameterized by a horizon *i*
- At *h=1*, the drone chooses the immediate neighbour
- At *h=3* the drone looks three steps ahead and selects the move that maximizes the cumulative score

### 2. Top-N planner
1. Initializes by selecting the Top *N%* highest-value planes
2. Orders them first by proximity to the start, then by inter-plane distances.
3. Builds a shortest path through these targets, and refines the path if extra time is available. 
    - The best path is chosen based on the score-to-steps ratio. 


#### Multi-Drone implementation
The grid is divided into balanced subgrids using a round-robin BFS expansion from each drone’s start. Each drone is restricted to its subgrid (via ID checks), though in the Top-N planner a known glitch sometimes allows paths to cross regions.



## Notes
- The single-drone planners are functional and deliver stable results for *grid_size=[20, 100]* but is sometimes unstable for *grid_size=1000* (Top-N planner). 
- The multi-drone version works is partially functional for *grid_size=[20, 100]* but not for *grid_size=1000*. (long runtimes) 
- Visualizations show the paths and collected scores (per drone)