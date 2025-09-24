"""
Main file to run the grid and planner modules.
"""

from multi_planner import *
from single_planner import *
from visualize import visualize_results
from grid import Grid

import numpy as np
import time

np.random.seed(42)

abbs = {
    "MultiDroneTopNPlanner": "MD_TNP",
    "MultiDroneGreedyRHPlanner": "MD_GPRHP",
}

# Hyperparameter options
grids_drones = {    # per different grid different spawn positions of the drones
    20: [[(10, 10)], 
         [(2, 2), (2, 4)],  # simulate coming give or take from the same base
         [(2, 2), (18, 18)],
         [(2, 2), (18, 18), (2, 18), (18, 2)]],
    100: [[(50, 50)], 
          [(48, 48), (52, 52)],
          [(5, 5), (15, 5)],
          [(5, 5), (20, 5), (40, 5), (60, 5)], # simulate coming from the same base
          [tuple(xy) for xy in 2*np.random.randint(0, 50, size=(10, 2))],], # 10 drones random positions (force a little bit of spacing)
    1000: [[(500, 500)], 
           [(50, 50), (950, 950)],
           [(50, 50), (200, 50), (50, 200), (200, 200)], # simulate coming from the same base
           [(450, 450), (450, 550), (550, 450), (550, 550)],
           [tuple(xy) for xy in 5*np.random.randint(0, 200, size=(20, 2))]] # 20 drones random positions (force a little bit of spacing)
}
horizons = [1, 2, 3, 4, 5]  # Greedy
N = [1, 2, 5, 10] # Top-N
t_extra = [0.5, 1, 2] # Top-N
beams = [2, 4, 8, 16] # Top-N
rollouts = [2, 4, 8, 16] # Top-N
eps = [0.0, 0.15, 0.3] # Top-N
t = [20, 50, 100, 300, 500] # both
T_ms = [500, 1000, 5000, 10000] # both
patience = [10, 50, 100, 250] # both

# TODO: test all hyperparameters against each other...

if __name__ == "__main__":
    # # Single drone scenarios
    # scenarios = [grids_drones[20][0][0], grids_drones[100][0][0], grids_drones[1000][0][0]]
    # grids = [20, 100, 1000]
    # Ns = [1, 5, 10]
    # t_extra_ = [1, 1, 1]
    # horizons = [[1, 3], [1, 3], [1, 3]]
    # ts = [25, 100, 300]
    # T_mss = [500, 2000, 5000]
    # ps = [50, 100, 250]

    # for i, drones in enumerate(scenarios):
    #     for h in horizons[i]:
    #         grid = Grid(grids[i], f"grids/{grids[i]}.txt", patience=ps[i])
    #         planner = GreedyRHPlanner(grid, t=ts[i], T_ms=T_mss[i], start=drones, horizon=h, seed=42)
    #         path, score, time_ms = planner.run()
    #         title = f"{score}_SD_GRHP_d{len(drones)}_t{ts[i]}_T{T_mss[i]}_p{ps[i]}_h{h}"
    #         extra = f"horizon={h}"
    #         visualize_results(grid,
    #                             [path],
    #                             planner_name="GreedyRHPlanner",
    #                             max_steps=ts[i],
    #                             scores=[score],
    #                             time_ms=time_ms,
    #                             max_time_ms=T_mss[i],
    #                             patience=ps[i],
    #                             extra=extra,
    #                             show=False,
    #                             save_path=f"results/{grids[i]}/1/{title}.png"
    #                             )
    #     grid = Grid(grids[i], f"grids/{grids[i]}.txt", patience=ps[i])
    #     planner = TopNPlanner(grid, t=ts[i], T_ms=T_mss[i], start=drones, N=Ns[i], t_extra=t_extra_[i], seed=42)
    #     path, score, time_ms = planner.run()
    #     title = f"{score}_SD_TNP_d{len(drones)}_t{ts[i]}_T{T_mss[i]}_p{ps[i]}_N{Ns[i]}_te{t_extra_[i]}"
    #     extra = f"N={Ns[i]} | t_extra={1+t_extra_[i]}"
    #     visualize_results(grid,
    #                         [path],
    #                         planner_name="TopNPlanner",
    #                         max_steps=ts[i],
    #                         scores=[score],
    #                         time_ms=time_ms,
    #                         max_time_ms=T_mss[i],
    #                         patience=ps[i],
    #                         extra=extra,
    #                         show=False,
    #                         save_path=f"results/{grids[i]}/1/{title}.png"
    #                         )
        
    # Multi drone scenarios
    scenarios = [grids_drones[20][3], grids_drones[100][3]]
    # scenarios = [grids_drones[100][4]]
    grids = [20, 100, 1000]
    Ns = [1, 5, 10]
    t_extra_ = [1, 1, 1]
    horizons = [[1, 3], [1, 3], [1, 3]]
    ts = [25, 100, 300]
    T_mss = [500, 2000, 5000]
    ps = [50, 100, 250]

    for i, drones in enumerate(scenarios):
        for h in horizons[i]:
            grid = Grid(grids[i], f"grids/{grids[i]}.txt", patience=ps[i])
            planner = MultiDroneGreedyRHPlanner(grid, t=ts[i], T_ms=T_mss[i], starts=drones, horizon=h, seed=42)
            paths, scores, total_time, greedy_moves = planner.run()
            title = f"{sum(scores)}_MD_GRHP_d{len(drones)}_t{ts[i]}_T{T_mss[i]}_p{ps[i]}_h{h}"
            extra = f"horizon={h}"
            if greedy_moves is not None:
                extra += f"\n$\\mathbf{{Incomplete\ algorithm:}}$ time budget exceeded after {ts[i] - greedy_moves} steps, filled remaining {greedy_moves} steps greedily."

            visualize_results(grid,
                                paths,
                                planner_name="MultiDroneGreedyRHPlanner",
                                max_steps=ts[i],
                                scores=scores,
                                time_ms=total_time,
                                max_time_ms=T_mss[i],
                                patience=ps[i],
                                extra=extra,
                                show=False,
                                save_path=f"results/{grids[i]}/multi/a/{title}.png"
                                )
        grid = Grid(grids[i], f"grids/{grids[i]}.txt", patience=ps[i])
        planner = MultiDroneTopNPlanner(grid, t=ts[i], T_ms=T_mss[i], starts=drones, N=Ns[i], t_extra=t_extra_[i], seed=42)
        paths, scores, total_time, greedy_moves = planner.run()
        title = f"{sum(scores)}_MD_TNP_d{len(drones)}_t{ts[i]}_T{T_mss[i]}_p{ps[i]}_N{Ns[i]}_te{t_extra_[i]}"
        extra = f"N={planner.N} | t_extra={1+planner.t_extra} | beam={planner.beam} | rollout={planner.rollout_d} | eps={planner.eps}"
        if greedy_moves is not None:
            extra += f"\n$\\mathbf{{Incomplete\ algorithm:}}$ time budget exceeded after {ts[i] - greedy_moves} steps, filled remaining {greedy_moves} steps greedily."

        visualize_results(grid,
                            paths,
                            planner_name="MultiDroneTopNPlanner",
                            max_steps=ts[i],
                            scores=scores,
                            time_ms=total_time,
                            max_time_ms=T_mss[i],
                            patience=ps[i],
                            extra=extra,
                            show=False,
                            save_path=f"results/{grids[i]}/multi/a/{title}.png"
                            )
        
        

