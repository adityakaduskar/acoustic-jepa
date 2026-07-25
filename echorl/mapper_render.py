"""Render true vs predicted maps for a few held-out cluttered doorway rooms.

Each row = one room: (left) TRUE occupancy with the exploratory path + true goal;
(right) the mapper's PREDICTED occupancy (scatter-CNN) with the fused doorway estimate.
Makes the M1 story legible -- room walls recovered, interior pillars largely missed,
doorway localized to ~2-3 m -- and why the planner still navigates on it.
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import torch
import mapper, mapper_fuse, mapper_collect
from mapping import (GRID_SPAN, true_occupancy, goal_from_openings, world_to_grid)
from ppo import opening_seeds


def main():
    seeds = opening_seeds(600, 4, 3)
    eps = mapper_collect.collect(seeds, 1, nobs=3, t_max=40, workers=4, base_seed=99)
    occ_m = mapper.OccMapper().to(mapper.DEV)
    occ_m.load_state_dict(torch.load("echorl/ckpt/mapper_nobs3.pt", map_location=mapper.DEV)); occ_m.eval()
    goal_m = mapper_fuse.EgoVecNet().to(mapper.DEV)
    goal_m.load_state_dict(torch.load("echorl/ckpt/egovec_goal_nobs3.pt", map_location=mapper.DEV)); goal_m.eval()

    n = len(eps)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n))
    ext = [-GRID_SPAN / 2, GRID_SPAN / 2, -GRID_SPAN / 2, GRID_SPAN / 2]
    for i, e in enumerate(eps):
        poses = e["poses"]; r = e["range"].astype(np.float32); c = e["coch"].astype(np.float32)
        occ_t = true_occupancy(e["Lx"], e["Lz"], e["obstacles"])
        gx, gz = goal_from_openings(e["openings"])
        occ_p, gest_cnn, _ = mapper.predict_grid(occ_m, poses, r, c)
        gxh, gzh = mapper_fuse.fuse(goal_m, poses, r, c)
        for ax, grid, title in ((axes[i, 0], occ_t, f"seed {e['seed']}: TRUE"),
                                (axes[i, 1], occ_p, "PREDICTED (scatter-CNN)")):
            ax.imshow(grid.T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1, alpha=0.9)
            ax.plot(poses[:, 0], poses[:, 1], ".-", ms=3, lw=0.8, color="tab:blue", alpha=0.7)
            ax.plot(gx, gz, "*", ms=16, color="tab:green", label="true door")
            ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        axes[i, 1].plot(gxh, gzh, "X", ms=11, color="tab:red", label="fused door est")
        if i == 0:
            axes[i, 1].legend(fontsize=7, loc="upper right")
    fig.suptitle("EchoRL mapping — cluttered doorway (nobs=3): true vs learned map\n"
                 "walls recovered · pillars largely missed · doorway ~2.8 m", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs("results", exist_ok=True)
    out = "results/echorl_mapping_nobs3.png"
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()
