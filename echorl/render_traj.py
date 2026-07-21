"""EchoRL -- top-down trajectory maps: model-free PPO vs model-based MPC.

Draws sample doorway rooms (walls, the opening gap, obstacle pillars, the goal + its
success radius) and overlays each controller's closed-loop paths from a shared set of
start poses, coloured by outcome (green = reached the doorway, red = failed). Columns =
controllers, rows = rooms. Makes the behavioural difference behind the success numbers
(§5) legible: how PPO curves in on the bearing vs how the MPC wanders / gets stuck.

Run (nobs=0):
  python echorl/render_traj.py --nobs 0 --ppo-ckpt echorl/ckpt/ppo_goal_nobs0_full_s0_best.pt \
      --corpus echorl/corpus_goal.pkl --wm echorl/ckpt/mpc_wm.pt --out results/echorl_traj_nobs0.png
"""
from __future__ import annotations
import argparse, math, os, sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from env import EchoLocEnv, EpisodeServer, EAR_Y, DYAW_MAX, DSTEP_MAX   # noqa: E402
from mpc import (WorldModel, load_corpus, cem_plan, oracle_action, raw_to_norm,  # noqa: E402
                 COCH_SHAPE)
from mpc_direct import train_outcome_head, cem_outcome_plan            # noqa: E402
from ppo import opening_seeds, RecurrentAC, obs_to_t                   # noqa: E402


# --------------------------------------------------------------- geometry draw
def draw_room(ax, srv, goal, goal_radius=1.2):
    Lx, Lz = srv.Lx, srv.Lz
    # opening spans, keyed by wall id: 0:+x 1:-x 4:+z 5:-z  (record = wall,cx,cy,cz,w,h)
    gaps = {0: [], 1: [], 4: [], 5: []}
    for (wall, cx, cy, cz, w, h) in srv.openings:
        gaps.setdefault(int(wall), []).append((cx if int(wall) in (4, 5) else cz, w))

    def wall_with_gap(p0, p1, along, gap_list):
        """Draw a wall segment from p0..p1 (a horizontal or vertical line), removing the
        opening spans. `along` picks which coord varies; gap_list = [(centre,width)]."""
        lo, hi = (p0[along], p1[along]); segs = [(lo, hi)]
        for (c, w) in gap_list:
            new = []
            for (a, b) in segs:
                if c - w/2 <= a and b <= c + w/2:      # fully inside gap
                    continue
                if c + w/2 <= a or b <= c - w/2:       # no overlap
                    new.append((a, b)); continue
                if a < c - w/2:
                    new.append((a, c - w/2))
                if c + w/2 < b:
                    new.append((c + w/2, b))
            segs = new
        for (a, b) in segs:
            if along == 0:
                ax.plot([a, b], [p0[1], p1[1]], color="k", lw=2.5, solid_capstyle="butt")
            else:
                ax.plot([p0[0], p1[0]], [a, b], color="k", lw=2.5, solid_capstyle="butt")

    wall_with_gap((Lx/2, -Lz/2), (Lx/2, Lz/2), 1, gaps[0])     # +x wall (varies z)
    wall_with_gap((-Lx/2, -Lz/2), (-Lx/2, Lz/2), 1, gaps[1])   # -x wall
    wall_with_gap((-Lx/2, Lz/2), (Lx/2, Lz/2), 0, gaps[4])     # +z wall (varies x)
    wall_with_gap((-Lx/2, -Lz/2), (Lx/2, -Lz/2), 0, gaps[5])   # -z wall

    for (cx, cy, cz, hx, hy, hz, _m) in srv.obstacles:         # ear-height pillars
        if cy - hy - 1e-3 <= EAR_Y <= cy + hy + 1e-3:
            ax.add_patch(Rectangle((cx - hx, cz - hz), 2*hx, 2*hz, color="0.55", zorder=1))
    ax.add_patch(Circle(goal, goal_radius, fill=False, ls="--", ec="0.4", lw=1))
    ax.plot(*goal, marker="*", ms=15, color="gold", mec="k", mew=0.6, zorder=5)
    ax.set_xlim(-Lx/2 - 0.3, Lx/2 + 0.3); ax.set_ylim(-Lz/2 - 0.3, Lz/2 + 0.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


# --------------------------------------------------------------- controllers
def make_controllers(args, dev):
    ctrls = {}
    if args.ppo_ckpt:
        net = RecurrentAC().to(dev)
        net.load_state_dict(torch.load(args.ppo_ckpt, map_location=dev)["model"]); net.eval()
        ctrls["model-free PPO"] = ("ppo", net)
    wm = None
    if args.wm:
        wm = WorldModel().to(dev); wm.load_state_dict(torch.load(args.wm, map_location=dev)["model"]); wm.eval()
        for p in wm.parameters():
            p.requires_grad_(False)
        ctrls["learned-WM MPC"] = ("mpc", wm)
    if args.corpus and wm is not None:
        g = train_outcome_head(wm, load_corpus(args.corpus), dev, H=args.H, seed=0)
        ctrls["direct-outcome MPC"] = ("direct", (wm, g))
    return ctrls


@torch.no_grad()
def _rollout(kind, model, env, obs, info, dev, H):
    """Closed-loop from an already-reset (obs, info). Returns (xs, zs, success)."""
    xs, zs = [info["pose"][0]], [info["pose"][1]]
    h = torch.zeros(1, 1, model.H, device=dev) if kind == "ppo" else None
    done = False; ok = False
    while not done:
        if kind == "ppo":
            f = model.feats(*[t.unsqueeze(0) for t in obs_to_t(obs, dev)])
            mean, _std, _v, h = model.step(f, h)
            a = mean.clamp(-1, 1).cpu().numpy()[0]
        elif kind == "mpc":
            z = model.enc(torch.as_tensor(obs["range"][None], device=dev), torch.as_tensor(obs["coch"][None], device=dev))
            a = cem_plan(model, z, H, dev)
        else:  # direct-outcome
            wm, g = model
            z = wm.enc(torch.as_tensor(obs["range"][None], device=dev), torch.as_tensor(obs["coch"][None], device=dev))
            a = cem_outcome_plan(g, z, H, dev)
        obs, _, term, trunc, info = env.step(a)
        xs.append(info["pose"][0]); zs.append(info["pose"][1])
        ok = bool(info.get("success")); done = term or trunc
    return xs, zs, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nobs", type=int, default=0)
    ap.add_argument("--rooms", type=int, default=4, help="number of sample rooms (rows)")
    ap.add_argument("--starts", type=int, default=6, help="trajectories per room per method")
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--ppo-ckpt", default="")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--wm", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = opening_seeds(600, args.rooms, args.nobs)          # held-out rooms (as in eval)
    ctrls = make_controllers(args, dev)
    names = list(ctrls.keys())
    print(f"rendering nobs={args.nobs} | rooms {seeds} | methods {names}")

    nrows, ncols = len(seeds), len(names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4*ncols, 3.4*nrows), squeeze=False)
    for r, seed in enumerate(seeds):
        srv = EpisodeServer(seed, args.nobs, threads=0)       # geometry for drawing
        for c, name in enumerate(names):
            ax = axes[r][c]
            kind, model = ctrls[name]
            env = EchoLocEnv(seed_pool=[seed], action_mode="continuous", nobs=args.nobs,
                             sim_threads=2, t_max=40, task="goal", base_seed=seed)
            n_ok = 0; goal = None
            for s in range(args.starts):
                obs, info = env.reset(seed=seed*1000 + s); goal = info["goal"]
                xs, zs, ok = _rollout(kind, model, env, obs, info, dev, args.H)
                col = "#1a9850" if ok else "#d73027"
                ax.plot(xs, zs, color=col, lw=1.3, alpha=0.85, zorder=3)
                ax.plot(xs[0], zs[0], marker="o", ms=4, color=col, mec="k", mew=0.4, zorder=4)
                n_ok += int(ok)
            env.close()
            draw_room(ax, srv, goal)                          # frame the paths
            if r == 0:
                ax.set_title(name, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"room {seed}", fontsize=10)
            ax.text(0.02, 0.98, f"{n_ok}/{args.starts}", transform=ax.transAxes,
                    va="top", ha="left", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))
        srv.close()
        print(f"  room {seed} done")

    fig.suptitle(f"EchoRL doorway-finding — closed-loop trajectories (nobs={args.nobs})\n"
                 f"green = reached the doorway, red = failed; * = goal, o = start",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
