"""EchoRL -- top-down trajectory maps for BOTH tasks, all controllers, both clutter tiers.

Draws sample rooms (walls, opening gap, obstacle pillars, and for the goal task the goal
+ its success radius) and overlays each controller's closed-loop paths from a shared set of
starts, coloured by outcome (green = success, red = failure). Columns = controllers, rows =
rooms. One figure per (task, clutter) case; together they cover every condition and make the
behaviour behind the success numbers legible.

  task=avoid : controllers DQN (discrete, fixed-speed) / PPO (continuous) / oracle (privileged
               planner). success = survived the 40-step cap without colliding; a red x marks
               where a failed run crashed.
  task=goal  : controllers PPO / learned-WM MPC / direct-outcome MPC. success = reached the
               doorway (within goal_radius).

Run:
  python echorl/render_traj.py --task avoid --nobs 0 --out results/echorl_avoid_nobs0.png
  python echorl/render_traj.py --task goal  --nobs 3 --out results/echorl_goal_nobs3.png
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
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from env import EchoLocEnv, EpisodeServer, EAR_Y, DYAW_MAX, DSTEP_MAX   # noqa: E402
from models import DuelingQNet                                          # noqa: E402
from ppo import RecurrentAC, obs_to_t, opening_seeds                    # noqa: E402
from mpc import (WorldModel, load_corpus, cem_plan, oracle_action, raw_to_norm)  # noqa: E402
from mpc_direct import train_outcome_head, cem_outcome_plan            # noqa: E402
from oracle import Geom, plan_action                                    # noqa: E402

GREEN, RED = "#1a9850", "#d73027"


# --------------------------------------------------------------- geometry draw
def draw_room(ax, srv, goal=None, goal_radius=1.2):
    Lx, Lz = srv.Lx, srv.Lz
    gaps = {0: [], 1: [], 4: [], 5: []}
    for (wall, cx, cy, cz, w, h) in srv.openings:
        gaps.setdefault(int(wall), []).append((cx if int(wall) in (4, 5) else cz, w))

    def wall_with_gap(p0, p1, along, gap_list):
        lo, hi = (p0[along], p1[along]); segs = [(lo, hi)]
        for (c, w) in gap_list:
            new = []
            for (a, b) in segs:
                if c - w/2 <= a and b <= c + w/2: continue
                if c + w/2 <= a or b <= c - w/2: new.append((a, b)); continue
                if a < c - w/2: new.append((a, c - w/2))
                if c + w/2 < b: new.append((c + w/2, b))
            segs = new
        for (a, b) in segs:
            if along == 0: ax.plot([a, b], [p0[1], p1[1]], color="k", lw=2.5, solid_capstyle="butt")
            else: ax.plot([p0[0], p1[0]], [a, b], color="k", lw=2.5, solid_capstyle="butt")

    wall_with_gap((Lx/2, -Lz/2), (Lx/2, Lz/2), 1, gaps[0])
    wall_with_gap((-Lx/2, -Lz/2), (-Lx/2, Lz/2), 1, gaps[1])
    wall_with_gap((-Lx/2, Lz/2), (Lx/2, Lz/2), 0, gaps[4])
    wall_with_gap((-Lx/2, -Lz/2), (Lx/2, -Lz/2), 0, gaps[5])
    for (cx, cy, cz, hx, hy, hz, _m) in srv.obstacles:
        if cy - hy - 1e-3 <= EAR_Y <= cy + hy + 1e-3:
            ax.add_patch(Rectangle((cx - hx, cz - hz), 2*hx, 2*hz, color="0.55", zorder=1))
    if goal is not None:
        ax.add_patch(Circle(goal, goal_radius, fill=False, ls="--", ec="0.4", lw=1))
        ax.plot(*goal, marker="*", ms=15, color="gold", mec="k", mew=0.6, zorder=5)
    ax.set_xlim(-Lx/2 - 0.3, Lx/2 + 0.3); ax.set_ylim(-Lz/2 - 0.3, Lz/2 + 0.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


# --------------------------------------------------------------- controllers
def make_controllers(task, args, dev):
    ctrls = {}
    if task == "goal":
        if args.ppo_ckpt:
            net = RecurrentAC().to(dev); net.load_state_dict(torch.load(args.ppo_ckpt, map_location=dev)["model"]); net.eval()
            ctrls["model-free PPO"] = ("ppo", net)
        wm = None
        if args.wm:
            wm = WorldModel().to(dev); wm.load_state_dict(torch.load(args.wm, map_location=dev)["model"]); wm.eval()
            for p in wm.parameters(): p.requires_grad_(False)
            ctrls["learned-WM MPC"] = ("mpc", wm)
        if args.corpus and wm is not None:
            g = train_outcome_head(wm, load_corpus(args.corpus), dev, H=args.H, seed=0)
            ctrls["direct-outcome MPC"] = ("direct", (wm, g))
    else:  # avoid
        if args.dqn_ckpt:
            q = DuelingQNet().to(dev); q.load_state_dict(torch.load(args.dqn_ckpt, map_location=dev)["model"]); q.eval()
            ctrls[args.dqn_label] = ("dqn", q)
        if args.ppo_ckpt:
            net = RecurrentAC().to(dev); net.load_state_dict(torch.load(args.ppo_ckpt, map_location=dev)["model"]); net.eval()
            ctrls["model-free PPO"] = ("ppo", net)
        ctrls["privileged oracle"] = ("oracle_avoid", None)
    return ctrls


@torch.no_grad()
def rollout(kind, model, env, obs, info, dev, H, srv=None):
    """Closed-loop from a reset (obs, info). Returns (xs, zs, success, crash_xy|None)."""
    xs, zs = [info["pose"][0]], [info["pose"][1]]
    h = torch.zeros(1, 1, model.H, device=dev) if kind == "ppo" else None
    rng_np = np.random.default_rng(0); g = Geom(srv) if kind == "oracle_avoid" else None
    done = False; success = False; crash = None
    while not done:
        if kind == "ppo":
            f = model.feats(*[t.unsqueeze(0) for t in obs_to_t(obs, dev)])
            mean, _s, _v, h = model.step(f, h); a = mean.clamp(-1, 1).cpu().numpy()[0]
        elif kind == "dqn":
            q = model(*[t.unsqueeze(0) for t in obs_to_t(obs, dev)])
            a = int(q.argmax(1).item())                       # discrete env takes the int
        elif kind == "oracle_avoid":
            x, z, yaw = info["pose"]; dyaw, dstep = plan_action(x, z, yaw, g, rng_np)
            a = np.array(raw_to_norm(dyaw, dstep), np.float32)
        elif kind == "oracle_goal":
            a = oracle_action(*info["pose"], info["goal"])
        elif kind == "mpc":
            z = model.enc(torch.as_tensor(obs["range"][None], device=dev), torch.as_tensor(obs["coch"][None], device=dev))
            a = cem_plan(model, z, H, dev)
        else:  # direct
            wm, gg = model
            z = wm.enc(torch.as_tensor(obs["range"][None], device=dev), torch.as_tensor(obs["coch"][None], device=dev))
            a = cem_outcome_plan(gg, z, H, dev)
        obs, _, term, trunc, info = env.step(a)
        xs.append(info["pose"][0]); zs.append(info["pose"][1])
        if info.get("collision"): crash = (info["pose"][0], info["pose"][1])
        success = bool(info.get("success")) if env.task == "goal" else (not info.get("collision"))
        done = term or trunc
    return xs, zs, success, (crash if env.task == "avoid" else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["avoid", "goal"], default="goal")
    ap.add_argument("--nobs", type=int, default=0)
    ap.add_argument("--rooms", type=int, default=4)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--ppo-ckpt", default="")
    ap.add_argument("--dqn-ckpt", default="")
    ap.add_argument("--dqn-label", default="DQN (discrete)")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--wm", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # goal rooms must have an opening; avoid rooms are any held-out rooms
    seeds = opening_seeds(600, args.rooms, args.nobs) if args.task == "goal" else list(range(600, 600 + args.rooms))
    ctrls = make_controllers(args.task, args, dev)
    names = list(ctrls.keys())
    amode = "discrete" if (args.task == "avoid" and "dqn" in [c[0] for c in ctrls.values()]) else "continuous"
    print(f"task={args.task} nobs={args.nobs} | rooms {seeds} | methods {names}")

    nrows, ncols = len(seeds), len(names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4*ncols, 3.4*nrows), squeeze=False)
    for r, seed in enumerate(seeds):
        srv = EpisodeServer(seed, args.nobs, threads=0)
        for c, name in enumerate(names):
            ax = axes[r][c]; kind, model = ctrls[name]
            am = "discrete" if kind == "dqn" else "continuous"
            env = EchoLocEnv(seed_pool=[seed], action_mode=am, nobs=args.nobs, sim_threads=2,
                             t_max=40, task=args.task, base_seed=seed)
            n_ok = 0; goal = None
            for s in range(args.starts):
                obs, info = env.reset(seed=seed*1000 + s)
                goal = info.get("goal")
                xs, zs, ok, crash = rollout(kind, model, env, obs, info, dev, args.H, srv=srv)
                col = GREEN if ok else RED
                ax.plot(xs, zs, color=col, lw=1.3, alpha=0.85, zorder=3)
                ax.plot(xs[0], zs[0], marker="o", ms=4, color=col, mec="k", mew=0.4, zorder=4)
                if crash is not None:
                    ax.plot(*crash, marker="x", ms=7, color=RED, mew=1.8, zorder=6)
                n_ok += int(ok)
            env.close()
            draw_room(ax, srv, goal if args.task == "goal" else None)
            if r == 0: ax.set_title(name, fontsize=12, fontweight="bold")
            if c == 0: ax.set_ylabel(f"room {seed}", fontsize=10)
            ax.text(0.02, 0.98, f"{n_ok}/{args.starts}", transform=ax.transAxes, va="top", ha="left",
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85))
        srv.close(); print(f"  room {seed} done")

    clutter = "cluttered (3 pillars)" if args.nobs else "empty (no pillars)"
    what = ("reach the doorway" if args.task == "goal" else "survive without crashing")
    goalnote = "★ goal · " if args.task == "goal" else "✕ = crash · "
    fig.suptitle(f"EchoRL — {('doorway-finding' if args.task=='goal' else 'obstacle avoidance')}, "
                 f"{clutter}\ntrajectories to {what}; green = success, red = failure; "
                 f"{goalnote}○ = start · grey = pillars", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130); print(f"saved {args.out}")


if __name__ == "__main__":
    main()
