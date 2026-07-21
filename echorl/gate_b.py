"""EchoRL Experiment B -- GATE: does the cluttered doorway task need planning?

Before spending a ~2 h PPO run on nobs=3 goal-finding (RESULTS.md 5 is nobs=0), check
the task is even discriminative: on privileged geometry (no echoes), does a
RECEDING-HORIZON planner beat a GREEDY head-to-goal controller? If planning ~ greedy,
the interior pillars don't actually block the route to the doorway, so model-free-vs-
model-based can't separate there and Experiment B is not worth running. If planning >>
greedy, clutter forces detours and B is worth it.

Sim-free: spawn each room's server ONCE for its geometry (dims + obstacle AABBs +
openings), then simulate the goal task analytically with the env's exact goal-mode
rules (NON-TERMINAL collision: a bump blocks translation but the agent still turns;
success = reach within goal_radius). Mirrors oracle.py's privileged-geometry approach.
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from env import (EpisodeServer, _swept_collision, _obstacle_clearance, _wall_clearance,   # noqa: E402
                 DYAW_MAX, DSTEP_MAX)
from mpc import oracle_action                              # noqa: E402
from ppo import opening_seeds                              # noqa: E402


class Geom:
    """Lightweight geometry shim with the attrs env's collision helpers read."""
    def __init__(self, srv):
        self.Lx, self.Lz, self.obstacles = srv.Lx, srv.Lz, srv.obstacles


def goal_pos(srv):
    """Point 1.5 m in front of the largest opening -- identical to env._goal_pos."""
    ops = srv.openings
    if ops:
        o = max(ops, key=lambda r: r[4] * r[5])
        wall, cx, cz, off = int(o[0]), float(o[1]), float(o[3]), 1.5
        if wall == 0: return cx - off, cz
        if wall == 1: return cx + off, cz
        if wall == 4: return cx, cz - off
        if wall == 5: return cx, cz + off
    return 0.0, 0.0


def sample_start(g, goal, rng, start_margin=0.8, goal_radius=1.2):
    far = max(2.5, goal_radius + 1.5)
    for _ in range(200):
        x = rng.uniform(-g.Lx/2 + start_margin, g.Lx/2 - start_margin)
        z = rng.uniform(-g.Lz/2 + start_margin, g.Lz/2 - start_margin)
        if _obstacle_clearance(x, z, g.obstacles) > start_margin and \
           math.hypot(x - goal[0], z - goal[1]) > far:
            return float(x), float(z), float(rng.uniform(0, 360))
    return 0.0, 0.0, float(rng.uniform(0, 360))


def step_goal(x, z, yaw, dyaw, dstep, g):
    """One env goal-mode step (non-terminal collision). Returns (x,z,yaw)."""
    t = math.radians(yaw); nx = x + dstep*math.sin(t); nz = z - dstep*math.cos(t)
    if _swept_collision(x, z, nx, nz, g):
        return x, z, yaw + dyaw                  # bump: block translation, still turn
    return nx, nz, yaw + dyaw


def plan_action(x, z, yaw, g, goal, rng, K=256, H=8):
    """Receding-horizon random shooting on true dynamics: minimise final distance to
    goal (collisions block translation, so detours emerge). Returns first (dyaw,dstep)."""
    dyaw = rng.uniform(-DYAW_MAX, DYAW_MAX, (K, H)); dstep = rng.uniform(0, DSTEP_MAX, (K, H))
    px = np.full(K, x); pz = np.full(K, z); pyaw = np.full(K, yaw)
    mind = np.full(K, np.hypot(x - goal[0], z - goal[1]))
    for h in range(H):
        t = np.radians(pyaw); nx = px + dstep[:, h]*np.sin(t); nz = pz - dstep[:, h]*np.cos(t)
        blocked = _swept_collision_vec(px, pz, nx, nz, g)
        px = np.where(blocked, px, nx); pz = np.where(blocked, pz, nz); pyaw = pyaw + dyaw[:, h]
        mind = np.minimum(mind, np.hypot(px - goal[0], pz - goal[1]))   # best approach along the path
    b = int(mind.argmin())
    return dyaw[b, 0], dstep[b, 0]


def _swept_collision_vec(x0, z0, x1, z1, g, n=6):
    """Vectorized swept collision (walls + obstacle AABBs) over K candidates."""
    hit = np.zeros_like(x0, dtype=bool)
    obs = [(o[0], o[2], o[3], o[5]) for o in g.obstacles]     # cx,cz,hx,hz (ear-height pillars span it)
    AR = 0.3
    for i in range(1, n + 1):
        tt = i / n; x = x0 + (x1 - x0)*tt; z = z0 + (z1 - z0)*tt
        h = (x <= -g.Lx/2 + AR) | (x >= g.Lx/2 - AR) | (z <= -g.Lz/2 + AR) | (z >= g.Lz/2 - AR)
        for (cx, cz, hx, hz) in obs:
            h |= (np.abs(x - cx) <= hx + AR) & (np.abs(z - cz) <= hz + AR)
        hit |= h
    return hit


def run(kind, seeds, nobs, episodes_per=10, t_max=40, goal_radius=1.2, base_seed=100):
    succ = []
    for s in seeds:
        srv = EpisodeServer(s, nobs, threads=0)
        g = Geom(srv); goal = goal_pos(srv); srv.close()      # geometry only, no echoes
        rng = np.random.default_rng(base_seed + s)
        for _ in range(episodes_per):
            x, z, yaw = sample_start(g, goal, rng, goal_radius=goal_radius)
            ok = False
            for _ in range(t_max):
                if kind == "greedy":
                    a = oracle_action(x, z, yaw, goal)        # normalized (dyaw/45, dstep/.75-1)
                    dyaw = float(a[0]) * DYAW_MAX; dstep = (float(a[1]) + 1) * 0.5 * DSTEP_MAX
                else:
                    dyaw, dstep = plan_action(x, z, yaw, g, goal, rng)
                x, z, yaw = step_goal(x, z, yaw, dyaw, dstep, g)
                if math.hypot(x - goal[0], z - goal[1]) < goal_radius:
                    ok = True; break
            succ.append(1.0 if ok else 0.0)
    return float(np.mean(succ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--episodes-per", type=int, default=10)
    args = ap.parse_args()
    print("Experiment B GATE -- privileged-geometry greedy vs planning on the doorway task.")
    print("If planning >> greedy at nobs=3, clutter forces detours -> B worth running.\n")
    for nobs in (0, 3):
        seeds = opening_seeds(600, args.eval_rooms, nobs)
        gr = run("greedy", seeds, nobs, args.episodes_per)
        pl = run("planning", seeds, nobs, args.episodes_per)
        gap = pl - gr
        verdict = "NEEDS PLANNING" if gap > 0.10 else "reactive-sufficient"
        print(f"  nobs={nobs}: greedy {gr:.2f}  planning {pl:.2f}  gap {gap:+.2f}  -> {verdict}")
    print("\n(nobs=0 is the RESULTS.md 5 task -- expect ~no gap; nobs=3 is the question.)")


if __name__ == "__main__":
    main()
