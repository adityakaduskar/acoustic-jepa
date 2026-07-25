"""EchoRL mapping experiment -- shared geometry/grid/planner utilities.

Motivation (RESULTS.md §5.7): on the cluttered doorway task (nobs=3 goal) every
model-based fix failed for ONE reason -- the observation is myopic at the source
(a single ping fixes action-consequence at only R2~=0.43), and the honest next
lever is "a temporal model that integrates pings over time to build STATE before
planning." A mapping agent is the most explicit form of that lever: accumulate the
echo sequence into an OCCUPANCY MAP, then run the already-known-good planner on the
reconstructed geometry.

Why it is a clean decomposition: the privileged planner scores 1.00 on TRUE
geometry (gate_b.py). So swapping true->estimated geometry isolates exactly how
much geometry the echoes recover and whether that is enough to navigate. This file
provides the geometry side that both the true-map upper bound (M0) and the
learned-map controller (M2) share:

  * a fixed room-centred occupancy grid (48x48, 0.25 m cells, 12 m span),
  * true-geometry rasterization (walls + interior pillars) from the server handshake,
  * config-space inflation by the agent radius,
  * goal-from-opening (identical to env._goal_pos / gate_b.goal_pos),
  * a GRID-BASED receding-horizon random-shooting planner with the env's exact
    goal-mode rules (non-terminal collision: a bump blocks translation, agent still
    turns), so M0 on the true grid must reproduce gate_b's 1.00 ceiling.

Scope (honest): KNOWN-POSE mapping -- the agent knows its own (x,z,yaw) and infers
only GEOMETRY from echoes. This factors out localization to isolate the echo->map
question; dead-reckoned-pose / SLAM is the follow-up rung.
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from env import EpisodeServer, AGENT_R, DYAW_MAX, DSTEP_MAX, EAR_Y     # noqa: E402

# ------------------------------------------------------------------ grid frame
GRID_G   = 48          # cells per side
GRID_SPAN = 12.0       # metres spanned (room-centred): covers the largest rooms (~11.8 m)
CELL      = GRID_SPAN / GRID_G     # 0.25 m
INFLATE_CELLS = int(math.ceil(AGENT_R / CELL))   # config-space dilation (~2 cells)


def world_to_grid(x, z):
    """(x,z) metres -> integer cell (ix,iz), clipped into the frame."""
    ix = np.clip(np.floor((np.asarray(x) + GRID_SPAN / 2) / CELL).astype(np.int64), 0, GRID_G - 1)
    iz = np.clip(np.floor((np.asarray(z) + GRID_SPAN / 2) / CELL).astype(np.int64), 0, GRID_G - 1)
    return ix, iz


def grid_cell_centres():
    """(G,G) arrays of the world (x,z) at each cell centre."""
    c = (np.arange(GRID_G) + 0.5) * CELL - GRID_SPAN / 2
    X, Z = np.meshgrid(c, c, indexing="ij")
    return X, Z


# ------------------------------------------------------------- true occupancy
def ear_height_obstacles(obstacles):
    """AABBs spanning ear height, as (cx,cz,hx,hz) -- the pillars the sim renders."""
    out = []
    for (cx, cy, cz, hx, hy, hz, _m) in obstacles:
        if cy - hy - 1e-3 <= EAR_Y <= cy + hy + 1e-3:
            out.append((cx, cz, hx, hz))
    return out


def true_occupancy(Lx, Lz, obstacles):
    """(G,G) float {0=free interior, 1=solid}. Solid = outside the room rectangle
    (walls) OR inside an ear-height pillar AABB. Room is centred at the origin."""
    X, Z = grid_cell_centres()
    occ = (np.abs(X) > Lx / 2) | (np.abs(Z) > Lz / 2)          # beyond the walls -> solid
    for (cx, cz, hx, hz) in ear_height_obstacles(obstacles):
        occ |= (np.abs(X - cx) <= hx) & (np.abs(Z - cz) <= hz)
    return occ.astype(np.float32)


def inflate(occ, r_cells=INFLATE_CELLS):
    """Binary dilation by r_cells (Chebyshev) -> config-space grid for a point robot."""
    o = occ > 0.5
    out = o.copy()
    for dx in range(-r_cells, r_cells + 1):
        for dz in range(-r_cells, r_cells + 1):
            out |= np.roll(np.roll(o, dx, axis=0), dz, axis=1)
    return out.astype(np.float32)


def goal_from_openings(openings):
    """Point 1.5 m in front of the largest opening -- identical to env._goal_pos.
    Wall ids: 0:+x 1:-x 4:+z 5:-z. Fallback: room centre."""
    if openings:
        o = max(openings, key=lambda r: r[4] * r[5])
        wall, cx, cz, off = int(o[0]), float(o[1]), float(o[3]), 1.5
        if wall == 0: return cx - off, cz
        if wall == 1: return cx + off, cz
        if wall == 4: return cx, cz - off
        if wall == 5: return cx, cz + off
    return 0.0, 0.0


# ------------------------------------------------------- grid receding-horizon planner
def _grid_swept_hit(x0, z0, x1, z1, occ_infl, n=6):
    """Vectorized: does the segment (x0,z0)->(x1,z1) enter an inflated-occupied cell?
    x0.. are (K,) arrays; occ_infl is (G,G). Out-of-frame counts as solid."""
    hit = np.zeros_like(x0, dtype=bool)
    for i in range(1, n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t
        z = z0 + (z1 - z0) * t
        oob = (np.abs(x) > GRID_SPAN / 2) | (np.abs(z) > GRID_SPAN / 2)
        ix, iz = world_to_grid(x, z)
        hit |= oob | (occ_infl[ix, iz] > 0.5)
    return hit


def grid_plan_action(x, z, yaw, occ_infl, goal, rng, K=256, H=8):
    """Receding-horizon random shooting on the grid map (non-terminal collision:
    a blocked step keeps the pose but still applies the turn). Minimise best approach
    distance to `goal` along the horizon. Returns the first (dyaw, dstep). Mirrors
    gate_b.plan_action but reads collisions from an occupancy grid, not true AABBs."""
    dyaw = rng.uniform(-DYAW_MAX, DYAW_MAX, (K, H))
    dstep = rng.uniform(0, DSTEP_MAX, (K, H))
    px = np.full(K, x, np.float64); pz = np.full(K, z, np.float64); pyaw = np.full(K, yaw, np.float64)
    mind = np.full(K, math.hypot(x - goal[0], z - goal[1]))
    for h in range(H):
        t = np.radians(pyaw)
        nx = px + dstep[:, h] * np.sin(t); nz = pz - dstep[:, h] * np.cos(t)
        blocked = _grid_swept_hit(px, pz, nx, nz, occ_infl)
        px = np.where(blocked, px, nx); pz = np.where(blocked, pz, nz)
        pyaw = pyaw + dyaw[:, h]
        mind = np.minimum(mind, np.hypot(px - goal[0], pz - goal[1]))
    b = int(mind.argmin())
    return float(dyaw[b, 0]), float(dstep[b, 0])


# ------------------------------------------------------ true-env goal-mode step
def _true_swept_hit_scalar(x0, z0, x1, z1, Lx, Lz, obs, n=8):
    for i in range(1, n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t; z = z0 + (z1 - z0) * t
        if abs(x) > Lx / 2 - AGENT_R or abs(z) > Lz / 2 - AGENT_R:
            return True
        for (cx, cz, hx, hz) in obs:
            if abs(x - cx) <= hx + AGENT_R and abs(z - cz) <= hz + AGENT_R:
                return True
    return False


def run_map_nav(occ_provider, seeds, nobs=3, episodes_per=10, t_max=40,
                goal_radius=1.2, start_margin=0.8, base_seed=100, verbose=False):
    """Closed-loop goal-task eval driven by a MAP-based planner. `occ_provider(srv)`
    returns a callable step_map(x,z,yaw,history) -> (occ_infl, goal_est): the map and
    goal the planner uses THIS step (M0 = constant true map; M2 = mapper re-run on the
    accumulated echoes). Ground-truth collision/success use the TRUE geometry (env
    rules), exactly like gate_b: the map only informs the PLAN, never the physics."""
    from ppo import opening_seeds  # local import to avoid cycle at module load
    succ, per_room = [], {}
    for s in seeds:
        srv = EpisodeServer(s, nobs, threads=0)
        Lx, Lz = srv.Lx, srv.Lz
        obs_aabb = ear_height_obstacles(srv.obstacles)
        goal_true = goal_from_openings(srv.openings)
        step_map = occ_provider(srv)
        rng = np.random.default_rng(base_seed + s)
        room_succ = []
        for _ in range(episodes_per):
            # sample a start far from the goal, clear of pillars (mirrors gate_b.sample_start)
            far = max(2.5, goal_radius + 1.5)
            x = z = 0.0; yaw = float(rng.uniform(0, 360))
            for _ in range(200):
                cx = rng.uniform(-Lx / 2 + start_margin, Lx / 2 - start_margin)
                cz = rng.uniform(-Lz / 2 + start_margin, Lz / 2 - start_margin)
                clear = min(abs(cx) and Lx / 2 - abs(cx), Lz / 2 - abs(cz))
                pil_ok = all(max(abs(cx - c0) - h0, abs(cz - c1) - h1) > start_margin
                             for (c0, c1, h0, h1) in obs_aabb)
                if pil_ok and math.hypot(cx - goal_true[0], cz - goal_true[1]) > far:
                    x, z = float(cx), float(cz); break
            history = []
            ok = False
            for _t in range(t_max):
                occ_infl, goal_est = step_map(x, z, yaw, history)
                dyaw, dstep = grid_plan_action(x, z, yaw, occ_infl, goal_est, rng)
                t = math.radians(yaw)
                nx = x + dstep * math.sin(t); nz = z - dstep * math.cos(t)
                if not _true_swept_hit_scalar(x, z, nx, nz, Lx, Lz, obs_aabb):
                    x, z = nx, nz               # non-terminal collision: bump blocks translation
                yaw = yaw + dyaw
                if math.hypot(x - goal_true[0], z - goal_true[1]) < goal_radius:
                    ok = True; break
            room_succ.append(1.0 if ok else 0.0)
        srv.close()
        per_room[s] = float(np.mean(room_succ))
        succ.extend(room_succ)
        if verbose:
            print(f"    seed {s}: {per_room[s]:.2f}")
    macro = float(np.mean(list(per_room.values())))
    return macro, per_room


# ------------------------------------------------------------------- M0 sanity
def _m0_true_provider(srv):
    """occ_provider that always returns the TRUE inflated grid + true goal (upper bound)."""
    occ = true_occupancy(srv.Lx, srv.Lz, srv.obstacles)
    occ_infl = inflate(occ)
    goal = goal_from_openings(srv.openings)
    return lambda x, z, yaw, hist: (occ_infl, goal)


def main():
    ap = argparse.ArgumentParser(description="M0: grid-planner on the TRUE map (upper bound).")
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--episodes-per", type=int, default=10)
    ap.add_argument("--nobs", type=int, default=3)
    args = ap.parse_args()
    from ppo import opening_seeds
    seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    macro, per = run_map_nav(_m0_true_provider, seeds, nobs=args.nobs,
                             episodes_per=args.episodes_per)
    print(f"M0 grid-planner on TRUE map (nobs={args.nobs}, {len(seeds)} held-out doorway rooms "
          f"x {args.episodes_per}):")
    print(f"  success = {macro:.3f}   (target: reproduce gate_b planning-oracle 1.00)")
    lo = min(per.values()); worst = [s for s, v in per.items() if v == lo]
    print(f"  per-room range [{lo:.2f}, {max(per.values()):.2f}]; worst seeds {worst[:5]}")


if __name__ == "__main__":
    main()
