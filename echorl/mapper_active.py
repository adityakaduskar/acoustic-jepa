"""EchoRL §5.10 -- ACTIVE SENSING (fix C): does moving to disambiguate actually help?

§5.9 left the doorway half as the binding limit (perfect-doorway lifts the controller
0.587 -> 0.785, vs 0.674 for perfect obstacles), and proved the *consumer* is not the
problem (true bearing with no range navigates at 0.938 vs the 0.948 ceiling). So the
residual is bearing ESTIMATION error, and the standing hypothesis for its floor is the
2-mic front-back ambiguity: one ping cannot tell a doorway ahead from its mirror image
behind, and §5.8 showed the resulting per-ping errors are CORRELATED across nearby poses,
so passive fusion saturates (3.7 -> 2.8 m, then flat).

Motion is the textbook resolution -- what a bat does with head movements, and what a
2-mic array needs geometrically: two bearings taken from poses separated PERPENDICULAR to
the target direction intersect at one point, while a baseline ALONG that direction adds
nothing. Before building an info-gain planner, this file runs the gate that decides
whether such a planner could possibly pay:

  GATE: at matched #pings and matched distance-to-goal, does vantage geometry change the
  estimate quality?  orbit (perpendicular baseline, max parallax)
                     vs radial (parallel baseline, no parallax)
                     vs spin   (rotate in place, zero baseline -- aspect change only)
                     vs random (the corpus policy, the reference)

The motion policies are driven by the TRUE goal direction, so this is an ORACLE-MOTION
UPPER BOUND: if even perfectly-aimed strafing does not beat radial motion, then no learned
info-gain term can, and C is dead with a mechanism. It also tests something uncomfortable
about §5.9 itself -- the reactive controller drives STRAIGHT AT the goal, which is exactly
the minimum-parallax trajectory, so it may be sabotaging its own perception.

Run: python echorl/mapper_active.py gate --eval-rooms 12 --ep-per-room 2
"""
from __future__ import annotations
import argparse, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from env import EpisodeServer, DYAW_MAX, DSTEP_MAX                      # noqa: E402
import mapping                                                          # noqa: E402
from mapping import (goal_from_openings, ear_height_obstacles,          # noqa: E402
                     _true_swept_hit_scalar)
from mapper_probe import ego_bearing                                    # noqa: E402
import mapper                                                           # noqa: E402
import mapper_fuse                                                      # noqa: E402
from mapper_react import ReactiveBearing, egovec_of                     # noqa: E402

DEV = mapper.DEV
POLICIES = ("random", "spin", "radial", "orbit", "approach", "scan")


# ------------------------------------------------------------------ motion policies
def _heading_to(x, z, tx, tz):
    """Yaw (deg) that points the agent at (tx,tz). Forward = (sin yaw, -cos yaw)."""
    return math.degrees(math.atan2(tx - x, -(tz - z)))


def _policy_action(policy, t, x, z, yaw, gx, gz, r0, rng, state):
    """-> (target_yaw, dstep). All policies are driven by the TRUE goal direction, so
    they are oracle-motion upper bounds on what an info-gain planner could achieve."""
    to_goal = _heading_to(x, z, gx, gz)
    d = math.hypot(x - gx, z - gz)
    if policy == "random":
        return yaw + float(rng.uniform(-DYAW_MAX, DYAW_MAX)), \
            float(rng.uniform(0, DSTEP_MAX)) * (1.0 if rng.random() < 0.6 else 0.4)
    if policy == "spin":
        return yaw + DYAW_MAX, 0.0                     # aspect changes, vantage does not
    if policy == "radial":
        # in/out along the bearing: baseline PARALLEL to the target direction (no parallax)
        if d < r0 - 0.6:
            state["out"] = True
        elif d > r0 + 0.6:
            state["out"] = False
        return (to_goal + 180.0 if state.get("out") else to_goal), DSTEP_MAX
    if policy == "orbit":
        # tangential: baseline PERPENDICULAR to the target direction (max parallax).
        # Re-aim at the goal when the orbit radius drifts, so distance stays matched.
        tang = to_goal + 90.0 * state.get("sign", 1.0)
        if d > r0 + 0.8:
            tang = to_goal
        elif d < r0 - 0.8:
            tang = to_goal + 180.0
        return tang, DSTEP_MAX
    if policy == "approach":
        return to_goal, DSTEP_MAX                      # what the §5.9 controller does
    if policy == "scan":
        # approach while sweeping the head: same radial profile as `approach`, but the
        # target is seen from a spread of ASPECTS. Costs ~cos(45 deg) of forward speed.
        state["s"] = -state.get("s", 1.0)
        return to_goal + 45.0 * state["s"], DSTEP_MAX
    raise ValueError(policy)


def collect_policy(seeds, policy, ep_per_room=2, nobs=3, t_max=40, base_seed=99,
                   workers=12, start_far=3.0):
    """Drive each held-out room under `policy`, logging (pose, echo) like mapper_collect.
    Physics use the env's own rules (non-terminal collision: a bump blocks translation)."""
    workers = min(workers, max(1, len(seeds)))
    shards = [s for s in ([seeds[i::workers] for i in range(workers)]) if s]

    def worker(wi):
        out = []
        for si, seed in enumerate(shards[wi]):
            srv = EpisodeServer(seed, nobs, threads=2)
            Lx, Lz = srv.Lx, srv.Lz
            obs_aabb = ear_height_obstacles(srv.obstacles)
            gx, gz = goal_from_openings(srv.openings)
            for ei in range(ep_per_room):
                rng = np.random.default_rng(base_seed * 100000 + seed * 10 + ei)
                # start: matched distance-to-goal across policies, clear of pillars
                x = z = 0.0
                for _ in range(400):
                    cx = rng.uniform(-Lx / 2 + 0.8, Lx / 2 - 0.8)
                    cz = rng.uniform(-Lz / 2 + 0.8, Lz / 2 - 0.8)
                    if not all(max(abs(cx - c0) - h0, abs(cz - c1) - h1) > 0.8
                               for (c0, c1, h0, h1) in obs_aabb):
                        continue
                    if abs(math.hypot(cx - gx, cz - gz) - start_far) < 0.6:
                        x, z = float(cx), float(cz); break
                yaw = float(rng.uniform(0, 360))
                r0 = math.hypot(x - gx, z - gz)
                state = {"sign": 1.0 if rng.random() < 0.5 else -1.0}
                poses, rngs, cochs, dists = [], [], [], []
                for t in range(t_max):
                    r_o, c_o = srv.observe(x, z, yaw)
                    poses.append((x, z, yaw)); rngs.append(r_o.astype(np.float16))
                    cochs.append(c_o.astype(np.float16))
                    dists.append(math.hypot(x - gx, z - gz))
                    tyaw, dstep = _policy_action(policy, t, x, z, yaw, gx, gz, r0, rng, state)
                    # the action space turns at most DYAW_MAX per step
                    dy = (tyaw - yaw + 180.0) % 360.0 - 180.0
                    yaw = yaw + float(np.clip(dy, -DYAW_MAX, DYAW_MAX))
                    th = math.radians(yaw)
                    nx = x + dstep * math.sin(th); nz = z - dstep * math.cos(th)
                    if not _true_swept_hit_scalar(x, z, nx, nz, Lx, Lz, obs_aabb):
                        x, z = nx, nz
                    else:
                        state["sign"] = -state.get("sign", 1.0)      # bounce the orbit
                out.append(dict(seed=seed, Lx=Lx, Lz=Lz, obstacles=list(srv.obstacles),
                                openings=list(srv.openings),
                                poses=np.array(poses, np.float32),
                                range=np.stack(rngs), coch=np.stack(cochs),
                                dist=np.array(dists, np.float32)))
            srv.close()
        return out

    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        res = list(pool.map(worker, range(len(shards))))
    return [e for r in res for e in r]


# ------------------------------------------------------------------ estimate quality
def _ang_err(a, b):
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


@torch.no_grad()
def eval_estimates(model, eps, ks=(2, 5, 10, 20, 40), alpha=0.5):
    """Per #pings: fused doorway error (m, the §5.8 metric), reactive-EMA DIRECTION error
    (deg, the §5.9 quantity that actually steers), and the mean distance-to-goal (the
    control that makes the policies comparable)."""
    fuse_e = {k: [] for k in ks}; dir_e = {k: [] for k in ks}; dmean = {k: [] for k in ks}
    per_ping = []
    for e in eps:
        gx, gz = goal_from_openings(e["openings"])
        rs = e["range"].astype(np.float32); cs = e["coch"].astype(np.float32)
        T = len(e["poses"])
        # the reactive filter is causal, so run it once and snapshot at each k
        rb = ReactiveBearing(alpha=alpha)
        snap = {}
        for t in range(T):
            px, pz, pyaw = e["poses"][t]
            d, beta, w = egovec_of(model, rs[t], cs[t])
            rb.update(px, pz, pyaw, d, beta, w)
            per_ping.append(_ang_err(math.radians(pyaw) + beta,
                                     math.atan2(gx - px, -(gz - pz))))
            if (t + 1) in ks:
                snap[t + 1] = (np.array(rb.dir, copy=True), px, pz)
        for k in ks:
            kk = min(k, T)
            gxh, gzh = mapper_fuse.fuse(model, e["poses"][:kk], rs[:kk], cs[:kk])
            fuse_e[k].append(math.hypot(gxh - gx, gzh - gz))
            v, px, pz = snap.get(kk, snap[max(snap)])
            dir_e[k].append(_ang_err(math.atan2(v[0], -v[1]),
                                     math.atan2(gx - px, -(gz - pz))))
            dmean[k].append(float(e["dist"][:kk].mean()))
    return ({k: float(np.mean(v)) for k, v in fuse_e.items()},
            {k: float(np.mean(v)) for k, v in dir_e.items()},
            {k: float(np.mean(v)) for k, v in dmean.items()},
            float(np.mean(per_ping)))


def cmd_gate(args):
    from ppo import opening_seeds
    m = mapper_fuse.EgoVecNet().to(DEV)
    m.load_state_dict(torch.load(args.goal_ckpt, map_location=DEV)); m.eval()
    seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    ks = (2, 5, 10, 20, args.t_max)
    print(f"§5.10 ACTIVE-SENSING GATE (nobs={args.nobs}, {len(seeds)} held-out rooms x "
          f"{args.ep_per_room}, oracle-driven motion)")
    print("question: at matched pings AND matched distance-to-goal, does vantage geometry")
    print("          change the doorway estimate?  orbit = perpendicular baseline (parallax),")
    print("          radial = parallel baseline (none), spin = zero baseline, random = corpus.")
    print("reference: §5.8 passive fusion saturates 3.7 -> 2.8 m; §5.9 per-ping bearing ~37-43 deg\n")
    rows = {}
    for pol in args.policies:
        t0 = time.time()
        eps = collect_policy(seeds, pol, ep_per_room=args.ep_per_room, nobs=args.nobs,
                             t_max=args.t_max, workers=min(12, len(seeds)))
        fe, de, dm, pp = eval_estimates(m, eps, ks=ks)
        rows[pol] = (fe, de, dm, pp)
        print(f"  [{pol:>6}] {len(eps)} traj, per-ping bearing {pp:5.1f} deg  "
              f"({time.time()-t0:.0f}s)", flush=True)
    print(f"\n{'policy':>7} {'#pings':>7} {'fused doorway err (m)':>22} "
          f"{'reactive DIR err (deg)':>23} {'mean dist (m)':>14}")
    for pol in args.policies:
        fe, de, dm, _ = rows[pol]
        for k in ks:
            print(f"{pol:>7} {k:>7} {fe[k]:>22.2f} {de[k]:>23.1f} {dm[k]:>14.2f}")
    kk = args.t_max
    print(f"\nverdict at {kk} pings:")
    base = rows["radial"] if "radial" in rows else rows[args.policies[0]]
    for pol in args.policies:
        fe, de, _, _ = rows[pol]
        print(f"  {pol:>7}: fused {fe[kk]:.2f} m ({fe[kk]-base[0][kk]:+.2f} vs radial), "
              f"dir {de[kk]:.1f} deg ({de[kk]-base[1][kk]:+.1f})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="does vantage geometry change the estimate at all?")
    g.add_argument("--goal-ckpt", default="echorl/ckpt/egovec_goal_nobs3.pt")
    g.add_argument("--policies", nargs="+", default=list(POLICIES))
    g.add_argument("--eval-rooms", type=int, default=12)
    g.add_argument("--ep-per-room", type=int, default=2)
    g.add_argument("--nobs", type=int, default=3)
    g.add_argument("--t-max", type=int, default=40)
    g.set_defaults(func=cmd_gate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
