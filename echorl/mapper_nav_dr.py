"""Dead-reckoned (UNPRIVILEGED-pose) mapper-nav -- removes §5.8's known-pose caveat.

§5.8's headline (full learned map 0.49) assumed the agent knows its absolute pose,
a rung more privileged than PPO/MPC (which get proprioception only). This closes that
gap: the agent integrates its OWN COMMANDED ACTIONS into an estimated pose (dead
reckoning), builds/fuses the map in that estimated frame, and plans there. Physics and
success use the TRUE pose; echoes are rendered at the TRUE pose but ATTRIBUTED to the
estimated pose -- so any drift corrupts the map exactly like real odometry error.

In this deterministic sim, dead reckoning is EXACT except when a non-terminal collision
blocks a commanded translation (the DR integrator assumes the step succeeded). The
doorway sits near a wall, so bumps -- hence drift -- concentrate right where precision
matters. Two variants bracket how much pose knowledge matters:
  * pure  -- no bump sensing (integrate every commanded step; hardest)
  * bump  -- a robot FEELS a blocked move and does not integrate it (realistic)
vs the §5.8 known-pose number (upper rung).
"""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import mapping                                             # noqa: E402
from mapping import (ear_height_obstacles, true_occupancy, inflate, goal_from_openings,
                     grid_plan_action, _true_swept_hit_scalar)   # noqa: E402
from ppo import opening_seeds                              # noqa: E402


def run_dr(occ_model, goal_model, seeds, dr_mode="pure", nobs=3, episodes_per=8, t_max=40,
           goal_radius=1.2, start_margin=0.8, base_seed=100, thr=0.5):
    import mapper, mapper_fuse
    per_room = {}
    for s in seeds:
        srv = mapping.EpisodeServer(s, nobs, threads=0)
        Lx, Lz = srv.Lx, srv.Lz
        obs_aabb = ear_height_obstacles(srv.obstacles)
        goal_true = goal_from_openings(srv.openings)
        rng = np.random.default_rng(base_seed + s)
        room = []
        for _ in range(episodes_per):
            far = max(2.5, goal_radius + 1.5)
            tx = tz = 0.0; tyaw = float(rng.uniform(0, 360))
            for _ in range(200):
                cx = rng.uniform(-Lx / 2 + start_margin, Lx / 2 - start_margin)
                cz = rng.uniform(-Lz / 2 + start_margin, Lz / 2 - start_margin)
                if all(max(abs(cx - c0) - h0, abs(cz - c1) - h1) > start_margin
                       for (c0, c1, h0, h1) in obs_aabb) and \
                   math.hypot(cx - goal_true[0], cz - goal_true[1]) > far:
                    tx, tz = float(cx), float(cz); break
            ex, ez, eyaw = tx, tz, tyaw            # known START pose; drift accrues from bumps
            hist = []                               # (ex,ez,eyaw, range, coch) in ESTIMATED frame
            ok = False
            for _t in range(t_max):
                rng_o, coch_o = srv.observe(tx, tz, tyaw)          # echo at TRUE pose
                hist.append((ex, ez, eyaw, rng_o.astype(np.float32), coch_o.astype(np.float32)))
                poses = np.array([[h[0], h[1], h[2]] for h in hist], np.float32)
                rseq = np.stack([h[3] for h in hist]); cseq = np.stack([h[4] for h in hist])
                occ_p, _, _ = mapper.predict_grid(occ_model, poses, rseq, cseq)
                occ_infl = inflate(occ_p > thr)
                goal_est = mapper_fuse.fuse(goal_model, poses, rseq, cseq)   # in ESTIMATED frame
                dyaw, dstep = grid_plan_action(ex, ez, eyaw, occ_infl, goal_est, rng)
                # execute on TRUE env
                t = math.radians(tyaw)
                nx = tx + dstep * math.sin(t); nz = tz - dstep * math.cos(t)
                blocked = _true_swept_hit_scalar(tx, tz, nx, nz, Lx, Lz, obs_aabb)
                if not blocked:
                    tx, tz = nx, nz
                tyaw += dyaw
                # dead-reckon the ESTIMATED pose from COMMANDED action
                if dr_mode == "pure" or not blocked:
                    te = math.radians(eyaw)
                    ex = ex + dstep * math.sin(te); ez = ez - dstep * math.cos(te)
                eyaw += dyaw
                if math.hypot(tx - goal_true[0], tz - goal_true[1]) < goal_radius:
                    ok = True; break
            room.append(1.0 if ok else 0.0)
        srv.close()
        per_room[s] = float(np.mean(room))
    return float(np.mean(list(per_room.values()))), per_room


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="model-seed suffixes (1 = default ckpt with no suffix)")
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--episodes-per", type=int, default=8)
    ap.add_argument("--nobs", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["pure", "bump"])
    args = ap.parse_args()
    import mapper, mapper_fuse

    def load(seed):
        sfx = "" if seed == 1 else f"_s{seed}"
        om = mapper.OccMapper().to(mapper.DEV)
        om.load_state_dict(torch.load(f"echorl/ckpt/mapper_nobs3{sfx}.pt", map_location=mapper.DEV)); om.eval()
        gm = mapper_fuse.EgoVecNet().to(mapper.DEV)
        gm.load_state_dict(torch.load(f"echorl/ckpt/egovec_goal_nobs3{sfx}.pt", map_location=mapper.DEV)); gm.eval()
        return om, gm

    eval_seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    print(f"Dead-reckoned (unprivileged-pose) mapper-nav (nobs={args.nobs}, {len(eval_seeds)} rooms "
          f"x {args.episodes_per}, {len(args.seeds)} model seeds)")
    print("vs §5.8 known-pose full = 0.49 +/- 0.08 | greedy 0.54 | MPC 0.24 | PPO 0.08-0.18\n")
    for mode in args.modes:
        vals = []
        for sd in args.seeds:
            om, gm = load(sd)
            macro, _ = run_dr(om, gm, eval_seeds, dr_mode=mode, nobs=args.nobs,
                              episodes_per=args.episodes_per)
            vals.append(macro)
        vals = np.array(vals)
        desc = {"pure": "no bump sensing (hardest)", "bump": "robot feels blocked moves"}[mode]
        print(f"  DR-{mode:>4} : {vals.mean():.3f} +/- {vals.std():.3f}  "
              f"({'/'.join(f'{v:.2f}' for v in vals)})  -- {desc}")


if __name__ == "__main__":
    main()
