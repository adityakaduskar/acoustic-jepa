"""Collect a MAPPING corpus for the cluttered doorway task.

Random exploratory trajectories in nobs=3 goal rooms, logging per step the absolute
pose (x,z,yaw) and the echo (range[256], coch[2,64,48]), plus the room geometry once
(Lx,Lz, obstacle AABBs, openings). The occupancy target + goal point are derived from
the geometry at load time (mapping.true_occupancy / goal_from_openings), so this file
just captures raw (pose, echo, geometry).

Unlike mpc.collect_traj (which stored only range/coch/act/dist), we need POSES and
GEOMETRY -- the mapper is a known-pose occupancy predictor. Threaded over rooms
(observe() releases the GIL). Echoes stored float16 to keep the pickle manageable.
"""
from __future__ import annotations
import argparse, os, sys, time, pickle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from env import EchoLocEnv, DYAW_MAX, DSTEP_MAX          # noqa: E402
from ppo import opening_seeds                            # noqa: E402


def _run_episode(env, rng, t_max, forward_bias):
    """One random exploratory rollout in a goal room -> per-step (pose,range,coch)."""
    obs, info = env.reset()
    poses, rngs, cochs = [], [], []
    for _ in range(t_max):
        px, pz, pyaw = info["pose"]
        poses.append((px, pz, pyaw))
        rngs.append(obs["range"].astype(np.float16))
        cochs.append(obs["coch"].astype(np.float16))
        dyaw = rng.uniform(-DYAW_MAX, DYAW_MAX)
        dstep = rng.uniform(0, DSTEP_MAX) * (1.0 if rng.random() < forward_bias else 0.4)
        an = np.array([dyaw / DYAW_MAX, dstep / (0.5 * DSTEP_MAX) - 1.0], np.float32)
        obs, _, term, trunc, info = env.step(an)
        if term or trunc:
            break
    srv = env.srv
    return dict(seed=srv.seed, Lx=srv.Lx, Lz=srv.Lz,
                obstacles=list(srv.obstacles), openings=list(srv.openings),
                poses=np.array(poses, np.float32),
                range=np.stack(rngs), coch=np.stack(cochs))


def collect(seeds, ep_per_room, nobs=3, t_max=40, forward_bias=0.6, workers=12, base_seed=7, ears=2):
    from concurrent.futures import ThreadPoolExecutor
    workers = min(workers, max(1, len(seeds)))
    shards = [seeds[i::workers] for i in range(workers)]
    shards = [s for s in shards if s]

    def worker(wi):
        sh = shards[wi]
        out = []
        for si, seed in enumerate(sh):
            env = EchoLocEnv(seed_pool=[seed], action_mode="continuous", nobs=nobs, sim_threads=2,
                             t_max=t_max, resample_room_every=0, task="goal",
                             base_seed=base_seed * 100000 + wi * 1000 + si, ears=ears)
            rng = np.random.default_rng(base_seed * 100000 + wi * 1000 + si)
            for _ in range(ep_per_room):
                try:
                    out.append(_run_episode(env, rng, t_max, forward_bias))
                except Exception as e:
                    print(f"  [w{wi}] seed {seed} episode failed: {e}", flush=True)
            env.close()
        return out

    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        results = list(pool.map(worker, range(len(shards))))
    return [ep for r in results for ep in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=300)
    ap.add_argument("--ep-per-room", type=int, default=3)
    ap.add_argument("--nobs", type=int, default=3)
    ap.add_argument("--t-max", type=int, default=40)
    ap.add_argument("--start-seed", type=int, default=3000, help="room-seed band (disjoint from eval 600+)")
    ap.add_argument("--out", default="echorl/mapper_corpus_nobs3.pkl")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ears", type=int, default=2, choices=[2, 4],
                    help="4 = the §5.11 non-collinear array (second head yawed 90 deg)")
    args = ap.parse_args()

    seeds = opening_seeds(args.start_seed, args.rooms, args.nobs)
    print(f"Collecting mapping corpus: {len(seeds)} opening-rooms x {args.ep_per_room} eps "
          f"(nobs={args.nobs}, t_max={args.t_max}), seeds {seeds[0]}..{seeds[-1]}")
    t0 = time.time()
    eps = collect(seeds, args.ep_per_room, nobs=args.nobs, t_max=args.t_max, workers=args.workers,
                  ears=args.ears)
    dt = time.time() - t0
    nsteps = sum(len(e["poses"]) for e in eps)
    print(f"  {len(eps)} episodes, {nsteps} steps in {dt:.1f}s ({nsteps/dt:.0f} obs/s)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(eps, f, protocol=4)
    mb = os.path.getsize(args.out) / 1e6
    print(f"  saved -> {args.out}  ({mb:.0f} MB)")


if __name__ == "__main__":
    main()
