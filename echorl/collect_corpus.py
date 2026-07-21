"""EchoRL offline corpus + norm-stats collector (plan §6.2).

Scatters random VALID poses across many procedural rooms and records the echo
observation at each -- decoupled from (short) random-episode length, so the norm
statistics are representative. Saves:
  * echorl/norm_stats.npz  -- per-channel mean/std for range & coch (RL input norm);
  * echorl/corpus.npz       -- raw (range, coch, pose, room, clearance) for later
                               encoder pretraining / BC / analysis (float16 to stay small).

Usage: python collect_corpus.py [--rooms 24] [--per 84] [--nobs 3] [--no-raw]
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import (EpisodeServer, RANGE_BINS, COCH_SHAPE, AGENT_R,          # noqa: E402
                 _obstacle_clearance, _wall_clearance)

HERE = os.path.dirname(os.path.abspath(__file__))


def sample_pose(srv, rng, margin=0.5):
    for _ in range(200):
        x = rng.uniform(-srv.Lx / 2 + margin, srv.Lx / 2 - margin)
        z = rng.uniform(-srv.Lz / 2 + margin, srv.Lz / 2 - margin)
        clr = min(_wall_clearance(x, z, srv.Lx, srv.Lz), _obstacle_clearance(x, z, srv.obstacles))
        if clr > AGENT_R:                       # valid, non-colliding
            return float(x), float(z), float(rng.uniform(0, 360)), float(clr)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=24)
    ap.add_argument("--per", type=int, default=84, help="poses per room")
    ap.add_argument("--nobs", type=int, default=3)
    ap.add_argument("--no-raw", action="store_true", help="skip saving the raw corpus")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    R = np.zeros((RANGE_BINS,), np.float64)          # running mean/M2 for range
    R2 = np.zeros((RANGE_BINS,), np.float64)
    C = np.zeros(COCH_SHAPE, np.float64)
    C2 = np.zeros(COCH_SHAPE, np.float64)
    n = 0
    raw_r, raw_c, raw_pose, raw_room, raw_clr = [], [], [], [], []

    t0 = time.time()
    for k in range(args.rooms):
        seed = 3000 + k                              # disjoint from train/eval seed bands
        srv = EpisodeServer(seed, args.nobs, threads=0)
        for _ in range(args.per):
            sp = sample_pose(srv, rng)
            if sp is None:
                continue
            x, z, yaw, clr = sp
            rng_o, coch_o = srv.observe(x, z, yaw)
            n += 1
            dr = rng_o - R;  R += dr / n;  R2 += dr * (rng_o - R)
            dc = coch_o - C; C += dc / n;  C2 += dc * (coch_o - C)
            if not args.no_raw:
                raw_r.append(rng_o.astype(np.float16)); raw_c.append(coch_o.astype(np.float16))
                raw_pose.append([x, 1.6, z, yaw]); raw_room.append([srv.Lx, srv.Ly, srv.Lz])
                raw_clr.append(clr)
        srv.close()
        print(f"  room {k+1}/{args.rooms} (seed {seed}): {n} obs so far", end="\r")

    dt = time.time() - t0
    r_std = np.sqrt(R2 / max(1, n - 1)).astype(np.float32)
    c_std = np.sqrt(C2 / max(1, n - 1)).astype(np.float32)
    np.savez(os.path.join(HERE, "norm_stats.npz"),
             range_mean=R.astype(np.float32), range_std=r_std,
             coch_mean=C.astype(np.float32), coch_std=c_std, n_obs=n)
    print(f"\n{n} observations across {args.rooms} rooms in {dt:.1f}s ({n/dt:.1f} obs/s)")
    print(f"  range: mean[{R.min():.2f},{R.max():.2f}] std[{r_std.min():.2f},{r_std.max():.2f}]")
    print(f"  coch : mean[{C.min():.3f},{C.max():.3f}] std[{c_std.min():.3f},{c_std.max():.3f}]")
    print(f"  saved norm_stats.npz")

    if not args.no_raw:
        out = os.path.join(HERE, "corpus.npz")
        np.savez_compressed(out,
                            range=np.stack(raw_r), coch=np.stack(raw_c),
                            pose=np.array(raw_pose, np.float32), room=np.array(raw_room, np.float32),
                            clearance=np.array(raw_clr, np.float32))
        mb = os.path.getsize(out) / 1e6
        print(f"  saved corpus.npz ({len(raw_r)} obs, {mb:.1f} MB)")

    if (r_std < 1e-6).any():
        print("WARN: some range channels have ~zero variance");
    print("DONE")


if __name__ == "__main__":
    main()
