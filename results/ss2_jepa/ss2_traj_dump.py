#!/usr/bin/env python3
"""
ss2_traj_dump.py — SoundSpaces 2.0 port of src/jepa_dump.cpp.

Monostatic trajectory dataset: 48-step explore-and-avoid walks through
procedural box rooms, one .npz per episode, feeding the action-conditioned
JEPA world model and the latent-MPC control experiment.

Schema per ss2_jepa_manifest.md §5(b). The C++ original also writes a `raw`
key ([K,2,9600] float32 binaural waveforms); the manifest schema omits it and
no script in jepa/ loads it, so it is omitted here too (it would be 3/4 of the
dataset size). `beacon` is kept for compatibility and is NaN in the monostatic
default, exactly as jepa_dump.cpp writes it.

Usage: ss2_traj_dump.py [--episodes N] [--out DIR] [--steps K]
                        [--seed0 S] [--workers N] [--beacon]
"""
import os, sys, math, shutil, time, argparse
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ss2_jepa_env import (SS2Env, Rng, make_box_room, analytic_label,
                          export_room_scene, free_scene,
                          COCH_F, COCH_T, RANGE_BINS)

DEF_OUT = os.path.expanduser("~/ss2_results/ss2_jepa/traj_data")
SCENE_ROOT = os.path.expanduser("~/ss2_results/ss2_jepa/scenes")


def apply_action(rm, pose, yaw, dyaw, dstep):
    """acoustic_env.h::apply_action — step, clamp 0.3 m off the walls, then turn."""
    dyaw = max(-45.0, min(45.0, dyaw))
    dstep = max(0.0, min(1.5, dstep))
    t = math.radians(yaw)
    pose = pose.copy()
    pose[0] += dstep * math.sin(t)
    pose[2] += dstep * -math.cos(t)
    m = 0.3
    pose[0] = max(rm.origin[0] + m, min(rm.origin[0] + rm.Lx - m, pose[0]))
    pose[2] = max(rm.origin[2] + m, min(rm.origin[2] + rm.Lz - m, pose[2]))
    return pose, yaw + dyaw


def run_episode(env, seed, K, path, scene_dir, use_beacon):
    rm = make_box_room(seed)                      # no obstacles, as jepa_dump.cpp
    export_room_scene(rm, scene_dir)
    env.set_scene_dir(scene_dir)

    pol = Rng(np.uint32(seed) * np.uint32(2654435761) + np.uint32(1))
    m = 0.6
    pose = np.array([rm.origin[0] + m + pol.uf() * (rm.Lx - 2 * m), 1.6,
                     rm.origin[2] + m + pol.uf() * (rm.Lz - 2 * m)])
    yaw = pol.rf(0.0, 360.0)

    bm = 0.8
    beacon = np.array([rm.origin[0] + bm + pol.uf() * (rm.Lx - 2 * bm),
                       rm.origin[1] + 0.5 + pol.uf() * (rm.Ly - 1.0),
                       rm.origin[2] + bm + pol.uf() * (rm.Lz - 2 * bm)])
    if not use_beacon:
        beacon = np.array([np.nan, np.nan, np.nan])

    coch = np.empty((K, 2, COCH_F, COCH_T), np.float16)
    rng = np.empty((K, RANGE_BINS), np.float16)
    actions = np.empty((K, 2), np.float32)
    labels_dist = np.empty((K, 8), np.float32)
    open_az = np.empty(K, np.float32)
    room_vol = np.empty(K, np.float32)
    material = np.empty(K, np.int32)
    poses = np.empty((K, 4), np.float32)

    for t in range(K):
        src = beacon if use_beacon else None
        coch[t], rng[t] = env.observe(pose, yaw, source=src)
        lb = analytic_label(rm, pose, yaw)
        labels_dist[t] = lb["dist_wall"]
        open_az[t] = lb["open_az"]
        room_vol[t] = lb["room_vol"]
        material[t] = lb["material"]
        poses[t] = (pose[0], pose[1], pose[2], yaw)

        # explore-and-avoid policy: wander open space, steer AWAY from a near wall
        dw = lb["dist_wall"]
        ko = int(np.argmax(dw))
        dmin = float(dw.min())
        if dmin < 1.5:
            rel = 45.0 * ko
            if rel > 180.0:
                rel -= 360.0
            dyaw = max(-45.0, min(45.0, rel))
        else:
            dyaw = pol.rf(-25.0, 25.0)
        dstep = pol.rf(0.5, 1.5)
        actions[t] = (dyaw, dstep)
        pose, yaw = apply_action(rm, pose, yaw, dyaw, dstep)

    room_meta = np.array([rm.Lx, rm.Ly, rm.Lz, len(rm.openings)] +
                         [float(x) for x in rm.wall_material], np.float32)
    tmp = path + ".tmp.npz"
    np.savez(tmp, coch=coch, range=rng, actions=actions, labels_dist=labels_dist,
             open_az=open_az, room_vol=room_vol, material=material, pose=poses,
             room_meta=room_meta, beacon=beacon.astype(np.float32))
    os.replace(tmp, path)


def worker(wid, args, eps):
    scene_root = os.path.join(SCENE_ROOT, "traj_w%02d" % wid)
    os.makedirs(scene_root, exist_ok=True)
    env = SS2Env(scene_root, threads=1)
    env.calibrate(free_scene(SCENE_ROOT))
    prev = None
    t0 = time.time()
    for n, ep in enumerate(eps):
        path = os.path.join(args.out, "ep%05d.npz" % ep)
        if os.path.exists(path):
            continue
        sd = os.path.join(scene_root, "r%06d" % ep)
        run_episode(env, args.seed0 + ep, args.steps, path, sd, args.beacon)
        if prev:
            shutil.rmtree(prev, ignore_errors=True)
        prev = sd
        if wid == 0 and n % 5 == 0:
            el = time.time() - t0
            print("  w0 %d/%d  %.1fs/ep  eta %.1f min"
                  % (n + 1, len(eps), el / (n + 1),
                     el / (n + 1) * (len(eps) - n - 1) / 60), flush=True)
    env.close()
    shutil.rmtree(prev, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--beacon", action="store_true",
                    help="displaced-emitter (bistatic) ablation; default monostatic")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(SCENE_ROOT, exist_ok=True)
    free_scene(SCENE_ROOT)
    todo = [e for e in range(args.episodes)
            if not os.path.exists(os.path.join(args.out, "ep%05d.npz" % e))]
    print("traj_dump: %d/%d episodes x %d steps [%s], %d workers -> %s"
          % (len(todo), args.episodes, args.steps,
             "displaced beacon" if args.beacon else "monostatic",
             args.workers, args.out), flush=True)
    if not todo:
        return
    shards = [todo[i::args.workers] for i in range(args.workers)]
    procs = [mp.Process(target=worker, args=(i, args, s))
             for i, s in enumerate(shards) if s]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    done = len([f for f in os.listdir(args.out) if f.endswith(".npz")])
    print("done: %d episodes in %s in %.1f min"
          % (done, args.out, (time.time() - t0) / 60), flush=True)
    bad = [p.exitcode for p in procs if p.exitcode]
    if bad:
        print("WARNING: %d worker(s) exited nonzero: %s" % (len(bad), bad), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
