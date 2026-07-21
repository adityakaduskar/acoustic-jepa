#!/usr/bin/env python3
"""
ss2_swarm_dump.py — SoundSpaces 2.0 port of src/swarm_dump.cpp.

Cross-view dataset for the Acoustic-JEPA pretext task. Per episode: one
procedural room + N agents; writes each agent's MONOSTATIC self-observation
(context candidates) and every ordered PAIR's cross-view observation (listener
L's reflection-only reception of emitter E's ping), plus the true and MEASURED
relative transforms and analytic per-agent geometry labels.

The .npz schema is byte-for-byte the contract in ss2_jepa_manifest.md §5(a) and
matches what swarm_dump.cpp writes, so jepa/*.py runs unchanged.

Usage: ss2_swarm_dump.py [--episodes N] [--out DIR] [--agents N]
                         [--seed0 S] [--obstacles N] [--workers N]
"""
import os, sys, json, shutil, time, argparse
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ss2_jepa_env import (SS2Env, make_box_room, place_agents, cross_meta,
                          analytic_label, export_room_scene, free_scene,
                          COCH_F, COCH_T, RANGE_BINS)

DEF_OUT = os.path.expanduser("~/ss2_results/ss2_jepa/swarm_data")
SCENE_ROOT = os.path.expanduser("~/ss2_results/ss2_jepa/scenes")


def run_episode(env, seed, N, n_ob, path, scene_dir):
    rm = make_box_room(seed, n_ob)
    export_room_scene(rm, scene_dir)
    env.set_scene_dir(scene_dir)
    pos, yaw = place_agents(rm, seed, N)

    mono_coch = np.empty((N, 2, COCH_F, COCH_T), np.float16)
    mono_range = np.empty((N, RANGE_BINS), np.float16)
    dist_wall = np.empty((N, 8), np.float32)
    room_vol = np.empty(N, np.float32)
    material = np.empty(N, np.int32)
    open_az = np.empty(N, np.float32)
    agent_pose = np.empty((N, 4), np.float32)

    for i in range(N):
        mono_coch[i], mono_range[i] = env.observe(pos[i], yaw[i])
        lb = analytic_label(rm, pos[i], yaw[i])
        dist_wall[i] = lb["dist_wall"]
        room_vol[i] = lb["room_vol"]
        material[i] = lb["material"]
        open_az[i] = lb["open_az"]
        agent_pose[i] = (pos[i][0], pos[i][1], pos[i][2], yaw[i])

    P = N * (N - 1)
    cross_coch = np.empty((P, 2, COCH_F, COCH_T), np.float16)
    cross_range = np.empty((P, RANGE_BINS), np.float16)
    pair = np.empty((P, 2), np.int32)
    T_true = np.empty((P, 6), np.float32)
    T_meas = np.empty((P, 6), np.float32)
    occluded = np.empty(P, np.int32)

    k = 0
    for l in range(N):
        for e in range(N):
            if e == l:
                continue
            # CONTEXT = emitter's monostatic obs (mono_*); TARGET = listener L's
            # reflection-only reception of E's ping. SS2 renders a real direct
            # path, so the `direct` flag is OFF for this render (ss2_jepa_env
            # SENSORS) -- the Steam Audio equivalent renders no direct path at all.
            cross_coch[k], cross_range[k] = env.observe(pos[l], yaw[l], source=pos[e])
            gt = cross_meta(rm, pos[l], yaw[l], pos[e], yaw[e])
            mt = env.measure_tab(pos[l], yaw[l], pos[e])
            pair[k] = (l, e)
            T_true[k] = (gt["dx"], gt["dy"], gt["dz"], gt["range"],
                         gt["rel_yaw"], gt["bearing"])
            T_meas[k] = (mt["dx"], mt["dy"], mt["dz"], mt["range"],
                         mt["bearing"], mt["fb_ok"])
            occluded[k] = gt["occluded"]
            k += 1

    room_meta = np.array([rm.Lx, rm.Ly, rm.Lz, len(rm.openings)] +
                         [float(m) for m in rm.wall_material], np.float32)
    if rm.obstacles:
        obstacles = np.array([list(o["center"]) + list(o["half"]) + [float(o["material"])]
                              for o in rm.obstacles], np.float32)
    else:
        obstacles = np.zeros((1, 7), np.float32)

    tmp = path + ".tmp.npz"
    np.savez(tmp, mono_coch=mono_coch, mono_range=mono_range,
             cross_coch=cross_coch, cross_range=cross_range, pair=pair,
             T_true=T_true, T_meas=T_meas, occluded=occluded,
             agent_pose=agent_pose, dist_wall=dist_wall, room_vol=room_vol,
             material=material, open_az=open_az, room_meta=room_meta,
             obstacles=obstacles)
    os.replace(tmp, path)


def worker(wid, args, eps):
    scene_root = os.path.join(SCENE_ROOT, "swarm_w%02d" % wid)
    os.makedirs(scene_root, exist_ok=True)
    env = SS2Env(scene_root, threads=1)
    env.calibrate(free_scene(SCENE_ROOT))
    prev = None
    t0 = time.time()
    for n, ep in enumerate(eps):
        path = os.path.join(args.out, "ep%05d.npz" % ep)
        if os.path.exists(path):
            continue
        # a fresh directory per episode: never let habitat resolve a new room
        # through a path it has already loaded
        sd = os.path.join(scene_root, "r%06d" % ep)
        run_episode(env, args.seed0 + ep, args.agents, args.obstacles, path, sd)
        if prev:
            shutil.rmtree(prev, ignore_errors=True)
        prev = sd
        if wid == 0 and n % 10 == 0:
            el = time.time() - t0
            print("  w0 %d/%d  %.1fs/ep  eta %.1f min"
                  % (n + 1, len(eps), el / (n + 1),
                     el / (n + 1) * (len(eps) - n - 1) / 60), flush=True)
    env.close()
    shutil.rmtree(prev, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--obstacles", type=int, default=3)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(SCENE_ROOT, exist_ok=True)
    free_scene(SCENE_ROOT)
    todo = [e for e in range(args.episodes)
            if not os.path.exists(os.path.join(args.out, "ep%05d.npz" % e))]
    print("swarm_dump: %d/%d episodes to render, %d agents, %d obstacle(s), "
          "%d workers -> %s" % (len(todo), args.episodes, args.agents,
                                args.obstacles, args.workers, args.out), flush=True)
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
