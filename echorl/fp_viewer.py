"""EchoRL -- first-person interactive viewer for the Steam Audio agent (episode_server.exe).

Rolls a controller out closed-loop in EchoLocEnv (exactly the setup of render_traj.py), and
writes ONE self-contained HTML page with a WebGL first-person camera riding on the agent's
ear position, plus the sensor it is actually navigating by (matched-filter range profile and
binaural L/R cochleagram energy) and a mesh-derived top-down minimap.

The geometry is NOT redrawn: for every room seed the page loads the exact triangle mesh the
simulator committed to Steam Audio, dumped by tools/scene_dump.exe (make_box_room +
build_room_mesh, the same functions episode_server.exe uses). The script cross-checks the dump
against the collision geometry episode_server streamed in its handshake before embedding it.

Steam Audio only -- ECHORL_EPISODE_SERVER is ignored on purpose (the SS2 replica has no
scene_dump counterpart, so its mesh could not be shown without redrawing it).

Run (from AcousticJEPA/):
  python echorl/fp_viewer.py --task avoid --nobs 3 --out results/echorl_fp_viewer.html
  python echorl/fp_viewer.py --task goal  --nobs 3 --ppo-ckpt echorl/ckpt/ppo_goal_nobs3_showdown2_s0_best.pt \
                             --wm echorl/ckpt/mpc_wm_nobs3.pt --out results/echorl_fp_goal.html
Then open the HTML in any browser (no server, no CDN).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _runtime_root():
    """Repo root that actually holds tools/*.exe (walk up: a git worktree under
    .claude/worktrees/ has the sources but not the untracked binaries)."""
    d = os.path.abspath(os.path.join(_HERE, ".."))
    while True:
        if os.path.exists(os.path.join(d, "tools", "scene_dump.exe")):
            return d
        up = os.path.dirname(d)
        if up == d:
            return os.path.abspath(os.path.join(_HERE, ".."))
        d = up


_ROOT = _runtime_root()
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
import env as envmod                                                     # noqa: E402
from env import EchoLocEnv, EpisodeServer, EAR_Y, DYAW_MAX, DSTEP_MAX, RANGE_BINS   # noqa: E402
from ppo import opening_seeds, obs_to_t                                  # noqa: E402
from mpc import cem_plan, oracle_action, raw_to_norm                     # noqa: E402
from mpc_direct import cem_outcome_plan                                  # noqa: E402
from oracle import Geom, plan_action                                     # noqa: E402
from render_traj import make_controllers                                 # noqa: E402

SCENE_DUMP = os.path.join(_ROOT, "tools", "scene_dump.exe")
C_SOUND, SR, RANGE_POOL, RANGE_GUARD = 343.0, 48000, 16, 96
MAT_NAMES = ["plaster", "hard/tile", "carpet", "mirror"]


# ------------------------------------------------------------------ scene mesh
def dump_scene(seed, nobs):
    """Run scene_dump.exe for this seed -> parse the OBJ (real simulator triangles) + txt."""
    subprocess.run([SCENE_DUMP, str(seed), "1", str(nobs)], cwd=_ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    obj = os.path.join(_ROOT, "results", "scenes", f"scene_{seed}.obj")
    txt = os.path.join(_ROOT, "results", "scenes", f"scene_{seed}.txt")
    V, F = [], []
    for line in open(obj):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            F.append([int(p.split("/")[0]) - 1 for p in t[1:4]])
    meta = dict(wall_mat=[0] * 6, obstacles=[])
    for line in open(txt):
        t = line.split()
        if not t:
            continue
        if t[0] == "materials":
            meta["wall_mat"] = [int(x) for x in t[1:7]]
        elif t[0] == "obstacle":
            meta["obstacles"].append(dict(c=[float(x) for x in t[2:5]],
                                          h=[float(x) for x in t[6:9]], mat=int(t[10])))
    return np.array(V, np.float32), np.array(F, np.int32), meta


def check_dump_matches_server(meta, srv, seed):
    """The dump must be the SAME room the server is rendering; abort loudly otherwise."""
    if len(meta["obstacles"]) != len(srv.obstacles):
        raise RuntimeError(f"seed {seed}: scene_dump has {len(meta['obstacles'])} obstacles, "
                           f"server {len(srv.obstacles)}")
    for d, s in zip(meta["obstacles"], srv.obstacles):
        if max(abs(a - b) for a, b in zip(d["c"] + d["h"], s[:6])) > 1e-3:
            raise RuntimeError(f"seed {seed}: obstacle mismatch {d} vs {s}")


def face_material(V, F, meta, Lx, Lz):
    """Per-triangle material id + kind (0 floor, 1 ceiling, 2 wall, 3 obstacle), read off
    the mesh itself (which plane each triangle lies in)."""
    tri = V[F]
    c = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ymax = float(V[:, 1].max())
    kind = np.full(len(F), 3, np.int8)
    mat = np.zeros(len(F), np.int8)
    eps = 1e-3
    for i in range(len(F)):
        ax_ = int(np.argmax(np.abs(n[i])))
        if ax_ == 1 and abs(c[i, 1]) < eps:
            kind[i], mat[i] = 0, meta["wall_mat"][3]
        elif ax_ == 1 and abs(c[i, 1] - ymax) < eps:
            kind[i], mat[i] = 1, meta["wall_mat"][2]
        elif ax_ == 0 and abs(abs(c[i, 0]) - Lx / 2) < eps:
            kind[i], mat[i] = 2, meta["wall_mat"][0 if c[i, 0] > 0 else 1]
        elif ax_ == 2 and abs(abs(c[i, 2]) - Lz / 2) < eps:
            kind[i], mat[i] = 2, meta["wall_mat"][4 if c[i, 2] > 0 else 5]
        else:
            for ob in meta["obstacles"]:
                if all(abs(c[i, k] - ob["c"][k]) <= ob["h"][k] + eps for k in range(3)):
                    mat[i] = ob["mat"]
                    break
    return kind, mat


# --------------------------------------------------------------------- rollout
@torch.no_grad()
def rollout_logged(kind, model, env, obs, info, dev, H, srv):
    """Same closed loop as render_traj.rollout, but keeps every obs + action for the HUD."""
    def rec(obs, info, a=None, r=0.0):
        coch = obs["coch"]                      # (ears, F, T) log energy
        return dict(pose=[round(float(v), 4) for v in info["pose"]],
                    range=[round(float(v), 3) for v in obs["range"]],
                    lr=[round(float(coch[0].sum()), 2), round(float(coch[1].sum()), 2)],
                    coch_t=[round(float(v), 3) for v in coch[:2].mean(axis=1).mean(axis=0)],
                    act=None if a is None else [round(float(v), 3) for v in a],
                    reward=round(float(r), 4),
                    collision=bool(info.get("collision", False)),
                    success=bool(info.get("success", False)),
                    clear=None if info.get("clearance") is None else round(float(info["clearance"]), 3))
    steps = [rec(obs, info)]
    h = torch.zeros(1, 1, model.H, device=dev) if kind == "ppo" else None
    rng_np = np.random.default_rng(0)
    g = Geom(srv) if kind == "oracle_avoid" else None
    done = False
    success = False
    while not done:
        if kind == "ppo":
            f = model.feats(*[t.unsqueeze(0) for t in obs_to_t(obs, dev)])
            mean, _s, _v, h = model.step(f, h)
            a = mean.clamp(-1, 1).cpu().numpy()[0]
        elif kind == "dqn":
            q = model(*[t.unsqueeze(0) for t in obs_to_t(obs, dev)])
            a = int(q.argmax(1).item())
        elif kind == "oracle_avoid":
            x, z, yaw = info["pose"]
            dyaw, dstep = plan_action(x, z, yaw, g, rng_np)
            a = np.array(raw_to_norm(dyaw, dstep), np.float32)
        elif kind == "oracle_goal":
            a = oracle_action(*info["pose"], info["goal"])
        elif kind == "mpc":
            z = model.enc(torch.as_tensor(obs["range"][None], device=dev),
                          torch.as_tensor(obs["coch"][None], device=dev))
            a = cem_plan(model, z, H, dev)
        else:
            wm, gg = model
            z = wm.enc(torch.as_tensor(obs["range"][None], device=dev),
                       torch.as_tensor(obs["coch"][None], device=dev))
            a = cem_outcome_plan(gg, z, H, dev)
        dyaw, dstep = env._decode(a)
        obs, r, term, trunc, info = env.step(a)
        steps.append(rec(obs, info, (dyaw, dstep), r))
        success = bool(info.get("success")) if env.task == "goal" else (not info.get("collision"))
        done = term or trunc
    return steps, success


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["avoid", "goal", "explore"], default="avoid")
    ap.add_argument("--nobs", type=int, default=3)
    ap.add_argument("--rooms", type=int, default=2)
    ap.add_argument("--seeds", default="", help="explicit comma list of room seeds (overrides --rooms)")
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--t-max", type=int, default=40)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--ppo-ckpt", default=""); ap.add_argument("--ppo-label", default="model-free PPO")
    ap.add_argument("--ppo2-ckpt", default=""); ap.add_argument("--ppo2-label", default="PPO (2nd)")
    ap.add_argument("--dqn-ckpt", default=""); ap.add_argument("--dqn-label", default="DQN (discrete)")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--corpus", default=""); ap.add_argument("--wm", default="")
    ap.add_argument("--out", default="results/echorl_fp_viewer.html")
    args = ap.parse_args()
    if not os.path.exists(SCENE_DUMP):
        sys.exit(f"missing {SCENE_DUMP} (build with src/build.bat)")
    if os.environ.get("ECHORL_EPISODE_SERVER"):
        print("NOTE: ECHORL_EPISODE_SERVER is set but ignored -- this viewer is Steam Audio only")
        os.environ.pop("ECHORL_EPISODE_SERVER")
    envmod.EXE = os.path.join(_ROOT, "tools", "episode_server.exe")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = (opening_seeds(600, args.rooms, args.nobs) if args.task == "goal"
                 else list(range(600, 600 + args.rooms)))
    ctrls = make_controllers(args.task, args, dev)
    if args.task == "goal" and not ctrls:
        ctrls["privileged oracle"] = ("oracle_goal", None)
    if not ctrls:
        sys.exit("no controllers (pass --ppo-ckpt / --dqn-ckpt / --wm, or drop --no-oracle)")
    print(f"task={args.task} nobs={args.nobs} rooms={seeds} controllers={list(ctrls)} dev={dev}")

    rooms, episodes = [], []
    for seed in seeds:
        srv = EpisodeServer(seed, args.nobs, threads=0)
        V, F, meta = dump_scene(seed, args.nobs)
        check_dump_matches_server(meta, srv, seed)
        kind_f, mat_f = face_material(V, F, meta, srv.Lx, srv.Lz)
        rooms.append(dict(seed=seed, L=[srv.Lx, srv.Ly, srv.Lz],
                          v=[round(float(x), 4) for x in V.ravel()], f=F.ravel().tolist(),
                          kind=kind_f.tolist(), mat=mat_f.tolist(),
                          openings=[list(map(float, o)) for o in srv.openings],
                          obstacles=[list(map(float, o)) for o in srv.obstacles]))
        for name, (kind, model) in ctrls.items():
            am = "discrete" if kind == "dqn" else "continuous"
            env = EchoLocEnv(seed_pool=[seed], action_mode=am, nobs=args.nobs, sim_threads=2,
                             t_max=args.t_max, task=args.task, base_seed=seed)
            for s in range(args.starts):
                obs, info = env.reset(seed=seed * 1000 + s)
                steps, ok = rollout_logged(kind, model, env, obs, info, dev, args.H, srv)
                episodes.append(dict(room=len(rooms) - 1, ctrl=name, start=s, success=ok,
                                     goal=None if args.task != "goal" else [float(v) for v in info["goal"]],
                                     steps=steps))
                print(f"  room {seed} | {name} | start {s}: {len(steps)-1} steps, {'OK' if ok else 'FAIL'}")
            env.close()
        srv.close()

    data = dict(task=args.task, nobs=args.nobs, ear_y=EAR_Y, agent_r=envmod.AGENT_R,
                dyaw_max=DYAW_MAX, dstep_max=DSTEP_MAX, goal_radius=1.2,
                range_m=[round(C_SOUND * ((b * RANGE_POOL + RANGE_POOL * 0.5) + RANGE_GUARD) / SR / 2.0, 3)
                         for b in range(RANGE_BINS)],
                mat_names=MAT_NAMES, rooms=rooms, episodes=episodes)
    html = open(os.path.join(_HERE, "fp_viewer_template.html"), encoding="utf-8").read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(html)
    print(f"wrote {args.out}  ({len(episodes)} episodes, {len(rooms)} rooms, "
          f"{os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
