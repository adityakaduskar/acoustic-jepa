"""Latent-MPC navigation control test (v2).

Different question from the probes: is the learned latent a useful CONTROL
substrate even though its frozen features don't beat raw pooling on geometry?
v1 noted the model learns predictable *dynamics* even where static geometry
decode is weak, so control could work where probes don't.

Task: from a random start pose, plan H actions to APPROACH THE NEAREST WALL.
An action-conditioned two-stream world model (TwoStreamEncoder + action-
conditioned latent dynamics + a nearest-wall-distance readout) is trained on
monostatic trajectory data (jepa_dump). Open-loop CEM-MPC picks the action
sequence minimising the imagined readout over the latent rollout; the plan is
then executed in TRUE pose coordinates (pose dynamics are analytic -- no live
sim needed) and scored by the true reduction in nearest-wall distance.

Controllers compared:
  * learned   -- world model with a trained encoder;
  * random    -- same, encoder FROZEN at init (predictor+readout trained on the
                 random features) -- the control analogue of the random-encoder
                 probe baseline;
  * analytic  -- greedy oracle heading to the true nearest wall (upper bound);
  * rand-act  -- random actions (lower bound).

If learned >> random here, learning helps for control even though it didn't for
geometry probes. If learned ~ random, the negative extends to control too.
"""
from __future__ import annotations
import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn

from model_xview import TwoStreamEncoder
from model import Predictor, vicreg_var_cov

DYAW_MAX, DSTEP_MAX = 45.0, 1.5


# ------------------------------------------------------------- data
def load_traj(data_dir):
    """Return per-episode dicts with coch/range/actions/pose/room."""
    eps = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        with np.load(f) as d:
            eps.append({
                "coch": d["coch"].astype(np.float32),        # [K,2,64,48]
                "range": d["range"].astype(np.float32),      # [K,256]
                "act": d["actions"].astype(np.float32),      # [K,2] (dyaw,dstep); act[t]: obs t->t+1
                "pose": d["pose"].astype(np.float32),        # [K,4] x,y,z,yaw
                "room": d["room_meta"].astype(np.float32),   # [10] Lx,Ly,Lz,...
            })
    return eps


def norm_act(a):
    o = a.copy()
    o[..., 0] /= DYAW_MAX
    o[..., 1] = o[..., 1] / DSTEP_MAX * 2 - 1
    return o


def nwd_from_pose(x, z, room):
    """True nearest side-wall distance (perpendicular)."""
    Lx, Lz = room[0], room[2]
    x0, x1 = -Lx / 2, Lx / 2
    z0, z1 = -Lz / 2, Lz / 2
    return min(x1 - x, x - x0, z1 - z, z - z0)


def step_pose(x, z, yaw, dyaw, dstep, room):
    """Analytic pose dynamics, mirror acoustic_env.h::apply_action."""
    dyaw = max(-DYAW_MAX, min(DYAW_MAX, dyaw)); dstep = max(0.0, min(DSTEP_MAX, dstep))
    t = math.radians(yaw)
    x += dstep * math.sin(t); z += dstep * (-math.cos(t))
    Lx, Lz = room[0], room[2]; m = 0.3
    x = max(-Lx / 2 + m, min(Lx / 2 - m, x))
    z = max(-Lz / 2 + m, min(Lz / 2 - m, z))
    return x, z, yaw + dyaw


# ------------------------------------------------------------- world model
class WorldModel(nn.Module):
    def __init__(self, emb=256, ema=0.99):
        super().__init__()
        self.encoder = TwoStreamEncoder(emb // 2, emb // 2)
        import copy
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(emb, action_dim=2)
        self.readout = nn.Sequential(nn.Linear(emb, 128), nn.GELU(), nn.Linear(128, 1))
        self.ema = ema

    def enc(self, coch, rng):
        zr, zc = self.encoder(coch, rng)
        return torch.cat([zr, zc], 1)

    @torch.no_grad()
    def enc_target(self, coch, rng):
        zr, zc = self.target(coch, rng)
        return torch.cat([zr, zc], 1)

    @torch.no_grad()
    def update_target(self):
        for p, tp in zip(self.encoder.parameters(), self.target.parameters()):
            tp.mul_(self.ema).add_(p, alpha=1 - self.ema)
        for b, tb in zip(self.encoder.buffers(), self.target.buffers()):
            tb.copy_(b)

    def rollout(self, z0, actions_n):
        """z0 [B,emb], actions_n [B,H,2] normalised -> predicted latents [B,H,emb]."""
        zs = []; z = z0
        for k in range(actions_n.shape[1]):
            z = self.predictor(z, actions_n[:, k]); zs.append(z)
        return torch.stack(zs, 1)


def build_windows(eps, H):
    """Yield flattened arrays of H+1-frame windows across episodes."""
    C, R, A, NWD = [], [], [], []
    for e in eps:
        K = e["act"].shape[0]
        for s in range(0, K - H):
            C.append(e["coch"][s:s + H + 1]); R.append(e["range"][s:s + H + 1])
            A.append(e["act"][s:s + H]);
            nwd = [nwd_from_pose(e["pose"][t, 0], e["pose"][t, 2], e["room"]) for t in range(s, s + H + 1)]
            NWD.append(nwd)
    return (np.array(C), np.array(R), np.array(A, np.float32), np.array(NWD, np.float32))


def train_wm(eps, dev, H=6, epochs=25, bs=128, freeze_encoder=False, seed=0):
    torch.manual_seed(seed)
    C, R, A, NWD = build_windows(eps, H)
    n = len(C); idx = np.arange(n)
    wm = WorldModel().to(dev)
    if freeze_encoder:
        for p in wm.encoder.parameters():
            p.requires_grad_(False)
    params = [p for p in wm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    An = norm_act(A)
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(idx)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            coch = torch.tensor(C[b], device=dev); rng = torch.tensor(R[b], device=dev)
            act = torch.tensor(An[b], device=dev); nwd = torch.tensor(NWD[b], device=dev)
            Bn, T = coch.shape[0], coch.shape[1]
            z = wm.enc(coch.reshape(Bn * T, 2, 64, 48), rng.reshape(Bn * T, 256)).reshape(Bn, T, -1)
            with torch.no_grad():
                zt = wm.enc_target(coch.reshape(Bn * T, 2, 64, 48), rng.reshape(Bn * T, 256)).reshape(Bn, T, -1)
            pred = wm.rollout(z[:, 0], act)                       # [B,H,emb]
            dyn = ((pred - zt[:, 1:]) ** 2).mean()
            ro = wm.readout(z.reshape(Bn * T, -1)).reshape(Bn, T)
            ro_loss = ((ro - nwd) ** 2).mean()
            vr, cv = vicreg_var_cov(z[:, 0])
            loss = dyn + ro_loss + 1.0 * vr + 0.04 * cv
            opt.zero_grad(); loss.backward(); opt.step()
            if not freeze_encoder:
                wm.update_target()
    wm.eval()
    return wm


# ------------------------------------------------------------- controllers
@torch.no_grad()
def cem_plan(wm, z0, H, dev, goal="wall", iters=3, N=256, elite=32):
    # wall  -> MINIMISE imagined nearest-wall distance (approach a wall)
    # center-> MAXIMISE it (stay away from all walls); flip the cost sign.
    sign = 1.0 if goal == "wall" else -1.0
    mu = torch.zeros(H, 2, device=dev); sd = torch.tensor([[30.0, 0.6]], device=dev).repeat(H, 1)
    for _ in range(iters):
        eps = torch.randn(N, H, 2, device=dev) * sd + mu
        dyaw = eps[..., 0].clamp(-DYAW_MAX, DYAW_MAX)
        dstep = eps[..., 1].clamp(0, DSTEP_MAX)
        an = torch.stack([dyaw / DYAW_MAX, dstep / DSTEP_MAX * 2 - 1], -1)
        z = z0.expand(N, -1)
        preds = wm.rollout(z, an)                                # [N,H,emb]
        cost = sign * wm.readout(preds.reshape(N * H, -1)).reshape(N, H).sum(1)
        top = cost.argsort()[:elite]
        chosen = torch.stack([dyaw, dstep], -1)[top]
        mu = chosen.mean(0); sd = chosen.std(0) + 1e-3
    return mu.cpu().numpy()                                       # [H,2] raw actions


def analytic_action(x, z, yaw, room, goal="wall"):
    """Greedy oracle. wall: face the true nearest wall. center: face room centre (0,0)."""
    if goal == "wall":
        Lx, Lz = room[0], room[2]
        ds = {"+x": Lx / 2 - x, "-x": x + Lx / 2, "+z": Lz / 2 - z, "-z": z + Lz / 2}
        tgt = min(ds, key=ds.get)
        want = {"+x": 90.0, "-x": -90.0, "+z": 180.0, "-z": 0.0}[tgt]
    else:                                                        # head toward centre (0,0)
        want = math.degrees(math.atan2(0.0 - x, -(0.0 - z)))
    d = ((want - yaw + 180) % 360) - 180
    return max(-DYAW_MAX, min(DYAW_MAX, d)), DSTEP_MAX


def eval_controller(kind, wm, eps, dev, H=6, n_starts=150, seed=1, goal="wall"):
    g = np.random.default_rng(seed)
    scores = []
    for _ in range(n_starts):
        e = eps[g.integers(len(eps))]
        K = e["act"].shape[0]; s = int(g.integers(0, K - H))
        x, z, yaw = float(e["pose"][s, 0]), float(e["pose"][s, 2]), float(e["pose"][s, 3])
        room = e["room"]; start_nwd = nwd_from_pose(x, z, room)
        if kind in ("learned", "random"):
            coch = torch.tensor(e["coch"][s:s + 1], device=dev)
            rng = torch.tensor(e["range"][s:s + 1], device=dev)
            z0 = wm.enc(coch, rng)
            acts = cem_plan(wm, z0, H, dev, goal=goal)
            for k in range(H):
                x, z, yaw = step_pose(x, z, yaw, float(acts[k, 0]), float(acts[k, 1]), room)
        elif kind == "analytic":
            for _ in range(H):
                dy, dstp = analytic_action(x, z, yaw, room, goal=goal)
                x, z, yaw = step_pose(x, z, yaw, dy, dstp, room)
        else:  # rand-act
            for _ in range(H):
                x, z, yaw = step_pose(x, z, yaw, float(g.uniform(-DYAW_MAX, DYAW_MAX)),
                                      float(g.uniform(0, DSTEP_MAX)), room)
        final_nwd = nwd_from_pose(x, z, room)
        # score = progress toward the goal (higher always better)
        scores.append((start_nwd - final_nwd) if goal == "wall" else (final_nwd - start_nwd))
    return float(np.mean(scores)), float(np.std(scores) / math.sqrt(len(scores)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--starts", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=1, help="training seeds for error bars")
    ap.add_argument("--goal", choices=["wall", "center"], default="wall")
    args = ap.parse_args()
    dev = args.device
    G = args.goal

    eps = load_traj(args.data_dir)
    ntr = int(0.8 * len(eps)); tr, te = eps[:ntr], eps[ntr:]
    print(f"trajectory data: {len(eps)} episodes ({ntr} train / {len(te)} test), H={args.H}, "
          f"repeats={args.repeats}, goal={G}")

    EVAL_SEED = 100          # FIXED eval starts across repeats -> variance is from TRAINING seed only
    learned_r, random_r = [], []
    for r in range(args.repeats):
        print(f"[seed {r}] training learned + random-encoder world models...")
        wm = train_wm(tr, dev, H=args.H, freeze_encoder=False, seed=r)
        wm_rand = train_wm(tr, dev, H=args.H, freeze_encoder=True, seed=r)
        lr, _ = eval_controller("learned", wm, te, dev, H=args.H, n_starts=args.starts, seed=EVAL_SEED, goal=G)
        rr, _ = eval_controller("random", wm_rand, te, dev, H=args.H, n_starts=args.starts, seed=EVAL_SEED, goal=G)
        learned_r.append(lr); random_r.append(rr)
        print(f"[seed {r}] learned {lr:+.3f}  random-enc {rr:+.3f}")

    # seed-independent bounds (need a model arg but ignore it)
    dummy = wm
    an, _ = eval_controller("analytic", dummy, te, dev, H=args.H, n_starts=args.starts, seed=EVAL_SEED, goal=G)
    ra, _ = eval_controller("rand-act", dummy, te, dev, H=args.H, n_starts=args.starts, seed=EVAL_SEED, goal=G)

    def ms(v):
        v = np.array(v); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
    lm, ls = ms(learned_r); rm, rs = ms(random_r)
    metric = "nearest-wall-distance REDUCTION (approach wall)" if G == "wall" else \
             "wall-distance INCREASE (reach centre)"
    print(f"\n{metric} over {args.H} steps (mean +/- SD over "
          f"{args.repeats} training seeds; held-out rooms, {args.starts} fixed starts):")
    print(f"  analytic (oracle)    {an:+.3f}            (seed-independent upper bound)")
    print(f"  learned MPC          {lm:+.3f} +/- {ls:.3f}")
    print(f"  random actions       {ra:+.3f}            (seed-independent lower-ish bound)")
    print(f"  random-enc MPC       {rm:+.3f} +/- {rs:.3f}")
    if args.repeats > 1:
        gap = np.array(learned_r) - np.array(random_r)
        print(f"\nlearned - random-enc per-seed: {[round(g,3) for g in gap]}  "
              f"-> mean {gap.mean():+.3f} +/- {gap.std(ddof=1):.3f} "
              f"({'ROBUST: learned>random every seed' if (gap>0).all() else 'NOT robust'})")


if __name__ == "__main__":
    main()
