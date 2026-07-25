"""Explicit per-ping ego-vector estimation + known-pose FUSION (the mapping lever).

The probe (mapper_probe.py) showed that in cluttered rooms a SINGLE ping decodes
geometry only weakly (dist-to-goal R2~0.25, ego-bearings ~40 deg MAE). The whole
point of mapping is that KNOWN POSES let you fuse many such weak pings into a strong
estimate. A global scatter-CNN did not learn this fusion; here we do it explicitly
and analytically, which is both stronger and interpretable:

  per ping t:  (range,coch) -> ego (dist, bearing) to the target, with a learned
               heteroscedastic CONFIDENCE (log-variance);
  transform:   target_world_t = pose_t + dist * (sin(yaw+bearing), -cos(yaw+bearing));
  fuse:        goal_est = confidence-weighted mean of target_world_t over the pings.

Heteroscedastic loss makes the net down-weight pings it cannot read (far / occluded),
so the weighted mean is a near-optimal multilateration of the doorway at known poses.
Trained for the GOAL (doorway) target; the same head structure is reused for the
nearest-pillar target to seed obstacle avoidance.
"""
from __future__ import annotations
import argparse, os, sys, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from models import Normalizer
from model_xview import TwoStreamEncoder
from mapping import goal_from_openings, ear_height_obstacles
import mapper
from mapper_probe import ego_bearing, nearest_pillar

DEV = mapper.DEV


# ------------------------------------------------------------------- model
def ears_of(state_dict):
    """Recover the ear count from a checkpoint: coch_proj.0.weight is [emb, ears*emb/2]."""
    w = state_dict["enc.coch_proj.0.weight"]
    return int(w.shape[1] // (w.shape[0] // 2))


class EgoVecNet(nn.Module):
    """(range,coch) -> per-ping ego (dist, sin b, cos b) to a target + log-variances.
    Confidence = exp(-logvar) weights the fusion. `ears`=4 is the §5.11 non-collinear array."""
    def __init__(self, ears=2):
        super().__init__()
        self.norm = Normalizer(); self.enc = TwoStreamEncoder(128, 128, ears=ears)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 5))
        # outputs: dist, bsin, bcos, logvar_dist, logvar_bear

    def forward(self, r, c):
        rn, cn = self.norm(r, c); zr, zc = self.enc(cn, rn)
        o = self.head(torch.cat([zr, zc], 1))
        dist = F.softplus(o[:, 0])
        bnorm = torch.linalg.norm(o[:, 1:3], dim=1, keepdim=True).clamp(min=1e-6)
        bsc = o[:, 1:3] / bnorm
        return dist, bsc[:, 0], bsc[:, 1], o[:, 3], o[:, 4]


# ------------------------------------------------------------------- data
def build_samples(eps, target="goal", ear_idx=None):
    """ear_idx selects cochleagram channels: None = all, (0,1) = the original head,
    (2,3) = the +90 deg rotated head (§5.11). Training a net on (2,3) alone tests whether
    the second pair carries usable bearing at all -- i.e. whether a null 4-ear result means
    'no extra information' or 'the net failed to exploit it'."""
    R, C, POSE, D, BS, BC = [], [], [], [], [], []
    for e in eps:
        gx, gz = goal_from_openings(e["openings"])
        obs = ear_height_obstacles(e["obstacles"])
        for t in range(len(e["poses"])):
            px, pz, pyaw = e["poses"][t]
            if target == "goal":
                tx, tz = gx, gz
            else:
                _, pc = nearest_pillar(px, pz, obs); tx, tz = pc if pc else (0.0, 0.0)
            ct = e["coch"][t] if ear_idx is None else e["coch"][t][list(ear_idx)]
            R.append(e["range"][t]); C.append(ct); POSE.append([px, pz, pyaw])
            b = ego_bearing(px, pz, pyaw, tx, tz)
            D.append(math.hypot(px - tx, pz - tz)); BS.append(math.sin(b)); BC.append(math.cos(b))
    return (np.array(R, np.float32), np.stack(C).astype(np.float32), np.array(POSE, np.float32),
            np.array(D, np.float32), np.array(BS, np.float32), np.array(BC, np.float32))


def fuse(model, poses, rseq, cseq, min_conf=0.0):
    """Confidence-weighted fusion of per-ping ego-vectors -> world target (x,z)."""
    with torch.no_grad():
        r = torch.as_tensor(rseq, dtype=torch.float32, device=DEV)
        c = torch.as_tensor(cseq, dtype=torch.float32, device=DEV)
        dist, bs, bc, lvd, lvb = model(r, c)
    dist = dist.cpu().numpy(); bs = bs.cpu().numpy(); bc = bc.cpu().numpy()
    w = np.exp(-(lvd.cpu().numpy() + lvb.cpu().numpy()))          # combined confidence
    px, pz, pyaw = poses[:, 0], poses[:, 1], np.radians(poses[:, 2])
    beta = np.arctan2(bs, bc)
    ang = pyaw + beta
    tx = px + dist * np.sin(ang); tz = pz - dist * np.cos(ang)
    W = w.sum()
    if W < 1e-8:
        return float(tx.mean()), float(tz.mean())
    return float((w * tx).sum() / W), float((w * tz).sum() / W)


# ------------------------------------------------------------------- train
def train(args):
    eps = mapper.load_corpus(args.corpus)
    rng = np.random.default_rng(0); perm = rng.permutation(len(eps))
    nval = max(8, int(0.1 * len(eps)))
    tr = [eps[i] for i in perm[nval:]]; va = [eps[i] for i in perm[:nval]]
    ear_idx = None if not getattr(args, "ear_idx", "") else tuple(int(v) for v in args.ear_idx.split(","))
    Rtr, Ctr, Ptr, Dtr, BStr, BCtr = build_samples(tr, args.target, ear_idx)
    ears = Ctr.shape[1]
    print(f"[{args.target}] train steps {len(Rtr)}  val eps {len(va)}  ears={ears}")
    m = EgoVecNet(ears=ears).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), args.lr, weight_decay=1e-4)
    dmu, dsd = float(Dtr.mean()), float(Dtr.std() + 1e-6)
    N = len(Rtr); bs = 512
    best = 1e9
    for ep in range(args.epochs):
        m.train(); idx = np.random.permutation(N); t0 = time.time()
        for i in range(0, N, bs):
            j = idx[i:i + bs]
            r = torch.tensor(Rtr[j], device=DEV); c = torch.tensor(Ctr[j], device=DEV)
            td = torch.tensor(Dtr[j], device=DEV); ts = torch.tensor(BStr[j], device=DEV)
            tc = torch.tensor(BCtr[j], device=DEV)
            dist, bsx, bcx, lvd, lvb = m(r, c)
            # heteroscedastic Gaussian NLL on dist (normalized) and on bearing vector
            ed = ((dist - td) / dsd)
            lvd = lvd.clamp(-6, 6); lvb = lvb.clamp(-6, 6)
            loss_d = (0.5 * torch.exp(-lvd) * ed ** 2 + 0.5 * lvd).mean()
            eb = ((bsx - ts) ** 2 + (bcx - tc) ** 2)
            loss_b = (0.5 * torch.exp(-lvb) * eb + 0.5 * lvb).mean()
            loss = loss_d + loss_b
            opt.zero_grad(); loss.backward(); opt.step()
        # val: fused localization error vs #pings
        gerr = eval_fusion(m, va, args.target, ks=(args.t_max,), ear_idx=ear_idx)
        e_full = gerr[args.t_max]
        tag = ""
        if e_full < best:
            best = e_full; tag = " *"
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(m.state_dict(), args.out)
        print(f"ep {ep:2d}  fused {args.target}-err@full {e_full:.2f} m  ({time.time()-t0:.0f}s){tag}",
              flush=True)
    print(f"best fused {args.target}-err {best:.2f} m -> {args.out}")


@torch.no_grad()
def eval_fusion(model, eps, target="goal", ks=(2, 5, 10, 20, 40), ear_idx=None):
    model.eval()
    errs = {k: [] for k in ks}
    sel = (lambda c: c) if ear_idx is None else (lambda c: c[:, list(ear_idx)])
    for e in eps:
        gx, gz = goal_from_openings(e["openings"])
        obs = ear_height_obstacles(e["obstacles"])
        T = len(e["poses"])
        for k in ks:
            kk = min(k, T)
            gxh, gzh = fuse(model, e["poses"][:kk], e["range"][:kk].astype(np.float32),
                            sel(e["coch"][:kk].astype(np.float32)))
            if target == "goal":
                errs[k].append(math.hypot(gxh - gx, gzh - gz))
            else:
                dmin = min(math.hypot(gxh - c[0], gzh - c[1]) for c in obs) if obs else 0.0
                errs[k].append(dmin)
    return {k: float(np.mean(v)) for k, v in errs.items()}


def cmd_evalq(args):
    from ppo import opening_seeds
    import mapper_collect
    sd = torch.load(args.ckpt, map_location=DEV); ears = ears_of(sd)
    m = EgoVecNet(ears=ears).to(DEV); m.load_state_dict(sd); m.eval()
    seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    print(f"Collecting held-out eval trajectories ({len(seeds)} rooms, ears={ears})...")
    eps = mapper_collect.collect(seeds, args.ep_per_room, nobs=args.nobs, t_max=args.t_max,
                                 workers=min(12, len(seeds)), base_seed=99, ears=ears)
    ks = (1, 2, 5, 10, 20, args.t_max)
    gerr = eval_fusion(m, eps, args.target, ks=ks)
    print(f"\nFused {args.target}-localization error vs #pings (held-out nobs={args.nobs}, "
          f"{len(eps)} trajectories):")
    print(f"  {'#pings':>7} {'err(m)':>9}")
    for k in ks:
        print(f"  {k:>7} {gerr[k]:>9.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--corpus", default="echorl/mapper_corpus_nobs3.pkl")
    p.add_argument("--target", choices=["goal", "pillar"], default="goal")
    p.add_argument("--out", default="echorl/ckpt/egovec_goal_nobs3.pt")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--t-max", type=int, default=40)
    p.add_argument("--ear-idx", default="", help="e.g. '2,3' to train on the rotated pair only")
    p.set_defaults(func=train)
    q = sub.add_parser("evalq")
    q.add_argument("--ckpt", default="echorl/ckpt/egovec_goal_nobs3.pt")
    q.add_argument("--target", choices=["goal", "pillar"], default="goal")
    q.add_argument("--eval-rooms", type=int, default=12)
    q.add_argument("--ep-per-room", type=int, default=2)
    q.add_argument("--nobs", type=int, default=3)
    q.add_argument("--t-max", type=int, default=40)
    q.set_defaults(func=cmd_evalq)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
