"""Diagnostic: is the per-PING geometry signal present in the mapping corpus?

Before trusting/blaming the scatter-map architecture, check what a single observation
(range,coch) linearly-ish decodes, with a fresh encoder trained on my corpus:
  * dist-to-goal            (R2) -- §5 claims R2~0.98 from the WM encoder; reproduce?
  * ego-bearing-to-goal     (mean angular error, deg) -- the binaural doorway direction
  * dist-to-nearest-pillar  (R2) -- the obstacle signal the occupancy map needs
  * ego-bearing-to-pillar   (deg)
If these decode well, the signal is there and the global-CNN mapper is the wrong
inductive bias (it cannot multilaterate distance-at-known-poses); if they don't, the
bottleneck is perception/encoder, not the map head.
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

DEV = mapper.DEV


def ego_bearing(px, pz, pyaw_deg, tx, tz):
    """Ego bearing (rad) of target (tx,tz) relative to agent heading. World angle of a
    vector (vx,vz) is atan2(vx,-vz) so forward=(sin yaw,-cos yaw) maps to yaw."""
    world = math.atan2(tx - px, -(tz - pz))
    return math.atan2(math.sin(world - math.radians(pyaw_deg)),
                      math.cos(world - math.radians(pyaw_deg)))


def nearest_pillar(px, pz, obs):
    best_d, best = 1e9, None
    for (cx, cz, hx, hz) in obs:
        dx = abs(px - cx) - hx; dz = abs(pz - cz) - hz
        d = math.hypot(max(dx, 0), max(dz, 0)) if (dx > 0 or dz > 0) else max(dx, dz)
        if d < best_d:
            best_d, best = d, (cx, cz)
    return best_d, best


def build_samples(eps):
    R, C, Y, T = [], [], [], []
    for e in eps:
        gx, gz = goal_from_openings(e["openings"])
        obs = ear_height_obstacles(e["obstacles"])
        for t in range(len(e["poses"])):
            px, pz, pyaw = e["poses"][t]
            R.append(e["range"][t]); C.append(e["coch"][t])
            Y.append([math.sin(math.radians(pyaw)), math.cos(math.radians(pyaw))])
            dg = math.hypot(px - gx, pz - gz)
            bg = ego_bearing(px, pz, pyaw, gx, gz)
            dp, pc = nearest_pillar(px, pz, obs)
            bp = ego_bearing(px, pz, pyaw, pc[0], pc[1]) if pc else 0.0
            T.append([dg, math.sin(bg), math.cos(bg), dp, math.sin(bp), math.cos(bp)])
    return (np.array(R, np.float32), np.stack(C).astype(np.float32),
            np.array(Y, np.float32), np.array(T, np.float32))


class Probe(nn.Module):
    def __init__(self, use_yaw=True):
        super().__init__()
        self.norm = Normalizer(); self.enc = TwoStreamEncoder(128, 128)
        self.use_yaw = use_yaw
        self.head = nn.Sequential(nn.Linear(256 + (2 if use_yaw else 0), 256), nn.GELU(),
                                  nn.Linear(256, 6))

    def forward(self, r, c, y):
        rn, cn = self.norm(r, c); zr, zc = self.enc(cn, rn)
        z = torch.cat([zr, zc], 1)
        if self.use_yaw:
            z = torch.cat([z, y], 1)
        return self.head(z)


def r2(pred, tgt):
    ss_res = ((pred - tgt) ** 2).sum(); ss_tot = ((tgt - tgt.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="echorl/mapper_corpus_nobs3.pkl")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    eps = mapper.load_corpus(args.corpus)
    rng = np.random.default_rng(0); perm = rng.permutation(len(eps))
    nval = max(8, int(0.1 * len(eps)))
    tr = [eps[i] for i in perm[nval:]]; va = [eps[i] for i in perm[:nval]]
    Rtr, Ctr, Ytr, Ttr = build_samples(tr)
    Rva, Cva, Yva, Tva = build_samples(va)
    print(f"train steps {len(Rtr)}  val steps {len(Rva)}")
    m = Probe(use_yaw=True).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=1e-4)
    # normalize targets for balanced loss
    tmu = torch.tensor(Ttr.mean(0), device=DEV); tsd = torch.tensor(Ttr.std(0) + 1e-6, device=DEV)
    def batches(R, C, Y, T, bs=512, shuf=True):
        idx = np.random.permutation(len(R)) if shuf else np.arange(len(R))
        for i in range(0, len(R), bs):
            j = idx[i:i + bs]
            yield (torch.tensor(R[j], device=DEV), torch.tensor(C[j], device=DEV),
                   torch.tensor(Y[j], device=DEV), torch.tensor(T[j], device=DEV))
    for ep in range(args.epochs):
        m.train()
        for r, c, y, t in batches(Rtr, Ctr, Ytr, Ttr):
            p = m(r, c, y); loss = F.mse_loss(p, (t - tmu) / tsd)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        preds = []
        for r, c, y, t in batches(Rva, Cva, Yva, Tva, shuf=False):
            preds.append((m(r, c, y) * tsd + tmu).cpu())
        P = torch.cat(preds); Tv = torch.tensor(Tva)
    dg_r2 = r2(P[:, 0], Tv[:, 0]).item()
    dp_r2 = r2(P[:, 3], Tv[:, 3]).item()
    def bearing_err(ps, pc, ts, tc):
        pa = torch.atan2(ps, pc); ta = torch.atan2(ts, tc)
        return math.degrees(float(torch.atan2((pa - ta).sin().abs().mean(), (pa - ta).cos().mean().abs())))
    bg_err = math.degrees(float(torch.atan2(
        ((torch.atan2(P[:, 1], P[:, 2]) - torch.atan2(Tv[:, 1], Tv[:, 2])).sin()).abs().mean(),
        ((torch.atan2(P[:, 1], P[:, 2]) - torch.atan2(Tv[:, 1], Tv[:, 2])).cos()).mean())))
    bp_err = math.degrees(float(torch.atan2(
        ((torch.atan2(P[:, 4], P[:, 5]) - torch.atan2(Tv[:, 4], Tv[:, 5])).sin()).abs().mean(),
        ((torch.atan2(P[:, 4], P[:, 5]) - torch.atan2(Tv[:, 4], Tv[:, 5])).cos()).mean())))
    print("\nPer-ping decodability (held-out, single observation + yaw):")
    print(f"  dist-to-goal        R2 = {dg_r2:+.3f}   (§5 WM readout claims ~0.98)")
    print(f"  ego-bearing-to-goal  MAE = {bg_err:5.1f} deg")
    print(f"  dist-to-pillar      R2 = {dp_r2:+.3f}")
    print(f"  ego-bearing-pillar   MAE = {bp_err:5.1f} deg")


if __name__ == "__main__":
    main()
