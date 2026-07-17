"""Frozen-latent allocentric probes for the cross-view Acoustic-JEPA (v2).

The actual hypothesis test. Freeze the trained encoder, embed each listener's
cross-view observation, and linearly decode targets with closed-form ridge
(no sklearn), episode-split. Compare three feature sets:

  * JEPA    -- the trained encoder's [z_range | z_coch] (256-d);
  * raw     -- pooled raw features (avg-pooled range profile + time-pooled
               cochleagram), the strong baseline that BEAT JEPA everywhere in v1;
  * random  -- an untrained encoder of the same shape (lottery-ticket control).

Targets, chosen to separate the ALLOCENTRIC claim from raw-favoured controls:
  * peer_pos  (dx,dz of the emitter rel. to the listener, from T_true) --
               the headline allocentric probe. The cross obs is REFLECTION-ONLY
               (no direct arrival), so decoding where the emitter is requires
               inferring it from the bistatic wall reflections -> a genuine
               allocentric inference raw pooling should struggle with.
  * room_vol  -- listener's room volume: a low-order, raw-friendly control
               (raw pooling won this in v1).
  * occluded  -- is the E->L path blocked (binary): higher-order relational.

The v2 bet is that JEPA >= raw on peer_pos / occluded even if raw >= JEPA on
room_vol. If raw wins everything (as in v1), that is the honest negative result.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from model_xview import CrossViewJEPA
from dataset_xview import make_loader, split_files


# ------------------------------------------------------------- feature sets
def raw_features(coch, rng):
    """Pooled raw baseline, dim-matched to the JEPA latent (~256)."""
    r = F.adaptive_avg_pool1d(rng.unsqueeze(1), 128).squeeze(1)   # [B,128]
    c = coch.mean(dim=3).reshape(coch.shape[0], -1)               # [B,2*64=128] time-pooled
    return torch.cat([r, c], dim=1)                               # [B,256]


@torch.no_grad()
def collect(loader, dev, jepa, rand):
    feats = {"jepa": [], "raw": [], "random": []}
    tgt = {"peer_pos": [], "room_vol": [], "occluded": []}
    for b in loader:
        coch = b["tgt_coch"].to(dev); rng = b["tgt_range"].to(dev)
        zr, zc = jepa.embed(coch, rng); feats["jepa"].append(torch.cat([zr, zc], 1).cpu())
        rr, rc = rand.embed(coch, rng); feats["random"].append(torch.cat([rr, rc], 1).cpu())
        feats["raw"].append(raw_features(coch, rng).cpu())
        tt = b["T_true"]
        tgt["peer_pos"].append(tt[:, [0, 2]])            # dx, dz
        tgt["room_vol"].append(b["L_room_vol"].unsqueeze(1))
        tgt["occluded"].append(b["occluded"].float().unsqueeze(1))
    feats = {k: torch.cat(v).numpy() for k, v in feats.items()}
    tgt = {k: torch.cat(v).numpy() for k, v in tgt.items()}
    return feats, tgt


# ------------------------------------------------------------- closed-form ridge
def zstat(X):
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-6
    return mu, sd


def ridge_fit(Xtr, ytr, lam=1.0):
    mu, sd = zstat(Xtr); Xn = (Xtr - mu) / sd
    Xa = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)
    d = Xa.shape[1]; R = lam * np.eye(d); R[-1, -1] = 0
    W = np.linalg.solve(Xa.T @ Xa + R, Xa.T @ ytr)
    return W, mu, sd


def ridge_pred(W, mu, sd, X):
    Xn = (X - mu) / sd
    return np.concatenate([Xn, np.ones((len(Xn), 1))], 1) @ W


def r2(y, yh):
    ss = ((y - yh) ** 2).sum(0); tot = ((y - y.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss / tot).mean())


def probe(Xtr, ytr, Xte, yte, lam=1.0):
    W, mu, sd = ridge_fit(Xtr, ytr, lam)
    return r2(yte, ridge_pred(W, mu, sd, Xte))


def probe_acc(Xtr, ytr, Xte, yte, lam=1.0):
    W, mu, sd = ridge_fit(Xtr, ytr, lam)
    pred = (ridge_pred(W, mu, sd, Xte) > 0.5).astype(np.float32)
    return float((pred == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--ckpt", default="ckpt_xview.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    ck = torch.load(args.ckpt, map_location=dev)
    jepa = CrossViewJEPA(num_hrtf=ck["num_hrtf"]).to(dev); jepa.load_state_dict(ck["model"]); jepa.eval()
    rand = CrossViewJEPA(num_hrtf=ck["num_hrtf"]).to(dev); rand.eval()

    tr_files, te_files = split_files(args.data_dir, 0.8, seed=0)
    trl = make_loader(tr_files, batch_size=256, shuffle=False)
    tel = make_loader(te_files, batch_size=256, shuffle=False)
    Ftr, Ytr = collect(trl, dev, jepa, rand)
    Fte, Yte = collect(tel, dev, jepa, rand)
    print(f"probe: {len(tr_files)} train / {len(te_files)} test rooms | "
          f"{len(Ftr['jepa'])} train / {len(Fte['jepa'])} test pairs\n")

    modes = ["jepa", "raw", "random"]
    print(f"{'target':12s} {'metric':6s} {'chance':>7s} " + " ".join(f"{m:>9s}" for m in modes) + "   winner")
    for name, metric in [("peer_pos", "R2"), ("room_vol", "R2"), ("occluded", "acc")]:
        vals = {}
        for m in modes:
            if metric == "R2":
                vals[m] = probe(Ftr[m], Ytr[name], Fte[m], Yte[name])
            else:
                vals[m] = probe_acc(Ftr[m], Ytr[name], Fte[m], Yte[name])
        # chance: R2 baseline is 0 (predict the mean); acc baseline is majority class
        if metric == "R2":
            chance = 0.0
        else:
            p = Yte[name].mean(); chance = max(p, 1 - p)
        win = max(vals, key=vals.get)
        note = "  <-- JEPA>raw & >chance" if (vals["jepa"] > vals["raw"] and
               (metric == "R2" and vals["jepa"] > 0.02 or metric != "R2" and vals["jepa"] > chance + 0.02)) else ""
        print(f"{name:12s} {metric:6s} {chance:>7.3f} " + " ".join(f"{vals[m]:>9.3f}" for m in modes) +
              f"   {win}{note}")

    print("\nheadline: peer_pos (allocentric) is where JEPA should beat raw; "
          "room_vol is the raw-favoured control. A win only counts if it also clears chance.")


if __name__ == "__main__":
    main()
