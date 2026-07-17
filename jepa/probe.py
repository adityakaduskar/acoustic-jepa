"""Frozen-latent geometry probes (ACOUSTIC_JEPA_SPEC.md §4a) -- the first
reportable result.

Freeze a trained Acoustic-JEPA encoder, embed every observation, and fit LINEAR
probes from the frozen latent to the analytic ground-truth labels:

    distance-to-wall (8 sectors)  ridge   -> MAE (m), R^2
    opening azimuth               ridge   -> circular MAE (deg)   [openings only]
    room volume                   ridge   -> R^2, MAE (m^3)
    material (nearest wall)       linear  -> accuracy (%)

The claim lands if the frozen JEPA latent carries all four JOINTLY and beats the
self-contained baselines reported alongside:
    raw-cochleagram : linear probe on pooled input features (learned rep vs input)
    random-encoder  : same architecture, untrained weights (training vs init)

The single-cue analytic baselines from the spec table (matched-filter range,
ITD/ILD bearing) live in the C++ tools (matched_range.*, echo_nav.*); cross-refer
those numbers when writing up. Splits are by EPISODE (room), so train and test
never share a room -- this measures generalisation to unseen geometry.

No sklearn: ridge is closed-form, the classifier is a short torch fit. Run in the
torch venv, e.g.:  python probe.py runs/jepa/final.pt <data_dir>
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

from dataset import ProbeDataset
from model import AcousticJEPA
from torch.utils.data import DataLoader


class ProbeWithFi(ProbeDataset):
    """ProbeDataset that also returns the episode id (fi) for episode-level splits."""

    def __getitem__(self, i):
        coch, lab = super().__getitem__(i)
        lab["fi"] = torch.tensor(self.index[i][0])
        return coch, lab


# --------------------------------------------------------------------------- #
#  feature collection
# --------------------------------------------------------------------------- #
def raw_features(coch: torch.Tensor) -> torch.Tensor:
    """Pooled cochleagram baseline: per (ear,band) mean+std over time -> [B,256].
    If given a window [B,W,2,64,48], use the newest frame (the 'current' echo)."""
    if coch.dim() == 5:
        coch = coch[:, -1]                     # [B,2,64,48]
    m = coch.mean(dim=3)                       # [B,2,64]
    s = coch.std(dim=3)                        # [B,2,64]
    return torch.cat([m.flatten(1), s.flatten(1)], dim=1)   # [B,256]


@torch.no_grad()
def collect(loader, trained, rand_enc, device):
    Z, Zr, Xraw = [], [], []
    dist, oaz, vol, mat, fi = [], [], [], [], []
    for coch, lab in loader:
        c = coch.to(device, non_blocking=True)
        Z.append(trained.embed(c).float().cpu().numpy())
        Zr.append(rand_enc.embed(c).float().cpu().numpy())
        Xraw.append(raw_features(coch).numpy())
        dist.append(lab["dist_wall"].numpy())
        oaz.append(lab["open_az"].numpy())
        vol.append(lab["room_vol"].numpy())
        mat.append(lab["material"].numpy())
        fi.append(lab["fi"].numpy())
    return {
        "jepa": np.concatenate(Z), "random": np.concatenate(Zr),
        "raw": np.concatenate(Xraw), "dist": np.concatenate(dist),
        "open_az": np.concatenate(oaz), "vol": np.concatenate(vol),
        "material": np.concatenate(mat), "fi": np.concatenate(fi),
    }


# --------------------------------------------------------------------------- #
#  linear probes
# --------------------------------------------------------------------------- #
def _standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def ridge_fit_predict(Xtr, Ytr, Xte, lam=1.0):
    """Closed-form ridge, multi-output. Y may be [N] or [N,K]."""
    Xtr, Xte = _standardize(Xtr, Xte)
    single = Ytr.ndim == 1
    Y = Ytr.reshape(-1, 1) if single else Ytr
    ybar = Y.mean(0, keepdims=True)
    d = Xtr.shape[1]
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ (Y - ybar))
    pred = Xte @ W + ybar
    return pred[:, 0] if single else pred


def r2(y, yh):
    ss = ((y - y.mean(0)) ** 2).sum()
    return float(1.0 - ((y - yh) ** 2).sum() / (ss + 1e-12))


def mae(y, yh):
    return float(np.abs(y - yh).mean())


def circular_mae_deg(az_true_deg, Xtr, Xte, ytr_deg):
    """Regress sin/cos of azimuth, reconstruct angle, wrapped MAE in degrees."""
    rad = np.deg2rad(ytr_deg)
    s = ridge_fit_predict(Xtr, np.sin(rad), Xte)
    c = ridge_fit_predict(Xtr, np.cos(rad), Xte)
    pred = np.rad2deg(np.arctan2(s, c))
    diff = (pred - az_true_deg + 180.0) % 360.0 - 180.0
    return float(np.abs(diff).mean())


def fit_classifier(Xtr, ytr, Xte, yte, n_cls, device, steps=400):
    Xtr, Xte = _standardize(Xtr, Xte)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    xe = torch.tensor(Xte, dtype=torch.float32, device=device)
    clf = torch.nn.Linear(Xtr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = clf(xe).argmax(1).cpu().numpy()
    return float((pred == yte).mean() * 100.0)


# --------------------------------------------------------------------------- #
def evaluate(feats, name, test_mask, device, lam):
    tr, te = ~test_mask, test_mask
    X = feats[name]
    out = {}

    dpred = ridge_fit_predict(X[tr], feats["dist"][tr], X[te], lam)
    out["dist_mae_m"] = mae(feats["dist"][te], dpred)
    out["dist_r2"] = r2(feats["dist"][te], dpred)

    vpred = ridge_fit_predict(X[tr], feats["vol"][tr], X[te], lam)
    out["vol_r2"] = r2(feats["vol"][te], vpred)
    out["vol_mae_m3"] = mae(feats["vol"][te], vpred)

    # openings only (open_az is NaN when a room/pose has no opening)
    has = ~np.isnan(feats["open_az"])
    otr, ote = tr & has, te & has
    if ote.sum() > 5 and otr.sum() > 5:
        out["open_circmae_deg"] = circular_mae_deg(
            feats["open_az"][ote], X[otr], X[ote], feats["open_az"][otr])
    else:
        out["open_circmae_deg"] = float("nan")

    n_cls = int(feats["material"].max()) + 1
    out["material_acc"] = fit_classifier(
        X[tr], feats["material"][tr], X[te], feats["material"][te], n_cls, device)
    return out


def main():
    ap = argparse.ArgumentParser(description="frozen-latent geometry probes")
    ap.add_argument("ckpt")
    ap.add_argument("data_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1.0, help="ridge regularisation")
    ap.add_argument("--test-frac", type=float, default=0.2, help="fraction of EPISODES held out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location=device)
    emb = ck.get("cfg", {}).get("emb", 256)
    window = ck.get("cfg", {}).get("window", 1)
    trained = AcousticJEPA(emb=emb, window=window).to(device).eval()
    trained.load_state_dict(ck["model"])
    rand_enc = AcousticJEPA(emb=emb, window=window).to(device).eval()   # untrained baseline
    torch.manual_seed(args.seed)
    print(f"probe: emb={emb} window={window}")

    # ProbeDataset with the episode id (fi) attached, for episode-level splits
    ds = ProbeWithFi(args.data_dir, window)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    feats = collect(loader, trained, rand_enc, device)

    # split by episode so train/test never share a room
    rng = np.random.default_rng(args.seed)
    n_ep = len(ds.files)
    perm = rng.permutation(n_ep)
    n_test = max(1, int(round(args.test_frac * n_ep)))
    test_eps = set(perm[:n_test].tolist())
    test_mask = np.array([fi in test_eps for fi in feats["fi"]], dtype=bool)
    print(f"samples={len(test_mask)}  test_episodes={n_test}/{n_ep}  "
          f"test_samples={int(test_mask.sum())}  emb={emb}")

    rows = {}
    for name in ("jepa", "raw", "random"):
        rows[name] = evaluate(feats, name, test_mask, device, args.lam)

    hdr = f"{'probe':<10}{'dist MAE(m)':>12}{'dist R2':>9}{'open cMAE°':>11}{'vol R2':>8}{'mat acc%':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    label = {"jepa": "JEPA", "raw": "raw-coch", "random": "rand-enc"}
    for name in ("jepa", "raw", "random"):
        r = rows[name]
        print(f"{label[name]:<10}{r['dist_mae_m']:>12.3f}{r['dist_r2']:>9.3f}"
              f"{r['open_circmae_deg']:>11.1f}{r['vol_r2']:>8.3f}{r['material_acc']:>9.1f}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "probe_results.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
