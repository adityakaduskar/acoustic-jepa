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


# ------------------------------------------------------------- nonlinear (MLP) probe
# Same probe applied IDENTICALLY to every feature set (jepa/raw/random). If JEPA
# encodes geometry non-linearly that the ridge probe can't read, it should
# overtake raw here; if raw still wins, the linear negative hardens.
def mlp_probe(Xtr, ytr, Xte, yte, task="reg", dev="cuda", hidden=256,
              epochs=400, patience=40, seed=0):
    import torch.nn as nn
    g = np.random.default_rng(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_n = (Xtr - mu) / sd; Xte_n = (Xte - mu) / sd
    out = ytr.shape[1]
    if task == "reg":
        ymu, ysd = ytr.mean(0, keepdims=True), ytr.std(0, keepdims=True) + 1e-6
        ytr_s = (ytr - ymu) / ysd
    else:
        ytr_s = ytr
    n = len(Xtr_n); idx = g.permutation(n); nv = max(1, int(0.15 * n))
    va, trn = idx[:nv], idx[nv:]

    def T(a): return torch.tensor(a, dtype=torch.float32, device=dev)
    Xt, yt = T(Xtr_n[trn]), T(ytr_s[trn])
    Xv, yv = T(Xtr_n[va]), T(ytr_s[va])
    Xe = T(Xte_n)

    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr_n.shape[1], hidden), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(hidden, out)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.MSELoss() if task == "reg" else nn.BCEWithLogitsLoss()
    best, best_state, wait = 1e9, None, 0
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        loss = lossf(net(Xt), yt); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(lossf(net(Xv), yv))
        if vl < best - 1e-4:
            best, best_state, wait = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        pe = net(Xe).cpu().numpy()
    if task == "reg":
        pe = pe * ysd + ymu
        return r2(yte, pe)
    return float(((pe > 0) == (yte > 0.5)).mean())


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
    targets = [("peer_pos", "R2", "reg"), ("room_vol", "R2", "reg"), ("occluded", "acc", "cls")]

    def run_table(kind):
        print(f"\n=== {kind.upper()} probe ===")
        print(f"{'target':12s} {'metric':6s} {'chance':>7s} " + " ".join(f"{m:>9s}" for m in modes) + "   winner")
        for name, metric, task in targets:
            vals = {}
            for m in modes:
                if kind == "linear":
                    vals[m] = probe(Ftr[m], Ytr[name], Fte[m], Yte[name]) if metric == "R2" \
                        else probe_acc(Ftr[m], Ytr[name], Fte[m], Yte[name])
                else:
                    vals[m] = mlp_probe(Ftr[m], Ytr[name], Fte[m], Yte[name], task=task, dev=dev)
            chance = 0.0 if metric == "R2" else float(max(Yte[name].mean(), 1 - Yte[name].mean()))
            win = max(vals, key=vals.get)
            beats = vals["jepa"] > vals["raw"] and (vals["jepa"] > chance + 0.02)
            print(f"{name:12s} {metric:6s} {chance:>7.3f} " + " ".join(f"{vals[m]:>9.3f}" for m in modes) +
                  f"   {win}" + ("  <-- JEPA>raw & >chance" if beats else ""))

    run_table("linear")
    run_table("nonlinear")
    print("\nheadline: does a NONLINEAR probe change the ranking? If JEPA overtakes raw only "
          "under the MLP, it encodes geometry non-linearly; if raw still wins, the negative hardens.")


if __name__ == "__main__":
    main()
