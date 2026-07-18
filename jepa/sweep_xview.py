"""Hyperparameter sweep: give cross-view JEPA its best shot on the geometry probes.

The negative was "learned rep loses to raw pooling on geometry decode." Before
concluding that, sweep the knobs most likely to help JEPA RETAIN geometry -- the
ones controlling how hard the objective compresses variance: VICReg variance
weight (var_coeff), coch-loss weight (lam_coch), training length (epochs), and
capacity (emb). For each config, train, freeze, and nonlinear-probe room_vol +
peer_pos on held-out rooms. If the BEST config still loses to raw, the negative
is robust to tuning; if any config beats raw, it was a tuning artifact.
"""
from __future__ import annotations
import argparse, math
import numpy as np
import torch

from model_xview import CrossViewJEPA
from dataset_xview import make_loader, split_files
from probe_xview import raw_features, mlp_probe


def train_one(cfg, tr_files, dev):
    tr = make_loader(tr_files, batch_size=cfg["batch"], shuffle=True)
    torch.manual_seed(0)
    m = CrossViewJEPA(emb_range=cfg["emb"], emb_coch=cfg["emb"], num_hrtf=8,
                      ema=cfg["ema"], lam_coch=cfg["lam"], var_coeff=cfg["var"],
                      cov_coeff=cfg["cov"]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps = max(1, len(tr)) * cfg["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    g = 0
    for ep in range(cfg["epochs"]):
        for b in tr:
            b = {k: v.to(dev) for k, v in b.items()}
            out = m(b["ctx_coch"], b["ctx_range"], b["tgt_coch"], b["tgt_range"], b["T"], b["hrtf_id"])
            opt.zero_grad(set_to_none=True); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step(); sched.step()
            m.ema = 1.0 - (1.0 - cfg["ema"]) * (0.5 * (1 + math.cos(math.pi * g / steps)))
            m.update_target(); g += 1
    m.eval(); return m


@torch.no_grad()
def feats(loader, dev, enc):
    F = []
    for b in loader:
        zr, zc = enc.embed(b["tgt_coch"].to(dev), b["tgt_range"].to(dev))
        F.append(torch.cat([zr, zc], 1).cpu())
    return torch.cat(F).numpy()


@torch.no_grad()
def raw_targets(loader, dev):
    R, peer, vol = [], [], []
    for b in loader:
        R.append(raw_features(b["tgt_coch"].to(dev), b["tgt_range"].to(dev)).cpu())
        peer.append(b["T_true"][:, [0, 2]]); vol.append(b["L_room_vol"].unsqueeze(1))
    return (torch.cat(R).numpy(),
            {"peer_pos": torch.cat(peer).numpy(), "room_vol": torch.cat(vol).numpy()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    tr_files, te_files = split_files(args.data_dir, 0.8, seed=0)
    trl = make_loader(tr_files, batch_size=256, shuffle=False)
    tel = make_loader(te_files, batch_size=256, shuffle=False)
    raw_tr, Ytr = raw_targets(trl, dev)
    raw_te, Yte = raw_targets(tel, dev)

    D = dict(batch=256, emb=128, ema=0.996, lam=0.25, var=1.0, cov=0.04, lr=1e-3, epochs=60)
    def cfg(**kw): d = dict(D); d.update(kw); return d
    configs = [
        ("default",            cfg()),
        ("var=4",              cfg(var=4.0)),
        ("var=16",             cfg(var=16.0)),
        ("lam=1.0",            cfg(lam=1.0)),
        ("lam=0 (range-only)", cfg(lam=0.0)),
        ("epochs=20",          cfg(epochs=20)),
        ("epochs=20,var=4",    cfg(epochs=20, var=4.0)),
        ("lr=3e-4",            cfg(lr=3e-4)),
        ("emb=256",            cfg(emb=256)),
        ("retain: var4,lam1,e30", cfg(var=4.0, lam=1.0, epochs=30)),
    ]

    # baselines
    raw_v = mlp_probe(raw_tr, Ytr["room_vol"], raw_te, Yte["room_vol"], task="reg", dev=dev, seed=0)
    raw_p = mlp_probe(raw_tr, Ytr["peer_pos"], raw_te, Yte["peer_pos"], task="reg", dev=dev, seed=0)
    torch.manual_seed(1000)
    rnd = CrossViewJEPA(num_hrtf=8).to(dev).eval()
    rf_tr, rf_te = feats(trl, dev, rnd), feats(tel, dev, rnd)
    rnd_v = mlp_probe(rf_tr, Ytr["room_vol"], rf_te, Yte["room_vol"], task="reg", dev=dev, seed=0)
    rnd_p = mlp_probe(rf_tr, Ytr["peer_pos"], rf_te, Yte["peer_pos"], task="reg", dev=dev, seed=0)
    print(f"{len(tr_files)} train / {len(te_files)} test rooms")
    print(f"BASELINES  room_vol: raw {raw_v:.3f}, random {rnd_v:.3f} | "
          f"peer_pos: raw {raw_p:.3f}, random {rnd_p:.3f}\n")

    print(f"{'config':22s} {'room_vol':>9s} {'peer_pos':>9s}   {'beats raw?':>10s}")
    best = None
    for name, c in configs:
        m = train_one(c, tr_files, dev)
        jt, je = feats(trl, dev, m), feats(tel, dev, m)
        v = mlp_probe(jt, Ytr["room_vol"], je, Yte["room_vol"], task="reg", dev=dev, seed=0)
        p = mlp_probe(jt, Ytr["peer_pos"], je, Yte["peer_pos"], task="reg", dev=dev, seed=0)
        flag = "YES" if (v > raw_v or p > raw_p) else "no"
        print(f"{name:22s} {v:>9.3f} {p:>9.3f}   {flag:>10s}", flush=True)
        if best is None or v > best[1]:
            best = (name, v, p)
    print(f"\nbest room_vol config: {best[0]} at {best[1]:.3f} (raw {raw_v:.3f}, random {rnd_v:.3f})")
    print("verdict: " + ("a config BEAT raw -> negative was a tuning artifact"
          if best[1] > raw_v else "even the best-tuned JEPA loses to raw -> negative robust to tuning"))


if __name__ == "__main__":
    main()
