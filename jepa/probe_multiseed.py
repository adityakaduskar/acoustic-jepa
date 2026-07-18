"""Error bars for the probe (geometry-decode) result.

Puts variance on the experiment-1 claim (raw/random >= learned) by aggregating:
  * JEPA over multiple trained checkpoints x probe seeds;
  * random over multiple random-encoder draws x probe seeds;
  * raw over probe seeds (raw features are deterministic).
Nonlinear (MLP) probe, room-level split. Reports mean +/- SD and whether raw
beats JEPA robustly (mean gap beyond the combined spread).
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import torch

from model_xview import CrossViewJEPA
from dataset_xview import make_loader, split_files
from probe_xview import raw_features, mlp_probe


@torch.no_grad()
def feats_of(loader, dev, enc):
    F = []
    for b in loader:
        zr, zc = enc.embed(b["tgt_coch"].to(dev), b["tgt_range"].to(dev))
        F.append(torch.cat([zr, zc], 1).cpu())
    return torch.cat(F).numpy()


@torch.no_grad()
def raw_and_targets(loader, dev):
    R, peer, vol = [], [], []
    for b in loader:
        R.append(raw_features(b["tgt_coch"].to(dev), b["tgt_range"].to(dev)).cpu())
        peer.append(b["T_true"][:, [0, 2]]); vol.append(b["L_room_vol"].unsqueeze(1))
    return (torch.cat(R).numpy(),
            {"peer_pos": torch.cat(peer).numpy(), "room_vol": torch.cat(vol).numpy()})


def ms(v):
    v = np.array(v); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--ckpts", default="ckpt_xview400*.pt")
    ap.add_argument("--rand-draws", type=int, default=5)
    ap.add_argument("--probe-seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    ck_files = sorted(glob.glob(args.ckpts))
    assert ck_files, f"no checkpoints match {args.ckpts}"
    tr_files, te_files = split_files(args.data_dir, 0.8, seed=0)
    trl = make_loader(tr_files, batch_size=256, shuffle=False)
    tel = make_loader(te_files, batch_size=256, shuffle=False)
    raw_tr, Ytr = raw_and_targets(trl, dev)
    raw_te, Yte = raw_and_targets(tel, dev)
    num_hrtf = torch.load(ck_files[0], map_location=dev)["num_hrtf"]
    pseeds = list(range(args.probe_seeds))
    print(f"{len(tr_files)} train / {len(te_files)} test rooms | {len(ck_files)} JEPA ckpt(s) | "
          f"{args.rand_draws} random draws | {args.probe_seeds} probe seeds\n")

    # collect JEPA features per checkpoint
    jepa_feats = []
    for cf in ck_files:
        m = CrossViewJEPA(num_hrtf=num_hrtf).to(dev); m.load_state_dict(torch.load(cf, map_location=dev)["model"]); m.eval()
        jepa_feats.append((feats_of(trl, dev, m), feats_of(tel, dev, m)))
    # collect random-encoder features per draw
    rand_feats = []
    for s in range(args.rand_draws):
        torch.manual_seed(1000 + s)
        m = CrossViewJEPA(num_hrtf=num_hrtf).to(dev); m.eval()
        rand_feats.append((feats_of(trl, dev, m), feats_of(tel, dev, m)))

    print(f"{'target':10s} {'JEPA (mean+/-SD)':>20s} {'raw':>16s} {'random':>18s}   verdict")
    for t in ("peer_pos", "room_vol"):
        j = [mlp_probe(ftr, Ytr[t], fte, Yte[t], task="reg", dev=dev, seed=ps)
             for (ftr, fte) in jepa_feats for ps in pseeds]
        r = [mlp_probe(raw_tr, Ytr[t], raw_te, Yte[t], task="reg", dev=dev, seed=ps) for ps in pseeds]
        n = [mlp_probe(ftr, Ytr[t], fte, Yte[t], task="reg", dev=dev, seed=ps)
             for (ftr, fte) in rand_feats for ps in pseeds]
        jm, js = ms(j); rm, rs = ms(r); nm, ns = ms(n)
        # robust if raw mean exceeds JEPA mean by more than the summed SDs
        robust = (rm - jm) > (js + rs)
        verdict = "raw>JEPA robust" if robust else ("raw>JEPA (within noise)" if rm > jm else "JEPA>=raw")
        print(f"{t:10s} {jm:>9.3f} +/- {js:<6.3f} {rm:>7.3f} +/- {rs:<4.3f} "
              f"{nm:>8.3f} +/- {ns:<4.3f}   {verdict}")
    print("\n(R2, nonlinear MLP probe; higher=better. random = untrained-encoder feature bank.)")


if __name__ == "__main__":
    main()
