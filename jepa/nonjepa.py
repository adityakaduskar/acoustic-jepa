"""Non-JEPA controls: is the negative specific to the PREDICTIVE objective?

The v2 negative was: cross-view JEPA discards geometric variance (its frozen
latent decodes geometry worse than a random projection). This script tests
whether that is specific to the *predictive* objective by training two non-JEPA
encoders on the SAME two-stream architecture / same inputs and probing them with
the SAME suite:

  * AE   -- a reconstruction autoencoder (non-predictive SSL). Reconstruction
            forces the latent to PRESERVE input variance, including geometry --
            the opposite pressure to JEPA's compress-toward-predictable. If AE
            >= raw and >> JEPA, the failure is the predictive objective, not SSL
            per se or the architecture.
  * SUP  -- supervised upper bound: the same encoder trained end-to-end on the
            probe target (held-out rooms). The ceiling of what this architecture
            can extract from the reflection-only cross observation.

Compared against the frozen JEPA encoder, pooled-raw, and a random encoder.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn

from model_xview import CrossViewJEPA, TwoStreamEncoder
from dataset_xview import make_loader, split_files
from probe_xview import raw_features, probe, probe_acc, mlp_probe, r2


class AE(nn.Module):
    """Two-stream encoder + a decoder that reconstructs the observation."""
    def __init__(self, emb_range=128, emb_coch=128):
        super().__init__()
        self.encoder = TwoStreamEncoder(emb_range, emb_coch)
        emb = emb_range + emb_coch
        self.dec_range = nn.Sequential(nn.Linear(emb, 512), nn.GELU(), nn.Linear(512, 256))
        self.dec_coch = nn.Sequential(nn.Linear(emb, 1024), nn.GELU(), nn.Linear(1024, 2 * 64 * 48))

    def embed(self, coch, rng):
        return self.encoder(coch, rng)

    def forward(self, coch, rng):
        zr, zc = self.encoder(coch, rng)
        z = torch.cat([zr, zc], 1)
        return self.dec_range(z), self.dec_coch(z).view(-1, 2, 64, 48)


def train_ae(files, dev, epochs=40, bs=256):
    dl = make_loader(files, batch_size=bs, shuffle=True)
    ae = AE().to(dev)
    opt = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.MSELoss()
    for ep in range(epochs):
        for b in dl:
            coch, rng = b["tgt_coch"].to(dev), b["tgt_range"].to(dev)
            rr, rc = ae(coch, rng)
            loss = lf(rr, rng) + lf(rc, coch)
            opt.zero_grad(); loss.backward(); opt.step()
    ae.eval()
    return ae


def train_supervised(files, te_files, dev, target="peer_pos", epochs=60, bs=256):
    """End-to-end supervised ceiling: encoder -> head -> target, test-room R2."""
    out = 2 if target == "peer_pos" else 1
    enc = TwoStreamEncoder().to(dev)
    head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, out)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
    tr = make_loader(files, batch_size=bs, shuffle=True)

    def tgt_of(b):
        if target == "peer_pos":
            return b["T_true"][:, [0, 2]]
        return b["L_room_vol"].unsqueeze(1)
    # target standardisation from train
    ys = torch.cat([tgt_of(b) for b in make_loader(files, batch_size=512, shuffle=False)])
    ymu, ysd = ys.mean(0, keepdim=True).to(dev), (ys.std(0, keepdim=True) + 1e-6).to(dev)
    lf = nn.MSELoss()
    for ep in range(epochs):
        for b in tr:
            coch, rng = b["tgt_coch"].to(dev), b["tgt_range"].to(dev)
            y = (tgt_of(b).to(dev) - ymu) / ysd
            zr, zc = enc(coch, rng)
            pred = head(torch.cat([zr, zc], 1))
            loss = lf(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); head.eval()
    P, Y = [], []
    with torch.no_grad():
        for b in make_loader(te_files, batch_size=512, shuffle=False):
            coch, rng = b["tgt_coch"].to(dev), b["tgt_range"].to(dev)
            zr, zc = enc(coch, rng)
            P.append((head(torch.cat([zr, zc], 1)) * ysd + ymu).cpu().numpy())
            Y.append(tgt_of(b).numpy())
    return r2(np.concatenate(Y), np.concatenate(P))


@torch.no_grad()
def collect(loader, dev, encoders):
    feats = {k: [] for k in encoders}; feats["raw"] = []
    tgt = {"peer_pos": [], "room_vol": [], "occluded": []}
    for b in loader:
        coch, rng = b["tgt_coch"].to(dev), b["tgt_range"].to(dev)
        for k, enc in encoders.items():
            zr, zc = enc.embed(coch, rng); feats[k].append(torch.cat([zr, zc], 1).cpu())
        feats["raw"].append(raw_features(coch, rng).cpu())
        tgt["peer_pos"].append(b["T_true"][:, [0, 2]])
        tgt["room_vol"].append(b["L_room_vol"].unsqueeze(1))
        tgt["occluded"].append(b["occluded"].float().unsqueeze(1))
    return ({k: torch.cat(v).numpy() for k, v in feats.items()},
            {k: torch.cat(v).numpy() for k, v in tgt.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--ckpt", default="ckpt_xview400.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    ck = torch.load(args.ckpt, map_location=dev)
    jepa = CrossViewJEPA(num_hrtf=ck["num_hrtf"]).to(dev); jepa.load_state_dict(ck["model"]); jepa.eval()
    rand = CrossViewJEPA(num_hrtf=ck["num_hrtf"]).to(dev); rand.eval()

    tr_files, te_files = split_files(args.data_dir, 0.8, seed=0)
    print(f"non-JEPA controls | {len(tr_files)} train / {len(te_files)} test rooms")
    print("training AE (reconstruction)...")
    ae = train_ae(tr_files, dev)

    encoders = {"jepa": jepa, "ae": ae, "random": rand}
    Ftr, Ytr = collect(make_loader(tr_files, batch_size=256, shuffle=False), dev, encoders)
    Fte, Yte = collect(make_loader(te_files, batch_size=256, shuffle=False), dev, encoders)

    modes = ["jepa", "ae", "raw", "random"]
    targets = [("peer_pos", "R2", "reg"), ("room_vol", "R2", "reg"), ("occluded", "acc", "cls")]
    for kind in ("linear", "nonlinear"):
        print(f"\n=== {kind.upper()} probe ===")
        print(f"{'target':12s} {'metric':6s} {'chance':>7s} " + " ".join(f"{m:>8s}" for m in modes) + "   winner")
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
            print(f"{name:12s} {metric:6s} {chance:>7.3f} " + " ".join(f"{vals[m]:>8.3f}" for m in modes) + f"   {win}")

    print("\nsupervised ceiling (encoder trained end-to-end on the target, test rooms):")
    for t in ("peer_pos", "room_vol"):
        print(f"  {t:10s} R2 = {train_supervised(tr_files, te_files, dev, target=t):.3f}")
    print("\nread: if AE >= raw and >> JEPA, the negative is specific to the PREDICTIVE "
          "objective. The supervised ceiling shows what the architecture CAN extract.")


if __name__ == "__main__":
    main()
