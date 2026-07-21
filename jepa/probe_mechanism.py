"""A2 mechanism probe: WHAT does the JEPA latent keep, given it discards
absolute geometry?

Experiment 1 (probe_multiseed.py) established the negative: the trained
cross-view JEPA latent decodes room_vol / peer_pos WORSE than raw pooling.
This asks the follow-up the negative leaves open -- is the loss *uniform*
(learning is just a worse feature extractor), or *selective* (the world-model
objective reorganises the latent toward control-relevant structure, paying for
it in absolute geometry)?

Same encoder, data, and probe methodology as probe_multiseed (frozen encoder,
episode split, nonlinear MLP + linear ridge, 3 JEPA ckpts x 5 random draws x
probe seeds), but an expanded target battery grouped by what the target IS:

  allo-absolute   room_vol        -- allocentric room scale (raw won this in E1)
  ego-directional nearest_wall    -- distance to the nearest wall (min over 8)
                  dist_wall8      -- the full 8-direction egocentric wall vector
                                     (this is exactly what wall-avoidance needs)
  relational      peer_pos        -- emitter dx,dz (raw/random won this in E1)
                  occluded        -- is the E->L sightline blocked (binary)
  surface         material        -- listener's dominant wall material (4-way)

The mechanism hypothesis predicts a CROSSOVER: raw >> JEPA on allo-absolute
room_vol, but JEPA catches raw on the ego-directional wall geometry a controller
actually uses. A flat "raw wins everything" is the honest alternative -> the
dissociation lives entirely in dynamics (see probe_plannability.py), not in the
static representation.

Usage: python probe_mechanism.py <swarm_data> --ckpts "ckpt_ss2_*.pt"
"""
from __future__ import annotations
import argparse, glob
import numpy as np
import torch
import torch.nn as nn

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
def raw_of(loader, dev):
    F = [raw_features(b["tgt_coch"].to(dev), b["tgt_range"].to(dev)).cpu() for b in loader]
    return torch.cat(F).numpy()


@torch.no_grad()
def targets_of(loader):
    dw, rv, mat, peer, occ = [], [], [], [], []
    for b in loader:
        dw.append(b["L_dist_wall"]); rv.append(b["L_room_vol"].unsqueeze(1))
        mat.append(b["L_material"]); peer.append(b["T_true"][:, [0, 2]])
        occ.append(b["occluded"].float().unsqueeze(1))
    dw = torch.cat(dw).numpy(); rv = torch.cat(rv).numpy()
    mat = torch.cat(mat).numpy().astype(np.int64)
    peer = torch.cat(peer).numpy(); occ = torch.cat(occ).numpy()
    near = dw.min(1, keepdims=True)
    # (task, y, group)
    return {
        "room_vol":     ("reg",  rv,   "allo-absolute"),
        "nearest_wall": ("reg",  near, "ego-directional"),
        "dist_wall8":   ("reg",  dw,   "ego-directional"),
        "peer_pos":     ("reg",  peer, "relational"),
        "occluded":     ("clf",  occ,  "relational"),
        "material":     ("mclf", mat,  "surface"),
    }


def mlp_probe_mclf(Xtr, ytr, Xte, yte, dev="cuda", hidden=256, epochs=400,
                   patience=40, seed=0):
    """Multiclass softmax MLP probe (material). Returns test accuracy."""
    g = np.random.default_rng(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr_n = (Xtr - mu) / sd; Xte_n = (Xte - mu) / sd
    nclass = int(max(ytr.max(), yte.max())) + 1
    n = len(Xtr_n); idx = g.permutation(n); nv = max(1, int(0.15 * n))
    va, trn = idx[:nv], idx[nv:]

    def T(a, dt=torch.float32): return torch.tensor(a, dtype=dt, device=dev)
    Xt, yt = T(Xtr_n[trn]), T(ytr[trn], torch.long)
    Xv, yv = T(Xtr_n[va]),  T(ytr[va], torch.long)
    Xe = T(Xte_n)
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr_n.shape[1], hidden), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
                        nn.Linear(hidden, nclass)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    best, best_state, wait = 1e9, None, 0
    for _ in range(epochs):
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
        pe = net(Xe).argmax(1).cpu().numpy()
    return float((pe == yte).mean())


def score(task, Xtr, ytr, Xte, yte, dev, seed):
    if task == "reg":
        return mlp_probe(Xtr, ytr, Xte, yte, task="reg", dev=dev, seed=seed)
    if task == "clf":
        return mlp_probe(Xtr, ytr, Xte, yte, task="clf", dev=dev, seed=seed)
    return mlp_probe_mclf(Xtr, ytr.ravel(), Xte, yte.ravel(), dev=dev, seed=seed)


def ms(v):
    v = np.array(v); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--ckpts", default="ckpt_ss2_*.pt")
    ap.add_argument("--rand-draws", type=int, default=5)
    ap.add_argument("--probe-seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    ck = sorted(glob.glob(args.ckpts))
    assert ck, f"no checkpoints match {args.ckpts}"
    tr_f, te_f = split_files(args.data_dir, 0.8, seed=0)
    trl = make_loader(tr_f, batch_size=256, shuffle=False)
    tel = make_loader(te_f, batch_size=256, shuffle=False)

    Ttr, Tte = targets_of(trl), targets_of(tel)
    raw_tr, raw_te = raw_of(trl, dev), raw_of(tel, dev)
    num_hrtf = torch.load(ck[0], map_location=dev)["num_hrtf"]
    pseeds = list(range(args.probe_seeds))

    jepa = []
    for cf in ck:
        m = CrossViewJEPA(num_hrtf=num_hrtf).to(dev)
        m.load_state_dict(torch.load(cf, map_location=dev)["model"]); m.eval()
        jepa.append((feats_of(trl, dev, m), feats_of(tel, dev, m)))
    rand = []
    for s in range(args.rand_draws):
        torch.manual_seed(1000 + s)
        m = CrossViewJEPA(num_hrtf=num_hrtf).to(dev); m.eval()
        rand.append((feats_of(trl, dev, m), feats_of(tel, dev, m)))

    print(f"{len(tr_f)} train / {len(te_f)} test rooms | {len(ck)} JEPA ckpt | "
          f"{args.rand_draws} random | {args.probe_seeds} probe seeds")
    print(f"metric: R2 (reg) or accuracy (clf/mclf), nonlinear MLP probe, higher=better\n")
    print(f"{'target':13s} {'group':15s} {'JEPA':>15s} {'raw':>15s} {'random':>15s}   verdict")
    for name, (task, ytr, grp) in Ttr.items():
        yte = Tte[name][1]
        j = [score(task, ftr, ytr, fte, yte, dev, ps) for (ftr, fte) in jepa for ps in pseeds]
        r = [score(task, raw_tr, ytr, raw_te, yte, dev, ps) for ps in pseeds]
        n = [score(task, ftr, ytr, fte, yte, dev, ps) for (ftr, fte) in rand for ps in pseeds]
        jm, js = ms(j); rm, rs = ms(r); nm, ns = ms(n)
        if rm - jm > js + rs:
            verdict = "DISCARDED (raw>JEPA)"
        elif jm >= rm:
            verdict = "KEPT (JEPA>=raw)"
        else:
            verdict = "within noise"
        print(f"{name:13s} {grp:15s} {jm:>7.3f}+/-{js:<6.3f} {rm:>7.3f}+/-{rs:<5.3f} "
              f"{nm:>7.3f}+/-{ns:<5.3f}   {verdict}")
    print("\nreg=R2, clf/mclf=accuracy. material chance=0.25 (4-way), occluded chance~0.71.")


if __name__ == "__main__":
    main()
