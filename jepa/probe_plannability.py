"""A2 mechanism probe, dynamics side: is the learned latent uniquely good at
exposing the CONSEQUENCE OF AN ACTION -- the property MPC actually exploits?

The control result (control_mpc.py) shows a learned world model plans well while
a random-encoder world model fails. That is end-to-end; this isolates WHY at the
representation level. For each frozen encoder we ask a single supervised
question on the trajectory data:

    given the current observation's latent z_t and the action a_t about to be
    taken, predict the nearest-wall distance AFTER the action, nwd_{t+1}.

This is exactly the quantity control_mpc's readout head predicts through a
rollout to plan. It is encoder-independent (the target is geometry, not another
encoder's latent), so JEPA / raw / random are directly comparable.

Two references frame it:
  * static   z_t -> nwd_t         (current nearest wall; the E1-style readout)
  * dynamics [z_t,a_t] -> nwd_{t+1}   (action consequence; the plannable readout)

The mechanism hypothesis: on the static readout raw >= JEPA (as in E1), but on
the dynamics readout JEPA >= raw -- the world-model objective buys action-
predictability at the cost of static geometry. If JEPA leads on BOTH, the latent
is simply better here; if it leads on NEITHER, plannability lives only in the
predictor/readout heads, not the encoder.

Usage: python probe_plannability.py <traj_data> --ckpts "../ckpt_ss2_*.pt"
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import torch

from model_xview import CrossViewJEPA
from probe_xview import raw_features, mlp_probe


def load_traj(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "ep*.npz")))
    assert files, f"no ep*.npz under {data_dir}"
    eps = []
    for f in files:
        d = np.load(f)
        eps.append(dict(coch=d["coch"].astype(np.float32),      # [K,2,64,48]
                        range=d["range"].astype(np.float32),    # [K,256]
                        actions=d["actions"].astype(np.float32),# [K,2]
                        nwd=d["labels_dist"].astype(np.float32).min(1)))  # [K]
    return eps


@torch.no_grad()
def encode_ep(enc, coch, rng, dev, kind):
    c = torch.tensor(coch, device=dev); r = torch.tensor(rng, device=dev)
    if kind == "raw":
        return raw_features(c, r).cpu().numpy()
    zr, zc = enc.embed(c, r)
    return torch.cat([zr, zc], 1).cpu().numpy()


def build(eps, enc, dev, kind):
    """Return (Z_static, y_static, Z_dyn, y_dyn) stacked over all episodes."""
    Zs, ys, Zd, yd = [], [], [], []
    for e in eps:
        z = encode_ep(enc, e["coch"], e["range"], dev, kind)   # [K,D]
        nwd = e["nwd"]; a = e["actions"]
        Zs.append(z); ys.append(nwd[:, None])
        Zd.append(np.concatenate([z[:-1], a[:-1]], 1)); yd.append(nwd[1:, None])
    return (np.concatenate(Zs), np.concatenate(ys),
            np.concatenate(Zd), np.concatenate(yd))


def ms(v):
    v = np.array(v); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--ckpts", default="../ckpt_ss2_*.pt")
    ap.add_argument("--rand-draws", type=int, default=5)
    ap.add_argument("--probe-seeds", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    ck = sorted(glob.glob(args.ckpts)); assert ck, f"no ckpts match {args.ckpts}"
    eps = load_traj(args.data_dir)
    n = len(eps); ntr = int(0.8 * n)
    g = np.random.default_rng(0); order = g.permutation(n)
    tr = [eps[i] for i in order[:ntr]]; te = [eps[i] for i in order[ntr:]]
    num_hrtf = torch.load(ck[0], map_location=dev)["num_hrtf"]
    pseeds = list(range(args.probe_seeds))
    print(f"{n} episodes ({ntr} train / {n-ntr} test), {len(ck)} JEPA ckpt, "
          f"{args.rand_draws} random draws, {args.probe_seeds} probe seeds")
    print("metric: R2, nonlinear MLP probe (higher=better)\n")

    def encoders():
        for cf in ck:
            m = CrossViewJEPA(num_hrtf=num_hrtf).to(dev)
            m.load_state_dict(torch.load(cf, map_location=dev)["model"]); m.eval()
            yield "jepa", m
        for s in range(args.rand_draws):
            torch.manual_seed(1000 + s)
            m = CrossViewJEPA(num_hrtf=num_hrtf).to(dev); m.eval()
            yield "random", m
        yield "raw", None

    res = {"static": {}, "dynamics": {}}
    for tag, enc in encoders():
        kind = "raw" if tag == "raw" else "jepa"
        Zs_tr, ys_tr, Zd_tr, yd_tr = build(tr, enc, dev, kind)
        Zs_te, ys_te, Zd_te, yd_te = build(te, enc, dev, kind)
        for ps in pseeds:
            res["static"].setdefault(tag, []).append(
                mlp_probe(Zs_tr, ys_tr, Zs_te, ys_te, task="reg", dev=dev, seed=ps))
            res["dynamics"].setdefault(tag, []).append(
                mlp_probe(Zd_tr, yd_tr, Zd_te, yd_te, task="reg", dev=dev, seed=ps))

    print(f"{'readout':22s} {'JEPA':>15s} {'raw':>15s} {'random':>15s}   verdict")
    for r, label in [("static", "static  z->nwd_t"),
                     ("dynamics", "dynamics [z,a]->nwd_t+1")]:
        jm, js = ms(res[r]["jepa"]); rm, rs = ms(res[r]["raw"]); nm, ns = ms(res[r]["random"])
        if jm - rm > js + rs:
            verdict = "JEPA>raw (learned wins)"
        elif rm - jm > js + rs:
            verdict = "raw>JEPA"
        else:
            verdict = "within noise"
        print(f"{label:22s} {jm:>7.3f}+/-{js:<6.3f} {rm:>7.3f}+/-{rs:<5.3f} "
              f"{nm:>7.3f}+/-{ns:<5.3f}   {verdict}")


if __name__ == "__main__":
    main()
