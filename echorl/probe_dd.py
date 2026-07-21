"""EchoRL Experiment A -- the Delta-d probe: WHY does the learned-WM MPC (mpc.py,
RESULTS.md 5) trail model-free PPO?

The 5 diagnostic showed the encoder DECODES distance-to-goal (readout R2=0.98) but
the action-conditioned rollout predicts the RESULTING distance no better than a
"distance-unchanged" baseline -- so CEM plans over a forward model blind to what
actions do. Three hypotheses for that blindness:

  H0  the info isn't there:  a single monostatic ping + action can't support
      Delta-d = d(t+1)-d(t) at all  -> raw+a probe FAILS. MPC gap is fundamental.
  H1  the encoder discards it: raw+a probe SUCCEEDS but z+a (trained encoder latent)
      FAILS -> the readout objective only asked for distance, so bearing/heading-to-
      goal was thrown away. Fix: add a Delta-d head to the WM training loss.
  H2  the predictor fails to use it: z+a probe SUCCEEDS -> the info survives into the
      latent; the trained Predictor+readout path is what's weak. Fix: better dynamics
      model, or plan on a direct (z,a)->Delta-d consequence head.

Method: identical small-MLP probes (held-out R^2, multi-seed) over inputs
{raw+a, z+a, z-only, a-only}, plus the WM's OWN predictor-path Delta-d as reference.
The raw probe gets a flexible MLP so a low held-out R^2 means the info is genuinely
absent, not under-fit.

Run: python echorl/probe_dd.py --corpus echorl/corpus_goal.pkl --wm echorl/ckpt/mpc_wm.pt
"""
from __future__ import annotations
import argparse, os, sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from mpc import WorldModel, load_corpus, COCH_SHAPE      # noqa: E402


def build_pairs(eps):
    """(range, coch, act, Delta-d) for every consecutive step across episodes."""
    R, C, A, DD = [], [], [], []
    for (r, c, a, d) in eps:
        K = a.shape[0]
        for t in range(K):
            R.append(r[t]); C.append(c[t]); A.append(a[t]); DD.append(d[t + 1] - d[t])
    return (np.asarray(R, np.float32), np.asarray(C, np.float32),
            np.asarray(A, np.float32), np.asarray(DD, np.float32))


class MLP(nn.Module):
    def __init__(self, d_in, hidden=128, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def r2(pred, targ):
    ss_res = float(((pred - targ) ** 2).sum())
    ss_tot = float(((targ - targ.mean()) ** 2).sum() + 1e-9)
    return 1.0 - ss_res / ss_tot


def train_probe(X, Y, tr, va, dev, seed=0, epochs=60, bs=256, wd=1e-3):
    torch.manual_seed(seed)
    net = MLP(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    Xt = torch.as_tensor(X, device=dev); Yt = torch.as_tensor(Y, device=dev)
    best = -1e9
    for ep in range(epochs):
        net.train(); idx = tr.copy(); np.random.default_rng(ep).shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            best = max(best, r2(net(Xt[va]), Yt[va]))     # best held-out over epochs
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="cached corpus from mpc.py --corpus-out")
    ap.add_argument("--wm", required=True, help="learned WM checkpoint from mpc.py --save-wm")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    eps = load_corpus(args.corpus)
    R, C, A, DD = build_pairs(eps)
    n = len(DD)
    print(f"corpus {os.path.relpath(args.corpus)}: {len(eps)} episodes, {n} step-pairs | "
          f"Delta-d mean {DD.mean():+.3f} sd {DD.std():.3f} m (target to explain)")

    # load the trained WM; produce z_t and the normalized raw features on the SAME footing
    ck = torch.load(args.wm, map_location=dev)
    wm = WorldModel().to(dev); wm.load_state_dict(ck["model"]); wm.eval()
    with torch.no_grad():
        Rt = torch.as_tensor(R, device=dev); Ct = torch.as_tensor(C, device=dev)
        At = torch.as_tensor(A, device=dev)
        Z, RAWr, RAWc, ddPred = [], [], [], []
        for i in range(0, n, 1024):
            r_, c_, a_ = Rt[i:i+1024], Ct[i:i+1024], At[i:i+1024]
            z = wm.enc(r_, c_)                                  # [b,256] trained latent
            rn, cn = wm.norm(r_, c_)                            # normalized raw, same as encoder input
            Z.append(z.cpu().numpy())
            RAWr.append(rn.cpu().numpy()); RAWc.append(cn.reshape(cn.shape[0], -1).cpu().numpy())
            # WM's OWN predictor path: implied Delta-d = readout(pred(z,a)) - readout(z)
            z1 = wm.predictor(z, a_)
            ddPred.append((wm.readout(z1).squeeze(-1) - wm.readout(z).squeeze(-1)).cpu().numpy())
    Z = np.concatenate(Z); RAW = np.concatenate([np.concatenate(RAWr), np.concatenate(RAWc)], 1)
    ddPred = np.concatenate(ddPred)

    inputs = {
        "raw + a  (H0 test)":       np.concatenate([RAW, A], 1),
        "z + a    (H1/H2 test)":    np.concatenate([Z, A], 1),
        "z only   (control)":       Z,
        "a only   (control)":       A.copy(),
    }
    rng = np.random.default_rng(0); perm = rng.permutation(n)
    ntr = int(0.85 * n); tr, va = perm[:ntr], perm[ntr:]

    print(f"\nDelta-d probe -- held-out R^2 (best over epochs; mean +/- sd over {args.seeds} probe seeds):")
    res = {}
    for name, X in inputs.items():
        rs = [train_probe(X, DD, tr, va, dev, seed=s) for s in range(args.seeds)]
        m, sd = float(np.mean(rs)), float(np.std(rs, ddof=1) if len(rs) > 1 else 0.0)
        res[name] = m
        print(f"  {name:26s}  R^2 {m:+.3f} +/- {sd:.3f}  (dim {X.shape[1]})")
    # the WM's own dynamics path, for reference (no probe training -- it's the model)
    wm_r2 = r2(ddPred[va], DD[va])
    print(f"  {'WM predictor path':26s}  R^2 {wm_r2:+.3f}         (readout(pred(z,a))-readout(z); what CEM plans on)")

    raw_a = res["raw + a  (H0 test)"]; z_a = res["z + a    (H1/H2 test)"]
    print("\n=== verdict ===")
    if raw_a < 0.15:
        print(f"  H0: single-ping+action barely predicts Delta-d (raw+a R^2 {raw_a:+.2f}). The MPC gap is")
        print(f"      FUNDAMENTAL to the cue -- doorway Delta-d isn't in one monostatic ping.")
    elif z_a < 0.15 <= raw_a:
        print(f"  H1: info is in the ping (raw+a R^2 {raw_a:+.2f}) but the trained encoder DISCARDS it")
        print(f"      (z+a R^2 {z_a:+.2f}). Fix: add a Delta-d head to the WM training loss.")
    else:
        print(f"  H2: info survives into the latent (z+a R^2 {z_a:+.2f}) -- the trained PREDICTOR path is")
        print(f"      the bottleneck (its Delta-d R^2 {wm_r2:+.2f}). Fix: better dynamics / plan on a (z,a)->Delta-d head.")


if __name__ == "__main__":
    main()
