"""A2 mechanism, decisive experiment: does a RAW-FEATURE-encoder world model
plan as well as the learned one?

probe_mechanism.py + probe_plannability.py showed that a FROZEN JEPA encoder is
a uniformly WORSE readout substrate than raw pooling -- raw wins every static
geometry target AND the action-consequence readout [z,a]->nwd_{t+1}. Yet the
control result (control_mpc.py) says the learned world model beats a
random-encoder one. If plannability were just "good decodable features", then
freezing the BEST decodable features (raw pooling) as the encoder and training
the predictor+readout+MPC on top should plan at least as well as learned.

This runs exactly that baseline on the unchanged offline control protocol
(same windows, CEM planner, fixed eval starts), adding a RAW-encoder world
model alongside the learned and random-encoder ones. Everything is imported
from control_mpc -- only the encoder is swapped for frozen raw pooling.

Outcome map:
  raw-enc ~ learned   -> plannability = decodable features; the learned encoder
                         is NOT necessary (a correction to the control claim).
  raw-enc ~ random    -> decodability != plannability; end-to-end learning of
                         the encoder (not just good features) is what makes the
                         latent plannable -- the sharp mechanism.

Usage: python control_raw_encoder.py <traj_data> --goal wall --repeats 5
"""
from __future__ import annotations
import argparse, math
import numpy as np
import torch

import control_mpc as CM
from control_mpc import (WorldModel, build_windows, norm_act, cem_plan,
                         eval_controller, load_traj)
from model import vicreg_var_cov
from probe_xview import raw_features


class RawWorldModel(WorldModel):
    """World model whose encoder is FROZEN raw pooling (dim-matched to emb=256).
    The predictor + readout are still trained; only the representation is fixed
    to the strong hand-designed features."""
    def enc(self, coch, rng):
        return raw_features(coch, rng)

    @torch.no_grad()
    def enc_target(self, coch, rng):
        return raw_features(coch, rng)


def train_raw_wm(eps, dev, H=6, epochs=25, bs=128, seed=0):
    """Mirror control_mpc.train_wm but with the raw-feature encoder (no target
    EMA -- the target features are the deterministic raw pooling)."""
    torch.manual_seed(seed)
    C, R, A, NWD = build_windows(eps, H)
    n = len(C); idx = np.arange(n)
    wm = RawWorldModel().to(dev)
    for p in wm.encoder.parameters():          # unused, keep them out of the optimiser
        p.requires_grad_(False)
    params = [p for p in wm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    An = norm_act(A)
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(idx)
        for i in range(0, n, bs):
            b = idx[i:i + bs]
            coch = torch.tensor(C[b], device=dev); rng = torch.tensor(R[b], device=dev)
            act = torch.tensor(An[b], device=dev); nwd = torch.tensor(NWD[b], device=dev)
            Bn, T = coch.shape[0], coch.shape[1]
            z = wm.enc(coch.reshape(Bn * T, 2, 64, 48), rng.reshape(Bn * T, 256)).reshape(Bn, T, -1)
            with torch.no_grad():
                zt = wm.enc_target(coch.reshape(Bn * T, 2, 64, 48), rng.reshape(Bn * T, 256)).reshape(Bn, T, -1)
            pred = wm.rollout(z[:, 0], act)
            dyn = ((pred - zt[:, 1:]) ** 2).mean()
            ro = wm.readout(z.reshape(Bn * T, -1)).reshape(Bn, T)
            ro_loss = ((ro - nwd) ** 2).mean()
            vr, cv = vicreg_var_cov(z[:, 0])
            loss = dyn + ro_loss + 1.0 * vr + 0.04 * cv
            opt.zero_grad(); loss.backward(); opt.step()
    wm.eval()
    return wm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--goal", choices=["wall", "center"], default="wall")
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--starts", type=int, default=150)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device; G = args.goal

    eps = load_traj(args.data_dir)
    ntr = int(0.8 * len(eps)); tr, te = eps[:ntr], eps[ntr:]
    EVAL_SEED = 100
    print(f"{len(eps)} episodes ({ntr} train / {len(te)} test), goal={G}, "
          f"H={args.H}, repeats={args.repeats}, {args.starts} fixed eval starts\n")

    def se(v): return np.std(v, ddof=1) / math.sqrt(len(v))
    learned, raw, random = [], [], []
    for r in range(args.repeats):
        wm_l = CM.train_wm(tr, dev, H=args.H, freeze_encoder=False, seed=r)
        wm_w = train_raw_wm(tr, dev, H=args.H, seed=r)
        wm_r = CM.train_wm(tr, dev, H=args.H, freeze_encoder=True, seed=r)
        learned.append(eval_controller("learned", wm_l, te, dev, H=args.H,
                                       n_starts=args.starts, seed=EVAL_SEED, goal=G)[0])
        raw.append(eval_controller("learned", wm_w, te, dev, H=args.H,
                                   n_starts=args.starts, seed=EVAL_SEED, goal=G)[0])
        random.append(eval_controller("random", wm_r, te, dev, H=args.H,
                                      n_starts=args.starts, seed=EVAL_SEED, goal=G)[0])
        print(f"  seed {r}: learned {learned[-1]:+.3f} | raw-enc {raw[-1]:+.3f} | "
              f"random-enc {random[-1]:+.3f}", flush=True)

    metric = "reduction (approach wall)" if G == "wall" else "increase (reach centre)"
    print(f"\nwall-distance {metric} over {args.H} steps, held-out rooms:")
    print(f"  learned-encoder MPC   {np.mean(learned):+.3f} +/- {se(learned):.3f} m")
    print(f"  RAW-encoder MPC       {np.mean(raw):+.3f} +/- {se(raw):.3f} m")
    print(f"  random-encoder MPC    {np.mean(random):+.3f} +/- {se(random):.3f} m")
    dl = np.array(learned) - np.array(raw)
    dr = np.array(raw) - np.array(random)
    print(f"\n  learned - raw   {dl.mean():+.3f} +/- {se(dl):.3f}  "
          f"({'learned encoder helps beyond raw' if dl.mean() > 2*se(dl) else 'raw matches learned'})")
    print(f"  raw - random    {dr.mean():+.3f} +/- {se(dr):.3f}  "
          f"({'raw beats random' if dr.mean() > 2*se(dr) else 'raw ~ random (decodability != plannability)'})")


if __name__ == "__main__":
    main()
