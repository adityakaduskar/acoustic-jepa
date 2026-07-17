"""World-model evaluation + latent MPC (ACOUSTIC_JEPA_SPEC.md §4b).

Uses the frozen Acoustic-JEPA encoder + predictor as an action-conditioned world
model and reports two things, both runnable from the dataset alone:

  (1) Rollout accuracy -- how well the predictor forecasts future observation
      latents over a horizon, vs a persistence ("no dynamics") baseline. This
      validates the predictor as a forward model.

  (2) Latent MPC -- plan action sequences with CEM, using the learned dynamics to
      minimise a distance-to-nearest-wall probe applied to *predicted* future
      latents (the roadmap-#2 "approach the wall" objective). Compares the planned
      approach against random and stay-put actions.

SCOPE: planning is evaluated *in imagination* (against the learned model), not by
executing in live phonon -- closed-loop navigation on real geometry needs a
Python<->phonon bridge (future work). The analytic controller in echo_nav.cpp
(~9 steps to target) is the closed-loop reference to beat once that bridge exists.

Run in the torch venv:  python plan.py runs/jepa/final.pt <data_dir>
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import JEPADataset, ProbeDataset
from model import AcousticJEPA


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    emb = ck.get("cfg", {}).get("emb", 256)
    window = ck.get("cfg", {}).get("window", 1)
    m = AcousticJEPA(emb=emb, window=window).to(device).eval()
    m.load_state_dict(ck["model"])
    return m, emb, window


class Ridge:
    """Standardised closed-form ridge, kept on-device for use inside CEM."""

    def fit(self, X, y, lam=1.0):
        self.mu = X.mean(0, keepdim=True)
        self.sd = X.std(0, keepdim=True) + 1e-6
        Xs = (X - self.mu) / self.sd
        self.ybar = y.mean()
        d = Xs.shape[1]
        I = torch.eye(d, device=X.device)
        self.W = torch.linalg.solve(Xs.T @ Xs + lam * I, Xs.T @ (y - self.ybar).unsqueeze(1))
        return self

    def predict(self, X):
        return (((X - self.mu) / self.sd) @ self.W).squeeze(1) + self.ybar


# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout_accuracy(model, loader, device):
    """Per-step predictor error vs persistence, measured in target-latent space."""
    pred_se, persist_se, n = None, None, 0
    W = model.window
    for coch, act, _ in loader:
        coch = coch.to(device); act = act.to(device)
        T = coch.shape[1]
        H = T - W
        z0 = model.encoder(coch[:, 0:W])                            # online context latent (window)
        ztrue = torch.stack([model.target(coch[:, k:k + W]) for k in range(1, H + 1)], 1)  # [B,H,emb]
        z = z0
        pe = torch.zeros(H, device=device)
        qe = torch.zeros(H, device=device)
        for k in range(H):
            z = model.predictor(z, act[:, W - 1 + k])
            pe[k] = (z - ztrue[:, k]).pow(2).sum(1).mean()          # predictor error
            qe[k] = (z0 - ztrue[:, k]).pow(2).sum(1).mean()         # persistence error
        B = coch.shape[0]
        pred_se = pe * B if pred_se is None else pred_se + pe * B
        persist_se = qe * B if persist_se is None else persist_se + qe * B
        n += B
    pred = (pred_se / n).cpu().numpy()
    persist = (persist_se / n).cpu().numpy()
    skill = 1.0 - pred / (persist + 1e-9)                           # >0 => beats persistence
    return pred, persist, skill


# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_probe_data(model, data_dir, device, batch, workers):
    """Encoder latents + nearest-wall labels over all observations (for the probe)."""
    dl = DataLoader(ProbeDataset(data_dir, model.window), batch_size=batch, shuffle=False, num_workers=workers)
    Z, near = [], []
    for coch, lab in dl:
        Z.append(model.encoder(coch.to(device)))
        near.append(lab["dist_wall"].min(dim=1).values)            # nearest wall (m)
    return torch.cat(Z), torch.cat(near).to(device)


@torch.no_grad()
def cem_plan(model, z0, g, L, iters, pop, elite, device):
    """CEM over action sequences to minimise predicted nearest-wall distance at step L."""
    mu = torch.zeros(L, 2, device=device)
    sd = torch.ones(L, 2, device=device)
    best_actions = None
    for _ in range(iters):
        a = (mu + sd * torch.randn(pop, L, 2, device=device)).clamp(-1, 1)   # [pop,L,2]
        z = z0.expand(pop, -1).clone()
        for l in range(L):
            z = model.predictor(z, a[:, l])
        cost = g.predict(z)                                        # predicted nearest-wall dist
        idx = torch.topk(-cost, elite).indices
        mu = a[idx].mean(0); sd = a[idx].std(0) + 1e-3
        best_actions = a[idx][0]
    # evaluate the mean plan
    z = z0.unsqueeze(0).clone()
    for l in range(L):
        z = model.predictor(z, mu[l:l + 1])
    return float(g.predict(z)[0]), best_actions


@torch.no_grad()
def eval_mpc(model, data_dir, device, args):
    Z, near = embed_probe_data(model, data_dir, device, args.batch, args.num_workers)
    # fit nearest-wall probe on the first half, plan from the second half's starts
    ntr = Z.shape[0] // 2
    g = Ridge().fit(Z[:ntr], near[:ntr], lam=args.lam)
    starts = torch.randperm(Z.shape[0] - ntr, device=device)[:args.mpc_starts] + ntr

    L = args.horizon
    d_start, d_plan, d_rand, d_stay = [], [], [], []
    stay = torch.tensor([0.0, -1.0], device=device)                # dyaw=0, dstep=0 (normalised)
    for si in starts.tolist():
        z0 = Z[si]
        d_start.append(float(g.predict(z0.unsqueeze(0))[0]))
        # plan
        dp, _ = cem_plan(model, z0, g, L, args.cem_iters, args.cem_pop, args.cem_elite, device)
        d_plan.append(dp)
        # random-action baseline (best of cem_pop random sequences)
        a = (torch.rand(args.cem_pop, L, 2, device=device) * 2 - 1)
        z = z0.expand(args.cem_pop, -1).clone()
        for l in range(L):
            z = model.predictor(z, a[:, l])
        d_rand.append(float(g.predict(z).min()))
        # stay-put baseline
        z = z0.unsqueeze(0).clone()
        for _ in range(L):
            z = model.predictor(z, stay.unsqueeze(0))
        d_stay.append(float(g.predict(z)[0]))
    return {
        "start_dist_m": float(np.mean(d_start)),
        "plan_dist_m": float(np.mean(d_plan)),
        "random_dist_m": float(np.mean(d_rand)),
        "stay_dist_m": float(np.mean(d_stay)),
        "plan_reduction_m": float(np.mean(d_start) - np.mean(d_plan)),
    }


def main():
    ap = argparse.ArgumentParser(description="world-model rollout + latent MPC")
    ap.add_argument("ckpt")
    ap.add_argument("data_dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mpc-starts", type=int, default=32)
    ap.add_argument("--cem-iters", type=int, default=4)
    ap.add_argument("--cem-pop", type=int, default=256)
    ap.add_argument("--cem-elite", type=int, default=32)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, emb, window = load_model(args.ckpt, device)

    loader = DataLoader(JEPADataset(args.data_dir, horizon=args.horizon, window=window),
                        batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    pred, persist, skill = rollout_accuracy(model, loader, device)
    print(f"world model (emb={emb}, window={window}) rollout latent MSE vs persistence:")
    print(f"  {'step':>4} {'pred':>10} {'persist':>10} {'skill':>7}")
    for k in range(len(pred)):
        print(f"  {k+1:>4} {pred[k]:>10.3f} {persist[k]:>10.3f} {skill[k]:>7.2f}")

    mpc = eval_mpc(model, args.data_dir, device, args)
    print("\nlatent MPC -- approach nearest wall (imagined, mean over "
          f"{args.mpc_starts} starts):")
    print(f"  start   {mpc['start_dist_m']:.2f} m")
    print(f"  planned {mpc['plan_dist_m']:.2f} m   (reduction {mpc['plan_reduction_m']:.2f} m)")
    print(f"  random  {mpc['random_dist_m']:.2f} m")
    print(f"  stay    {mpc['stay_dist_m']:.2f} m")
    print("  [imagined vs the learned model; echo_nav.cpp ~9 steps is the live-geometry reference]")

    res = {"rollout_pred_mse": pred.tolist(), "rollout_persist_mse": persist.tolist(),
           "rollout_skill": skill.tolist(), "mpc": mpc}
    out = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "plan_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
