"""EchoRL: uncertainty-penalised ENSEMBLE MPC — the fix three experiments converge on.

§5.2 (objective mismatch), §5.5 (DAgger fails), §5.6 (clearance cost fails) all point at
one bottleneck: **CEM exploits the dynamics model's confident-but-wrong predictions off the
training distribution.** The principled fix (PETS-style): train an ENSEMBLE of dynamics
predictors (shared encoder — the encoder is the good part, readout R²≈0.98), and penalise
action sequences where the ensemble DISAGREES, so the planner stays where the model is
reliable instead of exploiting its blind spots.

  plan cost(a) = mean_k Σ dist(pred_k(z,a))  +  λ · std_k( Σ dist(pred_k(z,a)) )

λ=0 recovers an averaged-model MPC (≈ §5); λ>0 is the trust-region. Because λ lives only in
the cost, one trained ensemble is swept over λ with no retraining.

Run: python echorl/mpc_ensemble.py --nobs 3 --corpus-in echorl/corpus_goal_nobs3_clear.pkl --lams 0,1,3
"""
from __future__ import annotations
import argparse, os, sys, time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from env import DYAW_MAX, DSTEP_MAX                                        # noqa: E402
from mpc import load_corpus, build_windows, raw_to_norm, oracle_action, COCH_SHAPE  # noqa: E402
from mpc_direct import eval_policy                                        # noqa: E402
from models import Normalizer                                            # noqa: E402
from model_xview import TwoStreamEncoder                                 # noqa: E402
from model import Predictor, vicreg_var_cov                              # noqa: E402
from ppo import opening_seeds                                            # noqa: E402


class WorldModelE(nn.Module):
    """Shared encoder + readout, ENSEMBLE of K action-conditioned latent predictors."""
    def __init__(self, emb=256, K=5, ema=0.99):
        super().__init__()
        import copy
        self.norm = Normalizer()
        self.encoder = TwoStreamEncoder(emb // 2, emb // 2)
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictors = nn.ModuleList([Predictor(emb, action_dim=2) for _ in range(K)])
        self.readout = nn.Sequential(nn.Linear(emb, 128), nn.GELU(), nn.Linear(128, 1))
        self.K, self.ema = K, ema

    def enc(self, rng, coch):
        rn, cn = self.norm(rng, coch); zr, zc = self.encoder(cn, rn); return torch.cat([zr, zc], 1)

    @torch.no_grad()
    def enc_target(self, rng, coch):
        rn, cn = self.norm(rng, coch); zr, zc = self.target(cn, rn); return torch.cat([zr, zc], 1)

    @torch.no_grad()
    def update_target(self):
        for p, tp in zip(self.encoder.parameters(), self.target.parameters()):
            tp.mul_(self.ema).add_(p, alpha=1 - self.ema)
        for b, tb in zip(self.encoder.buffers(), self.target.buffers()):
            tb.copy_(b)

    def rollout_k(self, z0, acts_n, k):
        z = z0; zs = []
        for j in range(acts_n.shape[1]):
            z = self.predictors[k](z, acts_n[:, j]); zs.append(z)
        return torch.stack(zs, 1)


def _r2(pred, targ):
    ss = float(((pred - targ) ** 2).sum()); st = float(((targ - targ.mean()) ** 2).sum() + 1e-9)
    return 1.0 - ss / st


def train_wm_e(eps, dev, H=6, epochs=30, bs=128, K=5, seed=0):
    torch.manual_seed(seed)
    R, C, A, DD = build_windows(eps, H)
    n = len(R); perm = np.random.default_rng(seed).permutation(n)
    ntr = int(0.85 * n); tr, va = perm[:ntr], perm[ntr:]
    wm = WorldModelE(K=K).to(dev)
    opt = torch.optim.AdamW([p for p in wm.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device=dev)
    for ep in range(epochs):
        idx = tr.copy(); np.random.default_rng(ep).shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            rng = torch.tensor(R[b], device=dev); coch = torch.tensor(C[b], device=dev)
            act = torch.tensor(A[b], device=dev); dist = torch.tensor(DD[b], device=dev)
            Bn, T = rng.shape[0], rng.shape[1]
            z = wm.enc(rng.reshape(Bn * T, 256), coch.reshape(Bn * T, *COCH_SHAPE)).reshape(Bn, T, -1)
            with torch.no_grad():
                zt = wm.enc_target(rng.reshape(Bn * T, 256),
                                   coch.reshape(Bn * T, *COCH_SHAPE)).reshape(Bn, T, -1)
            # each predictor on a BOOTSTRAP resample of the minibatch (diversity for disagreement)
            dyn = 0.0
            for k in range(wm.K):
                boot = torch.randint(0, Bn, (Bn,), device=dev)
                pred = wm.rollout_k(z[boot, 0], act[boot], k)
                dyn = dyn + ((pred - zt[boot, 1:]) ** 2).mean()
            dyn = dyn / wm.K
            ro = ((wm.readout(z.reshape(Bn * T, -1)).reshape(Bn, T) - dist) ** 2).mean()
            vr, cv = vicreg_var_cov(z[:, 0])
            loss = dyn + ro + 1.0 * vr + 0.04 * cv
            opt.zero_grad(); loss.backward(); opt.step()
            wm.update_target()
    wm.eval()
    # diagnostics on held-out windows
    with torch.no_grad():
        rng = torch.tensor(R[va], device=dev); coch = torch.tensor(C[va], device=dev)
        act = torch.tensor(A[va], device=dev); dist = torch.tensor(DD[va], device=dev)
        Bn, T = rng.shape[0], rng.shape[1]
        z0 = wm.enc(rng[:, 0], coch[:, 0])
        d_r2 = _r2(wm.readout(z0).squeeze(-1).cpu().numpy(), DD[va, 0])
        # per-predictor final-distance prediction + ensemble disagreement
        finals = []
        for k in range(wm.K):
            pr = wm.rollout_k(z0, act, k)                     # [B,H,emb]
            finals.append(wm.readout(pr[:, -1]).squeeze(-1))
        finals = torch.stack(finals)                          # [K,B]
        disagree = float(finals.std(0).mean())                # mean ensemble std of final-dist (m)
        ens_mae = float((finals.mean(0) - dist[:, -1]).abs().mean())
    wm.diag = dict(readout_r2=d_r2, ens_disagree=disagree, ens_rollout_mae=ens_mae)
    return wm


@torch.no_grad()
def cem_plan_e(wm, z0, H, dev, lam, iters=3, N=256, elite=32):
    """CEM minimising mean_k Σdist + λ·std_k(Σdist) over the ensemble rollout."""
    mu = torch.zeros(H, 2, device=dev); sd = torch.tensor([[30.0, 0.6]], device=dev).repeat(H, 1)
    z = z0.expand(N, -1)
    for _ in range(iters):
        eps = torch.randn(N, H, 2, device=dev) * sd + mu
        dyaw = eps[..., 0].clamp(-DYAW_MAX, DYAW_MAX); dstep = eps[..., 1].clamp(0, DSTEP_MAX)
        an = torch.stack([dyaw / DYAW_MAX, dstep / (0.5 * DSTEP_MAX) - 1.0], -1)
        costs = []
        for k in range(wm.K):
            preds = wm.rollout_k(z, an, k)                    # [N,H,emb]
            costs.append(wm.readout(preds.reshape(N * H, -1)).reshape(N, H).sum(1))
        costs = torch.stack(costs)                            # [K,N]
        total = costs.mean(0) + lam * costs.std(0)            # mean cost + disagreement penalty
        top = total.argsort()[:elite]
        chosen = torch.stack([dyaw, dstep], -1)[top]
        mu = chosen.mean(0); sd = chosen.std(0) + 1e-3
    return np.array(raw_to_norm(float(mu[0, 0]), float(mu[0, 1])), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nobs", type=int, default=3)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--train-rooms", type=int, default=60)
    ap.add_argument("--train-ep", type=int, default=360)
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--episodes-per", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lams", type=str, default="0,1,3", help="disagreement-penalty weights to sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus-in", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    eval_seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    print(f"device={dev} | ensemble-MPC K={args.K} | nobs={args.nobs} H={args.H} | "
          f"{len(eval_seeds)} eval rooms")

    t0 = time.time()
    if args.corpus_in and os.path.exists(args.corpus_in):
        raw = load_corpus(args.corpus_in)
        eps = [tuple(e[:4]) for e in raw]                    # clearance corpus -> (r,c,a,d)
        print(f"loaded corpus {os.path.relpath(args.corpus_in)}: {len(eps)} eps")
    else:
        from mpc import collect_traj
        train_seeds = opening_seeds(1, args.train_rooms, args.nobs)
        print(f"collecting {args.train_ep} random trajectories...")
        eps = collect_traj(train_seeds, args.train_ep, dev, nobs=args.nobs, base_seed=7)
    print(f"  {len(eps)} episodes ({time.time()-t0:.0f}s)")

    wm = train_wm_e(eps, dev, H=args.H, epochs=args.epochs, K=args.K, seed=args.seed)
    print(f"  WM-E diagnostic: readout-R2 {wm.diag['readout_r2']:+.2f} | "
          f"ensemble disagreement {wm.diag['ens_disagree']:.2f} m | "
          f"ensemble rollout-MAE {wm.diag['ens_rollout_mae']:.2f} m")
    if wm.diag['ens_disagree'] < 0.05:
        print("  !! low disagreement -> uncertainty penalty may have little to bite on")

    orc = eval_policy(lambda o, i: oracle_action(*i["pose"], i["goal"]), eval_seeds, dev,
                      episodes_per=args.episodes_per, nobs=args.nobs)
    print(f"greedy oracle (privileged): {orc:.2f}")

    print("sweeping disagreement weight λ (same ensemble, cost-only; λ=0 == averaged-model MPC):")
    results = {}
    for lam in [float(x) for x in args.lams.split(",")]:
        def plan(o, i, lam=lam):
            z = wm.enc(torch.as_tensor(o["range"][None], device=dev), torch.as_tensor(o["coch"][None], device=dev))
            return cem_plan_e(wm, z, args.H, dev, lam)
        sr = eval_policy(plan, eval_seeds, dev, episodes_per=args.episodes_per, nobs=args.nobs)
        results[lam] = sr
        tag = "  (averaged-model, no trust region)" if lam == 0 else ""
        print(f"  λ={lam:5.1f}  success {sr:.2f}{tag}")

    print(f"\n=== ensemble-MPC (nobs={args.nobs}), {len(eval_seeds)} rooms x {args.episodes_per} ===")
    print(f"  greedy oracle            {orc:.2f}")
    for lam, sr in results.items():
        print(f"  ensemble-MPC λ={lam:<4}  {sr:.2f}")
    print(f"  (§5 single-model MPC baseline ~0.25)")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
