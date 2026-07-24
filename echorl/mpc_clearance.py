"""EchoRL: clearance-aware MPC — fix Problem A (the greedy distance-only objective).

The §5 MPC minimises distance-to-goal ONLY, so in clutter it drives straight at the
goal, wedges behind pillars, and inherits the privileged GREEDY oracle's 0.54 ceiling
(landing below it, ~0.24, from model error). This adds an obstacle-CLEARANCE readout to
the world model and a clearance penalty to the CEM cost, so the planner can *see* pillars
and route around them — attacking the objective half of the 1.00 → 0.24 gap.

  plan cost = Σ_k dist(z_k)  +  w_pen · Σ_k relu(d_safe − clearance(z_k))

The penalty only fires within d_safe of a surface, so it forces detours around pillars
without fighting the near-wall final approach (goal clearance ~1.2 m > d_safe). Because
w_pen lives only in the COST (not the model), one trained world model can be evaluated at
several w_pen values with no retraining — an A/B where w_pen=0 reproduces the §5 MPC.

Run: python echorl/mpc_clearance.py --nobs 3 --train-ep 360 --eval-rooms 12
"""
from __future__ import annotations
import argparse, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from env import (EchoLocEnv, _wall_clearance, _obstacle_clearance,          # noqa: E402
                 DYAW_MAX, DSTEP_MAX)
from mpc import load_corpus, save_corpus, raw_to_norm, oracle_action, COCH_SHAPE  # noqa: E402
from mpc_direct import eval_policy                                          # noqa: E402
from models import Normalizer                                              # noqa: E402
from model_xview import TwoStreamEncoder                                   # noqa: E402
from model import Predictor, vicreg_var_cov                                # noqa: E402
from ppo import opening_seeds                                              # noqa: E402


# ------------------------------------------------------------------ world model
class WorldModelC(nn.Module):
    """Like mpc.WorldModel but with TWO readouts: distance-to-goal AND clearance."""
    def __init__(self, emb=256, ema=0.99):
        super().__init__()
        import copy
        self.norm = Normalizer()
        self.encoder = TwoStreamEncoder(emb // 2, emb // 2)
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(emb, action_dim=2)
        self.dist_readout = nn.Sequential(nn.Linear(emb, 128), nn.GELU(), nn.Linear(128, 1))
        self.clear_readout = nn.Sequential(nn.Linear(emb, 128), nn.GELU(), nn.Linear(128, 1))
        self.ema = ema

    def enc(self, rng, coch):
        rn, cn = self.norm(rng, coch)
        zr, zc = self.encoder(cn, rn)
        return torch.cat([zr, zc], 1)

    @torch.no_grad()
    def enc_target(self, rng, coch):
        rn, cn = self.norm(rng, coch)
        zr, zc = self.target(cn, rn)
        return torch.cat([zr, zc], 1)

    @torch.no_grad()
    def update_target(self):
        for p, tp in zip(self.encoder.parameters(), self.target.parameters()):
            tp.mul_(self.ema).add_(p, alpha=1 - self.ema)
        for b, tb in zip(self.encoder.buffers(), self.target.buffers()):
            tb.copy_(b)

    def rollout(self, z0, acts_n):
        zs = []; z = z0
        for k in range(acts_n.shape[1]):
            z = self.predictor(z, acts_n[:, k]); zs.append(z)
        return torch.stack(zs, 1)


# ------------------------------------------------------------------ data + train
def _clearance(x, z, srv):
    return min(_wall_clearance(x, z, srv.Lx, srv.Lz), _obstacle_clearance(x, z, srv.obstacles))


def collect_traj_c(seeds, n_ep, nobs=3, t_max=40, base_seed=7, forward_bias=0.6, workers=12):
    """Random exploratory rollouts logging (range, coch, act[-1,1], dist, CLEARANCE)."""
    workers = min(workers, max(1, len(seeds)))
    per = [n_ep // workers + (1 if i < n_ep % workers else 0) for i in range(workers)]
    shards = [seeds[i::workers] or [seeds[i % len(seeds)]] for i in range(workers)]

    def worker(wi):
        env = EchoLocEnv(seed_pool=shards[wi], action_mode="continuous", nobs=nobs, sim_threads=2,
                         t_max=t_max, resample_room_every=1, task="goal", base_seed=base_seed * 1000 + wi)
        rng = np.random.default_rng(base_seed * 1000 + wi); out = []
        for _ in range(per[wi]):
            obs, info = env.reset(); gx, gz = info["goal"]
            r_ep, c_ep, a_ep, d_ep, cl_ep = [], [], [], [], []
            done = False
            while not done:
                dyaw = rng.uniform(-DYAW_MAX, DYAW_MAX)
                dstep = rng.uniform(0, DSTEP_MAX) * (1.0 if rng.random() < forward_bias else 0.4)
                an = np.array(raw_to_norm(dyaw, dstep), np.float32)
                r_ep.append(obs["range"]); c_ep.append(obs["coch"]); a_ep.append(an)
                px, pz, _ = info["pose"]
                d_ep.append(math.hypot(px - gx, pz - gz)); cl_ep.append(_clearance(px, pz, env.srv))
                obs, _, term, trunc, info = env.step(an); done = term or trunc
            px, pz, _ = info["pose"]
            d_ep.append(math.hypot(px - gx, pz - gz)); cl_ep.append(_clearance(px, pz, env.srv))
            out.append((np.array(r_ep, np.float32), np.array(c_ep, np.float32),
                        np.array(a_ep, np.float32), np.array(d_ep, np.float32), np.array(cl_ep, np.float32)))
        env.close(); return out
    with ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(worker, range(workers)))
    return [ep for r in res for ep in r]


def build_windows_c(eps, H):
    R, C, A, DD, CL = [], [], [], [], []
    for (r, c, a, d, cl) in eps:
        K = a.shape[0]
        for s in range(0, K - H):
            R.append(r[s:s + H + 1]); C.append(c[s:s + H + 1]); A.append(a[s:s + H])
            DD.append(d[s:s + H + 1]); CL.append(cl[s:s + H + 1])
    return (np.array(R, np.float32), np.array(C, np.float32), np.array(A, np.float32),
            np.array(DD, np.float32), np.array(CL, np.float32))


def _r2(pred, targ):
    ss = float(((pred - targ) ** 2).sum()); st = float(((targ - targ.mean()) ** 2).sum() + 1e-9)
    return 1.0 - ss / st


def train_wm_c(eps, dev, H=6, epochs=30, bs=128, seed=0):
    torch.manual_seed(seed)
    R, C, A, DD, CL = build_windows_c(eps, H)
    n = len(R); perm = np.random.default_rng(seed).permutation(n)
    ntr = int(0.85 * n); tr, va = perm[:ntr], perm[ntr:]
    wm = WorldModelC().to(dev)
    opt = torch.optim.AdamW([p for p in wm.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        idx = tr.copy(); np.random.default_rng(ep).shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            rng = torch.tensor(R[b], device=dev); coch = torch.tensor(C[b], device=dev)
            act = torch.tensor(A[b], device=dev); dist = torch.tensor(DD[b], device=dev)
            clr = torch.tensor(CL[b], device=dev)
            Bn, T = rng.shape[0], rng.shape[1]
            z = wm.enc(rng.reshape(Bn * T, 256), coch.reshape(Bn * T, *COCH_SHAPE)).reshape(Bn, T, -1)
            with torch.no_grad():
                zt = wm.enc_target(rng.reshape(Bn * T, 256),
                                   coch.reshape(Bn * T, *COCH_SHAPE)).reshape(Bn, T, -1)
            pred = wm.rollout(z[:, 0], act)
            dyn = ((pred - zt[:, 1:]) ** 2).mean()
            d_hat = wm.dist_readout(z.reshape(Bn * T, -1)).reshape(Bn, T)
            c_hat = wm.clear_readout(z.reshape(Bn * T, -1)).reshape(Bn, T)
            ro = ((d_hat - dist) ** 2).mean() + ((c_hat - clr) ** 2).mean()
            vr, cv = vicreg_var_cov(z[:, 0])
            loss = dyn + ro + 1.0 * vr + 0.04 * cv
            opt.zero_grad(); loss.backward(); opt.step()
            wm.update_target()
    wm.eval()
    # diagnostic on held-out windows: readout R^2 for dist & clearance (encoded, step 0)
    with torch.no_grad():
        rng = torch.tensor(R[va, 0], device=dev); coch = torch.tensor(C[va, 0], device=dev)
        z0 = wm.enc(rng, coch)
        dp = wm.dist_readout(z0).squeeze(-1).cpu().numpy(); dt = DD[va, 0]
        cp = wm.clear_readout(z0).squeeze(-1).cpu().numpy(); ct = CL[va, 0]
    wm.diag = dict(dist_r2=_r2(dp, dt), clear_r2=_r2(cp, ct))
    return wm


# ------------------------------------------------------------------ planner
@torch.no_grad()
def cem_plan_c(wm, z0, H, dev, w_pen, d_safe, iters=3, N=256, elite=32):
    """Minimise Σ dist + w_pen·Σ relu(d_safe − clearance) over the latent rollout."""
    mu = torch.zeros(H, 2, device=dev); sd = torch.tensor([[30.0, 0.6]], device=dev).repeat(H, 1)
    z = z0.expand(N, -1)
    for _ in range(iters):
        eps = torch.randn(N, H, 2, device=dev) * sd + mu
        dyaw = eps[..., 0].clamp(-DYAW_MAX, DYAW_MAX); dstep = eps[..., 1].clamp(0, DSTEP_MAX)
        an = torch.stack([dyaw / DYAW_MAX, dstep / (0.5 * DSTEP_MAX) - 1.0], -1)
        preds = wm.rollout(z, an)                                       # [N,H,emb]
        flat = preds.reshape(N * H, -1)
        dist = wm.dist_readout(flat).reshape(N, H)
        clr = wm.clear_readout(flat).reshape(N, H)
        cost = dist.sum(1) + w_pen * torch.relu(d_safe - clr).sum(1)
        top = cost.argsort()[:elite]
        chosen = torch.stack([dyaw, dstep], -1)[top]
        mu = chosen.mean(0); sd = chosen.std(0) + 1e-3
    return np.array(raw_to_norm(float(mu[0, 0]), float(mu[0, 1])), np.float32)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nobs", type=int, default=3)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--train-rooms", type=int, default=60)
    ap.add_argument("--train-ep", type=int, default=360)
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--episodes-per", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--d-safe", type=float, default=0.6)
    ap.add_argument("--w-pens", type=str, default="0,2,5,10", help="comma list of clearance weights to sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus-in", default=None)
    ap.add_argument("--corpus-out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train_seeds = opening_seeds(1, args.train_rooms, args.nobs)
    eval_seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    print(f"device={dev} | clearance-MPC | nobs={args.nobs} H={args.H} d_safe={args.d_safe} | "
          f"{len(train_seeds)} train / {len(eval_seeds)} eval rooms")

    t0 = time.time()
    if args.corpus_in and os.path.exists(args.corpus_in):
        eps = load_corpus(args.corpus_in); print(f"loaded corpus {os.path.relpath(args.corpus_in)}: {len(eps)} eps")
    else:
        print(f"collecting {args.train_ep} random trajectories (w/ clearance labels)...")
        eps = collect_traj_c(train_seeds, args.train_ep, nobs=args.nobs)
        if args.corpus_out:
            save_corpus(args.corpus_out, eps); print(f"  saved -> {os.path.relpath(args.corpus_out)}")
    print(f"  {len(eps)} episodes ({time.time()-t0:.0f}s)")

    wm = train_wm_c(eps, dev, H=args.H, epochs=args.epochs, seed=args.seed)
    print(f"  WM diagnostic: dist readout-R2 {wm.diag['dist_r2']:+.2f} | "
          f"clearance readout-R2 {wm.diag['clear_r2']:+.2f}")

    # greedy privileged oracle for reference (the ceiling a distance-only objective hits)
    orc = eval_policy(lambda o, i: oracle_action(*i["pose"], i["goal"]), eval_seeds, dev,
                      episodes_per=args.episodes_per, nobs=args.nobs)
    print(f"greedy oracle (privileged): {orc:.2f}")

    print("sweeping clearance weight w_pen (same WM, cost-only change; w_pen=0 == §5 MPC):")
    results = {}
    for w in [float(x) for x in args.w_pens.split(",")]:
        def plan(o, i, w=w):
            z = wm.enc(torch.as_tensor(o["range"][None], device=dev), torch.as_tensor(o["coch"][None], device=dev))
            return cem_plan_c(wm, z, args.H, dev, w, args.d_safe)
        sr = eval_policy(plan, eval_seeds, dev, episodes_per=args.episodes_per, nobs=args.nobs)
        results[w] = sr
        tag = "  (== §5 distance-only MPC)" if w == 0 else ""
        print(f"  w_pen={w:5.1f}  success {sr:.2f}{tag}")

    print(f"\n=== clearance-MPC (nobs={args.nobs}), {len(eval_seeds)} rooms x {args.episodes_per} ===")
    print(f"  greedy oracle           {orc:.2f}   (distance-only objective ceiling)")
    for w, sr in results.items():
        print(f"  clearance-MPC w={w:<4} {sr:.2f}")
    print(f"  (§5 learned-WM MPC baseline was ~0.25)")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
