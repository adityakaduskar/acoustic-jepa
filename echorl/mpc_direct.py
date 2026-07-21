"""EchoRL Experiment C (H2 branch) -- the informed MPC rematch.

Experiment A (probe_dd.py, RESULTS.md 5.1) diagnosed the learned-WM MPC gap as H2:
the action-consequence info survives into the latent (z+a predicts one-step Delta-d at
R^2=0.42) but the trained compositional predictor path readout(pred(z,a))-readout(z)
delivers only 0.24 -- CEM plans on a near-blind forward model.

Two design facts drive the fix:
  * the cue is myopic (single-ping Delta-d caps at R^2~0.42), and
  * the env is STEP-THEN-TURN (translation uses the CURRENT yaw, then yaw+=dyaw), so a
    dyaw only pays off on LATER steps -> a 1-step consequence head can't value turning.
So the fix is a MULTI-STEP OUTCOME HEAD g_H(z0, a_1..H) -> d(t+H)-d(t): one network maps
the initial latent + the whole action sequence to the resulting distance change, learned
directly (no latent rollout to go wrong). CEM optimises the sequence against g_H; execute
the first action; re-observe (closed-loop). This replaces the broken readout-of-rollout
with the thing the probe showed is learnable.

Controllers, all closed-loop through the SAME EchoLocEnv as PPO / 5:
  * oracle            -- privileged greedy to the true goal (ceiling);
  * learned-WM MPC    -- the 5 controller (compositional rollout), re-run here for parity;
  * direct-outcome    -- the fix (g_H sequence-outcome planner).
Reported vs model-free PPO ~0.85.

Run: python echorl/mpc_direct.py --corpus echorl/corpus_goal.pkl --wm echorl/ckpt/mpc_wm.pt
"""
from __future__ import annotations
import argparse, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from env import EchoLocEnv, DYAW_MAX, DSTEP_MAX               # noqa: E402
from mpc import (WorldModel, load_corpus, build_windows, cem_plan, oracle_action,   # noqa: E402
                 raw_to_norm, COCH_SHAPE)
from ppo import opening_seeds                                 # noqa: E402


# ------------------------------------------------------ multi-step outcome head
class OutcomeHead(nn.Module):
    """g_H(z0[256], a_1..H flattened[2H]) -> predicted d(t+H)-d(t)."""
    def __init__(self, emb=256, H=6, hidden=256, p=0.1):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(nn.Linear(emb + 2 * H, hidden), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p),
                                 nn.Linear(hidden, 1))

    def forward(self, z0, aseq):                              # aseq [B,H,2]
        return self.net(torch.cat([z0, aseq.reshape(aseq.shape[0], -1)], 1)).squeeze(-1)


def train_outcome_head(wm, eps, dev, H=6, epochs=40, bs=256, seed=0):
    """Train g_H on the FROZEN learned encoder (isolates the H2 fix: same perception,
    a direct outcome model instead of the compositional rollout). Target = CUMULATIVE
    distance sum_{k=1..H} d(t+k) -- the EXACT cost the compositional CEM minimises (sum
    of readout over the H rolled latents), so the A/B swaps ONLY compositional-rollout
    vs direct-outcome, not the objective. Reports held-out R^2 on that cumulative cost."""
    torch.manual_seed(seed)
    R, C, A, DD = build_windows(eps, H)                      # A[N,H,2], DD[N,H+1]
    tgt = DD[:, 1:].sum(1).astype(np.float32)                # cumulative future distance
    n = len(R); perm = np.random.default_rng(seed).permutation(n)
    ntr = int(0.85 * n); tr, va = perm[:ntr], perm[ntr:]
    # precompute z0 from the frozen encoder (no grad through it)
    with torch.no_grad():
        Z0 = np.empty((n, 256), np.float32)
        for i in range(0, n, 512):
            r_ = torch.tensor(R[i:i+512, 0], device=dev); c_ = torch.tensor(C[i:i+512, 0], device=dev)
            Z0[i:i+512] = wm.enc(r_, c_).cpu().numpy()
    Z0 = torch.tensor(Z0, device=dev); At = torch.tensor(A, device=dev); Yt = torch.tensor(tgt, device=dev)
    g = OutcomeHead(H=H).to(dev)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3, weight_decay=1e-3)
    best_r2, best_state = -1e9, None
    for ep in range(epochs):
        g.train(); idx = tr.copy(); np.random.default_rng(ep).shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i+bs]
            loss = ((g(Z0[b], At[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        g.eval()
        with torch.no_grad():
            pred = g(Z0[va], At[va])
            ssr = float(((pred - Yt[va]) ** 2).sum()); sst = float(((Yt[va] - Yt[va].mean()) ** 2).sum() + 1e-9)
            r2 = 1.0 - ssr / sst
        if r2 > best_r2:
            best_r2 = r2; best_state = {k: v.clone() for k, v in g.state_dict().items()}
    g.load_state_dict(best_state); g.eval()
    g.val_r2 = best_r2
    return g


# --------------------------------------------------------------- planners
@torch.no_grad()
def cem_outcome_plan(g, z0, H, dev, iters=3, N=256, elite=32):
    """CEM over action SEQUENCES against the outcome head (minimise terminal Delta-d =
    maximise distance reduction). Returns the first action in env [-1,1] convention."""
    mu = torch.zeros(H, 2, device=dev); sd = torch.tensor([[30.0, 0.6]], device=dev).repeat(H, 1)
    z = z0.expand(N, -1)
    for _ in range(iters):
        eps = torch.randn(N, H, 2, device=dev) * sd + mu
        dyaw = eps[..., 0].clamp(-DYAW_MAX, DYAW_MAX); dstep = eps[..., 1].clamp(0, DSTEP_MAX)
        an = torch.stack([dyaw / DYAW_MAX, dstep / (0.5 * DSTEP_MAX) - 1.0], -1)
        cost = g(z, an)                                       # predicted terminal Delta-d
        top = cost.argsort()[:elite]
        chosen = torch.stack([dyaw, dstep], -1)[top]
        mu = chosen.mean(0); sd = chosen.std(0) + 1e-3
    return np.array(raw_to_norm(float(mu[0, 0]), float(mu[0, 1])), np.float32)


# --------------------------------------------------------------- closed-loop eval
@torch.no_grad()
def eval_policy(plan_fn, seeds, dev, episodes_per=10, nobs=0, t_max=40, base_seed=100):
    """Run any plan_fn(obs, info)->action closed-loop through EchoLocEnv, concurrently
    over rooms (same harness/metric as ppo.evaluate and mpc.eval_controller)."""
    def eval_room(s):
        env = EchoLocEnv(seed_pool=[s], action_mode="continuous", nobs=nobs, sim_threads=2,
                         t_max=t_max, resample_room_every=0, task="goal", base_seed=base_seed + s)
        rs = []
        for _ in range(episodes_per):
            obs, info = env.reset(); done = False; ok = False
            goal = info["goal"]                              # only reset-info carries it
            while not done:
                info["goal"] = goal                          # step-info lacks it; keep it available
                a = plan_fn(obs, info)
                obs, _, term, trunc, info = env.step(a)
                ok = bool(info.get("success")); done = term or trunc
            rs.append(1.0 if ok else 0.0)
        env.close()
        return rs
    with ThreadPoolExecutor(max_workers=min(len(seeds), 12)) as pool:
        res = list(pool.map(eval_room, seeds))
    return float(np.mean([v for r in res for v in r]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--episodes-per", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3, help="outcome-head training seeds (CIs)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    eps = load_corpus(args.corpus)
    ck = torch.load(args.wm, map_location=dev)
    wm = WorldModel().to(dev); wm.load_state_dict(ck["model"]); wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    eval_seeds = opening_seeds(600, args.eval_rooms, 0)
    print(f"device={dev} | Exp C (direct-outcome MPC) | H={args.H} | {len(eval_seeds)} held-out rooms "
          f"x {args.episodes_per} starts | {len(eps)} corpus episodes")

    t0 = time.time()
    # baselines (seed-independent): oracle + the 5 learned-WM MPC, same harness
    orc = eval_policy(lambda o, i: oracle_action(*i["pose"], i["goal"]), eval_seeds, dev,
                      episodes_per=args.episodes_per)
    def learned_plan(o, i):
        z = wm.enc(torch.as_tensor(o["range"][None], device=dev), torch.as_tensor(o["coch"][None], device=dev))
        return cem_plan(wm, z, args.H, dev)
    lrn = eval_policy(learned_plan, eval_seeds, dev, episodes_per=args.episodes_per)
    print(f"oracle {orc:.2f} | learned-WM MPC (compositional rollout) {lrn:.2f}  ({time.time()-t0:.0f}s)")

    # the fix: direct-outcome MPC, multi-seed
    direct_sr, head_r2 = [], []
    for s in range(args.seeds):
        g = train_outcome_head(wm, eps, dev, H=args.H, seed=s)
        def direct_plan(o, i, g=g):
            z = wm.enc(torch.as_tensor(o["range"][None], device=dev), torch.as_tensor(o["coch"][None], device=dev))
            return cem_outcome_plan(g, z, args.H, dev)
        sr = eval_policy(direct_plan, eval_seeds, dev, episodes_per=args.episodes_per)
        direct_sr.append(sr); head_r2.append(g.val_r2)
        print(f"[seed {s}] outcome-head cumulative-dist R2 {g.val_r2:+.2f} | direct-outcome MPC success {sr:.2f}")

    dm = float(np.mean(direct_sr)); dsd = float(np.std(direct_sr, ddof=1) if len(direct_sr) > 1 else 0.0)
    print(f"\n=== Exp C: informed MPC rematch (doorway, nobs=0, {len(eval_seeds)}x{args.episodes_per}) ===")
    print(f"  oracle (privileged)        {orc:.2f}   (ceiling)")
    print(f"  model-free PPO             ~0.85")
    print(f"  direct-outcome MPC (FIX)   {dm:.2f} +/- {dsd:.2f}   [outcome-head cumulative-dist R2 {np.mean(head_r2):+.2f}]")
    print(f"  learned-WM MPC (§5)        {lrn:.2f}   (compositional rollout)")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
