"""EchoRL Experiment D -- does on-policy data cure the objective mismatch?

RESULTS.md 5.2 found the direct-outcome MPC's failure is NOT model inaccuracy (it
predicts the planning cost at R^2=0.96) but the planner-model interaction under
DISTRIBUTION SHIFT: the outcome head is trained on RANDOM wandering, but CEM queries
it about DELIBERATE goal-directed maneuvers it never saw, and exploits its confident
off-distribution mistakes. The indicated fix (flagged, not naive "make it accurate"):
DAgger-style aggregation -- let the model see the states/actions the planner actually
visits.

Loop (frozen learned encoder throughout; only the outcome head is retrained):
  round 0:  train head on the RANDOM corpus; eval MPC.
  round k:  run the current MPC in the goal env, LOG its own (range,coch,action,dist)
            trajectories, ADD them to the training set, RETRAIN the head, eval MPC.
If success climbs across rounds, the objective-mismatch diagnosis is confirmed and the
distribution-shift fix works. If it plateaus low, the myopic-cue ceiling (partial H0,
5.1) dominates instead.

Run: python echorl/mpc_dagger.py --corpus echorl/corpus_goal.pkl --wm echorl/ckpt/mpc_wm.pt \
        --nobs 0 --rounds 4
"""
from __future__ import annotations
import argparse, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from env import EchoLocEnv, DYAW_MAX, DSTEP_MAX                # noqa: E402
from mpc import WorldModel, load_corpus                        # noqa: E402
from mpc_direct import train_outcome_head, cem_outcome_plan, eval_policy   # noqa: E402
from ppo import opening_seeds                                  # noqa: E402

COCH_SHAPE = (2, 64, 48)


@torch.no_grad()
def collect_onpolicy(g, wm, seeds, n_ep, dev, H, nobs=0, t_max=40, base_seed=500, workers=12):
    """Run the current direct-outcome MPC in the goal env and LOG its OWN trajectories
    as (range, coch, action[-1,1], dist-to-goal) -- the on-distribution data the head
    was missing. Parallel over rooms (GIL released during the C observe())."""
    workers = min(workers, max(1, len(seeds)))
    per = [n_ep // workers + (1 if i < n_ep % workers else 0) for i in range(workers)]
    shards = [seeds[i::workers] or [seeds[i % len(seeds)]] for i in range(workers)]

    def worker(wi):
        env = EchoLocEnv(seed_pool=shards[wi], action_mode="continuous", nobs=nobs, sim_threads=2,
                         t_max=t_max, resample_room_every=1, task="goal", base_seed=base_seed * 1000 + wi)
        out = []
        for _ in range(per[wi]):
            obs, info = env.reset(); gx, gz = info["goal"]
            r_ep, c_ep, a_ep, d_ep = [], [], [], []
            done = False
            while not done:
                z = wm.enc(torch.as_tensor(obs["range"][None], device=dev),
                           torch.as_tensor(obs["coch"][None], device=dev))
                a = cem_outcome_plan(g, z, H, dev)
                r_ep.append(obs["range"]); c_ep.append(obs["coch"]); a_ep.append(a)
                px, pz, _ = info["pose"]; d_ep.append(math.hypot(px - gx, pz - gz))
                obs, _, term, trunc, info = env.step(a); done = term or trunc
            px, pz, _ = info["pose"]; d_ep.append(math.hypot(px - gx, pz - gz))
            out.append((np.array(r_ep, np.float32), np.array(c_ep, np.float32),
                        np.array(a_ep, np.float32), np.array(d_ep, np.float32)))
        env.close()
        return out
    with ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(worker, range(workers)))
    return [ep for r in res for ep in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--nobs", type=int, default=0)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--onpolicy-ep", type=int, default=180, help="MPC trajectories collected per round")
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--episodes-per", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    eps = load_corpus(args.corpus)                             # round-0 random data (grows each round)
    ck = torch.load(args.wm, map_location=dev)
    wm = WorldModel().to(dev); wm.load_state_dict(ck["model"]); wm.eval()
    for p in wm.parameters():
        p.requires_grad_(False)
    train_seeds = opening_seeds(1, 60, args.nobs)
    eval_seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    print(f"device={dev} | Exp D DAgger | nobs={args.nobs} H={args.H} | {args.rounds} rounds x "
          f"{args.onpolicy_ep} on-policy ep | eval {len(eval_seeds)} rooms x {args.episodes_per}")

    t0 = time.time(); hist = []
    for rd in range(args.rounds + 1):
        g = train_outcome_head(wm, eps, dev, H=args.H, seed=args.seed)
        def plan(o, i, g=g):
            z = wm.enc(torch.as_tensor(o["range"][None], device=dev), torch.as_tensor(o["coch"][None], device=dev))
            return cem_outcome_plan(g, z, args.H, dev)
        sr = eval_policy(plan, eval_seeds, dev, episodes_per=args.episodes_per, nobs=args.nobs)
        hist.append(sr)
        n_win = sum(max(0, a.shape[0] - args.H) for (_, _, a, _) in eps)
        print(f"[round {rd}] head-R2 {g.val_r2:+.2f} | MPC success {sr:.2f} | "
              f"train windows {n_win} ({len(eps)} eps) | {time.time()-t0:.0f}s")
        if rd < args.rounds:                                   # aggregate the MPC's own trajectories
            new = collect_onpolicy(g, wm, train_seeds, args.onpolicy_ep, dev, args.H,
                                   nobs=args.nobs, base_seed=500 + rd)
            eps = eps + new

    print(f"\n=== Exp D: DAgger on the direct-outcome MPC (nobs={args.nobs}) ===")
    print("  success by round: " + " -> ".join(f"{s:.2f}" for s in hist))
    delta = hist[-1] - hist[0]
    verdict = ("DISTRIBUTION SHIFT was the problem -- on-policy data fixes it" if delta > 0.08
               else "no lift -- myopic-cue ceiling (partial H0) dominates, not distribution shift")
    print(f"  round 0 (random-only) {hist[0]:.2f}  ->  final {hist[-1]:.2f}   (delta {delta:+.2f})")
    print(f"  verdict: {verdict}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
