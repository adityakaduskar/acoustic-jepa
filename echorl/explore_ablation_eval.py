"""Unified eval of the coverage-task (explore) cue/memory ablations.

Loads each best checkpoint (full / no-memory / range-only / coch-only), evaluates it
with many episodes per room for a STABLE coverage number (best-ckpt selection during
training used only 6 eps/room, which is noisy for coverage), and prints coverage +
collision side by side. The question: unlike survive-only avoidance (where every
ablation tied because the winning policy "stop" needed no perception, §3.4/§3.4.1),
does the COVERAGE task — which requires sensing walls/pillars to sweep without crashing
— finally make the cues separate?

Run: python echorl/explore_ablation_eval.py --nobs 3 --episodes-per 20
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from ppo import RecurrentAC, evaluate                       # noqa: E402

CONFIGS = [                                                  # tag, arch kwargs, label
    ("cover",     dict(),                    "full (range+coch+mem)"),
    ("nomem",     dict(memory=False),        "no-memory"),
    ("rangeonly", dict(drop_coch=True),      "range-only (no bearing)"),
    ("cochonly",  dict(drop_range=True),     "coch-only (bearing)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nobs", type=int, default=3)
    ap.add_argument("--episodes-per", type=int, default=20)
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--t-max", type=int, default=60)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(600, 600 + args.eval_rooms))
    print(f"Coverage-task cue ablation — nobs={args.nobs}, {len(seeds)} held-out rooms x "
          f"{args.episodes_per} eps, t_max={args.t_max}\n")
    rows = []
    for tag, kw, label in CONFIGS:
        ck = os.path.join(_HERE, "ckpt", f"ppo_explore_nobs{args.nobs}_{tag}_s0_best.pt")
        if not os.path.exists(ck):
            print(f"  {label:26s}  MISSING ({os.path.relpath(ck)})"); continue
        net = RecurrentAC(**kw).to(dev); net.load_state_dict(torch.load(ck, map_location=dev)["model"]); net.eval()
        ml, cr, sr, cov = evaluate(net, seeds, dev, episodes_per=args.episodes_per,
                                   t_max=args.t_max, nobs=args.nobs, task="explore")
        rows.append((label, cov, cr, ml)); print(f"  {label:26s}  coverage {cov:5.1f}  collision {cr:.2f}  len {ml:.1f}")
    if rows:
        best = max(rows, key=lambda r: r[1]); worst = min(rows, key=lambda r: r[1])
        print(f"\n  spread: best {best[0]} {best[1]:.1f}  vs  worst {worst[0]} {worst[1]:.1f}  "
              f"(Δ {best[1]-worst[1]:.1f} cells)")
        print("  → cues SEPARATE (perception matters here)" if best[1] - worst[1] > 1.5
              else "  → cues still ~tie (task may still be perception-light)")


if __name__ == "__main__":
    main()
