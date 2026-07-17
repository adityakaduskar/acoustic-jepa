"""Train the cross-view Acoustic-JEPA (v2) on a swarm_dump dataset.

Minimal but complete: AdamW + cosine LR, EMA target momentum ramp, per-block
VICReg, a collapse watchdog, and a held-out check that the predictor beats the
"no-transform" baseline (regressing the target range block onto the context
block unchanged) -- if it does, the model is actually crossing the viewpoint
using the transform, which is the whole point.

Run in AcousticJEPA/.venv:
  .venv\\Scripts\\python.exe jepa\\train_xview.py <data_dir> --epochs N --device cuda
"""
from __future__ import annotations
import argparse
import math
import os

import torch

from model_xview import CrossViewJEPA, latent_stats
from dataset_xview import make_loader, split_files


def move(batch, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    n = 0; Lr = Lc = base = 0.0
    zr_all = []
    for b in loader:
        b = move(b, dev)
        out = model(b["ctx_coch"], b["ctx_range"], b["tgt_coch"], b["tgt_range"], b["T"], b["hrtf_id"])
        bs = b["T"].shape[0]; n += bs
        Lr += float(out["L_range"]) * bs; Lc += float(out["L_coch"]) * bs
        base += float(out["base_range"]) * bs
        zr_all.append(out["zr"])
    model.train()
    st = latent_stats(torch.cat(zr_all))
    return {"L_range": Lr / n, "L_coch": Lc / n, "base_range": base / n,
            "skill": 1.0 - (Lr / n) / max(1e-9, base / n), **st}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lam-coch", type=float, default=0.25)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--oracle-T", action="store_true", help="condition on T_true (ablation)")
    ap.add_argument("--out", default="ckpt_xview.pt")
    args = ap.parse_args()

    dev = args.device
    tr_files, te_files = split_files(args.data_dir, 0.8, seed=0)
    num_hrtf = 8
    tr = make_loader(tr_files, batch_size=args.batch, shuffle=True, use_measured=not args.oracle_T)
    te = make_loader(te_files, batch_size=args.batch, shuffle=False, use_measured=not args.oracle_T)
    print(f"device {dev} | {len(tr_files)} train / {len(te_files)} test rooms | "
          f"{len(tr.dataset)} train pairs | transform={'oracle' if args.oracle_T else 'measured'}")

    model = CrossViewJEPA(num_hrtf=num_hrtf, ema=args.ema, lam_coch=args.lam_coch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(tr)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    gstep = 0
    for ep in range(args.epochs):
        for b in tr:
            b = move(b, dev)
            out = model(b["ctx_coch"], b["ctx_range"], b["tgt_coch"], b["tgt_range"], b["T"], b["hrtf_id"])
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            # EMA momentum ramp base->1 over training (V-JEPA-2 style)
            model.ema = 1.0 - (1.0 - args.ema) * (0.5 * (1 + math.cos(math.pi * gstep / steps)))
            model.update_target(); gstep += 1
        if ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1:
            ev = evaluate(model, te, dev)
            warn = "  <-- COLLAPSE?" if ev["std"] < 0.1 else ""
            print(f"ep {ep:3d} | test L_range {ev['L_range']:.4f} base {ev['base_range']:.4f} "
                  f"skill {ev['skill']:+.3f} | L_coch {ev['L_coch']:.4f} | "
                  f"z_range std {ev['std']:.3f} rank {ev['rank']:.1f}{warn}")

    torch.save({"model": model.state_dict(), "args": vars(args), "num_hrtf": num_hrtf}, args.out)
    ev = evaluate(model, te, dev)
    print(f"\nFINAL test: L_range {ev['L_range']:.4f} vs no-transform baseline {ev['base_range']:.4f} "
          f"-> skill {ev['skill']:+.3f} (positive = predictor uses the transform to cross viewpoints)")
    print(f"z_range std {ev['std']:.3f} rank {ev['rank']:.1f} (healthy = no collapse)")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
