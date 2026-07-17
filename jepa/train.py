"""Acoustic-JEPA pretraining loop (ACOUSTIC_JEPA_SPEC.md §3).

Trains AcousticJEPA on the .npz transitions from jepa_dump.cpp: predict the latent
of future observations from the current one + motor actions, with VICReg keeping
the representation from collapsing. Logs the collapse watchdog (per-dim std +
participation-ratio rank) every log-interval and checkpoints the full model
(encoder + EMA target + predictor) for probe.py / plan.py to consume.

Run in a Python 3.11/3.12 torch venv (see requirements.txt). Example:
    python train.py <data_dir> --epochs 50 --batch 128 --horizon 4
Quick CPU sanity check before committing GPU time:
    python train.py <data_dir> --device cpu --batch 8 --horizon 2 --max-steps 20
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import time

import torch

from dataset import make_loader
from model import AcousticJEPA, latent_stats

# std below this for a sustained stretch => representation is collapsing
COLLAPSE_STD = 0.1


def build_param_groups(model, weight_decay):
    """Weight-decay the matmul weights only; skip norms/biases."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias") or ".gn" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_lr(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, p)))


def momentum(step, total, base, final):
    return base + (final - base) * min(1.0, step / max(1, total))


def save_ckpt(path, model, opt, step, epoch, cfg):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "epoch": epoch, "cfg": cfg}, path)


def main():
    ap = argparse.ArgumentParser(description="Acoustic-JEPA pretraining")
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="runs/jepa")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=8)   # longer horizon (needs episodes > W+H)
    ap.add_argument("--window", type=int, default=4)    # temporal encoder window (frames integrated per obs; 1 = single-frame)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.04)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--ema", type=float, default=0.996, help="base EMA momentum")
    ap.add_argument("--ema-final", type=float, default=1.0, help="EMA momentum at end of training")
    ap.add_argument("--var-coeff", type=float, default=1.0)
    ap.add_argument("--cov-coeff", type=float, default=0.04)
    ap.add_argument("--gamma", type=float, default=1.0, help="multi-step loss discount")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="mixed precision (cuda only)")
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--ckpt-interval", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (0 = full run)")
    ap.add_argument("--resume", default="", help="checkpoint to resume from")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    loader = make_loader(args.data_dir, horizon=args.horizon, window=args.window, batch_size=args.batch,
                         shuffle=True, num_workers=args.num_workers, probe=False)
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise SystemExit(
            f"no full batches: {len(loader.dataset)} windows < batch {args.batch}. "
            f"Lower --batch or --horizon, or generate more data.")
    total_steps = args.max_steps or args.epochs * steps_per_epoch

    model = AcousticJEPA(emb=args.emb, ema=args.ema, var_coeff=args.var_coeff,
                         cov_coeff=args.cov_coeff, gamma=args.gamma, window=args.window).to(device)
    opt = torch.optim.AdamW(build_param_groups(model, args.weight_decay),
                            lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    step, start_epoch = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, start_epoch = ck["step"], ck["epoch"]
        print(f"resumed from {args.resume} @ step {step}")

    logf = open(os.path.join(args.out, "log.csv"), "a", newline="")
    logw = csv.writer(logf)
    if step == 0:
        logw.writerow(["step", "epoch", "loss", "pred", "var", "cov", "std", "rank", "lr", "ema"])

    print(f"device={device} steps/epoch={steps_per_epoch} total={total_steps} "
          f"windows={len(loader.dataset)}")
    collapse_run = 0
    t0 = time.time()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        for coch, act, _ in loader:
            lr = cosine_lr(step, total_steps, args.lr, args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            model.ema = momentum(step, total_steps, args.ema, args.ema_final)

            coch = coch.to(device, non_blocking=True)
            act = act.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                out = model(coch, act)
            scaler.scale(out["loss"]).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            model.update_target()

            if step % args.log_interval == 0:
                st = latent_stats(out["z"].float())
                logw.writerow([step, epoch, f"{out['loss']:.5f}", f"{out['pred']:.5f}",
                               f"{out['var']:.5f}", f"{out['cov']:.5f}",
                               f"{st['std']:.4f}", f"{st['rank']:.2f}", f"{lr:.2e}",
                               f"{model.ema:.5f}"])
                logf.flush()
                ips = (step + 1) * args.batch / (time.time() - t0)
                print(f"step {step:6d} ep {epoch:3d} | loss {out['loss']:.4f} "
                      f"pred {out['pred']:.4f} var {out['var']:.4f} cov {out['cov']:.4f} "
                      f"| std {st['std']:.3f} rank {st['rank']:.1f} | lr {lr:.1e} "
                      f"ema {model.ema:.4f} | {ips:.0f} obs/s")
                # collapse watchdog
                collapse_run = collapse_run + 1 if st["std"] < COLLAPSE_STD else 0
                if collapse_run >= 5:
                    print(f"  !! WARNING: latent std < {COLLAPSE_STD} for {collapse_run} logs "
                          f"-- representation collapsing. Raise --var-coeff or lower --lr.")

            step += 1
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

        save_ckpt(os.path.join(args.out, "latest.pt"), model, opt, step, epoch + 1, vars(args))
        if stop:
            break

    save_ckpt(os.path.join(args.out, "final.pt"), model, opt, step, epoch + 1, vars(args))
    logf.close()
    print(f"done: {step} steps, checkpoints in {args.out}")


if __name__ == "__main__":
    main()
