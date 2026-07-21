"""EchoRL — model-based CEM-MPC controller for the doorway (goal) task.

The head-to-head the substrate was built for: **model-free RL (ppo.py) vs a
model-based latent-MPC controller on the SAME server, task, dynamics and metric.**
This also bridges the two research threads — AcousticJEPA showed a *learned* world
model plans well while a *random-encoder* world model fails (control-positive
dissociation); EchoRL showed model-free PPO is near-optimal. Here both meet on one
task.

Task: Stage-2 doorway-finding (env.py task="goal", nobs=0), echo-only. Success =
reach within goal_radius of a point in front of the room's largest opening.

World model (mirrors jepa/control_mpc.py, on episode_server data):
  * encoder   TwoStreamEncoder over normalized (range, coch) -> latent z[256];
  * dynamics  action-conditioned residual Predictor (z_t, a_t) -> z_{t+1},
              trained JEPA-style against an EMA target encoder;
  * readout   z -> predicted distance-to-goal (the CEM cost).
Trained on RANDOM exploratory trajectories (NOT the PPO policy -> no distillation,
so beating/losing to PPO is a fair model-based-vs-model-free result).

Controllers evaluated CLOSED-LOOP through the very same EchoLocEnv PPO uses
(identical collision, success, metric):
  * learned    -- world model with a trained encoder;
  * random-enc -- encoder FROZEN at init (predictor+readout trained on it) -- the
                  AcousticJEPA random-encoder control baseline (expected to fail);
  * oracle     -- privileged: knows the true goal point, greedy heading to it under
                  the true dynamics (the ceiling; the analytic upper bound).
Reported vs the model-free PPO number from ppo.py (--task goal).
"""
from __future__ import annotations
import argparse, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from env import EchoLocEnv, DYAW_MAX, DSTEP_MAX          # noqa: E402
from models import Normalizer                            # noqa: E402
from model_xview import TwoStreamEncoder                 # noqa: E402
from model import Predictor, vicreg_var_cov              # noqa: E402
from ppo import opening_seeds                            # noqa: E402

COCH_SHAPE = (2, 64, 48)


# --------------------------------------------------------------- action coding
def raw_to_norm(dyaw, dstep):
    """Raw (deg, m) -> env's [-1,1] action convention (matches env._decode inverse)."""
    return dyaw / DYAW_MAX, dstep / (0.5 * DSTEP_MAX) - 1.0


# ------------------------------------------------------------------ world model
class WorldModel(nn.Module):
    """Normalized echo encoder + action-conditioned latent dynamics + goal-distance
    readout. emb=256 = [z_range(128) | z_coch(128)] (no prop: the latent is the
    acoustic state the dynamics predictor rolls forward)."""

    def __init__(self, emb=256, ema=0.99):
        super().__init__()
        import copy
        self.norm = Normalizer()
        self.encoder = TwoStreamEncoder(emb // 2, emb // 2)
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(emb, action_dim=2)
        self.readout = nn.Sequential(nn.Linear(emb, 128), nn.GELU(), nn.Linear(128, 1))
        self.ema = ema

    def enc(self, rng, coch):
        rn, cn = self.norm(rng, coch)
        zr, zc = self.encoder(cn, rn)                    # TwoStreamEncoder(coch, rng)
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
        """z0 [B,emb], acts_n [B,H,2] (env [-1,1] convention) -> latents [B,H,emb]."""
        zs = []; z = z0
        for k in range(acts_n.shape[1]):
            z = self.predictor(z, acts_n[:, k]); zs.append(z)
        return torch.stack(zs, 1)


# ---------------------------------------------------------------- data + train
def _run_one_episode(env, rng, t_max, forward_bias):
    """One random exploratory rollout -> (range, coch, act[-1,1], dist)[per step]."""
    obs, info = env.reset()
    gx, gz = info["goal"]
    r_ep, c_ep, a_ep, d_ep = [], [], [], []
    for _ in range(t_max):
        dyaw = rng.uniform(-DYAW_MAX, DYAW_MAX)
        dstep = rng.uniform(0, DSTEP_MAX) * (1.0 if rng.random() < forward_bias else 0.4)
        an = np.array(raw_to_norm(dyaw, dstep), np.float32)
        r_ep.append(obs["range"]); c_ep.append(obs["coch"]); a_ep.append(an)
        px, pz, _ = info["pose"]; d_ep.append(math.hypot(px - gx, pz - gz))
        obs, _, term, trunc, info = env.step(an)
        if term or trunc:
            break
    px, pz, _ = info["pose"]; d_ep.append(math.hypot(px - gx, pz - gz))   # final-frame dist
    return (np.array(r_ep, np.float32), np.array(c_ep, np.float32),
            np.array(a_ep, np.float32), np.array(d_ep, np.float32))


def collect_traj(seeds, n_ep, dev, nobs=0, t_max=40, base_seed=7, forward_bias=0.6, workers=12):
    """Random exploratory rollouts in the goal env, gathered by `workers` concurrent
    envs (the C observe() releases the GIL, so threads give real parallelism -- the
    same trick vec_env uses). Each worker owns its rooms and rotates through them."""
    from concurrent.futures import ThreadPoolExecutor
    workers = min(workers, max(1, len(seeds)))
    shards = [seeds[i::workers] or [seeds[i % len(seeds)]] for i in range(workers)]
    per = [n_ep // workers + (1 if i < n_ep % workers else 0) for i in range(workers)]

    def worker(wi):
        env = EchoLocEnv(seed_pool=shards[wi], action_mode="continuous", nobs=nobs, sim_threads=2,
                         t_max=t_max, resample_room_every=1, task="goal", base_seed=base_seed * 1000 + wi)
        rng = np.random.default_rng(base_seed * 1000 + wi)
        out = [_run_one_episode(env, rng, t_max, forward_bias) for _ in range(per[wi])]
        env.close()
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(worker, range(workers)))
    return [ep for r in results for ep in r]


def save_corpus(path, eps):
    """Persist the ragged (range, coch, act, dist) episode list. The render is not
    bit-deterministic (~1% jitter), so this is a time-saver, not a reproducibility
    guarantee -- reuse it to avoid re-simulating across probe/experiment reruns."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(eps, f, protocol=4)


def load_corpus(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def build_windows(eps, H):
    C, R, A, DD = [], [], [], []
    for (r, c, a, d) in eps:
        K = a.shape[0]
        for s in range(0, K - H):
            R.append(r[s:s + H + 1]); C.append(c[s:s + H + 1])
            A.append(a[s:s + H]); DD.append(d[s:s + H + 1])
    return (np.array(R, np.float32), np.array(C, np.float32),
            np.array(A, np.float32), np.array(DD, np.float32))


def train_wm(eps, dev, H=6, epochs=30, bs=128, freeze_encoder=False, seed=0):
    torch.manual_seed(seed)
    R, C, A, DD = build_windows(eps, H)
    n = len(R)
    if n == 0:
        raise RuntimeError("no training windows -- collect more/longer trajectories")
    # 15% held-out windows for the readout/dynamics diagnostic (is the WM usable?)
    perm = np.random.default_rng(seed).permutation(n)
    ntr = int(0.85 * n); tr_idx, va_idx = perm[:ntr], perm[ntr:]
    wm = WorldModel().to(dev)
    if freeze_encoder:
        for p in wm.encoder.parameters():
            p.requires_grad_(False)
    params = [p for p in wm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        idx = tr_idx.copy(); np.random.default_rng(ep).shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            rng = torch.tensor(R[b], device=dev); coch = torch.tensor(C[b], device=dev)
            act = torch.tensor(A[b], device=dev); dist = torch.tensor(DD[b], device=dev)
            Bn, T = rng.shape[0], rng.shape[1]
            z = wm.enc(rng.reshape(Bn * T, 256), coch.reshape(Bn * T, *COCH_SHAPE)).reshape(Bn, T, -1)
            with torch.no_grad():
                zt = wm.enc_target(rng.reshape(Bn * T, 256),
                                   coch.reshape(Bn * T, *COCH_SHAPE)).reshape(Bn, T, -1)
            pred = wm.rollout(z[:, 0], act)                        # [B,H,emb]
            dyn = ((pred - zt[:, 1:]) ** 2).mean()
            ro = wm.readout(z.reshape(Bn * T, -1)).reshape(Bn, T)
            ro_loss = ((ro - dist) ** 2).mean()
            vr, cv = vicreg_var_cov(z[:, 0])
            loss = dyn + ro_loss + 1.0 * vr + 0.04 * cv
            opt.zero_grad(); loss.backward(); opt.step()
            if not freeze_encoder:
                wm.update_target()
    wm.eval()
    wm.diag = _wm_diagnostic(wm, R, C, A, DD, va_idx, dev)
    return wm


@torch.no_grad()
def _wm_diagnostic(wm, R, C, A, DD, va_idx, dev, bs=256):
    """On held-out windows: (a) readout R^2 of predicted vs true distance-to-goal
    (can the latent decode distance at all?); (b) H-step ROLLED-OUT distance error
    (does imagining actions predict the resulting distance?) -- the two things CEM
    relies on."""
    ro_p, ro_t, roll_err, base = [], [], [], []
    for i in range(0, len(va_idx), bs):
        b = va_idx[i:i + bs]
        rng = torch.tensor(R[b], device=dev); coch = torch.tensor(C[b], device=dev)
        act = torch.tensor(A[b], device=dev); dist = torch.tensor(DD[b], device=dev)
        Bn, T = rng.shape[0], rng.shape[1]
        z = wm.enc(rng.reshape(Bn * T, 256), coch.reshape(Bn * T, *COCH_SHAPE)).reshape(Bn, T, -1)
        ro = wm.readout(z.reshape(Bn * T, -1)).reshape(Bn, T)
        ro_p.append(ro[:, 0].cpu().numpy()); ro_t.append(dist[:, 0].cpu().numpy())
        pred = wm.rollout(z[:, 0], act)                            # [B,H,emb]
        d_hat = wm.readout(pred.reshape(Bn * H_pred(T), -1)).reshape(Bn, -1)
        roll_err.append((d_hat[:, -1] - dist[:, -1]).abs().cpu().numpy())
        base.append((dist[:, 0] - dist[:, -1]).abs().cpu().numpy())   # naive "distance unchanged"
    ro_p = np.concatenate(ro_p); ro_t = np.concatenate(ro_t)
    ss_res = float(((ro_p - ro_t) ** 2).sum()); ss_tot = float(((ro_t - ro_t.mean()) ** 2).sum() + 1e-9)
    r2 = 1.0 - ss_res / ss_tot
    return dict(readout_r2=r2, rollout_mae=float(np.concatenate(roll_err).mean()),
                base_mae=float(np.concatenate(base).mean()))


def H_pred(T):
    return T - 1


# ---------------------------------------------------------------- CEM planner
@torch.no_grad()
def cem_plan(wm, z0, H, dev, iters=3, N=256, elite=32):
    """Minimise imagined distance-to-goal over the latent rollout. Returns the
    elite-mean first action in the env's [-1,1] convention."""
    mu = torch.zeros(H, 2, device=dev)
    sd = torch.tensor([[30.0, 0.6]], device=dev).repeat(H, 1)
    for _ in range(iters):
        eps = torch.randn(N, H, 2, device=dev) * sd + mu
        dyaw = eps[..., 0].clamp(-DYAW_MAX, DYAW_MAX)
        dstep = eps[..., 1].clamp(0, DSTEP_MAX)
        an = torch.stack([dyaw / DYAW_MAX, dstep / (0.5 * DSTEP_MAX) - 1.0], -1)
        z = z0.expand(N, -1)
        preds = wm.rollout(z, an)                                  # [N,H,emb]
        cost = wm.readout(preds.reshape(N * H, -1)).reshape(N, H).sum(1)
        top = cost.argsort()[:elite]
        chosen = torch.stack([dyaw, dstep], -1)[top]
        mu = chosen.mean(0); sd = chosen.std(0) + 1e-3
    dyaw0, dstep0 = float(mu[0, 0]), float(mu[0, 1])
    return np.array(raw_to_norm(dyaw0, dstep0), np.float32)


def oracle_action(x, z, yaw, goal):
    """Privileged greedy: turn toward the true goal point, step forward."""
    gx, gz = goal
    want = math.degrees(math.atan2(gx - x, -(gz - z)))            # env heading convention
    d = ((want - yaw + 180) % 360) - 180
    dyaw = max(-DYAW_MAX, min(DYAW_MAX, d))
    return np.array(raw_to_norm(dyaw, DSTEP_MAX), np.float32)


# ---------------------------------------------------------------- closed-loop eval
@torch.no_grad()
def eval_controller(kind, wm, seeds, dev, H=6, episodes_per=10, nobs=0, t_max=40, base_seed=100):
    """Run the controller CLOSED-LOOP through EchoLocEnv (same collision/success/metric
    as ppo.evaluate). One env per seed evaluated concurrently (GIL released during the
    C observe()); `episodes_per` starts each."""
    from concurrent.futures import ThreadPoolExecutor

    def eval_room(s):
        env = EchoLocEnv(seed_pool=[s], action_mode="continuous", nobs=nobs, sim_threads=2,
                         t_max=t_max, resample_room_every=0, task="goal", base_seed=base_seed + s)
        rsucc, rsteps = [], []
        for _ in range(episodes_per):
            obs, info = env.reset()
            goal = info["goal"]; done = False; n = 0; ok = False
            while not done:
                if kind == "oracle":
                    x, z, yaw = info["pose"]
                    a = oracle_action(x, z, yaw, goal)
                else:
                    rng = torch.as_tensor(obs["range"][None], device=dev)
                    coch = torch.as_tensor(obs["coch"][None], device=dev)
                    a = cem_plan(wm, wm.enc(rng, coch), H, dev)
                obs, _, term, trunc, info = env.step(a)
                n += 1; ok = bool(info.get("success")); done = term or trunc
            rsucc.append(1.0 if ok else 0.0); rsteps.append(n)
        env.close()
        return rsucc, rsteps

    with ThreadPoolExecutor(max_workers=min(len(seeds), 12)) as pool:
        res = list(pool.map(eval_room, seeds))
    succ = [v for r in res for v in r[0]]; steps = [v for r in res for v in r[1]]
    return float(np.mean(succ)), float(np.mean(steps))


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nobs", type=int, default=0)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--train-rooms", type=int, default=60)
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--train-ep", type=int, default=400, help="random trajectories for WM training")
    ap.add_argument("--episodes-per", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=1, help="training seeds for CIs (corpus collected once)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus-in", default=None, help="load a cached trajectory corpus instead of simulating")
    ap.add_argument("--corpus-out", default=None, help="save the collected corpus here (reuse via --corpus-in)")
    ap.add_argument("--save-wm", default=None, help="checkpoint the first-seed learned WM here (for probe_dd.py)")
    ap.add_argument("--skip-eval", action="store_true",
                    help="collect + train + save only (no closed-loop controller eval); for building probe inputs")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.train_rooms, args.eval_rooms, args.train_ep, args.epochs, args.episodes_per = 6, 3, 40, 6, 3
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train_seeds = opening_seeds(1, args.train_rooms, args.nobs)
    eval_seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    print(f"device={dev} | goal task nobs={args.nobs} | H={args.H} | repeats={args.repeats} | "
          f"{len(train_seeds)} train / {len(eval_seeds)} eval doorway rooms")

    t0 = time.time()
    if args.corpus_in and os.path.exists(args.corpus_in):
        eps = load_corpus(args.corpus_in)
        print(f"loaded cached corpus {os.path.relpath(args.corpus_in)}: {len(eps)} episodes ({time.time()-t0:.0f}s)")
    else:
        print(f"collecting {args.train_ep} random exploratory trajectories (once, shared across seeds)...")
        eps = collect_traj(train_seeds, args.train_ep, dev, nobs=args.nobs, base_seed=7)
        if args.corpus_out:
            save_corpus(args.corpus_out, eps)
            print(f"  saved corpus -> {os.path.relpath(args.corpus_out)}")
    nwin = sum(max(0, a.shape[0] - args.H) for (_, _, a, _) in eps)
    print(f"  {len(eps)} episodes, {nwin} H-windows ({time.time()-t0:.0f}s)")

    if args.skip_eval:
        wm = train_wm(eps, dev, H=args.H, epochs=args.epochs, freeze_encoder=False, seed=args.seed)
        print(f"  WM diagnostic: readout-R2 {wm.diag['readout_r2']:+.2f} "
              f"rollout-MAE {wm.diag['rollout_mae']:.2f}m (naive {wm.diag['base_mae']:.2f}m)")
        if args.save_wm:
            torch.save({"model": wm.state_dict(), "H": args.H, "diag": wm.diag}, args.save_wm)
            print(f"  saved learned WM -> {os.path.relpath(args.save_wm)}")
        print(f"skip-eval: corpus + WM ready ({time.time()-t0:.0f}s)")
        return

    # oracle is seed-independent (deterministic given eval starts) -> compute once
    orc, orc_st = eval_controller("oracle", None, eval_seeds, dev, H=args.H,
                                  episodes_per=args.episodes_per, nobs=args.nobs)
    print(f"oracle (privileged ceiling): success {orc:.2f}  avg steps {orc_st:.1f}")

    learned_sr, random_sr, r2_l, r2_r, roll_l = [], [], [], [], []
    for r in range(args.repeats):
        sd = args.seed + r
        wm = train_wm(eps, dev, H=args.H, epochs=args.epochs, freeze_encoder=False, seed=sd)
        wm_rand = train_wm(eps, dev, H=args.H, epochs=args.epochs, freeze_encoder=True, seed=sd)
        if args.save_wm and r == 0:
            torch.save({"model": wm.state_dict(), "H": args.H, "diag": wm.diag}, args.save_wm)
            print(f"  saved learned WM (seed {sd}) -> {os.path.relpath(args.save_wm)}")
        ls, _ = eval_controller("learned", wm, eval_seeds, dev, H=args.H,
                                episodes_per=args.episodes_per, nobs=args.nobs)
        rs, _ = eval_controller("random-enc", wm_rand, eval_seeds, dev, H=args.H,
                                episodes_per=args.episodes_per, nobs=args.nobs)
        learned_sr.append(ls); random_sr.append(rs)
        r2_l.append(wm.diag["readout_r2"]); r2_r.append(wm_rand.diag["readout_r2"])
        roll_l.append(wm.diag["rollout_mae"])
        print(f"[seed {sd}] learned {ls:.2f}  random-enc {rs:.2f} | "
              f"readout-R2 learned {wm.diag['readout_r2']:+.2f} random {wm_rand.diag['readout_r2']:+.2f} | "
              f"learned rollout-MAE {wm.diag['rollout_mae']:.2f}m (naive {wm.diag['base_mae']:.2f}m)")

    def ms(v):
        v = np.array(v); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
    lm, lsd = ms(learned_sr); rm, rsd = ms(random_sr)
    print(f"\n=== MPC vs RL, doorway task (nobs={args.nobs}), {len(eval_seeds)} held-out rooms "
          f"x {args.episodes_per} starts, {args.repeats} seeds ===")
    print(f"  oracle (privileged)   {orc:.2f}             (ceiling)")
    print(f"  learned-WM MPC        {lm:.2f} +/- {lsd:.2f}")
    print(f"  model-free PPO        ~0.85            (ppo.py --task goal, from RESULTS.md)")
    print(f"  random-enc MPC        {rm:.2f} +/- {rsd:.2f}   (control lower bound)")
    print(f"  [WM] learned readout-R2 {ms(r2_l)[0]:+.2f}  random {ms(r2_r)[0]:+.2f}  | "
          f"learned rollout-MAE {ms(roll_l)[0]:.2f}m")
    if args.repeats > 1:
        gap = np.array(learned_sr) - np.array(random_sr)
        print(f"  learned - random per-seed: {[round(g,2) for g in gap]} -> "
              f"{gap.mean():+.2f} +/- {gap.std(ddof=1):.2f} "
              f"({'ROBUST: learned>random every seed' if (gap>0).all() else 'not robust'})")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
