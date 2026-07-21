"""Dueling Double-DQN for Stage-0 echolocation obstacle avoidance (EchoRL M3).

Paper-faithful baseline (Mohan & Vanderelst 2020): model-free deep RL, discrete
turn actions, reward = fly-far-without-crashing. Perception = the reused
TwoStreamEncoder trunk (models.py). Success = mean flight length climbs well above
the random ~4-step floor on HELD-OUT rooms.

Bootstrapping: collision is a true terminal (mask q_next); time-limit truncation is
NOT (we bootstrap), the standard fix so the agent isn't taught the episode "ends".
"""
from __future__ import annotations
import argparse, os, sys, time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from vec_env import VecEchoLocEnv, _stack          # noqa: E402
from env import EchoLocEnv                          # noqa: E402
from models import DuelingQNet                      # noqa: E402

COCH_SHAPE = (2, 64, 48)


# --------------------------------------------------------------- replay buffer
class Replay:
    def __init__(self, cap, device):
        self.cap, self.device, self.n, self.i = cap, device, 0, 0
        self.s_r = np.zeros((cap, 256), np.float16)
        self.s_c = np.zeros((cap, *COCH_SHAPE), np.float16)
        self.s_p = np.zeros((cap, 3), np.float16)
        self.n_r = np.zeros((cap, 256), np.float16)
        self.n_c = np.zeros((cap, *COCH_SHAPE), np.float16)
        self.n_p = np.zeros((cap, 3), np.float16)
        self.a = np.zeros(cap, np.int64)
        self.rew = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, np.float32)      # collision mask (bootstrap gate)

    def push(self, s, a, r, d, ns):
        i = self.i
        self.s_r[i], self.s_c[i], self.s_p[i] = s["range"], s["coch"], s["prop"]
        self.n_r[i], self.n_c[i], self.n_p[i] = ns["range"], ns["coch"], ns["prop"]
        self.a[i], self.rew[i], self.done[i] = a, r, d
        self.i = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs):
        idx = np.random.randint(0, self.n, size=bs)
        t = lambda x: torch.as_tensor(x, device=self.device)
        s = (t(self.s_r[idx].astype(np.float32)), t(self.s_c[idx].astype(np.float32)), t(self.s_p[idx].astype(np.float32)))
        ns = (t(self.n_r[idx].astype(np.float32)), t(self.n_c[idx].astype(np.float32)), t(self.n_p[idx].astype(np.float32)))
        return s, t(self.a[idx]), t(self.rew[idx]), t(self.done[idx]), ns


def obs_to_t(obs, device):
    return (torch.as_tensor(obs["range"], dtype=torch.float32, device=device),
            torch.as_tensor(obs["coch"], dtype=torch.float32, device=device),
            torch.as_tensor(obs["prop"], dtype=torch.float32, device=device))


# ------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(net, seeds, device, episodes_per=6, t_max=40, sim_threads=2, greedy=True, nobs=3):
    """Greedy (or random) policy on held-out rooms. Returns (mean_len, collision_rate, mean_return)."""
    vec = VecEchoLocEnv([[s] for s in seeds], action_mode="discrete", sim_threads=sim_threads,
                        t_max=t_max, resample_room_every=0, nobs=nobs)
    lengths, colls, rets = [], [], []
    cur_len = np.zeros(vec.n); cur_ret = np.zeros(vec.n)
    obs = vec.reset()
    n_eps = episodes_per * vec.n
    done_count = 0
    while done_count < n_eps:
        if greedy and net is not None:
            q = net(*obs_to_t(obs, device))
            acts = q.argmax(1).cpu().numpy()
        else:
            acts = np.random.randint(0, 5, size=vec.n)
        obs, rew, term, trunc, done, _, _ = vec.step(acts)
        cur_len += 1; cur_ret += rew
        for i in range(vec.n):
            if done[i]:
                lengths.append(cur_len[i]); colls.append(1.0 if term[i] else 0.0); rets.append(cur_ret[i])
                cur_len[i] = 0; cur_ret[i] = 0; done_count += 1
    vec.close()
    return float(np.mean(lengths)), float(np.mean(colls)), float(np.mean(rets))


# ------------------------------------------------------------------- training
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    net = DuelingQNet(n_actions=5).to(device)
    if args.init_ckpt:                               # curriculum warm-start from an easier tier
        sd = torch.load(args.init_ckpt, map_location=device)["model"]
        net.load_state_dict(sd)
        print(f"warm-started from {os.path.relpath(args.init_ckpt)}")
    tgt = DuelingQNet(n_actions=5).to(device)
    tgt.load_state_dict(net.state_dict())
    for p in tgt.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    buf = Replay(args.buffer, device)

    # K envs across disjoint seed shards for room diversity
    train_seeds = list(range(1, args.train_rooms + 1))
    shards = [train_seeds[i::args.envs] for i in range(args.envs)]
    vec = VecEchoLocEnv(shards, action_mode="discrete", sim_threads=args.sim_threads,
                        t_max=args.t_max, resample_room_every=args.resample_every, nobs=args.nobs)
    eval_seeds = list(range(600, 600 + args.eval_rooms))

    print(f"device={device} | {args.envs} envs x {args.sim_threads} thr | "
          f"train rooms {len(train_seeds)}, eval rooms {len(eval_seeds)} | target {args.steps} steps")
    if args.baseline:
        ml, cr, rr = evaluate(None, eval_seeds, device, greedy=False, t_max=args.t_max, nobs=args.nobs)
        print(f"[random baseline] mean_len {ml:.1f}  collision_rate {cr:.2f}  return {rr:+.2f}")

    os.makedirs(os.path.join(_HERE, "ckpt"), exist_ok=True)
    best_ck = os.path.join(_HERE, "ckpt", "dqn_stage0_best.pt")
    best_len = -1.0
    obs = vec.reset()
    step = 0; grad = 0; t0 = time.time(); last_eval = 0
    while step < args.steps:
        eps = max(args.eps_end, 1.0 - (1 - args.eps_end) * step / args.eps_decay)
        with torch.no_grad():
            q = net(*obs_to_t(obs, device))
            greedy_a = q.argmax(1).cpu().numpy()
        rand_mask = np.random.rand(vec.n) < eps
        acts = np.where(rand_mask, np.random.randint(0, 5, size=vec.n), greedy_a)

        next_obs, rew, term, trunc, done, term_obs, _ = vec.step(acts)
        for i in range(vec.n):
            ns = term_obs[i] if done[i] else {k: next_obs[k][i] for k in next_obs}
            s_i = {k: obs[k][i] for k in obs}
            buf.push(s_i, acts[i], rew[i], 1.0 if term[i] else 0.0, ns)
        obs = next_obs
        step += vec.n

        if buf.n >= args.learn_start:
            for _ in range(args.grad_per_iter):
                s, a, r, d, ns = buf.sample(args.batch)
                with torch.no_grad():
                    a_star = net(*ns).argmax(1)
                    q_next = tgt(*ns).gather(1, a_star[:, None]).squeeze(1)
                    y = r + args.gamma * (1 - d) * q_next
                q = net(*s).gather(1, a[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step(); grad += 1
                with torch.no_grad():
                    for p, tp in zip(net.parameters(), tgt.parameters()):
                        tp.mul_(1 - args.tau).add_(p, alpha=args.tau)

        if step - last_eval >= args.eval_every:
            last_eval = step
            ml, cr, rr = evaluate(net, eval_seeds, device, t_max=args.t_max, nobs=args.nobs)
            sps = step / (time.time() - t0)
            tag = ""
            if ml > best_len:                        # keep the peak policy (DQN greedy drifts)
                best_len = ml
                torch.save({"model": net.state_dict(), "eval": (ml, cr, rr), "step": step}, best_ck)
                tag = " *best"
            print(f"step {step:6d} eps {eps:.2f} grad {grad:6d} | eval len {ml:5.1f} "
                  f"coll {cr:.2f} ret {rr:+.2f} | {sps:.0f} env-steps/s{tag}")

    ml, cr, rr = evaluate(net, eval_seeds, device, episodes_per=10, t_max=args.t_max, nobs=args.nobs)
    print(f"\nFINAL (held-out): mean_len {ml:.1f}  collision_rate {cr:.2f}  return {rr:+.2f}")
    print(f"BEST eval mean_len over training: {best_len:.1f}  -> {os.path.relpath(best_ck)}")
    ck = os.path.join(_HERE, "ckpt", "dqn_stage0.pt")
    torch.save({"model": net.state_dict(), "final": (ml, cr, rr)}, ck)
    print(f"saved {os.path.relpath(ck)}")
    vec.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--sim-threads", type=int, default=2)
    ap.add_argument("--t-max", type=int, default=40)
    ap.add_argument("--train-rooms", type=int, default=60)
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--nobs", type=int, default=3)
    ap.add_argument("--resample-every", type=int, default=4)
    ap.add_argument("--buffer", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay", type=int, default=25000)
    ap.add_argument("--learn-start", type=int, default=2000)
    ap.add_argument("--grad-per-iter", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=6000)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--init-ckpt", type=str, default="", help="warm-start model from this checkpoint (curriculum)")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
