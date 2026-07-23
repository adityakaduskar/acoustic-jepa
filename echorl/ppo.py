"""Recurrent PPO, continuous actions — EchoRL M4.

The main agent. Continuous action (dyaw,dstep)∈[-1,1]² can slow/stop/pivot, which
the M3 fixed-step DQN could not — the hypothesis is that this cracks the clutter
tiers (nobs≥1) where DQN floored. A GRU over the ping sequence handles partial
observability (a single ping can't see behind the agent / around occluders).

Perception = the reused EchoTrunk (models.py: normalized range+coch via
TwoStreamEncoder, + prop). Trunk is FEEDFORWARD, so we encode the whole rollout in
one batch and run only the (cheap) GRU sequentially — the efficient recurrent-PPO
pattern. Hidden state is reset at episode boundaries via the done mask.
"""
from __future__ import annotations
import argparse, os, sys, time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from vec_env import VecEchoLocEnv                    # noqa: E402
from models import EchoTrunk                         # noqa: E402
from env import EpisodeServer                        # noqa: E402

COCH_SHAPE = (2, 64, 48)


def opening_seeds(start, count, nobs):
    """First `count` seeds from `start` whose room has an opening (goal-task eval targets)."""
    out, s = [], start
    while len(out) < count and s < start + 500:
        srv = EpisodeServer(s, nobs, threads=0)
        if len(srv.openings) > 0:
            out.append(s)
        srv.close(); s += 1
    return out


# --------------------------------------------------------------------- model
class RecurrentAC(nn.Module):
    def __init__(self, emb_range=128, emb_coch=128, gru_hidden=256, act_dim=2, init_logstd=-0.5,
                 memory=True, drop_range=False, drop_coch=False):
        super().__init__()
        self.trunk = EchoTrunk(emb_range, emb_coch)
        self.memory = memory
        self.drop_range, self.drop_coch = drop_range, drop_coch   # cue ablations
        if memory:
            self.gru = nn.GRU(self.trunk.out_dim, gru_hidden)
        else:                                                     # memory ablation: no hidden state
            self.proj = nn.Sequential(nn.Linear(self.trunk.out_dim, gru_hidden), nn.GELU())
        self.actor_mean = nn.Linear(gru_hidden, act_dim)
        self.critic = nn.Linear(gru_hidden, 1)
        self.logstd = nn.Parameter(torch.full((act_dim,), float(init_logstd)))
        self.H = gru_hidden

    def feats(self, rng, coch, prop):
        """Feedforward encode: any leading batch shape -> [..., trunk_out].
        A dropped stream is zeroed at the RAW input, so after normalization it is a fixed
        constant that carries no information -> a clean 'this cue removed' ablation."""
        if self.drop_range:
            rng = torch.zeros_like(rng)
        if self.drop_coch:
            coch = torch.zeros_like(coch)
        lead = rng.shape[:-1]
        f = self.trunk(rng.reshape(-1, 256), coch.reshape(-1, *COCH_SHAPE), prop.reshape(-1, 3))
        return f.reshape(*lead, -1)

    def step(self, feat, h):
        """One timestep. feat [N,F], h [1,N,H] -> mean,std,value [N,*], h' [1,N,H]."""
        if self.memory:
            out, h2 = self.gru(feat.unsqueeze(0), h)
            z = out.squeeze(0)
        else:
            z = self.proj(feat); h2 = h
        return self.actor_mean(z), self.logstd.exp(), self.critic(z).squeeze(-1), h2

    def rollout_eval(self, feats, dones, h0):
        """Replay over a stored rollout for PPO. feats [T,N,F], dones [T,N] (1 if step t
        ENDED the episode -> reset hidden before t+1). Returns mean,std,value [T,N,*]."""
        if not self.memory:
            z = self.proj(feats)
            return self.actor_mean(z), self.logstd.exp(), self.critic(z).squeeze(-1)
        T, N = feats.shape[0], feats.shape[1]
        h = h0
        means, values = [], []
        for t in range(T):
            out, h = self.gru(feats[t].unsqueeze(0), h)
            z = out.squeeze(0)
            means.append(self.actor_mean(z)); values.append(self.critic(z).squeeze(-1))
            h = h * (1.0 - dones[t]).view(1, N, 1)          # reset hidden after terminal step
        return torch.stack(means), self.logstd.exp(), torch.stack(values)


# ---------------------------------------------------------------- helpers
def obs_to_t(obs, device):
    return (torch.as_tensor(obs["range"], dtype=torch.float32, device=device),
            torch.as_tensor(obs["coch"], dtype=torch.float32, device=device),
            torch.as_tensor(obs["prop"], dtype=torch.float32, device=device))


@torch.no_grad()
def evaluate(net, seeds, device, episodes_per=6, t_max=40, sim_threads=2, nobs=3, task="avoid"):
    """Per-ROOM macro-average: exactly `episodes_per` counted episodes for each room,
    then averaged over rooms. NB: an earlier version counted a fixed TOTAL of episodes
    across autoresetting envs, which OVER-SAMPLED fast-finishing rooms — and since a
    success/collision ENDS the episode early, that biased success and collision rate
    (goal success was inflated: rooms solved in ~1 step autoreset and flooded the sample
    while rooms that time out barely got counted). This version caps counts per env."""
    vec = VecEchoLocEnv([[s] for s in seeds], action_mode="continuous", sim_threads=sim_threads,
                        t_max=t_max, resample_room_every=0, nobs=nobs, task=task)
    room_len = [[] for _ in range(vec.n)]
    room_coll = [[] for _ in range(vec.n)]
    room_succ = [[] for _ in range(vec.n)]
    room_cov = [[] for _ in range(vec.n)]                      # coverage (explore task)
    cur = np.zeros(vec.n)
    h = torch.zeros(1, vec.n, net.H, device=device)
    obs = vec.reset()
    while any(len(s) < episodes_per for s in room_succ):
        f = net.feats(*obs_to_t(obs, device))
        mean, std, _v, h = net.step(f, h)
        acts = mean.clamp(-1, 1).cpu().numpy()                 # deterministic (mean) for eval
        obs, rew, term, trunc, done, _, infos = vec.step(acts)
        cur += 1
        d = torch.as_tensor(done, device=device, dtype=torch.float32).view(1, vec.n, 1)
        h = h * (1.0 - d)
        for i in range(vec.n):
            if done[i]:
                if len(room_succ[i]) < episodes_per:           # only count the first N per room
                    room_len[i].append(cur[i])
                    room_coll[i].append(1.0 if infos[i].get("collision") else 0.0)
                    room_succ[i].append(1.0 if infos[i].get("success") else 0.0)
                    room_cov[i].append(float(infos[i].get("coverage", 0)))
                cur[i] = 0
    vec.close()
    rmean = lambda rr: float(np.mean([np.mean(r) for r in rr]))  # macro-average over rooms
    return rmean(room_len), rmean(room_coll), rmean(room_succ), rmean(room_cov)


# ---------------------------------------------------------------- training
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    net = RecurrentAC(memory=not args.no_memory,
                      drop_range=args.drop_range, drop_coch=args.drop_coch).to(device)
    if args.init_ckpt:
        net.load_state_dict(torch.load(args.init_ckpt, map_location=device)["model"])
        print(f"warm-started from {os.path.relpath(args.init_ckpt)}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    GOAL = args.task == "goal"
    EXPLORE = args.task == "explore"
    # goal mode: pre-filter to rooms WITH an opening so _spawn_room never churns servers
    # (spawn/close hunting for openings across 12 threads was crashing subprocesses).
    train_seeds = (opening_seeds(1, args.train_rooms, args.nobs) if GOAL
                   else list(range(1, args.train_rooms + 1)))
    shards = [train_seeds[i::args.envs] for i in range(args.envs)]
    vec = VecEchoLocEnv(shards, action_mode="continuous", sim_threads=args.sim_threads,
                        t_max=args.t_max, resample_room_every=args.resample_every, nobs=args.nobs,
                        task=args.task)
    eval_seeds = (opening_seeds(600, args.eval_rooms, args.nobs) if GOAL
                  else list(range(600, 600 + args.eval_rooms)))
    N, T = vec.n, args.rollout

    os.makedirs(os.path.join(_HERE, "ckpt"), exist_ok=True)
    tagn = args.tag or "full"
    best_ck = os.path.join(_HERE, "ckpt", f"ppo_{args.task}_nobs{args.nobs}_{tagn}_s{args.seed}_best.pt")
    best_metric = -1.0                                   # success rate (goal) or flight len (avoid)

    print(f"device={device} | {N} envs x {args.sim_threads} thr | task={args.task} nobs={args.nobs} | "
          f"tag={tagn} seed={args.seed} | rollout {T} | target {args.steps} steps")
    ml, cr, sr, cov = evaluate(net, eval_seeds, device, t_max=args.t_max, nobs=args.nobs, task=args.task)
    print(f"[init policy] len {ml:.1f} coll {cr:.2f} success {sr:.2f} coverage {cov:.1f}")

    obs = vec.reset()
    h = torch.zeros(1, N, net.H, device=device)
    step = 0; t0 = time.time(); last_eval = 0
    while step < args.steps:
        # ---- collect a rollout ----
        b_rng = torch.zeros(T, N, 256, device=device)
        b_coch = torch.zeros(T, N, *COCH_SHAPE, device=device)
        b_prop = torch.zeros(T, N, 3, device=device)
        b_act = torch.zeros(T, N, 2, device=device)
        b_logp = torch.zeros(T, N, device=device)
        b_val = torch.zeros(T, N, device=device)
        b_rew = torch.zeros(T, N, device=device)
        b_done = torch.zeros(T, N, device=device)
        h0 = h.detach().clone()
        for t in range(T):
            ro, co, po = obs_to_t(obs, device)
            b_rng[t], b_coch[t], b_prop[t] = ro, co, po
            with torch.no_grad():
                f = net.feats(ro, co, po)
                mean, std, val, h = net.step(f, h)
                dist = torch.distributions.Normal(mean, std)
                raw = dist.sample()                             # unclipped (for logprob)
                logp = dist.log_prob(raw).sum(-1)
            acts = raw.clamp(-1, 1).cpu().numpy()               # env clips too
            obs, rew, term, trunc, done, _, _ = vec.step(acts)
            b_act[t], b_logp[t], b_val[t] = raw, logp, val
            b_rew[t] = torch.as_tensor(rew, device=device)
            b_done[t] = torch.as_tensor(done.astype(np.float32), device=device)
            dmask = (1.0 - b_done[t]).view(1, N, 1)
            h = h * dmask                                       # reset hidden after episode end
            step += N

        # ---- bootstrap value + GAE ----
        with torch.no_grad():
            f = net.feats(*obs_to_t(obs, device))
            _, _, last_val, _ = net.step(f, h)
        adv = torch.zeros(T, N, device=device); gae = torch.zeros(N, device=device)
        term_mask = b_done                                     # collision AND truncation both end the ep
        for t in reversed(range(T)):
            nextval = last_val if t == T - 1 else b_val[t + 1]
            nonterm = 1.0 - term_mask[t]
            delta = b_rew[t] + args.gamma * nextval * nonterm - b_val[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + b_val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- PPO epochs (recurrent replay from h0) ----
        for _ in range(args.epochs):
            feats = net.feats(b_rng, b_coch, b_prop)            # [T,N,F] one batched encode
            mean, std, val = net.rollout_eval(feats, b_done, h0)
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(b_act).sum(-1)
            ent = dist.entropy().sum(-1).mean()
            ratio = (logp - b_logp).exp()
            s1 = ratio * adv
            s2 = ratio.clamp(1 - args.clip, 1 + args.clip) * adv
            pg = -torch.min(s1, s2).mean()
            vloss = 0.5 * (val - ret).pow(2).mean()
            loss = pg + args.vf_coef * vloss - args.ent_coef * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
            opt.step()

        if step - last_eval >= args.eval_every:
            last_eval = step
            ml, cr, sr, cov = evaluate(net, eval_seeds, device, t_max=args.t_max, nobs=args.nobs, task=args.task)
            sps = step / (time.time() - t0)
            # best-ckpt metric: goal=success, explore=coverage, avoid=flight length
            metric = cov if EXPLORE else (sr if GOAL else ml)
            mark = ""
            if metric > best_metric:
                best_metric = metric
                torch.save({"model": net.state_dict(), "eval": (ml, cr, sr, cov), "step": step}, best_ck)
                mark = " *best"
            print(f"step {step:6d} | eval len {ml:5.1f} coll {cr:.2f} success {sr:.2f} cov {cov:4.1f} | "
                  f"ent {ent.item():+.2f} pg {pg.item():+.3f} v {vloss.item():.2f} | {sps:.0f} steps/s{mark}")

    ml, cr, sr, cov = evaluate(net, eval_seeds, device, episodes_per=10, t_max=args.t_max, nobs=args.nobs, task=args.task)
    bm = "coverage" if EXPLORE else ("success" if GOAL else "len")
    print(f"\nFINAL (held-out): len {ml:.1f} coll {cr:.2f} success {sr:.2f} coverage {cov:.1f} | "
          f"BEST {bm} {best_metric:.2f} -> {os.path.relpath(best_ck)}")
    vec.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--sim-threads", type=int, default=2)
    ap.add_argument("--t-max", type=int, default=40)
    ap.add_argument("--nobs", type=int, default=0)
    ap.add_argument("--train-rooms", type=int, default=60)
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--resample-every", type=int, default=25)
    ap.add_argument("--rollout", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=12000)
    ap.add_argument("--init-ckpt", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", choices=["avoid", "goal", "explore"], default="avoid",
                    help="avoid = Stage-0 survive; goal = Stage-2 home to the opening; "
                         "explore = coverage avoidance (must move to cover new ground, no parking)")
    ap.add_argument("--tag", type=str, default="", help="checkpoint/label tag (e.g. nomem, rangeonly)")
    ap.add_argument("--no-memory", action="store_true", help="ablate the GRU (feedforward policy)")
    ap.add_argument("--drop-range", action="store_true", help="ablate the matched-filter range stream")
    ap.add_argument("--drop-coch", action="store_true", help="ablate the binaural cochleagram stream")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
