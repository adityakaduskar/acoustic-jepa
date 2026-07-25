"""Recurrent (temporal) doorway estimator -- breaks the fusion saturation (§5.8 follow-up).

The known-pose fusion (mapper_fuse.py) saturates at ~2.8 m because it predicts each
ping's doorway bearing INDEPENDENTLY, then averages -- so it cannot exploit the one cue
that resolves binaural front-back ambiguity: MOTION PARALLAX. As the agent turns and
translates, the true doorway stays fixed in the world while the front-back mirror
hypothesis is inconsistent across viewpoints. A recurrent model that integrates the
ping SEQUENCE (with poses) before predicting can track which hypothesis stays
consistent -- exactly §5.7's "temporal model that integrates pings over time to build
state before predicting", applied to the doorway estimator.

Architecture: per-ping TwoStreamEncoder feature + pose -> GRU -> direct (x,z) doorway
regression in the world frame. Trained end-to-end on trajectories (random prefix
lengths). Compared head-to-head with the independent-fusion estimator on held-out
doorway-localization error vs #pings, then dropped into closed-loop nav.
"""
from __future__ import annotations
import argparse, os, sys, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from models import Normalizer
from model_xview import TwoStreamEncoder
from mapping import goal_from_openings
import mapper

DEV = mapper.DEV
POSE_SCALE = 6.0


def pose_feats(pose):
    """[N,3] (x,z,yaw deg) -> [N,4] (x/s, z/s, sin yaw, cos yaw)."""
    y = torch.deg2rad(pose[:, 2])
    return torch.stack([pose[:, 0] / POSE_SCALE, pose[:, 1] / POSE_SCALE,
                        torch.sin(y), torch.cos(y)], 1)


class SeqGoalNet(nn.Module):
    def __init__(self, feat=64, hid=128):
        super().__init__()
        self.norm = Normalizer(); self.enc = TwoStreamEncoder(128, 128)
        self.proj = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, feat))
        self.gru = nn.GRU(feat + 4, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, 128), nn.GELU(), nn.Linear(128, 2))

    def step_feats(self, rng, coch, pose):
        rn, cn = self.norm(rng, coch); zr, zc = self.enc(cn, rn)
        return torch.cat([self.proj(torch.cat([zr, zc], 1)), pose_feats(pose)], 1)  # [N,feat+4]

    def forward(self, rng, coch, pose, lengths):
        """Flat steps [N,...] grouped by `lengths` (list, sum=N) -> [B,2] doorway (x,z)."""
        x = self.step_feats(rng, coch, pose)                      # [N,feat+4]
        B = len(lengths); Lmax = max(lengths); D = x.shape[1]
        pad = x.new_zeros(B, Lmax, D)
        off = 0
        for i, L in enumerate(lengths):
            pad[i, :L] = x[off:off + L]; off += L
        packed = nn.utils.rnn.pack_padded_sequence(pad, lengths, batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.gru(packed)                                   # h [1,B,hid]
        return self.head(h[-1]) * POSE_SCALE                      # [B,2] metres


# ------------------------------------------------------------------- data
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, eps, rng_seed=0):
        self.eps = eps; self.rng = np.random.default_rng(rng_seed)
        self.goal = [goal_from_openings(e["openings"]) for e in eps]

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, i):
        e = self.eps[i]; T = len(e["poses"]); k = int(self.rng.integers(1, T + 1))
        return dict(range=e["range"][:k].astype(np.float32), coch=e["coch"][:k].astype(np.float32),
                    pose=e["poses"][:k].astype(np.float32), goal=np.array(self.goal[i], np.float32))


def collate(batch):
    lengths = [len(b["pose"]) for b in batch]
    rng = np.concatenate([b["range"] for b in batch], 0)
    coch = np.concatenate([b["coch"] for b in batch], 0)
    pose = np.concatenate([b["pose"] for b in batch], 0)
    goal = np.stack([b["goal"] for b in batch], 0)
    return (torch.from_numpy(rng), torch.from_numpy(coch), torch.from_numpy(pose),
            lengths, torch.from_numpy(goal))


@torch.no_grad()
def predict(model, poses, rseq, cseq):
    r = torch.as_tensor(rseq, dtype=torch.float32, device=DEV)
    c = torch.as_tensor(cseq, dtype=torch.float32, device=DEV)
    p = torch.as_tensor(poses, dtype=torch.float32, device=DEV)
    out = model(r, c, p, [len(poses)])[0].cpu().numpy()
    return float(out[0]), float(out[1])


@torch.no_grad()
def eval_seq(model, eps, ks=(1, 2, 5, 10, 20, 40)):
    model.eval(); errs = {k: [] for k in ks}
    for e in eps:
        gx, gz = goal_from_openings(e["openings"]); T = len(e["poses"])
        for k in ks:
            kk = min(k, T)
            gxh, gzh = predict(model, e["poses"][:kk], e["range"][:kk].astype(np.float32),
                               e["coch"][:kk].astype(np.float32))
            errs[k].append(math.hypot(gxh - gx, gzh - gz))
    return {k: float(np.mean(v)) for k, v in errs.items()}


def train(args):
    eps = mapper.load_corpus(args.corpus)
    rng = np.random.default_rng(0); perm = rng.permutation(len(eps))
    nval = max(8, int(0.1 * len(eps)))
    va = [eps[i] for i in perm[:nval]]; tr = [eps[i] for i in perm[nval:]]
    print(f"corpus {len(eps)} -> train {len(tr)} / val {len(va)}")
    dl = torch.utils.data.DataLoader(SeqDataset(tr, 1), batch_size=args.batch, shuffle=True,
                                     collate_fn=collate, drop_last=True)
    m = SeqGoalNet().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), args.lr, weight_decay=1e-4)
    best = 1e9
    for ep in range(args.epochs):
        m.train(); t0 = time.time(); tot = 0.0; nb = 0
        for r, c, p, lengths, goal in dl:
            r, c, p, goal = r.to(DEV), c.to(DEV), p.to(DEV), goal.to(DEV)
            pred = m(r, c, p, lengths)
            loss = F.mse_loss(pred / POSE_SCALE, goal / POSE_SCALE)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        e_full = eval_seq(m, va, ks=(args.t_max,))[args.t_max]
        tag = ""
        if e_full < best:
            best = e_full; tag = " *"
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(m.state_dict(), args.out)
        print(f"ep {ep:2d}  loss {tot/nb:.4f}  val seq-goal-err@full {e_full:.2f} m  "
              f"({time.time()-t0:.0f}s){tag}", flush=True)
    print(f"best seq-goal-err {best:.2f} m -> {args.out}")


def cmd_evalq(args):
    from ppo import opening_seeds
    import mapper_collect
    m = SeqGoalNet().to(DEV); m.load_state_dict(torch.load(args.ckpt, map_location=DEV)); m.eval()
    seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    print(f"Collecting held-out eval trajectories ({len(seeds)} rooms)...")
    eps = mapper_collect.collect(seeds, args.ep_per_room, nobs=args.nobs, t_max=args.t_max,
                                 workers=min(12, len(seeds)), base_seed=99)
    ks = (1, 2, 5, 10, 20, args.t_max)
    ge = eval_seq(m, eps, ks=ks)
    print(f"\nRecurrent doorway-loc error vs #pings (held-out nobs={args.nobs}, {len(eps)} traj):")
    print(f"  {'#pings':>7} {'seq err(m)':>11}   (fusion saturates ~2.8 m)")
    for k in ks:
        print(f"  {k:>7} {ge[k]:>11.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--corpus", default="echorl/mapper_corpus_nobs3.pkl")
    p.add_argument("--out", default="echorl/ckpt/seqgoal_nobs3.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--t-max", type=int, default=40)
    p.set_defaults(func=train)
    q = sub.add_parser("evalq")
    q.add_argument("--ckpt", default="echorl/ckpt/seqgoal_nobs3.pt")
    q.add_argument("--eval-rooms", type=int, default=12)
    q.add_argument("--ep-per-room", type=int, default=3)
    q.add_argument("--nobs", type=int, default=3)
    q.add_argument("--t-max", type=int, default=40)
    q.set_defaults(func=cmd_evalq)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
