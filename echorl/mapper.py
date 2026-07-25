"""Known-pose occupancy MAPPER for the cluttered doorway task (EchoRL mapping expt).

The §5.7 lever made explicit: integrate the echo sequence over time into a metric
occupancy MAP, then plan on it (mapping.grid_plan_action). This module is the
perception half -- a scatter-and-refine neural mapper:

  1. per ping, encode (range,coch) with the codebase TwoStreamEncoder -> feature f_t;
  2. SCATTER f_t into a room-centred canvas at the KNOWN pose cell (index_add,
     mean-pooled per cell) + a visitation/coverage channel;
  3. a small U-Net refines the sparse canvas into (a) an OCCUPANCY grid [G,G]
     (walls + interior pillars) and (b) a DOORWAY heatmap [G,G] whose argmax is the
     goal point.

Trained supervised on random-exploration trajectories (mapper_collect.py) with true
grids as targets (mapping.true_occupancy / goal_from_openings). Because targets are
STATIC geometry (not action-consequences), and the map is conditioned on absolute
poses, this sidesteps the myopic-dynamics ceiling that capped the latent world model
(§5.1). Open-loop quality (M1) = occupancy IoU + doorway-localization error vs #pings.
"""
from __future__ import annotations
import argparse, os, sys, math, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from models import Normalizer                              # noqa: E402
from model_xview import TwoStreamEncoder                   # noqa: E402
import mapping                                             # noqa: E402
from mapping import GRID_G, CELL, GRID_SPAN, world_to_grid, true_occupancy, goal_from_openings  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FEAT_C = 24            # echo feature channels scattered onto the canvas

_GG = None
def gaussian_heatmap(goal_flat, sigma=1.5):
    """[B] flat goal-cell -> [B, G*G] normalized Gaussian soft target (label smoothing)."""
    global _GG
    if _GG is None:
        ii, jj = torch.meshgrid(torch.arange(GRID_G), torch.arange(GRID_G), indexing="ij")
        _GG = (ii.reshape(-1).to(DEV), jj.reshape(-1).to(DEV))
    gi = (goal_flat // GRID_G).float().unsqueeze(1)
    gj = (goal_flat % GRID_G).float().unsqueeze(1)
    d2 = (_GG[0].float().unsqueeze(0) - gi) ** 2 + (_GG[1].float().unsqueeze(0) - gj) ** 2
    hm = torch.exp(-d2 / (2 * sigma ** 2))
    return hm / hm.sum(1, keepdim=True)


# ------------------------------------------------------------------- model
def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class UNet(nn.Module):
    """Compact U-Net: (Cin,48,48) -> two logit maps (occupancy, doorway)."""
    def __init__(self, cin, base=32):
        super().__init__()
        def cbr(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), _gn(co), nn.GELU())
        self.e1 = nn.Sequential(cbr(cin, base), cbr(base, base))          # 48
        self.e2 = nn.Sequential(cbr(base, base * 2), cbr(base * 2, base * 2))    # 24
        self.e3 = nn.Sequential(cbr(base * 2, base * 4), cbr(base * 4, base * 4))  # 12
        self.b  = nn.Sequential(cbr(base * 4, base * 4), cbr(base * 4, base * 4))  # 6
        self.d3 = nn.Sequential(cbr(base * 8, base * 4), cbr(base * 4, base * 2))
        self.d2 = nn.Sequential(cbr(base * 4, base * 2), cbr(base * 2, base))
        self.d1 = nn.Sequential(cbr(base * 2, base), cbr(base, base))
        self.occ_head = nn.Conv2d(base, 1, 1)
        self.door_head = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)                     # 48
        e2 = self.e2(self.pool(e1))         # 24
        e3 = self.e3(self.pool(e2))         # 12
        b  = self.b(self.pool(e3))          # 6
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")
        d3 = self.d3(torch.cat([up(b), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        return self.occ_head(d1).squeeze(1), self.door_head(d1).squeeze(1)


class OccMapper(nn.Module):
    def __init__(self, feat_c=FEAT_C):
        super().__init__()
        self.norm = Normalizer()
        self.enc = TwoStreamEncoder(128, 128)
        # proj is CONDITIONED ON YAW (sin,cos): the coch bearing cue is egocentric, so the
        # world-frame direction to a surface/void is a function of (echo, heading). Without
        # yaw the network cannot place the binaural bearing on the allocentric map -- the
        # single biggest fix over the yaw-blind v0 (val goal-err was pinned at ~3 m).
        self.proj = nn.Sequential(nn.Linear(256 + 2, 128), nn.GELU(), nn.Linear(128, feat_c))
        self.feat_c = feat_c
        self.unet = UNet(feat_c + 1)        # +1 coverage channel

    def encode_steps(self, rng, coch, yaw_sc):
        """[N,256],[N,2,64,48],[N,2] -> [N,feat_c] world-aware echo features."""
        rn, cn = self.norm(rng, coch)
        zr, zc = self.enc(cn, rn)
        return self.proj(torch.cat([zr, zc, yaw_sc], 1))

    def scatter(self, feats, cell_ix, cell_iz, batch_idx, B):
        """Scatter [N,feat_c] into canvas [B, feat_c+1, G, G] at (batch, ix, iz),
        mean-pooled per cell; last channel = log1p(visit count)."""
        C = self.feat_c
        flat = (batch_idx * GRID_G + cell_ix) * GRID_G + cell_iz          # [N]
        canvas = feats.new_zeros(B * GRID_G * GRID_G, C)
        canvas.index_add_(0, flat, feats)
        cnt = feats.new_zeros(B * GRID_G * GRID_G)
        cnt.index_add_(0, flat, torch.ones_like(flat, dtype=feats.dtype))
        canvas = canvas / cnt.clamp(min=1).unsqueeze(1)                   # mean per cell
        canvas = canvas.view(B, GRID_G, GRID_G, C).permute(0, 3, 1, 2).contiguous()
        cov = torch.log1p(cnt.view(B, 1, GRID_G, GRID_G))
        return torch.cat([canvas, cov], 1)

    def forward(self, rng, coch, yaw_sc, cell_ix, cell_iz, batch_idx, B):
        feats = self.encode_steps(rng, coch, yaw_sc)
        canvas = self.scatter(feats, cell_ix, cell_iz, batch_idx, B)
        return self.unet(canvas)            # occ_logits[B,G,G], door_logits[B,G,G]


# ------------------------------------------------------------------- data
def load_corpus(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_targets(ep):
    """True occupancy grid [G,G] {0,1} and goal cell index (flattened) for one episode."""
    occ = true_occupancy(ep["Lx"], ep["Lz"], ep["obstacles"])
    gx, gz = goal_from_openings(ep["openings"])
    gix, giz = world_to_grid(gx, gz)
    return occ, int(gix) * GRID_G + int(giz), (gx, gz)


class MapDataset(torch.utils.data.Dataset):
    """Each item = one episode; __getitem__ picks a random prefix length so the mapper
    learns to predict from partial trajectories (needed for closed-loop early steps)."""
    def __init__(self, eps, min_k=1, rng_seed=0):
        self.eps = eps
        self.min_k = min_k
        self.rng = np.random.default_rng(rng_seed)
        self.tgt = [build_targets(e) for e in eps]

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, i):
        e = self.eps[i]
        T = len(e["poses"])
        k = int(self.rng.integers(self.min_k, T + 1))
        poses = e["poses"][:k]
        ix, iz = world_to_grid(poses[:, 0], poses[:, 1])
        yaw = np.radians(poses[:, 2].astype(np.float32))
        ysc = np.stack([np.sin(yaw), np.cos(yaw)], 1)          # [k,2]
        occ, goal_flat, _ = self.tgt[i]
        return dict(range=e["range"][:k].astype(np.float32),
                    coch=e["coch"][:k].astype(np.float32), ysc=ysc.astype(np.float32),
                    ix=ix.astype(np.int64), iz=iz.astype(np.int64),
                    occ=occ, goal=goal_flat)


def collate(batch):
    rngs = np.concatenate([b["range"] for b in batch], 0)
    cochs = np.concatenate([b["coch"] for b in batch], 0)
    ysc = np.concatenate([b["ysc"] for b in batch], 0)
    ix = np.concatenate([b["ix"] for b in batch], 0)
    iz = np.concatenate([b["iz"] for b in batch], 0)
    bidx = np.concatenate([np.full(len(b["ix"]), j, np.int64) for j, b in enumerate(batch)], 0)
    occ = np.stack([b["occ"] for b in batch], 0)
    goal = np.array([b["goal"] for b in batch], np.int64)
    return (torch.from_numpy(rngs), torch.from_numpy(cochs), torch.from_numpy(ysc),
            torch.from_numpy(ix), torch.from_numpy(iz), torch.from_numpy(bidx),
            torch.from_numpy(occ), torch.from_numpy(goal))


# ------------------------------------------------------------------- train
def train(args):
    eps = load_corpus(args.corpus)
    n_val = max(8, int(0.1 * len(eps)))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(eps))
    val_eps = [eps[i] for i in perm[:n_val]]
    tr_eps = [eps[i] for i in perm[n_val:]]
    print(f"corpus {len(eps)} eps -> train {len(tr_eps)} / val {len(val_eps)}")

    tr = MapDataset(tr_eps, rng_seed=1)
    dl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                     collate_fn=collate, num_workers=0, drop_last=True)
    model = OccMapper().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = -1.0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for rngs, cochs, ysc, ix, iz, bidx, occ, goal in dl:
            rngs, cochs, ysc = rngs.to(DEV), cochs.to(DEV), ysc.to(DEV)
            ix, iz, bidx = ix.to(DEV), iz.to(DEV), bidx.to(DEV)
            occ, goal = occ.to(DEV), goal.to(DEV)
            B = occ.shape[0]
            occ_l, door_l = model(rngs, cochs, ysc, ix, iz, bidx, B)
            loss_occ = F.binary_cross_entropy_with_logits(occ_l, occ)
            tgt_hm = gaussian_heatmap(goal, sigma=1.5)                # soft label (smoother gradient)
            loss_goal = -(tgt_hm * F.log_softmax(door_l.reshape(B, -1), 1)).sum(1).mean()
            loss = loss_occ + args.w_goal * loss_goal
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        piou, gerr, siou = evaluate_quality(model, val_eps, ks=(args.t_max,))
        # Select on OCCUPANCY quality: the doorway goal is handled by the fusion estimator
        # (mapper_fuse), so this checkpoint only needs to provide a good map for the planner.
        score = siou[args.t_max] + piou[args.t_max]
        tag = ""
        if score > best:
            best = score; tag = " *"
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(model.state_dict(), args.out)
        print(f"ep {ep:2d}  loss {tot/nb:.3f}  val pillar-IoU {piou[args.t_max]:.3f}  "
              f"solid-IoU {siou[args.t_max]:.3f}  goal-err {gerr[args.t_max]:.2f} m  "
              f"({time.time()-t0:.0f}s){tag}", flush=True)
    print(f"best score {best:.3f} -> {args.out}")


# ------------------------------------------------------------- open-loop M1 eval
@torch.no_grad()
def predict_grid(model, poses, rng_seq, coch_seq):
    """Run the mapper on a prefix of one trajectory -> (occ_prob[G,G], goal_xz, door_prob)."""
    ix, iz = world_to_grid(poses[:, 0], poses[:, 1])
    yaw = np.radians(poses[:, 2].astype(np.float32))
    ysc = np.stack([np.sin(yaw), np.cos(yaw)], 1)
    rngs = torch.as_tensor(rng_seq, dtype=torch.float32, device=DEV)
    cochs = torch.as_tensor(coch_seq, dtype=torch.float32, device=DEV)
    ysct = torch.as_tensor(ysc, dtype=torch.float32, device=DEV)
    ixt = torch.as_tensor(ix, dtype=torch.int64, device=DEV)
    izt = torch.as_tensor(iz, dtype=torch.int64, device=DEV)
    bidx = torch.zeros(len(ix), dtype=torch.int64, device=DEV)
    occ_l, door_l = model(rngs, cochs, ysct, ixt, izt, bidx, 1)
    occ_p = torch.sigmoid(occ_l[0]).cpu().numpy()
    di = int(door_l[0].reshape(-1).argmax())
    gix, giz = di // GRID_G, di % GRID_G
    gx = (gix + 0.5) * CELL - GRID_SPAN / 2
    gz = (giz + 0.5) * CELL - GRID_SPAN / 2
    return occ_p, (float(gx), float(gz)), torch.softmax(door_l[0].reshape(-1), 0).cpu().numpy()


@torch.no_grad()
def evaluate_quality(model, eps, ks=(5, 10, 20, 40), thr=0.5):
    """Per #pings k: solid-IoU (whole grid: walls+pillars, easy exterior included),
    PILLAR-IoU (occupied cells INSIDE the true room rectangle -- the hard interior
    obstacles), and doorway-localization error (m). Returns (pillar_iou, gerr, extra)
    where extra holds solid_iou for reference."""
    model.eval()
    X, Z = mapping.grid_cell_centres()
    piou = {k: [] for k in ks}; gerr = {k: [] for k in ks}; siou = {k: [] for k in ks}
    for e in eps:
        occ_t, _, goal_xz = build_targets(e)
        gt = occ_t > 0.5
        interior = (np.abs(X) < e["Lx"] / 2) & (np.abs(Z) < e["Lz"] / 2)
        T = len(e["poses"])
        for k in ks:
            kk = min(k, T)
            occ_p, gest, _ = predict_grid(model, e["poses"][:kk], e["range"][:kk].astype(np.float32),
                                          e["coch"][:kk].astype(np.float32))
            pred = occ_p > thr
            u = (pred | gt).sum()
            siou[k].append((pred & gt).sum() / max(u, 1))
            pg = gt & interior; pp = pred & interior
            pu = (pp | pg).sum()
            piou[k].append((pp & pg).sum() / pu if pu > 0 else np.nan)
            gerr[k].append(math.hypot(gest[0] - goal_xz[0], gest[1] - goal_xz[1]))
    return ({k: float(np.nanmean(v)) for k, v in piou.items()},
            {k: float(np.mean(v)) for k, v in gerr.items()},
            {k: float(np.mean(v)) for k, v in siou.items()})


def cmd_evalq(args):
    from ppo import opening_seeds
    model = OccMapper().to(DEV)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEV))
    # eval on a freshly collected held-out set (600+), NOT the training corpus
    import mapper_collect
    seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    print(f"Collecting held-out eval trajectories ({len(seeds)} rooms)...")
    eps = mapper_collect.collect(seeds, args.ep_per_room, nobs=args.nobs, t_max=args.t_max,
                                 workers=min(12, len(seeds)), base_seed=99)
    ks = (2, 5, 10, 20, args.t_max)
    piou, gerr, siou = evaluate_quality(model, eps, ks=ks)
    print(f"\nM1 open-loop mapping quality (held-out nobs={args.nobs}, {len(eps)} trajectories):")
    print(f"  {'#pings':>7} {'pillar-IoU':>11} {'solid-IoU':>10} {'goal-err(m)':>12}")
    for k in ks:
        print(f"  {k:>7} {piou[k]:>11.3f} {siou[k]:>10.3f} {gerr[k]:>12.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--corpus", default="echorl/mapper_corpus_nobs3.pkl")
    p.add_argument("--out", default="echorl/ckpt/mapper_nobs3.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w-goal", type=float, default=1.0)
    p.add_argument("--t-max", type=int, default=40)
    p.set_defaults(func=train)
    q = sub.add_parser("evalq")
    q.add_argument("--ckpt", default="echorl/ckpt/mapper_nobs3.pt")
    q.add_argument("--eval-rooms", type=int, default=12)
    q.add_argument("--ep-per-room", type=int, default=2)
    q.add_argument("--nobs", type=int, default=3)
    q.add_argument("--t-max", type=int, default=40)
    q.set_defaults(func=cmd_evalq)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
