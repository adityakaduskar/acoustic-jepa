"""EchoRL §5.9 -- the two fixes §5.8's diagnosis pointed to: a REACTIVE bearing goal
and a CLEARANCE-FIELD occupancy map.

§5.8 capped the full learned map+planner at 0.49 (perfect-map ceiling 0.96) and the
decomposition named both halves:

  * the DOORWAY half (perfect goal -> 0.67, the bigger lever) saturates at 2.8 m of
    localization error because the binaural bearing is *two-valued* (a 2-mic array
    cannot resolve front-back from one ping), so per-ping errors are CORRELATED across
    poses and averaging cannot drive them out. The task's own §5 result says this cue
    works REACTIVELY for PPO -- metric localization is the wrong consumer for it.
  * the OBSTACLE half (perfect occupancy -> 0.60) fails because the scatter-CNN gets
    pillar-IoU ~= 0.005 -- while §5.6 showed a per-ping CLEARANCE readout decodes
    obstacle proximity at R^2=0.95 on real observations. That signal is simply unused
    by the mapper.

So this module changes the CONSUMER of each signal rather than trying to make either
estimate more accurate (which §5.8's recurrent estimator did -- 2.84 -> 2.04 m -- while
navigating *worse*, 0.29 vs 0.49):

  A. REACTIVE BEARING GOAL (`react`). Stop localizing the doorway as a point. Each
     step, take the current ping's ego-bearing, low-pass it in the WORLD frame, and
     project a moving SUBGOAL a lookahead ahead of the *current* pose along it. The
     grid planner then routes to that subgoal around mapped obstacles -- "follow the
     bearing, plan around the pillars". No absolute doorway coordinate is ever formed,
     so the 2.8 m fusion floor cannot bind; a bearing that flips (front-back) is simply
     re-estimated next step instead of being permanently averaged into a wrong point.

  B. CLEARANCE-FIELD OCCUPANCY (`clear`). A per-ping heteroscedastic clearance net
     (distance to the nearest surface, the §5.6 readout, standalone) drives a classical
     sonar log-odds update at the known pose: CARVE the disc of radius c free (sound --
     a clearance reading asserts nothing is nearer than c), and add weak occupied
     evidence on the RING at radius c (only one point of the ring is truly a surface,
     but rings from many poses intersect on real geometry while free-carving from other
     poses erases the rest). Fully analytic given the scalar readout -- no map CNN.

The 2x2 (goal: fuse|react) x (occ: scatter|clear|both) ablation isolates each fix and
their interaction against the §5.8 baseline (scatter+fuse = 0.49).

Run:
  python echorl/mapper_react.py train-clear                    # B: per-ping clearance net
  python echorl/mapper_react.py mapq                           # B open-loop: pillar-IoU
  python echorl/mapper_react.py nav --modes scatter+fuse both+react ...
"""
from __future__ import annotations
import argparse, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from models import Normalizer                                            # noqa: E402
from model_xview import TwoStreamEncoder                                 # noqa: E402
from env import _wall_clearance, _obstacle_clearance                     # noqa: E402
import mapping                                                           # noqa: E402
from mapping import (GRID_G, CELL, grid_cell_centres, inflate,           # noqa: E402
                     true_occupancy, goal_from_openings, run_map_nav)
import mapper                                                            # noqa: E402
from mapper_probe import ego_bearing                                     # noqa: E402

DEV = mapper.DEV


# ============================================================ B. clearance net
class ClearNet(nn.Module):
    """(range, coch) -> distance to the NEAREST surface at the ping pose, with a
    heteroscedastic log-variance so unreadable pings can be down-weighted in the map."""

    def __init__(self):
        super().__init__()
        self.norm = Normalizer()
        self.enc = TwoStreamEncoder(128, 128)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, r, c):
        rn, cn = self.norm(r, c)
        zr, zc = self.enc(cn, rn)
        o = self.head(torch.cat([zr, zc], 1))
        return F.softplus(o[:, 0]), o[:, 1].clamp(-6, 6)      # clearance (m), logvar


def true_clearance(x, z, Lx, Lz, obstacles):
    """Distance from (x,z) to the nearest surface -- walls or ear-height pillars. NB the
    wall term treats the doorway opening as solid (env's own convention), so clearance is
    mildly pessimistic right at the goal; the ring evidence there lands on the wall plane
    the opening sits in, which is where the surface actually is either side of it."""
    return min(_wall_clearance(x, z, Lx, Lz), _obstacle_clearance(x, z, obstacles))


def nearest_surface_point(x, z, Lx, Lz, obstacles):
    """The actual (px,pz) of the nearest surface -- the DIRECTIONAL version of clearance.

    Needed because the corpus' median clearance is only 0.41 m (a random walk hugs walls),
    so an isotropic ring at radius c is dominated by the wall behind the agent and sprays
    phantom evidence into the room interior. Knowing *which way* the nearest surface lies
    turns one ping into a point instead of a circle. Walls: nearest point on the room
    rectangle's boundary. Pillars: nearest point on the AABB."""
    best_d, best_p = math.inf, (x, z)
    for (wx, wz) in ((Lx / 2, z), (-Lx / 2, z), (x, Lz / 2), (x, -Lz / 2)):
        d = math.hypot(wx - x, wz - z)
        if d < best_d:
            best_d, best_p = d, (wx, wz)
    for (cx, cz, hx, hz) in mapping.ear_height_obstacles(obstacles):
        qx = min(max(x, cx - hx), cx + hx)          # clamp to the AABB
        qz = min(max(z, cz - hz), cz + hz)
        d = math.hypot(qx - x, qz - z)
        if d < best_d:
            best_d, best_p = d, (qx, qz)
    return best_d, best_p


class SurfNet(nn.Module):
    """(range, coch) -> ego (dist, sin b, cos b) to the nearest SURFACE POINT + logvars.
    Same head structure as mapper_fuse.EgoVecNet, but the target is the nearest surface
    rather than the doorway, so its output places directed occupied evidence on the map."""

    def __init__(self):
        super().__init__()
        self.norm = Normalizer()
        self.enc = TwoStreamEncoder(128, 128)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 5))

    def forward(self, r, c):
        rn, cn = self.norm(r, c)
        zr, zc = self.enc(cn, rn)
        o = self.head(torch.cat([zr, zc], 1))
        dist = F.softplus(o[:, 0])
        bn = torch.linalg.norm(o[:, 1:3], dim=1, keepdim=True).clamp(min=1e-6)
        b = o[:, 1:3] / bn
        return dist, b[:, 0], b[:, 1], o[:, 3].clamp(-6, 6), o[:, 4].clamp(-6, 6)


def build_clear_samples(eps, directional=False):
    """Corpus episodes -> (range, coch[f16], targets) per ping. Labels are analytic from
    the logged pose + room geometry, so no re-collection is needed. targets = clearance
    [N] or, if directional, [N,3] = (dist, sin bearing, cos bearing) to the surface."""
    R, C, Y = [], [], []
    for e in eps:
        for t in range(len(e["poses"])):
            px, pz, pyaw = e["poses"][t]
            R.append(e["range"][t]); C.append(e["coch"][t])
            if directional:
                d, (sx, sz) = nearest_surface_point(px, pz, e["Lx"], e["Lz"], e["obstacles"])
                b = ego_bearing(px, pz, pyaw, sx, sz)
                Y.append([d, math.sin(b), math.cos(b)])
            else:
                Y.append(true_clearance(px, pz, e["Lx"], e["Lz"], e["obstacles"]))
    return (np.array(R, np.float32), np.stack(C), np.array(Y, np.float32))


def _r2(pred, targ):
    ss = float(((pred - targ) ** 2).sum())
    st = float(((targ - targ.mean()) ** 2).sum() + 1e-9)
    return 1.0 - ss / st


@torch.no_grad()
def eval_clear(m, R, C, Y, directional=False, bs=1024):
    """-> (dist R2, dist MAE m, bearing MAE deg or nan)."""
    m.eval(); ds, bs_, bc_ = [], [], []
    for i in range(0, len(R), bs):
        r = torch.tensor(R[i:i + bs], device=DEV)
        c = torch.tensor(C[i:i + bs].astype(np.float32), device=DEV)
        o = m(r, c)
        ds.append(o[0].cpu().numpy())
        if directional:
            bs_.append(o[1].cpu().numpy()); bc_.append(o[2].cpu().numpy())
    p = np.concatenate(ds)
    yd = Y[:, 0] if directional else Y
    if not directional:
        return _r2(p, yd), float(np.abs(p - yd).mean()), float("nan")
    pb = np.arctan2(np.concatenate(bs_), np.concatenate(bc_))
    tb = np.arctan2(Y[:, 1], Y[:, 2])
    err = np.degrees(np.abs(np.arctan2(np.sin(pb - tb), np.cos(pb - tb))))
    return _r2(p, yd), float(np.abs(p - yd).mean()), float(err.mean())


def train_clear(args):
    directional = args.directional
    if args.out is None:
        args.out = (f"echorl/ckpt/{'surfnet' if directional else 'clearnet'}_nobs3"
                    + (f"_s{args.seed}" if args.seed else "") + ".pt")
    eps = mapper.load_corpus(args.corpus)
    rng = np.random.default_rng(0); perm = rng.permutation(len(eps))
    nval = max(8, int(0.1 * len(eps)))
    va = [eps[i] for i in perm[:nval]]; tr = [eps[i] for i in perm[nval:]]
    Rtr, Ctr, Ytr = build_clear_samples(tr, directional)
    Rva, Cva, Yva = build_clear_samples(va, directional)
    dtr = Ytr[:, 0] if directional else Ytr
    print(f"{'surface (dist+bearing)' if directional else 'clearance (dist)'} net: "
          f"train {len(Rtr)} pings / val {len(Rva)}  "
          f"(range mean {dtr.mean():.2f} m, sd {dtr.std():.2f})", flush=True)

    torch.manual_seed(args.seed)
    m = (SurfNet() if directional else ClearNet()).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), args.lr, weight_decay=1e-4)
    ysd = float(dtr.std() + 1e-6)
    N, bs, best = len(Rtr), 512, -1e9
    for ep in range(args.epochs):
        m.train(); idx = np.random.default_rng(ep).permutation(N); t0 = time.time()
        for i in range(0, N, bs):
            j = idx[i:i + bs]
            r = torch.tensor(Rtr[j], device=DEV)
            c = torch.tensor(Ctr[j].astype(np.float32), device=DEV)
            y = torch.tensor(Ytr[j], device=DEV)
            if directional:
                dist, sb, cb, lvd, lvb = m(r, c)
                ed = (dist - y[:, 0]) / ysd
                eb = (sb - y[:, 1]) ** 2 + (cb - y[:, 2]) ** 2
                loss = ((0.5 * torch.exp(-lvd) * ed ** 2 + 0.5 * lvd).mean()
                        + (0.5 * torch.exp(-lvb) * eb + 0.5 * lvb).mean())
            else:
                ch, lv = m(r, c)
                e = (ch - y) / ysd
                loss = (0.5 * torch.exp(-lv) * e ** 2 + 0.5 * lv).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        r2, mae, bmae = eval_clear(m, Rva, Cva, Yva, directional)
        # rank on the map-relevant quantity: the placement error of the surface point
        score = r2 - (0.0 if math.isnan(bmae) else bmae / 90.0)
        tag = ""
        if score > best:
            best = score; tag = " *"
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save(m.state_dict(), args.out)
        print(f"ep {ep:2d}  val dist R2 {r2:+.3f}  MAE {mae:.3f} m"
              + ("" if math.isnan(bmae) else f"  bearing {bmae:5.1f} deg")
              + f"  ({time.time()-t0:.0f}s){tag}", flush=True)
    print(f"best -> {args.out}  (§5.6's in-WM clearance readout was R2=0.95)")


# ================================================ B. clearance -> occupancy field
class ClearField:
    """Incremental log-odds occupancy from per-ping clearance at known poses.

    Per ping (pose p, clearance c, confidence w):
      free : cells with |cell - p| < c - shrink   -> log-odds -= l_free * w   (sound)
      ring : cells with | |cell-p| - c | <= hw    -> log-odds += l_occ  * w   (weak)
    Unknown cells stay at 0 (p=0.5) and are treated as FREE by the planner threshold,
    which is the right prior for navigation (an all-unknown map must not block).

    Directional mode (`add_dir`) replaces the ring with a Gaussian BLOB at the estimated
    surface POINT, which matters here because the corpus' median clearance is only 0.41 m:
    an isotropic ring at that radius is dominated by the wall the agent is hugging and
    sprays phantom occupancy into the interior, exactly where the pillars must be read.
    """

    def __init__(self, l_free=0.7, l_occ=0.45, shrink=0.12, ring_hw=0.18, clip=6.0,
                 blob_sd=0.30):
        self.X, self.Z = grid_cell_centres()
        self.L = np.zeros((GRID_G, GRID_G), np.float32)
        self.l_free, self.l_occ = l_free, l_occ
        self.shrink, self.ring_hw, self.clip = shrink, ring_hw, clip
        self.blob_sd = blob_sd

    def _carve(self, px, pz, c, w):
        D = np.hypot(self.X - px, self.Z - pz)
        self.L[D < max(c - self.shrink, 0.0)] -= self.l_free * w
        return D

    def add(self, px, pz, c, w=1.0):
        """Isotropic: free disc of radius c, occupied ring at radius c."""
        D = self._carve(px, pz, c, w)
        self.L[np.abs(D - c) <= self.ring_hw] += self.l_occ * w
        np.clip(self.L, -self.clip, self.clip, out=self.L)

    def add_dir(self, px, pz, c, sx, sz, w=1.0):
        """Directional: free disc of radius c + a Gaussian occupied blob at (sx,sz)."""
        self._carve(px, pz, c, w)
        d2 = (self.X - sx) ** 2 + (self.Z - sz) ** 2
        self.L += (self.l_occ * w) * np.exp(-d2 / (2 * self.blob_sd ** 2)).astype(np.float32)
        np.clip(self.L, -self.clip, self.clip, out=self.L)

    def prob(self):
        return 1.0 / (1.0 + np.exp(-self.L))

    def logodds(self):
        return self.L


@torch.no_grad()
def clear_of(model, rng_o, coch_o):
    """One ping -> (clearance m, fusion weight from the learned confidence)."""
    r = torch.as_tensor(rng_o[None], dtype=torch.float32, device=DEV)
    c = torch.as_tensor(coch_o[None], dtype=torch.float32, device=DEV)
    ch, lv = model(r, c)
    return float(ch[0]), float(np.exp(-float(lv[0])))


@torch.no_grad()
def surf_of(model, rng_o, coch_o):
    """One ping -> (clearance m, ego bearing rad to that surface, confidence)."""
    r = torch.as_tensor(rng_o[None], dtype=torch.float32, device=DEV)
    c = torch.as_tensor(coch_o[None], dtype=torch.float32, device=DEV)
    dist, sb, cb, lvd, lvb = model(r, c)
    return (float(dist[0]), math.atan2(float(sb[0]), float(cb[0])),
            float(np.exp(-(float(lvd[0]) + float(lvb[0])))))


def field_update(field, model, x, z, yaw, rng_o, coch_o, directional):
    """Add one ping's surface evidence to the log-odds field; -> the clearance used."""
    if directional:
        c, beta, w = surf_of(model, rng_o, coch_o)
        ang = math.radians(yaw) + beta
        field.add_dir(x, z, c, x + c * math.sin(ang), z - c * math.cos(ang), min(w, 2.0))
    else:
        c, w = clear_of(model, rng_o, coch_o)
        field.add(x, z, c, min(w, 2.0))
    return c


# ============================================================ A. reactive bearing
class ReactiveBearing:
    """World-frame low-pass of the per-ping doorway bearing -> a moving SUBGOAL.

    The estimate is a DIRECTION held in the world frame (so the agent's own turning does
    not smear it) plus a lookahead range; the subgoal is re-anchored at the CURRENT pose
    every step. That is the whole point: no absolute doorway point is accumulated, so the
    2.8 m fusion floor (correlated front-back errors) never enters the controller.
    """

    def __init__(self, alpha=0.5, alpha_d=0.4, l_min=1.2, l_max=4.0, w_ref=1.0):
        self.alpha, self.alpha_d = alpha, alpha_d
        self.l_min, self.l_max, self.w_ref = l_min, l_max, w_ref
        self.dir = None          # unit (sx, cz) world direction to the doorway
        self.dist = None
        self.res = None          # UNnormalized EMA of the unit rays: |res| in [0,1] is
                                 # the circular mean-resultant length = how concentrated
                                 # the recent bearings are. Drives the scan anneal (§5.10).

    def concentration(self):
        """0 = bearings disagree (scan!), 1 = they agree (commit)."""
        return 0.0 if self.res is None else float(min(1.0, np.linalg.norm(self.res)))

    def update(self, x, z, yaw, dist, beta, w, scan=0.0):
        """dist/beta = this ping's ego range+bearing to the doorway, w = confidence.
        `scan` (rad) rotates only the emitted SUBGOAL, never the internal estimate: the
        agent looks off-axis to diversify aspect without corrupting what it believes."""
        ang = math.radians(yaw) + beta                       # world angle of the ray
        v = np.array([math.sin(ang), -math.cos(ang)], np.float64)
        a = self.alpha * min(1.0, w / self.w_ref)            # trust-scaled step
        if self.dir is None:
            self.dir, self.dist, self.res = v, dist, v.copy()
        else:
            self.res = (1 - a) * self.res + a * v
            self.dir = (1 - a) * self.dir + a * v
            n = np.linalg.norm(self.dir)
            self.dir = v if n < 1e-6 else self.dir / n
            ad = self.alpha_d * min(1.0, w / self.w_ref)
            self.dist = (1 - ad) * self.dist + ad * dist
        d = self.dir
        if scan:
            c, s = math.cos(scan), math.sin(scan)
            d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        L = float(np.clip(self.dist, self.l_min, self.l_max))
        return float(x + L * d[0]), float(z + L * d[1])


@torch.no_grad()
def egovec_of(model, rng_o, coch_o):
    """One ping -> (dist, bearing rad, confidence) to the doorway, via mapper_fuse's net."""
    r = torch.as_tensor(rng_o[None], dtype=torch.float32, device=DEV)
    c = torch.as_tensor(coch_o[None], dtype=torch.float32, device=DEV)
    dist, bs, bc, lvd, lvb = model(r, c)
    beta = math.atan2(float(bs[0]), float(bc[0]))
    w = float(np.exp(-(float(lvd[0]) + float(lvb[0]))))
    return float(dist[0]), beta, w


# ============================================================ C. active scanning
def _scan_offset(state, t):
    """§5.10 fix C: sweep the SUBGOAL off the estimated bearing by +/- scan_deg, so the
    agent approaches on a zig-zag and sees the doorway from a spread of ASPECTS. The gate
    (`mapper_active.py`) found aspect diversity -- not translational parallax -- is what
    the binaural estimate responds to. Annealed by the reactive filter's circular
    concentration: scan wide while the bearings disagree, commit once they agree (the
    head-movement schedule a bat uses). scan_deg=0 reproduces §5.9 exactly."""
    deg = state.get("scan_deg", 0.0)
    if not deg:
        return 0.0
    if state.get("scan_anneal", True):
        deg *= (1.0 - state["react"].concentration())
    sign = 1.0 if ((t - 1) // max(1, state.get("scan_period", 1) // 2)) % 2 == 0 else -1.0
    return math.radians(deg) * sign


def make_provider(occ_src, goal_src, occ_model=None, goal_model=None, clear_model=None,
                  thr_scatter=0.5, thr_clear=0.60, w_scatter=1.0, react_kw=None,
                  directional=True, ego_carve_r=0.35, scan_deg=0.0, scan_period=2,
                  scan_anneal=True):
    """occ_src in {true, scatter, clear, both}; goal_src in {true, fuse, react, reactfix,
    reacttrue}. `directional` selects the surface-point map update over the isotropic ring."""
    import mapper_fuse
    react_kw = react_kw or {}

    def provider(srv):
        occ_true_infl = inflate(true_occupancy(srv.Lx, srv.Lz, srv.obstacles))
        goal_true = goal_from_openings(srv.openings)
        state = {}

        def reset():
            state["field"] = ClearField()
            state["react"] = ReactiveBearing(**react_kw)
            state["scan_deg"] = scan_deg
            state["scan_period"] = scan_period
            state["scan_anneal"] = scan_anneal

        def step_map(x, z, yaw, history):
            rng_o, coch_o = srv.observe(x, z, yaw)
            rng_o = rng_o.astype(np.float32); coch_o = coch_o.astype(np.float32)
            history.append((x, z, yaw, rng_o, coch_o))
            if len(history) == 1:
                reset()                                   # new episode, same room/server

            # ---- occupancy
            if occ_src in ("clear", "both"):
                field_update(state["field"], clear_model, x, z, yaw, rng_o, coch_o, directional)
            if occ_src == "true":
                occ_infl = occ_true_infl
            elif occ_src == "clear":
                occ_infl = inflate(state["field"].prob() > thr_clear)
            else:
                poses = np.array([[h[0], h[1], h[2]] for h in history], np.float32)
                rseq = np.stack([h[3] for h in history])
                cseq = np.stack([h[4] for h in history])
                occ_p, _, _ = mapper.predict_grid(occ_model, poses, rseq, cseq)
                if occ_src == "scatter":
                    occ_infl = inflate(occ_p > thr_scatter)
                else:                                     # both: sum the log-odds
                    lo = np.log(np.clip(occ_p, 1e-4, 1 - 1e-4) / np.clip(1 - occ_p, 1e-4, 1))
                    tot = w_scatter * np.clip(lo, -3, 3) + state["field"].logodds()
                    occ_infl = inflate(1.0 / (1.0 + np.exp(-tot)) > thr_clear)

            # CONTRADICTION CARVE. The agent is demonstrably standing here, so if the map
            # says its own cell is solid the map is wrong -- and wrong in the one way that
            # is fatal, because then every sampled move reads as blocked and the planner
            # freezes. This fires only on that contradiction (config-space inflation is
            # 2 cells = 0.5 m while the corpus' median clearance is 0.41 m, so a slightly
            # mis-placed surface easily swallows the agent). An unconditional carve instead
            # HURTS accurate maps -- it invites planning through real inflated walls.
            if ego_carve_r > 0:
                aix, aiz = mapping.world_to_grid(x, z)
                if occ_infl[aix, aiz] > 0.5:
                    occ_infl = occ_infl.copy()
                    dx = state["field"].X - x; dz = state["field"].Z - z
                    occ_infl[np.hypot(dx, dz) < ego_carve_r] = 0.0

            # ---- goal
            if goal_src == "true":
                goal = goal_true
            elif goal_src == "reacttrue":
                # mechanism ceiling: TRUE bearing, NO range (fixed lookahead). Answers
                # "is following a direction enough to reach a 1.2 m radius?" separately
                # from "is the estimated direction good enough?"
                beta = ego_bearing(x, z, yaw, goal_true[0], goal_true[1])
                goal = state["react"].update(x, z, yaw, react_kw.get("l_max", 4.0), beta, 1e3)
            elif goal_src in ("react", "reactfix"):
                d, beta, w = egovec_of(goal_model, rng_o, coch_o)
                if goal_src == "reactfix":
                    d = react_kw.get("l_max", 4.0)         # bearing only, range discarded
                goal = state["react"].update(x, z, yaw, d, beta, w,
                                             scan=_scan_offset(state, len(history)))
            else:                                          # §5.8 known-pose fusion
                poses = np.array([[h[0], h[1], h[2]] for h in history], np.float32)
                rseq = np.stack([h[3] for h in history])
                cseq = np.stack([h[4] for h in history])
                goal = mapper_fuse.fuse(goal_model, poses, rseq, cseq)
            return occ_infl, goal

        return step_map
    return provider


# ======================================================================= evals
def _load_models(args, need_occ, need_goal, need_clear):
    import mapper_fuse
    occ_model = goal_model = clear_model = None
    if need_occ:
        occ_model = mapper.OccMapper().to(DEV)
        occ_model.load_state_dict(torch.load(args.occ_ckpt, map_location=DEV)); occ_model.eval()
    if need_goal:
        goal_model = mapper_fuse.EgoVecNet().to(DEV)
        goal_model.load_state_dict(torch.load(args.goal_ckpt, map_location=DEV)); goal_model.eval()
    if need_clear:
        clear_model = (SurfNet() if args.directional else ClearNet()).to(DEV)
        ck = args.surf_ckpt if args.directional else args.clear_ckpt
        clear_model.load_state_dict(torch.load(ck, map_location=DEV)); clear_model.eval()
    return occ_model, goal_model, clear_model


def cmd_mapq(args):
    """B, open loop: does the clearance field map the PILLARS the scatter-CNN misses?
    Compares the isotropic ring, the directional surface-point blob, the §5.8 scatter-CNN
    and the log-odds combination, at matched #pings."""
    from ppo import opening_seeds
    import mapper_collect
    occ_model = mapper.OccMapper().to(DEV)
    occ_model.load_state_dict(torch.load(args.occ_ckpt, map_location=DEV)); occ_model.eval()
    ring_model = ClearNet().to(DEV)
    ring_model.load_state_dict(torch.load(args.clear_ckpt, map_location=DEV)); ring_model.eval()
    surf_model = SurfNet().to(DEV)
    surf_model.load_state_dict(torch.load(args.surf_ckpt, map_location=DEV)); surf_model.eval()

    seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    print(f"Collecting held-out eval trajectories ({len(seeds)} rooms)...", flush=True)
    eps = mapper_collect.collect(seeds, args.ep_per_room, nobs=args.nobs, t_max=args.t_max,
                                 workers=min(12, len(seeds)), base_seed=99)
    X, Z = grid_cell_centres()
    ks = (2, 5, 10, 20, args.t_max)
    names = ("scatter", "ring", "dir", "both")
    rows = {k: {n: [[], []] for n in names} for k in ks}
    for e in eps:
        gt = true_occupancy(e["Lx"], e["Lz"], e["obstacles"]) > 0.5
        interior = (np.abs(X) < e["Lx"] / 2) & (np.abs(Z) < e["Lz"] / 2)
        T = len(e["poses"])
        f_ring, f_dir = ClearField(), ClearField()
        rs = e["range"].astype(np.float32); cs = e["coch"].astype(np.float32)
        added = 0
        for k in ks:
            kk = min(k, T)
            while added < kk:                              # incremental, in ping order
                px, pz, pyaw = e["poses"][added]
                field_update(f_ring, ring_model, px, pz, pyaw, rs[added], cs[added], False)
                field_update(f_dir, surf_model, px, pz, pyaw, rs[added], cs[added], True)
                added += 1
            occ_p, _, _ = mapper.predict_grid(occ_model, e["poses"][:kk], rs[:kk], cs[:kk])
            lo = np.log(np.clip(occ_p, 1e-4, 1 - 1e-4) / np.clip(1 - occ_p, 1e-4, 1))
            p_both = 1.0 / (1.0 + np.exp(-(np.clip(lo, -3, 3) + f_dir.logodds())))
            for name, pred in (("scatter", occ_p > 0.5),
                               ("ring", f_ring.prob() > args.thr_clear),
                               ("dir", f_dir.prob() > args.thr_clear),
                               ("both", p_both > args.thr_clear)):
                u = (pred | gt).sum()
                rows[k][name][0].append((pred & gt).sum() / max(u, 1))
                pg = gt & interior; pp = pred & interior
                pu = (pp | pg).sum()
                rows[k][name][1].append((pp & pg).sum() / pu if pu > 0 else np.nan)
    print(f"\nOpen-loop occupancy quality (held-out nobs={args.nobs}, {len(eps)} trajectories)")
    print(f"  §5.8 scatter-CNN reference: solid-IoU 0.78, pillar-IoU ~0.005\n")
    print(f"  {'#pings':>7} {'source':>9} {'solid-IoU':>10} {'pillar-IoU':>11}")
    for k in ks:
        for name in names:
            s, p = rows[k][name]
            print(f"  {k:>7} {name:>9} {np.mean(s):>10.3f} {np.nanmean(p):>11.3f}")


def cmd_nav(args):
    """Closed-loop 2x2: {scatter,clear,both,true} occupancy x {fuse,react,true} goal."""
    from ppo import opening_seeds
    need_occ = any(m.split("+")[0] in ("scatter", "both") for m in args.modes)
    need_clear = any(m.split("+")[0] in ("clear", "both") for m in args.modes)
    need_goal = any(m.split("+")[1] in ("fuse", "react", "reactfix") for m in args.modes)
    occ_model, goal_model, clear_model = _load_models(args, need_occ, need_goal, need_clear)
    seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    react_kw = dict(alpha=args.alpha, alpha_d=args.alpha_d, l_min=args.l_min, l_max=args.l_max)
    print(f"§5.9 reactive-bearing + clearance-map nav (nobs={args.nobs}, {len(seeds)} held-out "
          f"rooms x {args.episodes_per})")
    print("ladder: planning-oracle 1.00 | true-map planner 0.96 | greedy 0.54 | "
          "§5.8 map (scatter+fuse) 0.49 | MPC 0.24 | PPO 0.08-0.18")
    print(f"react: alpha={args.alpha} alpha_d={args.alpha_d} L in [{args.l_min},{args.l_max}] | "
          f"thr_clear={args.thr_clear}\n", flush=True)
    for mode in args.modes:
        osrc, gsrc = mode.split("+")
        prov = make_provider(osrc, gsrc, occ_model=occ_model, goal_model=goal_model,
                             clear_model=clear_model, thr_clear=args.thr_clear,
                             react_kw=react_kw, directional=args.directional,
                             ego_carve_r=args.ego_carve, scan_deg=args.scan_deg,
                             scan_period=args.scan_period, scan_anneal=not args.no_anneal)
        t0 = time.time()
        macro, per = run_map_nav(prov, seeds, nobs=args.nobs, episodes_per=args.episodes_per,
                                 t_max=args.t_max)
        lo, hi = min(per.values()), max(per.values())
        print(f"  {mode:>14} : {macro:.3f}  [{lo:.2f},{hi:.2f}]   ({time.time()-t0:.0f}s)",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train-clear", help="B: per-ping surface net (§5.6 readout, standalone)")
    t.add_argument("--corpus", default="echorl/mapper_corpus_nobs3.pkl")
    t.add_argument("--out", default=None)
    t.add_argument("--directional", action="store_true",
                   help="predict the nearest surface POINT (dist+bearing), not just clearance")
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=train_clear)

    q = sub.add_parser("mapq", help="B open-loop: solid/pillar IoU of the clearance field")
    q.add_argument("--occ-ckpt", default="echorl/ckpt/mapper_nobs3.pt")
    q.add_argument("--clear-ckpt", default="echorl/ckpt/clearnet_nobs3.pt")
    q.add_argument("--surf-ckpt", default="echorl/ckpt/surfnet_nobs3.pt")
    q.add_argument("--eval-rooms", type=int, default=12)
    q.add_argument("--ep-per-room", type=int, default=2)
    q.add_argument("--nobs", type=int, default=3)
    q.add_argument("--t-max", type=int, default=40)
    q.add_argument("--thr-clear", type=float, default=0.60)
    q.set_defaults(func=cmd_mapq)

    n = sub.add_parser("nav", help="closed-loop ablation")
    n.add_argument("--occ-ckpt", default="echorl/ckpt/mapper_nobs3.pt")
    n.add_argument("--goal-ckpt", default="echorl/ckpt/egovec_goal_nobs3.pt")
    n.add_argument("--clear-ckpt", default="echorl/ckpt/clearnet_nobs3.pt")
    n.add_argument("--surf-ckpt", default="echorl/ckpt/surfnet_nobs3.pt")
    n.add_argument("--isotropic", dest="directional", action="store_false",
                   help="use the isotropic clearance RING instead of the surface-point blob")
    n.add_argument("--modes", nargs="+",
                   default=["scatter+fuse", "scatter+react", "both+fuse", "both+react"])
    n.add_argument("--eval-rooms", type=int, default=12)
    n.add_argument("--episodes-per", type=int, default=8)
    n.add_argument("--nobs", type=int, default=3)
    n.add_argument("--t-max", type=int, default=40)
    n.add_argument("--thr-clear", type=float, default=0.60)
    n.add_argument("--alpha", type=float, default=0.5)
    n.add_argument("--alpha-d", type=float, default=0.4)
    n.add_argument("--l-min", type=float, default=1.2)
    n.add_argument("--l-max", type=float, default=4.0)
    n.add_argument("--ego-carve", type=float, default=0.35,
                   help="radius (m) around the agent forced free in the planning map")
    n.add_argument("--scan-deg", type=float, default=0.0,
                   help="§5.10 fix C: sweep the subgoal +/- this many degrees off the "
                        "estimated bearing (0 = §5.9 exactly)")
    n.add_argument("--scan-period", type=int, default=2, help="steps per full scan cycle")
    n.add_argument("--no-anneal", action="store_true",
                   help="do NOT shrink the scan as the bearing estimate concentrates")
    n.set_defaults(func=cmd_nav, directional=True)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
