"""EchoRL §5.11 -- fix D: is the doorway floor a PHYSICS limit or a 2-MIC limit?

Every §5 result up to §5.10 used a 2-ear head. Two ears give ITD/ILD that are symmetric
about the interaural axis, so a single ping cannot separate a source ahead from its mirror
image behind -- the cone of confusion. §5.8 traced the 2.8 m doorway-fusion floor to exactly
this: per-ping bearing errors are two-valued and therefore CORRELATED across poses, so
averaging cannot remove them. §5.10 then showed the estimate responds to head ASPECT rather
than translational parallax, which is the same story from the other side.

That makes the ambiguity a property of the SENSOR, not of the room acoustics -- a testable
claim. `episode_server --ears 4` decodes the same ambisonic field through a SECOND head
yawed +90 deg, so ears 2-3 have an interaural axis perpendicular to ears 0-1. A direction
ambiguous front/back for the first pair is left/right-distinct for the second. Physically
this is two 2-mic arrays at right angles (a non-collinear 4-mic array), and it costs one
extra ambisonics decode against a reflection sim that dominates the step budget.

The comparison is exactly paired: the corpus policy is a pure RNG (actions never depend on
observations), so at a fixed base_seed the 2-ear and 4-ear runs visit the SAME poses in the
SAME rooms. Only the sensor differs.

  python echorl/mapper_ears.py compare      # per-ping bearing MAE + fused doorway error
"""
from __future__ import annotations
import argparse, math, os, sys, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
import mapper, mapper_collect, mapper_fuse                                # noqa: E402
from mapping import goal_from_openings                                    # noqa: E402
from mapper_probe import ego_bearing                                      # noqa: E402

DEV = mapper.DEV


def _adiff(a, b):
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


@torch.no_grad()
def mirror_mask(model, eps, ear_idx=None):
    """Per-ping boolean: did the front-back-flipped hypothesis fit better than the truth?"""
    out = []
    for e in eps:
        gx, gz = goal_from_openings(e["openings"])
        c = e["coch"].astype(np.float32)
        if ear_idx is not None:
            c = c[:, list(ear_idx)]
        r = torch.as_tensor(e["range"].astype(np.float32), device=DEV)
        _, bs, bc, _, _ = model(r, torch.as_tensor(c, device=DEV))
        beta = np.arctan2(bs.cpu().numpy(), bc.cpu().numpy())
        for t in range(len(e["poses"])):
            px, pz, pyaw = e["poses"][t]
            tb = ego_bearing(px, pz, pyaw, gx, gz)
            out.append(_adiff(beta[t], math.pi - tb) < _adiff(beta[t], tb))
    return np.array(out, bool)


def cmd_pairs(args):
    """Is the residual ambiguity GEOMETRIC (fixable by array shape) or ACOUSTIC?

    Pair 0-1 and pair 2-3 are the same field heard by heads 90 deg apart, so their cones
    of confusion are PERPENDICULAR. If flips were a geometric property of a 2-mic array,
    the two pairs would flip on DIFFERENT pings and their combination would resolve them.
    If they flip on the SAME pings, no array geometry can help -- the doorway direction
    simply is not in the echo at those poses."""
    from ppo import opening_seeds
    seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    print(f"Collecting held-out 4-ear trajectories ({len(seeds)} rooms)...", flush=True)
    eps = mapper_collect.collect(seeds, args.ep_per_room, nobs=args.nobs, t_max=args.t_max,
                                 workers=min(12, len(seeds)), base_seed=99, ears=4)
    masks = {}
    for tag, ckpt, idx in (("pair 0-1 (original head)", args.ckpt01, (0, 1)),
                           ("pair 2-3 (rotated +90)", args.ckpt23, (2, 3))):
        sd = torch.load(ckpt, map_location=DEV)
        m = mapper_fuse.EgoVecNet(ears=2).to(DEV); m.load_state_dict(sd); m.eval()
        masks[idx] = mirror_mask(m, eps, idx)
        print(f"  {tag}: mirrored {masks[idx].mean()*100:.1f}%")
    a, b = masks[(0, 1)], masks[(2, 3)]
    p_a, p_b = a.mean(), b.mean()
    joint = (a & b).mean()
    print(f"\n  P(pair01 mirrored)          = {p_a*100:5.1f} %")
    print(f"  P(pair23 mirrored)          = {p_b*100:5.1f} %")
    print(f"  P(both mirrored)            = {joint*100:5.1f} %")
    print(f"  if INDEPENDENT would be     = {p_a*p_b*100:5.1f} %   (perpendicular cones)")
    print(f"  P(pair23 mirr | pair01 mirr)= {(joint/max(p_a,1e-9))*100:5.1f} %  "
          f"vs base rate {p_b*100:.1f} %")
    lift = (joint / max(p_a * p_b, 1e-9))
    print(f"\n  lift over independence = x{lift:.2f}")
    print("  x1 => flips are independent  => GEOMETRIC, a 4-mic array should fix them")
    print("  >>1 => flips coincide        => ACOUSTIC, the cue is absent at those poses")


@torch.no_grad()
def per_ping_bearing(model, eps):
    """-> (bearing MAE deg, MIRROR rate, residual MAE among non-mirrored pings).

    The MIRROR rate is the direct measurement of the cone of confusion, and it is the
    number this whole section is about. Ego bearing beta is 0 straight ahead and +/-90 deg
    to the sides, so reflecting a direction front<->back about the interaural axis maps
    beta -> pi - beta. A ping is counted as MIRRORED when the flipped hypothesis fits the
    truth better than the true one does. A 2-mic head cannot distinguish these two by
    construction; a non-collinear array can. (`err > 90 deg` would conflate a genuine flip
    with ordinary noise on a near-lateral target -- this does not.)"""
    errs, mirrored, resid = [], [], []
    for e in eps:
        gx, gz = goal_from_openings(e["openings"])
        r = torch.as_tensor(e["range"].astype(np.float32), device=DEV)
        c = torch.as_tensor(e["coch"].astype(np.float32), device=DEV)
        _, bs, bc, _, _ = model(r, c)
        beta = np.arctan2(bs.cpu().numpy(), bc.cpu().numpy())
        for t in range(len(e["poses"])):
            px, pz, pyaw = e["poses"][t]
            tb = ego_bearing(px, pz, pyaw, gx, gz)
            d_true = _adiff(beta[t], tb)
            d_mirr = _adiff(beta[t], math.pi - tb)
            errs.append(d_true)
            is_mirror = d_mirr < d_true
            mirrored.append(1.0 if is_mirror else 0.0)
            if not is_mirror:
                resid.append(d_true)
    return (float(np.mean(errs)), float(np.mean(mirrored)),
            float(np.mean(resid)) if resid else float("nan"))


def cmd_compare(args):
    from ppo import opening_seeds
    seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    ks = (1, 2, 5, 10, 20, args.t_max)
    rows = {}
    for ears, ckpt in ((2, args.ckpt2), (4, args.ckpt4)):
        sd = torch.load(ckpt, map_location=DEV)
        got = mapper_fuse.ears_of(sd)
        assert got == ears, f"{ckpt} is a {got}-ear checkpoint, expected {ears}"
        m = mapper_fuse.EgoVecNet(ears=ears).to(DEV); m.load_state_dict(sd); m.eval()
        t0 = time.time()
        # identical seeds + base_seed => identical poses; only the sensor differs
        eps = mapper_collect.collect(seeds, args.ep_per_room, nobs=args.nobs, t_max=args.t_max,
                                     workers=min(12, len(seeds)), base_seed=99, ears=ears)
        mae, mirror, resid = per_ping_bearing(m, eps)
        fus = mapper_fuse.eval_fusion(m, eps, "goal", ks=ks)
        rows[ears] = (mae, mirror, fus, len(eps), resid)
        print(f"  [{ears} ears] {len(eps)} traj, per-ping bearing {mae:5.1f} deg, "
              f"mirrored {mirror*100:4.1f}%, residual {resid:5.1f} deg  "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n§5.11 -- 2-ear head vs 4-ear non-collinear array "
          f"(held-out nobs={args.nobs}, paired poses)")
    print(f"reference: §5.8 fusion saturates 3.7 -> 2.8 m; goal radius is 1.2 m\n")
    print(f"  {'#pings':>7} {'2-ear err(m)':>13} {'4-ear err(m)':>13} {'delta':>8}")
    for k in ks:
        a, b = rows[2][2][k], rows[4][2][k]
        print(f"  {k:>7} {a:>13.2f} {b:>13.2f} {b-a:>+8.2f}")
    print(f"\n  per-ping bearing MAE : {rows[2][0]:5.1f} deg -> {rows[4][0]:5.1f} deg "
          f"({rows[4][0]-rows[2][0]:+.1f})")
    print(f"  MIRRORED (cone of confusion) : {rows[2][1]*100:5.1f} % -> {rows[4][1]*100:5.1f} % "
          f"({(rows[4][1]-rows[2][1])*100:+.1f})   <- the mechanism claim")
    print(f"  residual MAE, non-mirrored   : {rows[2][4]:5.1f} deg -> {rows[4][4]:5.1f} deg "
          f"({rows[4][4]-rows[2][4]:+.1f})")
    kk = args.t_max
    print(f"  fused doorway floor  : {rows[2][2][kk]:5.2f} m -> {rows[4][2][kk]:5.2f} m "
          f"({rows[4][2][kk]-rows[2][2][kk]:+.2f})")
    print("\nreading: a big drop in the FLIP RATE with little change elsewhere would mean the")
    print("         cone of confusion was real but not the binding constraint; a drop in the")
    print("         fused floor toward the 1.2 m goal radius would mean 2.8 m was HARDWARE.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="2-ear vs 4-ear, paired poses")
    c.add_argument("--ckpt2", default="echorl/ckpt/egovec_goal_nobs3.pt")
    c.add_argument("--ckpt4", default="echorl/ckpt/egovec_goal_nobs3_e4.pt")
    c.add_argument("--eval-rooms", type=int, default=12)
    c.add_argument("--ep-per-room", type=int, default=3)
    c.add_argument("--nobs", type=int, default=3)
    c.add_argument("--t-max", type=int, default=40)
    c.set_defaults(func=cmd_compare)
    q = sub.add_parser("pairs", help="are the flips geometric or acoustic?")
    q.add_argument("--ckpt01", default="echorl/ckpt/egovec_goal_nobs3_pair01.pt")
    q.add_argument("--ckpt23", default="echorl/ckpt/egovec_goal_nobs3_pair23.pt")
    q.add_argument("--eval-rooms", type=int, default=12)
    q.add_argument("--ep-per-room", type=int, default=3)
    q.add_argument("--nobs", type=int, default=3)
    q.add_argument("--t-max", type=int, default=40)
    q.set_defaults(func=cmd_pairs)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
