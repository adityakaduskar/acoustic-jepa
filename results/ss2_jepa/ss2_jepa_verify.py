#!/usr/bin/env python3
"""
ss2_jepa_verify.py — integrity check on a generated SS2 dataset.

The failure mode this exists to catch: habitat resolving a rebuilt scene
through a path it has already loaded, so every "different" room silently
renders the same audio. Also checks the schema against the manifest contract
and the honesty of the measured transform.

Usage: ss2_jepa_verify.py <swarm_dir> [traj_dir]
"""
import sys, os, glob, math
import numpy as np

SWARM_SPEC = {
    "mono_coch": ("float16", ("N", 2, 64, 48)), "mono_range": ("float16", ("N", 256)),
    "cross_coch": ("float16", ("P", 2, 64, 48)), "cross_range": ("float16", ("P", 256)),
    "pair": ("int32", ("P", 2)), "T_true": ("float32", ("P", 6)),
    "T_meas": ("float32", ("P", 6)), "occluded": ("int32", ("P",)),
    "agent_pose": ("float32", ("N", 4)), "dist_wall": ("float32", ("N", 8)),
    "room_vol": ("float32", ("N",)), "material": ("int32", ("N",)),
    "open_az": ("float32", ("N",)), "room_meta": ("float32", (10,)),
    "obstacles": ("float32", (None, 7)),
}
TRAJ_SPEC = {
    "coch": ("float16", ("K", 2, 64, 48)), "range": ("float16", ("K", 256)),
    "actions": ("float32", ("K", 2)), "labels_dist": ("float32", ("K", 8)),
    "open_az": ("float32", ("K",)), "room_vol": ("float32", ("K",)),
    "material": ("int32", ("K",)), "pose": ("float32", ("K", 4)),
    "room_meta": ("float32", (10,)), "beacon": ("float32", (3,)),
}
ok = True


def chk(c, msg):
    global ok
    print(("  PASS  " if c else "  FAIL  ") + msg, flush=True)
    if not c:
        ok = False


def check_schema(files, spec, dims):
    d = np.load(files[0])
    got = set(d.files)
    chk(got == set(spec), "keys exact (%d)%s" % (len(spec),
        "" if got == set(spec) else "  extra=%s missing=%s" % (got - set(spec), set(spec) - got)))
    res = {}
    for k, (dt, shp) in spec.items():
        a = d[k]
        chk(str(a.dtype) == dt, "%s dtype %s" % (k, a.dtype))
        exp = []
        for s in shp:
            if isinstance(s, str):
                res.setdefault(s, a.shape[len(exp)])
                exp.append(res[s])
            else:
                exp.append(a.shape[len(exp)] if s is None else s)
        chk(a.shape == tuple(exp), "%s shape %s" % (k, a.shape))
    for name in dims:
        print("        %s = %d" % (name, res.get(name, -1)))
    return res


def check_distinct(files, key, meta_key, n=40):
    """rooms differ AND their audio differs"""
    sel = files[:n]
    metas = np.stack([np.load(f)[meta_key] for f in sel])
    audio = np.stack([np.asarray(np.load(f)[key], np.float32).ravel()[:4000] for f in sel])
    umeta = len(np.unique(metas, axis=0))
    chk(umeta == len(sel), "room_meta distinct across episodes (%d/%d)" % (umeta, len(sel)))
    a = audio - audio.mean(1, keepdims=True)
    a /= (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    corr = a @ a.T
    off = corr[~np.eye(len(sel), dtype=bool)]
    chk(off.max() < 0.99, "audio distinct across episodes (max off-diag corr %.3f)" % off.max())
    print("        mean cross-episode audio corr %.3f" % off.mean())


def check_finite(files, spec, n=60):
    bad = []
    for f in files[:n]:
        d = np.load(f)
        for k in spec:
            # NaN is meaningful in two keys: open_az = "room has no opening",
            # beacon = "monostatic episode, no displaced emitter" (both match
            # what the C++ dumpers write).
            if k in ("open_az", "beacon"):
                continue
            a = np.asarray(d[k], np.float64)
            if not np.isfinite(a).all():
                bad.append((os.path.basename(f), k))
    chk(not bad, "all arrays finite (open_az exempt)%s" % ("" if not bad else " %s" % bad[:4]))


def check_nonsilent(files, keys, n=60):
    for k in keys:
        mx = np.array([np.asarray(np.load(f)[k], np.float32).max() for f in files[:n]])
        chk((mx > 1e-3).all(), "%s non-silent everywhere (min-of-max %.4f)" % (k, mx.min()))


def check_meas(files, n=200):
    er, eb, fb, occ = [], [], [], []
    for f in files[:n]:
        d = np.load(f)
        tt, tm, oc = d["T_true"], d["T_meas"], d["occluded"]
        er.append(np.abs(tm[:, 3] - tt[:, 3]))
        eb.append(np.abs((tm[:, 4] - tt[:, 5] + 540) % 360 - 180))
        fb.append(tm[:, 5])
        occ.append(oc)
    er, eb, fb, occ = map(np.concatenate, (er, eb, fb, occ))
    print("        range err  med %.3f m  p90 %.3f m" % (np.median(er), np.percentile(er, 90)))
    print("        bearing err med %.2f deg  p90 %.2f deg" % (np.median(eb), np.percentile(eb, 90)))
    print("        front/back correct %.1f%%   occluded pairs %.1f%%"
          % (100 * fb.mean(), 100 * occ.mean()))
    chk(np.median(er) < 0.30, "measured range median err < 30 cm")
    chk(np.median(eb) < 25.0, "measured bearing median err < 25 deg")
    chk(fb.mean() > 0.75, "front/back resolved > 75% of pairs")
    chk(0.0 < occ.mean() < 0.5, "occlusion present but not dominant")


def check_per_worker(files, workers=20, n=600):
    """Each worker self-calibrates independently (ITD-vs-azimuth curve + onset
    offset). A worker whose calibration went wrong would produce systematically
    worse T_meas on its own shard, so compare shards: episode e was rendered by
    worker e % workers."""
    med_r = np.full(workers, np.nan)
    med_b = np.full(workers, np.nan)
    for w in range(workers):
        er, eb = [], []
        for f in files[:n]:
            if int(os.path.basename(f)[2:7]) % workers != w:
                continue
            d = np.load(f)
            tt, tm = d["T_true"], d["T_meas"]
            er.append(np.abs(tm[:, 3] - tt[:, 3]))
            eb.append(np.abs((tm[:, 4] - tt[:, 5] + 540) % 360 - 180))
        if er:
            med_r[w] = np.median(np.concatenate(er))
            med_b[w] = np.median(np.concatenate(eb))
    good = ~np.isnan(med_r)
    print("        per-worker range err  med: min %.3f max %.3f m"
          % (med_r[good].min(), med_r[good].max()))
    print("        per-worker bearing err med: min %.2f max %.2f deg"
          % (med_b[good].min(), med_b[good].max()))
    chk(med_r[good].max() < 0.30, "no worker shard has range err > 30 cm")
    chk(med_b[good].max() < 25.0, "no worker shard has bearing err > 25 deg")


def main():
    swarm = sys.argv[1]
    files = sorted(glob.glob(os.path.join(swarm, "ep*.npz")))
    print("== cross-view: %d episodes in %s ==" % (len(files), swarm), flush=True)
    check_schema(files, SWARM_SPEC, ("N", "P"))
    check_distinct(files, "mono_coch", "room_meta")
    check_finite(files, SWARM_SPEC)
    check_nonsilent(files, ("mono_coch", "mono_range", "cross_coch", "cross_range"))
    check_meas(files)
    check_per_worker(files)

    if len(sys.argv) > 2:
        traj = sys.argv[2]
        tf = sorted(glob.glob(os.path.join(traj, "ep*.npz")))
        print("\n== trajectory: %d episodes in %s ==" % (len(tf), traj), flush=True)
        check_schema(tf, TRAJ_SPEC, ("K",))
        check_distinct(tf, "coch", "room_meta")
        check_finite(tf, TRAJ_SPEC)
        check_nonsilent(tf, ("coch", "range"))
        d = np.load(tf[0])
        p, a = d["pose"], d["actions"]
        moved = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1)
        print("        step length: mean %.2f m, max %.2f m; yaw span %.0f deg"
              % (moved.mean(), moved.max(), p[:, 3].ptp()))
        chk(moved.mean() > 0.2, "trajectory actually moves")
        chk((np.abs(a[:, 0]) <= 45.001).all(), "dyaw within [-45,45]")
        chk(((a[:, 1] >= -1e-6) & (a[:, 1] <= 1.5001)).all(), "dstep within [0,1.5]")

    print("\nVERIFY " + ("PASS" if ok else "FAIL"), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
