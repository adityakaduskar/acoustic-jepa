#!/usr/bin/env python3
"""
ss2_jepa_smoke.py — validate the SS2 observation operator before generating
any dataset (ss2_jepa_manifest.md §3: "Verify the operator on the smoke test
from the characterization manifest ... before generating any dataset").

Checks, in order:
  1. wall echo: co-located src/rcv vs a 20x20 m wall at d = 2/5/10 m ->
     echo exists, delay ~ 2d/c, level falls with d
  2. matched-filter range tower: the profile's peak bin tracks 2d/c
  3. cochleagram: non-degenerate, and its energy falls with distance
  4. self-calibration: ITD-vs-azimuth monotone, onset-vs-distance linear
  5. measure_tab: measured range/bearing vs oracle in a procedural room
  6. one real procedural room end-to-end + per-observation timing
"""
import os, sys, time, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ss2_collect as S
from ss2_jepa_env import (SS2Env, MatchedFilter, make_box_room, place_agents,
                          cross_meta, analytic_label, free_scene, export_room_scene,
                          material_specs, SR, C_NOM, RANGE_BINS, RANGE_POOL,
                          RANGE_GUARD, EAR_H, TOTAL)

OUT = os.path.expanduser("~/ss2_results/ss2_jepa")
SCENES = os.path.join(OUT, "scenes")
os.makedirs(SCENES, exist_ok=True)
ok = True


def chk(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg, flush=True)
    if not cond:
        ok = False


# ---------------------------------------------------------------- 1/2/3 wall
print("== 1-3. wall echo / range tower / cochleagram ==", flush=True)
S.SCENE_ROOT = SCENES
wall = S.scene_wall("_wall20", 20, 20)
S.write_materials(wall, "materials.json", material_specs())
# the wall slab's semantic label is "mirror", which is material 3 in our table
env = SS2Env(SCENES)
env.set_scene_dir(wall)

prev_e, prev_ce = None, None
for d in (2.0, 5.0, 10.0):
    p = [0.0, EAR_H, d]
    t0 = time.time()
    coch, rng = env.observe(p, 0.0)
    dt = time.time() - t0
    w = env.render("ambi", p, 0.0, p, n=TOTAL)[0]

    pred_ms = 2 * d / C_NOM * 1000
    n0 = int(0.7 * 2 * d / C_NOM * SR)
    pk_i = n0 + int(np.argmax(np.abs(w[n0:])))
    got_ms = pk_i / SR * 1000
    e = float(np.sum(w[n0:] ** 2))
    ce = float(np.asarray(coch, np.float32).mean())

    rb = int(np.argmax(np.asarray(rng, np.float32)))
    r_ms = ((rb * RANGE_POOL + RANGE_POOL * 0.5) + RANGE_GUARD) / SR * 1000

    print(f"  d={d:4.1f}  echo {got_ms:7.3f} ms (pred {pred_ms:7.3f})  "
          f"E={e:.3e}  mf-peak bin {rb:3d} = {r_ms:7.3f} ms  coch mean {ce:.3f}  "
          f"[{dt*1000:.0f} ms/obs]", flush=True)
    chk(abs(got_ms - pred_ms) < 0.05 * pred_ms + 1.0, f"echo delay at d={d}")
    chk(abs(r_ms - pred_ms) < 0.10 * pred_ms + 1.0, f"matched-filter range at d={d}")
    chk(ce > 1e-4, f"cochleagram non-degenerate at d={d}")
    if prev_e is not None:
        chk(e < prev_e, f"echo energy falls (d={d})")
        chk(ce < prev_ce, f"cochleagram energy falls (d={d})")
    prev_e, prev_ce = e, ce

# ------------------------------------------------------------ 4 calibration
print("== 4. self-calibration (free field) ==", flush=True)
cal = env.calibrate(free_scene(SCENES))
itd = np.array(cal["itd"])
az = np.array(cal["az"])
print("  ITD range %.2f..%.2f samples over az %.0f..%.0f deg"
      % (itd[0], itd[-1], az[0], az[-1]), flush=True)
print("  onset offset %.2f samples, effective c = %.2f m/s"
      % (cal["onset_offset"], cal["c_eff"]), flush=True)
chk(np.all(np.diff(itd) > 0), "ITD calibration strictly ordered")
chk(abs(az[0]) == 90 and abs(az[-1]) == 90 and az[0] != az[-1],
    "ITD extremes map to +-90 deg")
chk(itd[-1] - itd[0] > 20, "ITD span > 20 samples (~0.4 ms) across +-90 deg")
chk(300 < cal["c_eff"] < 380, "effective sound speed in 300-380 m/s")

# ------------------------------------------------ 5/6 procedural room e2e
print("== 5-6. procedural room end-to-end ==", flush=True)
rm = make_box_room(1, 3)
print("  room %.2f x %.2f x %.2f, %d opening(s), %d obstacle(s), mats %s"
      % (rm.Lx, rm.Ly, rm.Lz, len(rm.openings), len(rm.obstacles), rm.wall_material),
      flush=True)
t0 = time.time()
env.set_room(rm, "_smoke_room1")
print("  scene build+open %.2f s" % (time.time() - t0), flush=True)

pos, yaw = place_agents(rm, 1, 6)
t0 = time.time()
coch, rng = env.observe(pos[0], yaw[0])
t_mono = time.time() - t0
coch = np.asarray(coch, np.float32)
rng = np.asarray(rng, np.float32)
print("  monostatic obs %.0f ms  coch[min %.3f max %.3f]  range[min %.3f max %.3f]"
      % (t_mono * 1000, coch.min(), coch.max(), rng.min(), rng.max()), flush=True)
chk(np.isfinite(coch).all() and np.isfinite(rng).all(), "observation finite")
chk(coch.max() > 1e-3 and rng.max() > 1e-3, "observation non-silent")

t0 = time.time()
cc, cr = env.observe(pos[0], yaw[0], source=pos[1])
t_cross = time.time() - t0
cc = np.asarray(cc, np.float32)
chk(cc.max() > 1e-3, "cross-view observation non-silent")

errs_r, errs_b, fb = [], [], []
t0 = time.time()
for l in range(3):
    for e in range(3):
        if l == e:
            continue
        gt = cross_meta(rm, pos[l], yaw[l], pos[e], yaw[e])
        mt = env.measure_tab(pos[l], yaw[l], pos[e])
        errs_r.append(abs(mt["range"] - gt["range"]))
        db = math.fmod(mt["bearing"] - gt["bearing"] + 540, 360) - 180
        errs_b.append(abs(db))
        fb.append(mt["fb_ok"])
t_meas = (time.time() - t0) / len(errs_r)
print("  T_meas: range err med %.3f m (max %.3f) | bearing err med %.1f deg | "
      "front/back %d/%d correct  [%.0f ms/pair]"
      % (np.median(errs_r), np.max(errs_r), np.median(errs_b), sum(fb), len(fb),
         t_meas * 1000), flush=True)
chk(np.median(errs_r) < 0.30, "measured range median error < 30 cm")
chk(np.median(errs_b) < 25.0, "measured bearing median error < 25 deg")

lb = analytic_label(rm, pos[0], yaw[0])
print("  labels: vol %.1f m^3, mat %d, open_az %.1f, dist_wall %s"
      % (lb["room_vol"], lb["material"], lb["open_az"],
         np.round(lb["dist_wall"], 2)), flush=True)
chk(abs(float(lb["room_vol"]) - rm.Lx * rm.Ly * rm.Lz) < 1e-3, "room_vol label")
chk((lb["dist_wall"] > 0).all() and (lb["dist_wall"] < 20).all(), "dist_wall sane")

print("\n  budget: mono %.0f ms, cross %.0f ms, T_meas %.0f ms"
      % (t_mono * 1000, t_cross * 1000, t_meas * 1000))
print("  -> one 6-agent room ~ %.1f s; 2000 rooms ~ %.1f core-hours"
      % (6 * t_mono + 30 * (t_cross + t_meas),
         2000 * (6 * t_mono + 30 * (t_cross + t_meas)) / 3600))
env.close()
print("\nSMOKE " + ("PASS" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
