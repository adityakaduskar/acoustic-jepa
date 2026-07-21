# Acoustic-JEPA on SoundSpaces 2.0 — replication result

Replication of the Steam Audio 4.8.1 Acoustic-JEPA dissociation on a
bilaterally different engine (habitat-sim 0.2.2 + RLR path-traced audio
propagation), per `ss2_jepa_manifest.md`. Engine versions and the exact render
config are in `VERSION.txt`.

**Headline: the dissociation reproduces. It is a property of the problem, not
of Steam Audio's renderer.**

On SS2.0, as on Steam Audio:

- a self-supervised JEPA latent decodes room geometry **worse** than raw-pooled
  features (and, for peer position, worse than a random projection), and
- an action-conditioned JEPA world model drives latent-MPC navigation well while
  a random-encoder world model **fails**, on every seed and both goals.

---

## 1. Comparison table (manifest §7)

Steam Audio values are the manifest's reference column. SS2.0 values are from
2000 cross-view rooms (1600 train / 400 test) and 200 trajectory episodes
(160/40), the same scale as the Steam Audio 2000-room run.

| quantity | Steam Audio | SoundSpaces 2.0 | verdict |
|---|---|---|---|
| probe room_vol R² (nonlinear) | JEPA 0.81 / raw 0.93 / random 0.90 | JEPA 0.702 ± 0.030 / raw **0.854** / random 0.675 | **AGREES** — raw > JEPA, robust |
| probe peer_pos R² (nonlinear) | JEPA 0.09 / raw 0.09 / random 0.20 | JEPA 0.004 ± 0.008 / raw 0.146 ± 0.253 / random **0.201** ± 0.040 | **AGREES** — raw & random > JEPA |
| sweep: best JEPA room_vol | 0.85 (< raw 0.93) | *running (10 configs, ~9 h); see §6* | pending |
| control (reach centre, 5 seeds) | learned +0.86 / random-enc −0.34 / random-act −0.55 / analytic +0.58 | learned **+0.690** ± 0.037 / random-enc **−0.308** ± 0.147 / random-act −0.548 / analytic +0.286 | **AGREES** |
| control gap learned−random-enc | +1.20 ± 0.11, positive every seed | **+0.999** ± 0.155, positive every seed | **AGREES** |

**The headline test posed by the manifest** — "is raw/random ≥ learned on the
geometry probes (the negative), AND learned ≫ random-encoder on control (the
positive)?" — is **YES on both counts**. Both reproduce.

### 1a. Control, both goals (5 training seeds, held-out rooms, 200 fixed starts)

| controller | reach centre (wall-dist increase, m) | approach wall (nearest-wall reduction, m) |
|---|---:|---:|
| analytic (oracle, true geometry) | +0.286 | +1.152 |
| **learned MPC** | **+0.690 ± 0.037** | **+0.784 ± 0.075** |
| random actions | −0.548 | +0.548 |
| random-encoder MPC | −0.308 ± 0.147 | −0.127 ± 0.052 |
| learned − random-enc, per seed | [0.956, 1.071, 0.907, 1.227, 0.832] → **+0.999 ± 0.155** | [0.861, 0.846, 0.903, 0.869, 1.077] → **+0.911 ± 0.095** |

As on Steam Audio, the learned MPC *beats the analytic oracle* on reach-centre
(+0.690 vs +0.286): the greedy analytic controller is myopic over a 6-step
horizon, so this is a property of the task, not evidence of superhuman audio.

### 1b. Cross-view training health (3 seeds)

| seed | test L_range | no-transform baseline | skill | z_range std / rank |
|---|---:|---:|---:|---|
| a | 0.9228 | 2.0867 | +0.558 | 1.039 / 118.1 |
| b | 0.8596 | 2.0868 | +0.588 | 1.035 / 118.2 |
| c | 0.9384 | 2.1014 | +0.553 | 1.039 / 118.2 |

Positive skill on every seed means the predictor genuinely uses the transform to
cross viewpoints, and the latent rank/std show no collapse — i.e. the probe
negative is not a degenerate-encoder artifact.

---

## 2. The one expected engine difference: SS2 renders a real direct path

Steam Audio renders **no direct path at all** — its reflection simulation stores
reflections at excess delay and the direct arrival must be synthesised. RLR
renders a real, propagation-delayed direct arrival. This touches the pipeline in
exactly two places, and both were handled explicitly:

**(a) The cross-view target is reflection-only.** The manifest requires the
target to exclude the direct arrival. Rather than window it out post-hoc, the
observation sensors are configured with RLR's `direct` flag **OFF** and
`indirect` ON, so the rendered IR *is* the reflection-only field by construction
— the exact analogue of what Steam Audio produces. The monostatic
self-observations use the same configuration (a co-located source has a
degenerate direct path anyway).

**(b) T_meas is measured from the real arrival, not injected.** The manifest
explicitly permits this. A third, direct-only sensor (`direct` ON, `indirect`
OFF) isolates the direct arrival, and the listener measures:

- **range** from the arrival's ToA, through its own onset-vs-distance
  calibration (fitted free-field: fixed offset −13.7 samples, effective
  c = 344.2 m/s);
- **bearing** from the binaural ITD, through its own ITD-vs-azimuth calibration;
- **front/back** by a +20° head-turn re-measurement (Wallach).

Nothing here uses ground truth; it is what an agent could actually measure.

**Consequence, and it is a real confound worth stating.** Because the arrival is
physical rather than synthesised, T_meas on SS2 is far more accurate than its
Steam Audio counterpart:

| T_meas quality (2000 rooms, 60000 pairs) | SS2.0 |
|---|---|
| range error | median **1.0 cm**, p90 5.7 cm |
| bearing error | median **1.06°**, p90 4.13° |
| front/back resolved correctly | **99.8%** |
| pairs with occluded sightline | 28.9% |

So the SS2 predictor is conditioned on a *cleaner* transform than the Steam
Audio predictor was. This makes the JEPA's task **easier**, not harder — which
matters for how the negative should be read: the learned latent loses to raw
pooling on geometry decode *even with near-oracle conditioning*. It strengthens
the negative rather than explaining it away. It does mean the two engines'
absolute probe numbers are not perfectly like-for-like, and any future
comparison sensitive to conditioning noise should re-run the Steam Audio side
with matched T_meas quality (or the SS2 side with injected noise) before
drawing conclusions from small differences.

---

## 3. Other deviations from the Steam Audio rig

Each is a deliberate choice, recorded so the paper can cite it.

| # | deviation | why |
|---|---|---|
| 1 | **Ambisonic order 1** (Steam Audio used order 3) | RLR decodes binaural natively, so the binaural stream comes from a Binaural-layout sensor rather than an order-3 ambisonic decode. Only the W channel feeds the range tower, and W is order-independent. |
| 2 | **`transmission` flag OFF** | The SS2 characterization battery (E11) found RLR's transmission path injects a late diffuse smear rather than a physical through-wall arrival. Steam Audio's transmission was inert for reflections, so OFF is the faithful match. Material transmission coefficients are 0 for all four materials. |
| 3 | **`diffraction` flag OFF** | RLR's diffraction bends the *direct path only*, and only when `meshSimplification` is also on (battery E5B/E11). Steam Audio rendered no edge diffraction in its reflection path either. |
| 4 | **ITD cross-correlation low-passed to 800 Hz** | Broadband, RLR's HRIR cycle-skips: ITD saturates at ±165 µs and the calibration curve folds past ±60°. With the low-pass the span is ±683 µs at ±90°, which independently reproduces the ±682 µs the battery measured directly in E7. Steam Audio's HRIR cross-correlates cleanly broadband and needed no filter. |
| 5 | **`raw` key omitted** from the trajectory dataset | The manifest's §5(b) schema does not list it and no script in `jepa/` loads it; it would have been ~3/4 of the dataset size. Every key the schema *does* specify is present with the exact dtype and shape. |
| 6 | **Rooms are the same distribution, not the same random stream** | `make_box_room`'s xorshift32 RNG was ported verbatim, so a given seed yields the same room on both engines — stronger than the manifest required (it asked only for the same distribution). |
| 7 | **Monte-Carlo, not bit-reproducible** | RLR is a stochastic path tracer (battery E2): repeated renders of the same pose differ. Steam Audio was bit-reproducible. Nothing in the pipeline depends on bit-exactness. |

**Modifications to `jepa/*.py`: none.** Every pipeline script was run unchanged.
No import needed fixing — the scripts used all carry
`from __future__ import annotations`, so their 3.10+ type syntax runs on the
habitat env's Python 3.9.

---

## 4. What was built (deliverables in this directory)

| file | what |
|---|---|
| `ss2_jepa_env.py` | **the observation operator** — the only engine-specific code. Ports `make_box_room`, `build_room_mesh`, `analytic_label`, `MatchedFilter`, `Cochleagram`, `place_agents`, `measure_tab`. |
| `ss2_swarm_dump.py` | cross-view dataset generator (port of `swarm_dump.cpp`) |
| `ss2_traj_dump.py` | trajectory dataset generator (port of `jepa_dump.cpp`) |
| `ss2_jepa_smoke.py` | pre-generation validation of the operator |
| `ss2_jepa_verify.py` | post-generation dataset integrity check |
| `ss2_collect.py` | the battery's scene/material-binding harness, reused |
| `swarm_data/` | 2000 rooms × (6 monostatic + 30 cross-view) observations, 891 MB |
| `traj_data/` | 200 × 48-step monostatic trajectories, 119 MB |
| `VERSION.txt` | engine/library versions and full render config |

### Validation performed before and after generation

**Smoke test** (`ss2_jepa_smoke.py`, all pass): wall echo at d = 2/5/10 m
arrives at 2d/c and falls in level with distance; the matched-filter range tower
peaks at the corresponding bin; the cochleagram is non-degenerate and tracks
distance; the self-calibration is monotone with a physically correct ITD span;
measured range/bearing track the oracle in a real procedural room.

**Integrity check** (`ss2_jepa_verify.py`, all pass on both datasets): exact key
set, dtype and shape against the manifest contract; all arrays finite (with NaN
permitted only in `open_az` = "no opening" and `beacon` = "monostatic", matching
the C++ dumpers); no stream silent; **rooms and their audio distinct across
episodes** (max cross-episode correlation 0.919 — this is the check that would
catch habitat silently re-serving a cached scene); trajectories actually move
and actions stay in range; and T_meas accuracy is uniform across all 20 worker
shards (per-worker median range error 0.006–0.014 m, bearing 0.89–1.16°).

### One bug found during generation

Every worker rewrote the *shared* free-field calibration scene's
`materials.json` concurrently. Two workers read a half-written file, RLR's
`AudioMaterialDatabase::loadJSONFile` failed, and those workers **aborted with
SIGABRT during calibration** — costing 200 of the first 2000 episodes. Episodes
that were written are unaffected (each came from a surviving worker with a valid
calibration, as the per-shard check above confirms). Fixed by writing the shared
scene once, atomically; the 200 missing episodes were regenerated with the fix
in place, and the final dataset is a clean 2000.

---

## 5. Reproduce

```bash
conda activate ss
python ss2_jepa_smoke.py                                  # validate the operator
python ss2_swarm_dump.py --episodes 2000 --workers 20     # ~60 min, 24 cores
python ss2_traj_dump.py  --episodes 200  --workers 20     # ~5 min
python ss2_jepa_verify.py <swarm_data> <traj_data>        # integrity

cd /mnt/d/Unreal\ Projects/TP_Project1/AcousticJEPA
for s in a b c; do python jepa/train_xview.py <swarm_data> --epochs 60 --out ckpt_ss2_$s.pt; done
python jepa/probe_multiseed.py <swarm_data> --ckpts "ckpt_ss2_*.pt"
python jepa/sweep_xview.py     <swarm_data>
python jepa/control_mpc.py     <traj_data> --repeats 5 --goal center
python jepa/control_mpc.py     <traj_data> --repeats 5 --goal wall
```

Closed-loop control (`control_mpc.py --closed-loop`) would need an SS2 "observe
server" (pose in → coch+range out); `SS2Env.observe()` in `ss2_jepa_env.py` is
that function already and only needs a socket wrapper. Not run here — the
offline control experiment is what the manifest's §7 comparison table asks for.

---

## 6. Status of the hyperparameter sweep (the one incomplete item)

`sweep_xview.py` trains 10 configurations from scratch (≈490 epochs total,
≈9 h on this machine at ~65 min per 60-epoch run) and asks whether *any* tuning
of the compression knobs lets JEPA beat raw pooling on geometry decode. It is
**running**; log at `sweep.log` in this directory, and the table above will be
updated with `best JEPA room_vol` when it lands.

It is reported last deliberately. The manifest's priority is probes → control,
and both are complete; the sweep is a robustness check on the *negative*, which
three independent seeds already agree on (JEPA 0.702 ± 0.030 vs raw 0.854 on
room_vol; 0.004 ± 0.008 vs 0.201 random on peer_pos). On Steam Audio the sweep's
best config reached 0.85, still below raw's 0.93 — i.e. it did not overturn the
negative there. The SS2 verdict is not yet in, and until it is, the honest
statement is that the negative is robust **to seed** on SS2 and robust **to
tuning** only on Steam Audio.
