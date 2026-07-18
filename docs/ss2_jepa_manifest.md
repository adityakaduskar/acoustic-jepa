# Acoustic-JEPA on SoundSpaces 2.0 — replication manifest

**KEYWORD (hand this to the WSL2 session to start):** `JEPA-SS2-REPLICATE`

> Open a Claude Code session on WSL2 Ubuntu inside
> `/mnt/d/Unreal Projects/TP_Project1/AcousticJEPA/` and say:
> **"Run JEPA-SS2-REPLICATE — read docs/ss2_jepa_manifest.md and execute it."**
> That session has SoundSpaces 2.0 / habitat-sim installed; this Windows session
> does not. This document is self-contained; the WSL2 session does not have access
> to the conversation that produced it.

**Audience:** a Claude Code session on WSL2 Ubuntu with SoundSpaces 2.0 +
habitat-sim (RLR audio propagation) working — the same environment used for the
characterization battery in `../SteamAudioIR/docs/ss2_experiment_manifest.md`
(read that first for the RIR-rendering API; the same `AudioSensor` /
`RLRAudioPropagationConfiguration` path is used here).

---

## 1. Why — the result to reproduce

On **Steam Audio 4.8.1** we found a clean **dissociation** between two uses of a
learned acoustic representation (full writeup: `../AcousticJEPA/RESULTS.md`):

- **Geometry probes (reading state):** a self-supervised JEPA latent decodes
  room geometry (volume, peer position) *worse* than raw-pooled features or a
  random projection — robust across seeds and a hyperparameter sweep. Learning
  compresses away geometry that fixed operations already expose.
- **Control (planning):** an action-conditioned JEPA world model drives latent-
  MPC navigation near an analytic oracle, while a random-encoder world model
  *fails* — learning is essential for plannable dynamics.

The open question is **external validity**: is this a property of *acoustic
world models*, or an artifact of Steam Audio's specific renderer (10 ms energy-
histogram time base, no direct-path delay, listener-centric ray tracing)?
SoundSpaces 2.0 is a bilaterally different engine (path-traced RLR propagation).
**If the dissociation reproduces on SS2.0, it is a property of the problem, not
the engine. If it does not, the Steam Audio result is engine-specific.** Either
outcome is a strong, publishable finding.

## 2. The one fact that makes this cheap

Only **data generation** is engine-specific. Everything downstream — the model,
training, probes, sweep, and control experiments — is pure PyTorch that consumes
`.npz` files in a fixed schema and never touches an audio engine. So your job is:

1. build a SoundSpaces **observation operator** matching the Steam Audio one,
2. regenerate the two datasets in the **exact same `.npz` schema**,
3. run the **unchanged** `jepa/*.py` pipeline and compare the numbers.

Do **not** reimplement the model/probes/control — copy `jepa/*.py` as-is and
point them at your SS2-generated data. If a Python file imports a Steam-Audio-
specific path, fix only that import, and note it.

## 3. The observation operator (build this to match `acoustic_env.h::observe`)

For a listener at pose `(x, 1.6, z)` with head yaw `θ` (degrees; θ=0 faces −z,
+θ turns toward +x), produce a monostatic observation = two streams:

- **`range[256]` (float16)** — a matched-filter range/reflectivity profile:
  1. Render the RIR with a **co-located omni source+receiver** at the pose. Use
     the **W channel** (order-1 ambisonic W, or mono omni) — pre-binaural.
  2. Pulse-compress it exactly as `../SteamAudioIR/analysis/matched_range.py`
     does: excite with a 100 Hz→10 kHz, 50 ms exponential sweep (ESS), form the
     received signal (self-emission + RIR convolution), self-cancel the emission
     with a 1% residual, convolve with the Farina inverse filter, take `|·|`.
  3. Pool the profile from `RANGE_GUARD=96` samples past the zero-delay reference
     `z0 = len(ess)-1`, in `RANGE_BINS=256` bins of `RANGE_POOL=16` samples each,
     as `log1p(mean(x²)·1e4)`. (This is `MatchedFilter::profile`+`pool_log` in
     `../AcousticJEPA/src/acoustic_env.h` — port it verbatim; SS2's sample-
     accurate RIR should give *finer* range structure than Steam Audio's 10 ms
     bins, which is itself worth reporting.)
- **`coch[2,64,48]` (float16)** — a binaural gammatone cochleagram: decode the
  ambisonic RIR to binaural with an HRTF, crop the first 0.2 s (9600 samples @
  48 kHz), run the 64-band ERB gammatone filterbank in 48 time frames,
  `log1p(mean(x²)·1e4)`. (This is `Cochleagram::apply`.) If SS2 gives you
  binaural directly, use it and record the HRTF in `VERSION.txt`.

SR = 48000, IR ≥ 0.5 s, order-1 ambisonic (W suffices for range; full order-1
for the binaural decode). Verify the operator on the **smoke test** from the
characterization manifest (co-located source/receiver vs a 20×20 m wall at
d = 2/5/10 m: echo exists, delay ≈ 2d/c, level falls with d) before generating
any dataset.

## 4. Geometry (match `make_box_room` / `build_room_mesh`)

Procedural shoebox rooms, meshed with `trimesh` → GLB (as in the characterization
manifest). Per seed, using a reproducible RNG:
- dims: `Lx,Lz ∈ [3,12] m`, `Ly ∈ [2.5,4] m`; room centred on x/z (origin at
  `(-Lx/2, 0, -Lz/2)`), floor at y=0.
- 6 walls, each assigned one of 4 materials (plaster/tile/carpet/mirror —
  absorption {0.1,0.15,0.2}/{0.01,0.01,0.02}/{0.3,0.5,0.6}/{0,0,0}; map onto RLR's
  material schema, noting deviations).
- 0–3 rectangular **openings** carved on distinct side walls (make the box
  acoustically leaky).
- Interior **pillars** (v2 cross-view only): floor-standing AABB boxes rising
  past 1.6 m ear height; keep agents ≥0.4 m clear of them.

You do **not** need to match Steam Audio's random stream bit-for-bit — you need
the same *distribution* of rooms. Generate your own seeds; store the true dims,
openings, pillars, and agent poses so the labels are exact (below).

## 5. The two datasets (exact `.npz` schema — the contract)

Write float16 for the big arrays, float32/int32 for the rest, keys **exactly**:

**(a) Cross-view probe dataset** — one `.npz` per room (N=6 agents, P=N·(N−1)=30
pairs), mirroring `swarm_dump.cpp`:
`mono_coch[N,2,64,48] mono_range[N,256] cross_coch[P,2,64,48] cross_range[P,256]
pair[P,2]=(listener,emitter) T_true[P,6]=(dx,dy,dz,range,rel_yaw,bearing_local)
T_meas[P,6]=(dx,dy,dz,range,bearing,fb_ok) occluded[P] agent_pose[N,4]
dist_wall[N,8] room_vol[N] material[N] open_az[N] room_meta[10]=(Lx,Ly,Lz,nOpen,
mat0..5) obstacles[nOb,7]`.
- CONTEXT = emitter's monostatic obs; TARGET = listener's **reflection-only**
  reception of the emitter's ping (bistatic: source at emitter, receiver at
  listener; SS2 *does* render a direct path, so **exclude the direct arrival**
  from the target — window it out or render reflections-only — this is the one
  place SS2 differs from Steam Audio, which renders no direct path).
- `T_meas` = the transform the listener could *measure*: range from the direct
  arrival ToA, bearing from binaural ITD through the listener's own ITD-vs-
  azimuth calibration, front/back from a +20° head-turn (port `SwarmEnv::
  measure_tab`). Because SS2 renders a real direct path, you can measure ToA/ITD
  from the actual bistatic render instead of an injected arrival — note this.
- Analytic labels (`dist_wall`, `room_vol`, `material`, `open_az`) are computed
  from the geometry, not the audio (`analytic_label`).

**(b) Monostatic trajectory dataset** — one `.npz` per episode (48-step explore-
and-avoid walk), mirroring `jepa_dump.cpp`:
`coch[K,2,64,48] range[K,256] actions[K,2]=(dyaw,dstep) labels_dist[K,8]
open_az[K] room_vol[K] material[K] pose[K,4]=(x,y,z,yaw) room_meta[10]`.
Actions: dyaw∈[−45,45]°, dstep∈[0,1.5] m; action[t] maps obs t→t+1; pose
dynamics = step then clamp 0.3 m off walls, then turn.

Scale: **≥2000 cross-view rooms** and **≥200 trajectory episodes** (the Windows
side is scaling the cross-view set to 2000; match it). Save under
`results/ss2_jepa/swarm_data/` and `results/ss2_jepa/traj_data/`.

## 6. Run the pipeline (unchanged Python)

In a Python 3.11/3.12 venv with torch (CUDA if available):
```
python jepa/train_xview.py     results/ss2_jepa/swarm_data --epochs 60 --out ckpt_ss2.pt
python jepa/probe_multiseed.py results/ss2_jepa/swarm_data --ckpts "ckpt_ss2*.pt"
python jepa/sweep_xview.py     results/ss2_jepa/swarm_data
python jepa/control_mpc.py     results/ss2_jepa/traj_data --repeats 5 --goal center
python jepa/control_mpc.py     results/ss2_jepa/traj_data --repeats 5 --goal wall
```
For closed-loop control, port `observe_server.cpp` to an SS2 "observe server"
(pose in → coch+range out) so `control_mpc.py --closed-loop` works against the
live SS2 renderer.

## 7. Reference numbers (Steam Audio — compare against these)

| quantity | Steam Audio value | SS2.0 verdict |
|---|---|---|
| probe room_vol R² (nonlinear) | JEPA 0.81 / raw 0.93 / random 0.90 | AGREES / DIVERGES |
| probe peer_pos R² (nonlinear) | JEPA 0.09 / raw 0.09 / random 0.20 | |
| sweep: best JEPA room_vol | 0.85 (< raw 0.93) | |
| control (reach centre, 5 seeds) | learned +0.86 / random-enc −0.34 / random-act −0.55 / analytic +0.58 | |
| control gap learned−random-enc | +1.20 ± 0.11, positive every seed | |

**The headline to test:** on SS2.0, is raw/random ≥ learned on the geometry
probes (the negative), AND learned ≫ random-encoder on control (the positive)?
If both reproduce, the dissociation is engine-independent.

## 8. Definition of done

`results/ss2_jepa/` contains: the two datasets, the observation-operator script
(copied in, so the paper can cite it), `VERSION.txt` (SS2/habitat-sim/rlr
versions), and `SS2_JEPA_RESULTS.md` — a table mirroring section 7 with the SS2
numbers, each row flagged AGREES / DIVERGES, plus one paragraph on the one
expected difference (SS2 renders a real direct path; Steam Audio does not) and
how you handled it in the cross-view target and the measured transform. Partial
completion in the order (probes → control) is acceptable; silent gaps are not.
