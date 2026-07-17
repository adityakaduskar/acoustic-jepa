# Acoustic JEPA — implementation spec

An action-conditioned Joint-Embedding Predictive Architecture over binaural
echoes rendered by the Steam Audio (phonon) reflection engine. The model learns
to predict the *latent* representation of the next binaural observation from the
current one plus the agent's motor action, with no waveform reconstruction and
no geometric supervision. The scientific claim to establish: **the learned
latent linearly encodes scene geometry** (distance-to-wall, opening direction,
room volume, material), and the learned predictor is usable as a world model for
echolocation planning.

This spec is written against the real phonon v4.8.1 call sequence in
`echo_nav.cpp::sense()`; the observation operator is that function, lightly
refactored. Cross-refs: `steam-audio-ir-tool` memory, `echolocation-roadmap`
(this is a formalisation of items #11 "learned vs analytic controller" and a
prerequisite study for #2 wall-following), `windows-build-env`.

---

## 0. The observation operator (reuse of `sense()`)

One environment observation = one ping rendered to a binaural echo. The phonon
sequence, taken verbatim from `echo_nav.cpp` (lines 73–120):

```
hf = headFrame(pose, yawDeg)                          // IPLCoordinateSpace3
iplSimulatorSetSharedInputs(sim, REFLECTIONS, {        // per-ping
    listener=hf, numRays=NUM_RAYS, numBounces=NUM_BOUNCES,
    duration=IR_DUR, order=ORDER, irradianceMinDistance=0.1 })  // Gotcha 4
iplSourceSetInputs(src, REFLECTIONS, { source=hf, distAtten=DEFAULT,
    airAbs=DEFAULT, reverbScale={1,1,1} })
iplSimulatorRunReflections(sim)                        // Gotcha 1: re-run before EVERY render
out = iplSourceGetOutputs(src, REFLECTIONS)            // handle valid for ONE pass
rp  = out.reflections; rp.type=CONVOLUTION; rp.numChannels=AMBI(16); rp.irSize=IRN
// ambisonic IR from a unit impulse:
iplReflectionEffectReset(refl)
for frame in impulse: iplReflectionEffectApply(refl, &rp, inB, oB)  -> ambi[16][total]
// binaural decode with the head orientation:
iplAmbisonicsDecodeEffectApply({order=ORDER, hrtf=gHRTF,
    orientation=hf, binaural=IPL_TRUE}, ambi -> bL[2][total])
```

Constants (from `echo_nav.cpp`): `SR=48000, FRAME=1024, ORDER=3, AMBI=16,
IR_DUR=0.5, IRN=24000, total=IRN+FRAME=25024, C=343`.

**Observation tensor.** The raw binaural echo `bL[2][25024]` is 96% silence/tail.
Crop to the first `T_OBS = 0.2 s` (9600 samples) — covers echoes out to
`C*T_OBS/2 = 34 m`, past any room we train in — and convert to a **binaural
gammatone cochleagram** (bio-plausible front end; also the natural interface to
the TRIBEv2 auditory pathway in §6):

```
o  ∈ R[2, F=64, Tf≈48]     # 2 ears × 64 gammatone bands × ~48 time frames
```

Keep the cropped raw `bL[2][9600]` alongside `o` in the dataset so fine ITD is
recoverable for probes (the cochleagram envelope loses sub-sample ITD).

**Engine caveats already folded in.** Gotcha 1 (re-run reflections before each
render). Gotcha 4 (`irradianceMinDistance=0.1` or near surfaces go silent).
Gotcha 5 (echo time is quantised to 10 ms / 1.7 m bins — so onset is a *coarse*
range cue; the model must lean on ITD/ILD/spectrum for the rest, which is the
point). Since emitter and receiver are co-located, Gotcha 3's direct-delay
subtraction is a no-op (s≈0) and native reflection onset ≈ 2d/c is consistent
across the dataset — no per-sample timing correction needed for training.

---

## 1. Environment API

Refactor `sense()` into a reusable `AcousticEnv` with a Gym-like surface. Two
delivery modes share one renderer.

```cpp
struct Obs   { float coch[2*64*48]; float raw[2*9600]; };   // observation
struct Label { float dist_wall[8];   // GT range to nearest surface in 8 yaw sectors
               float open_azim;       // azimuth of largest gap (deg, allo)
               float room_vol;        // m^3
               int   material_id;     // 0..M-1
               float pose[3]; float yaw; };                 // GT pose (eval only)
struct Action{ float dyaw;            // [-45, +45] deg
               float dstep; };        // [0, 1.5] m forward

// core: render one observation at a pose (this IS sense(), minus the controller)
Obs   env_observe(IPLVector3 pose, float yawDeg);
Label env_label  (IPLVector3 pose, float yawDeg);   // analytic, from procedural geom
```

**Action semantics.** `dstep` along current heading then `dyaw` applied (or the
reverse — fix one convention and log it). Emission is implicit: every step pings.
Clamp candidate poses to stay inside the room and ≥ 0.3 m off walls.

**Scenes.** Two generators:
- *Procedural box rooms* (primary, for pretraining scale): axis-aligned boxes
  `[Lx,Ly,Lz]` sampled in 3–12 m per side, 0–3 rectangular openings punched in
  walls, per-wall material sampled from an M-class table (absorption/scattering
  triples). Mesh built exactly like the `icosphere()` helper builds triangles →
  `iplStaticMeshCreate`. All GT labels are analytic from the box params (no
  phonon geometry query needed).
- *Real Level 4 mesh* (transfer/eval): load `level4_geo.bin` via
  `iplStaticMeshLoad` (already working per `steam-audio-ir-tool`). Used to test
  that a model pretrained on procedural rooms transfers to the real wall.

---

## 2. Offline dataset (unblocks pretraining immediately)

A standalone `jepa_dump.cpp` (sibling of `matched_range.cpp`, same `build.bat`)
walks random trajectories and writes tuples. No Python↔C++ live coupling needed
for the JEPA pretraining phase.

**Trajectory sampling.** Per episode: random room, random start pose, then
`K=32` steps of a mildly-exploratory random policy (bias toward walls so echoes
are informative — e.g. step toward the nearest surface 50% of the time). At each
step record `(o_t, raw_t, a_t, label_t)`.

**On-disk format.** One `.npz` per episode (float16 for `coch`, float32 for
`raw`), fields:
```
coch   [K, 2, 64, 48]     actions [K, 2]      labels_dist [K, 8]
raw    [K, 2, 9600]       open_az [K]         room_vol [K]   material [K]
pose   [K, 4]             room_meta (Lx,Ly,Lz, openings, mats)
```
Target scale for a first result: **~2k rooms × 32 steps ≈ 64k transitions**
(a few GB at float16). The C++ dumper is the throughput bottleneck, not training
— parallelise over rooms with the existing `numThreads` sim setting, one
`IPLSimulator` per worker thread.

---

## 3. Model (JEPA recipe)

Non-contrastive, EMA-target, VICReg-regularised — the I-JEPA / V-JEPA line, with
V-JEPA-2-style multi-step latent rollout for planning.

```
Encoder  E_θ : o[2,64,48] -> z ∈ R^256      # small 2D CNN (5 conv blocks) or a
                                             # tiny ViT; shared weights per ear then concat
Target   E_ξ : EMA copy of E_θ, ξ ← m·ξ + (1-m)·θ, m=0.996, stop-grad
Predictor P_φ: (z_t, a_t) -> ẑ_{t+1}         # MLP (z ⊕ action-embed) or 2-layer
                                             # transformer for multi-step rollout
```

**Loss (single-step):**
```
L_pred = || P_φ(E_θ(o_t), a_t) − sg(E_ξ(o_{t+1})) ||²
L_vc   = VICReg variance+covariance terms on E_θ(o_t)      # anti-collapse (critical)
L      = L_pred + λ_v·L_vc         # λ_v ≈ 1.0, tune
```

**Multi-step (enables planning).** Roll P forward H=4 steps in latent space and
supervise each against the EMA target of the true future obs:
```
ẑ_{t+k} = P_φ(ẑ_{t+k-1}, a_{t+k-1}),   L = Σ_k γ^k || ẑ_{t+k} − sg(E_ξ(o_{t+k})) ||²
```

**Collapse watchdog.** Log latent std per dim and the rank of a batch of `z`;
a collapse (std→0) means bump `λ_v` or lower LR. This is the #1 failure mode for
non-contrastive audio SSL — instrument it from day one.

---

## 4. Evaluation — the actual scientific result

**(a) Linear geometry probes (headline).** Freeze `E_θ`. Fit ridge/logistic
probes `z → label`:
| target            | metric        | baseline to beat                    |
|-------------------|---------------|-------------------------------------|
| dist-to-wall (8×) | R² / m-MAE    | matched-filter onset range (you have it) |
| opening azimuth   | circ-MAE (°)  | ITD/ILD bearing (you have it)       |
| room volume       | R²            | total echo energy / RT60 heuristic  |
| material class    | accuracy      | ESS echo-spectrum classifier (roadmap #4) |
The claim lands if the frozen JEPA latent beats the analytic single-cue
baselines *jointly* — i.e. one representation carries all four, learned only from
next-latent prediction.

**(b) World-model planning.** Use `E_θ`+`P_φ` as a latent forward model; plan
actions by CEM/MPC in latent space to (i) reach the wall, (ii) wall-follow by
holding a target latent. Compare step count / success vs the analytic ITD/ILD
controller in `echo_nav.cpp`. This *is* roadmap #11, and a positive result is
the bridge to #2.

**(c) Ablations.** mono vs binaural obs; cochleagram vs raw; with/without action
conditioning (does the predictor actually use `a_t`?); single- vs multi-step.

---

## 5. Repo layout & build

```
SteamAudioIR/
  jepa_dump.cpp         # NEW: env + trajectory sampler + .npz writer (build.bat)
  acoustic_env.h        # NEW: env_observe/env_label refactored out of sense()
  jepa/                 # NEW: Python training (separate venv, see below)
    dataset.py  model.py  train.py  probe.py  plan.py
  ACOUSTIC_JEPA_SPEC.md # this file
```

**Build.** `jepa_dump.cpp` links exactly like the other exes — bundled
`phonon.lib`, MSVC via `vcvars64` (`windows-build-env`). Reuse `build.bat`; mind
the `echo`/`>` and `LIB`/`INCLUDE` batch gotchas noted there.

**Python/ML env caveat.** The on-PATH `python` is the Store stub and the real
3.14 interpreter has no reliable PyTorch wheels yet. Create a **dedicated
training venv on Python 3.11/3.12** for PyTorch, kept separate from the 3.14 used
for IR analysis. The dumper writes plain `.npz`, so the training venv never
touches phonon or the float-WAV parsing issue.

---

## 6. TRIBEv2 tie-in (Meta trimodal brain encoder)

TRIBEv2 predicts cortical responses to naturalistic video/audio/text; its audio
pathway is the relevant one here. Two feasible uses, neither a core dependency —
both are *evaluation axes* layered on top of the model above:

1. **Brain-alignment as a representation-quality metric.** Fit a linear encoding
   map from the frozen acoustic-JEPA latent to auditory-cortex responses in a
   public naturalistic-audio fMRI/MEG dataset, and report encoding accuracy
   (brain-score style). The sharp version uses **human-echolocation neuroimaging**
   (Thaler et al.: expert echolocators recruit calcarine/"visual" cortex for
   click echoes) — the question becomes whether a self-supervised *predictive*
   acoustic model aligns to echo-driven cortex better than reconstruction-trained
   or supervised audio encoders. That is a genuinely novel, publishable claim and
   the tightest JEPA↔TRIBE link.
2. **Architecture transfer.** TRIBEv2's recipe (frozen multimodal encoder +
   lightweight predictor head to a target space) is the same shape as §3. If you
   later add a vision channel in Unreal, a TRIBE-style trimodal encoder over
   {binaural echo, egocentric frame, action} is the natural extension — and the
   brain-alignment metric in (1) becomes the shared yardstick across modalities.

Scope note: TRIBE datasets are movie-watching, not echolocation, so (1) is a
transfer/alignment study, not a drop-in benchmark — the echolocation fMRI route
is the one that makes the connection real.

---

## 7. Build order

1. ~~Refactor `sense()` → `acoustic_env.h` (`AcousticEnv::observe`), procedural
   box-room generator + analytic `AcousticEnv::label`.~~ **DONE** — header-only,
   compiles & runs against the real SDK (`-DACOUSTIC_ENV_TEST` self-test,
   geometry check PASS). Openings are carved into the mesh (frame of quads around
   each hole, ≤1 hole/wall on distinct walls). NOTE the self-test also confirmed
   binaural onset ≠ wall distance (HRTF latency + reverb tail); geometry is read
   from labels / the W channel, not the binaural mix.
2. ~~`jepa_dump.cpp` → offline `.npz` dataset.~~ **DONE** — `#include`s
   `acoustic_env.h`, walls-biased random trajectories, self-contained stored-ZIP
   `.npz` writer (no deps; verified loading in NumPy). Run:
   `jepa_dump [episodes] [out_dir] [steps] [seed0] [threads]`. Scale to ~2k
   episodes × 32 steps for the ~64k-transition pretraining set. `.npz` schema is
   documented at the top of `jepa_dump.cpp`.
3. `jepa/dataset.py` + `jepa/model.py` **DONE** (syntax-checked; runtime needs the
   torch venv — 3.14 has no wheels, see `jepa/requirements.txt`). `dataset.py`:
   lazy `.npz` loader, `JEPADataset` (H-step transition windows) + `ProbeDataset`
   (single obs+labels), action normalisation. `model.py`: shared per-ear GroupNorm
   CNN `Encoder`, EMA `target`, residual action-conditioned `Predictor`, VICReg
   var/cov, `latent_stats` collapse watchdog, `AcousticJEPA` with H-step rollout.
   Smoke tests under each file's `__main__`.
   `jepa/train.py` **DONE** (syntax-checked): AdamW + cosine LR/warmup, EMA
   momentum ramp, VICReg objective, collapse watchdog (warns on sustained low
   std), CSV log, checkpoints (`latest.pt`/`final.pt`), resume, AMP, and a
   `--device cpu --max-steps N` quick path. Runtime needs the torch venv.
4. `jepa/probe.py` — the §4(a) table. **This is the first reportable result.**
   **DONE** (syntax-checked): freezes the trained encoder, fits closed-form ridge
   probes (dist-to-wall MAE/R², room-vol R², opening-azimuth circular MAE) + a
   short linear classifier (material acc), split BY EPISODE so train/test never
   share a room. Reports JEPA vs two self-contained baselines (pooled raw
   cochleagram, random-init encoder); the single-cue analytic baselines
   (matched-filter range, ITD/ILD bearing) cross-refer the C++ tools. Writes
   `probe_results.json`. Runtime needs the torch venv + a trained checkpoint.
5. `jepa/plan.py` — §4(b). **DONE**: world-model rollout accuracy (predictor vs
   persistence) + latent CEM-MPC to approach the nearest wall (imagined; live
   closed-loop needs a Python<->phonon bridge, `echo_nav` ~9 steps is the ref).
6. (optional) §6 brain-alignment eval.

## 8. First end-to-end run (CPU, torch 2.12 venv)

Whole pipeline (dump -> dataset -> train -> probe -> plan) is wired into
`run_all.ps1` (GPU defaults: batch 256, AMP, device auto-detected). One command on
a CUDA box: `./run_all.ps1 -SetupVenv -Episodes 2000 -Epochs 60`. Reuse data/venv
with `-SkipData`; CPU smoke via `-TorchIndex .../whl/cpu -Device cpu`.

Env: `uv`-managed CPython 3.12 venv with torch-cpu (3.14 has no wheels). Training healthy, no
collapse (latent std 0.04->0.57, rank ->~55); predictor beats persistence
(rollout skill 0.63->0.80 over 4 steps); CEM-MPC reduces predicted nearest-wall
distance ~0.6 m, more than random/stay.

Frozen-latent geometry probes (episode-level split), JEPA vs baselines:

| data      | JEPA dist R² | raw-coch | rand-enc | JEPA vol R² | JEPA mat% | rand mat% |
|-----------|--------------|----------|----------|-------------|-----------|-----------|
| 80 rooms  | 0.477        | 0.688    | 0.633    | 0.739       | 49        | 71        |
| 400 rooms | 0.669        | 0.768    | 0.709    | 0.775       | 57        | 64        |
| 2000 rooms (GPU) | 0.704 | 0.802    | 0.750    | 0.876       | 64        | 71        |

**Honest read (updated after the 2k-room GPU run):** scaling data improves JEPA in
absolute terms (dist R² 0.477 -> 0.669 -> 0.704) but it **plateaus ~0.1 R² below
raw-cochleagram and ~0.05 below the random encoder — data scale ALONE does not get
it past the baselines.** So the negative result is not merely a small-data
artifact; it is *structural*: the co-located-emitter, short-horizon,
action-conditioned prediction pretext task does not force the encoder to represent
geometry better than raw pooled acoustic features already do. Two corroborating
signs at 2k: the latent is high-variance/high-rank (std ~0.95, rank ~180/256 — not
collapsed), yet material acc (64%) still trails raw/random (~71%), and the
predictor underfits on the diverse set (pred loss ~0.38, rollout skill drops to
0.34->0.48 vs 0.63->0.80 at 400 rooms). Volume is the exception: JEPA (0.876) now
matches raw (0.895) and beats random.

**What to change (task, not scale):** displaced emitter so range/parallax is
informative; longer horizon; motion *through* the carved openings; I-JEPA-style
spatial/multi-view masking instead of pure forward prediction; a higher-capacity
predictor; and a var_coeff sweep (var_coeff=4 drove std to ~0.95, which may
over-spread the latent into noise dims). MPC still works throughout (2k: predicted
nearest-wall dist 0.49 -> 0.13 m planned, vs 0.17 random / 0.37 stay).

**Redesign v1 implemented (displaced emitter + longer horizon).** `acoustic_env.h`
gained a beacon (`set_beacon`); `observe()` sources from it when set (bistatic),
else co-located (monostatic). `jepa_dump.cpp` places a fixed per-episode beacon
(new `beacon` [3] npz key), uses 48-step episodes, and an explore-and-avoid policy
that covers the room (~44 m/episode). Training horizon default raised 4 -> 8.
Build- and mechanics-verified; the open question is whether it beats the
monostatic 400-room baseline (dist R² 0.669) — not yet run at scale. Compare via
`jepa_dump.exe 400 <out> 48 1 8` then `run_all.ps1 -SkipData -DataDir <out>`.

Run cost (RTX 5080, cu128, Blackwell sm_120): 2000-room dump 8.8 min (CPU),
train 80 epochs 19.4 min (GPU), probe 12 s, plan 17 s.
```
