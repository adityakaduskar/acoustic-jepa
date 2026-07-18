# Acoustic-JEPA v2 — cross-view results

## The v2 hypothesis

v1's negative result was that predicting your **own** next echo (action-conditioned
self-prediction) never beat a linear probe on raw-pooled cochleagrams for geometry
decode, at any scale (80→2000 rooms). Diagnosis: self-prediction over small steps
is solvable without an allocentric scene representation.

v2 tested a specific fix: **cross-view prediction**. Agent E emits; predict what
listener L *receives* (reflection-only), from E's own monostatic observation plus
the measured relative transform E↔L. Because egocentric interpolation cannot cross
viewpoints, a latent that solves this *should* be forced allocentric — and should
therefore beat raw pooling on allocentric probes raw can't cheat.

Supporting design (all built + verified, see git log):
- two-stream encoder: matched-filter range tower (HRTF-free) + binaural cochleagram
  tower, block-structured latent `[z_range | z_coch]`, shallow context fusion;
- self-supervised transform `T_meas` (ITD self-calibration + head-turn front/back;
  median 1.7° / 0.10 m vs oracle), not the oracle;
- interior obstacles + agent occlusion, so the task can't be solved by regressing
  the ~10 box-room parameters;
- occluded pairs kept in and their flag withheld from the model (inferring a
  blocked path is part of the task).

## Result (400 rooms, 6 agents, 3 obstacles; 320 train / 80 test rooms)

Training is healthy: cross-view skill +0.40 over the no-transform baseline (the
predictor genuinely uses the transform), latent high-rank (51/256) and
non-collapsed (std ≈ 1.0). So the objective is learnable and stable.

Frozen-latent linear probes, room-level split, JEPA vs pooled-raw vs random encoder:

| target (metric)   | chance | JEPA  | raw   | random | winner |
|-------------------|-------:|------:|------:|-------:|--------|
| peer_pos (R²)     |  0.000 | 0.094 | 0.121 | 0.098  | raw    |
| room_vol (R²)     |  0.000 | 0.794 | 0.879 | 0.878  | raw    |
| occluded (acc)    |  0.747 | 0.764 | 0.777 | 0.772  | raw    |

**Raw pooling wins or ties every probe; JEPA never beats it — including the
allocentric `peer_pos`, where JEPA is at the random-encoder level.** On `room_vol`
JEPA is *below* the random encoder.

## Interpretation (honest)

The v2 hypothesis is **not supported** at this scale. Cross-view prediction, despite
training well, produces a representation no better than raw pooled features for
linear geometry decode — and worse than a random projection for the low-order
control (volume).

The mechanism is the informative part: predictive SSL reshapes the latent toward
what is **predictable**, which is not the same as what is **linearly geometrically
informative**, and can actively **discard** linearly-accessible geometry a random
projection retains (volume here; cf. v1's temporal-encoder run, which also degraded
volume decodability). This generalises v1's finding: on this simulator the geometry
cues are low-order and linearly present in the raw input, so a strong linear
baseline is hard to beat, and a predictive objective that compresses toward
predictability tends to move *away* from linear geometric decodability, not toward
it. Changing *whose* echo you predict (self → cross-view) did not change this.

## Nonlinear probe (does JEPA hide geometry a linear probe can't read?)

Same MLP probe (2 hidden layers, dropout, weight decay, early-stopped on a val
split) applied **identically** to all three feature sets, on the same 400-room
checkpoint:

| target (metric) | chance | JEPA  | raw   | random |
|-----------------|-------:|------:|------:|-------:|
| peer_pos (R²)   |  0.000 | 0.088 | 0.093 | **0.204** |
| room_vol (R²)   |  0.000 | 0.795 | **0.930** | 0.901 |
| occluded (acc)  |  0.747 | 0.762 | 0.775 | 0.764 |

**The nonlinear probe does not rescue JEPA — it hardens the negative.** The
random encoder's `peer_pos` R² more than doubles under the MLP (0.10→0.20) while
JEPA stays flat (0.09→0.09); `room_vol` improves nonlinearly for raw and random
but not for JEPA (flat 0.79, now worst by a wide margin). So peer position and
volume have genuine **nonlinear** structure in the acoustic features that a
*random projection preserves and an MLP extracts* — and cross-view JEPA training
has **compressed it away**. The answer to "is the geometry there but nonlinear?"
is: yes it is there (in raw/random), and no, JEPA does not hold it — it holds
*less* geometry than a random projection, linearly and nonlinearly.

(Single probe seed; the pattern is consistent across all three targets, so it
does not hinge on one number.)

## Non-JEPA controls (is the negative specific to the predictive objective?)

Trained a reconstruction **autoencoder (AE)** and a **supervised** head on the
SAME two-stream architecture / inputs, probed identically (400 rooms):

| target (metric) | JEPA  | AE    | raw   | random | probe |
|-----------------|------:|------:|------:|-------:|-------|
| room_vol (R²)   | 0.795 | 0.867 | 0.930 | 0.903  | nonlinear |
| peer_pos (R²)   | 0.088 | 0.090 | 0.093 | **0.208** | nonlinear |
| room_vol (R²)   | 0.794 | 0.849 | 0.879 | 0.877  | linear |
| peer_pos (R²)   | 0.094 | 0.055 | 0.121 | 0.127  | linear |

Supervised ceiling (encoder trained end-to-end on the target, test rooms):
`room_vol` R² = 0.905 (ties raw/random — even the ceiling doesn't exceed raw);
`peer_pos` R² = 0.001 (the flexible encoder overfits room identity and fails to
generalise, whereas a *frozen* random projection decodes peer_pos at 0.208).

**The negative is not JEPA-specific — it generalises to learned representation
itself.** The `room_vol` ordering is monotonic in how hard the objective
compresses input variance:

    JEPA (predict) 0.795  <  AE (reconstruct) 0.867  <  random 0.903  <  raw 0.930

i.e. prediction discards the most geometry, reconstruction less, and *not
learning at all* (random projection / raw pooling) preserves the most. The AE
confirms the mechanism (the predictive objective is the worst) but no learned
SSL objective — predictive OR reconstructive — beats raw/random features, and
even direct supervision doesn't exceed the raw baseline (or fails to generalise).
On this simulator the decodable geometry is best accessed by *not* learning a
representation.

## Why raw / random win (mechanism, verified)

Hand-computed low-order statistics per observation, ridge-probed (room split):

| target   | 3 reverb stats | 7 hand-stats | raw (192-d) |
|----------|---------------:|-------------:|------------:|
| room_vol |          0.682 |        0.752 |       0.860 |
| peer_pos |            —   |        0.005 |       0.120 |

Three numbers -- the range profile's delay centroid, its spread, and the
late/early energy ratio -- recover 0.68 R² on room volume (~80% of the full raw
features). That is Sabine-style physics: bigger room -> longer, more-spread
reverberation -> the echo's energy-vs-delay distribution IS a volume gauge, and
pooling is exactly the operation that computes it. So:

- **raw wins** because pooling = computing the statistics the target is linearly
  encoded in; there is no high-order structure to disentangle, and any learned
  objective only compresses that variance away.
- **random wins** two ways: on room_vol (linearly accessible) a random linear
  projection is a near-isometry (Johnson-Lindenstrauss) that preserves linear
  decodability; on peer_pos (a distributed NONLINEAR cue -- the coarse hand-stats
  fail at 0.005 -- from the bistatic delay-ellipse + fine ITD/ILD) the untrained
  CNN is a random nonlinear feature bank (random features / kitchen sinks;
  conv+pool prior regardless of weights) that an MLP reads at 0.208, above any
  learned encoder.

Unifying point: the geometry is already exposed by FIXED operations (pooling,
random projection); representation learning's one job -- surfacing hidden
structure -- is unnecessary here, so training only discards variance. Hence the
monotonic ordering: the more a method compresses, the more exposed geometry it
loses.

## Control test (latent-MPC navigation) — the dissociation

"Poor geometry probe" and "poor control substrate" are different questions. An
action-conditioned two-stream world model (encoder + latent dynamics + nearest-
wall readout) was trained on monostatic trajectory data; open-loop CEM-MPC plans
H=6 actions to approach the nearest wall, executed in TRUE pose (analytic
dynamics), scored by true nearest-wall-distance reduction (held-out rooms, 200
starts):

| controller           | nwd reduction (m) |
|----------------------|------------------:|
| analytic (oracle)    | **+1.059** ± 0.075 |
| learned MPC          | **+0.922** ± 0.074 |
| random actions       |  +0.530 ± 0.089 |
| random-encoder MPC   |  **−0.106** ± 0.028 |

**This flips the probe result.** The learned world model navigates at 87% of the
analytic oracle, while the random-encoder world model — the encoder that *tied or
beat* the learned one on every geometry probe — FAILS, doing worse than random
actions (its confident plans move *away* from walls).

Mechanism (the resolution of the whole arc): MPC needs the latent's **dynamics**
to be predictable under actions, not its statics to be decodable. A random
encoder has decodable geometry (good probe) but *chaotic* dynamics — a small pose
change maps through the fixed random nonlinearity to an arbitrary latent jump —
so the predictor cannot roll it forward and CEM optimises a broken model
(worse-than-random). The learned encoder is trained *jointly* to make its latent
predictable under actions; it trades some static decodability for **dynamical
predictability**, which is exactly what model-based control consumes.

So: static decodability (probes) and dynamical predictability (control) are
different properties. Raw pooling / random projection win the first; only a
learned predictive representation delivers the second. Predictive SSL's value
here is not exposing geometry (raw already does) but making dynamics
plannable.

## Robustness (error bars — both results hold)

Both headline claims were re-run across seeds so the confidence is measured, not
asserted.

Experiment 1 (geometry negative), nonlinear probe over 3 JEPA checkpoints × 5
random-encoder draws × 3 probe seeds:

| target       | JEPA          | raw           | random        | verdict            |
|--------------|--------------:|--------------:|--------------:|--------------------|
| peer_pos R²  | 0.069 ± 0.012 | 0.091 ± 0.004 | 0.204 ± 0.012 | raw > JEPA (robust)|
| room_vol R²  | 0.805 ± 0.006 | 0.931 ± 0.001 | 0.904 ± 0.004 | raw > JEPA (robust)|

SDs are ~0.01 and raw beats JEPA by more than the combined spread on both — the
negative is tight and reproducible, and JEPA loses to random too.

Experiment 2 (control positive), 5 training seeds, fixed eval starts:

| controller         | reduction (m)   |
|--------------------|----------------:|
| analytic (oracle)  | +0.985          |
| learned MPC        | +0.798 ± 0.051  |
| random actions     | +0.551          |
| random-encoder MPC | −0.054 ± 0.138  |

learned − random-encoder = +0.852 ± 0.132, **positive on every one of the 5
seeds**. The dissociation survives error bars in both directions. (The learned
mean settled at +0.80, below the +0.92 of the first single run — averaging over
seeds corrected an optimistic draw, which is exactly why the error bars matter.)

Harder, confound-free goal (reach the room CENTRE — stay away from all walls;
5 seeds). "Approach the nearest wall" was criticised as too easy: a bounded box
drifts random walkers toward walls for free (that is why random actions scored
+0.55). Reaching the centre is the opposite — random drift moves *away* from it:

| controller (reach-centre) | wall-distance increase (m) |
|---------------------------|---------------------------:|
| learned MPC               | **+0.859 ± 0.070**         |
| analytic (greedy, true geometry) | +0.577              |
| random-encoder MPC        | −0.339 ± 0.088             |
| random actions            | −0.551                     |

Now random actions FAIL (−0.55) — the confound is gone, the task genuinely
requires purposeful navigation — yet the learned world model still succeeds
(+0.86), beats random actions by ~1.4 m, and the learned−random-encoder gap
GROWS to +1.198 ± 0.106, positive on every seed. The learned MPC even beats the
greedy analytic controller (+0.86 vs +0.58): the analytic heads straight at the
centre at full stride and overshoots a point target, while CEM-MPC plans the
whole 6-step horizon and settles — horizon planning beating myopic greedy is
exactly what a world model should buy. The control-positive is not an artifact
of an easy goal.

## Hyperparameter sweep (JEPA's best shot — negative holds)

To be fair to JEPA before concluding, swept the knobs most likely to help it
retain geometry (VICReg variance weight, coch-loss weight, epochs, capacity, lr),
10 configs, nonlinear-probed vs baselines (raw room_vol 0.930 / peer_pos 0.093;
random 0.898 / 0.186):

| config              | room_vol | peer_pos |
|---------------------|---------:|---------:|
| default             |   0.835  |  0.085   |
| var=4               |   0.810  |  0.073   |
| var=16              |   0.683  |  0.059   |
| lam=1.0             |   0.839  |  0.098   |
| lam=0 (range-only)  |   0.829  |  0.097   |
| epochs=20           |   0.815  |  0.074   |
| epochs=20, var=4    |   0.809  |  0.070   |
| lr=3e-4             | **0.849**|  0.091   |
| emb=256             |   0.842  | **0.113**|
| var4,lam1,e30       |   0.807  |  0.074   |

Verdict: **even the best-tuned JEPA loses.** On room_vol (the strong target) no
config approaches raw (best 0.849 vs raw 0.930, all below random 0.898). On the
weak peer_pos target a few configs marginally edge raw (0.093) but stay far below
random (0.186) — none beats the best baseline on either target. Mechanistic
note: cranking VICReg variance (var=16) made room_vol *worse* (0.683), so the
compression is not fixed by forcing variance retention — the predictive objective
reshapes toward predictability regardless. The geometry negative is robust to
tuning, not a hyperparameter artifact.

## What this does and does not establish

- It is consistent with v1's 2000-room plateau (~0.1 R² below raw); scaling 400→2000
  would likely strengthen, not reverse, the negative result.
- The negative is now shown under **both linear and nonlinear** probes, and the
  mechanism is confirmed: the predictive objective discards geometric variance
  (JEPA < random encoder), rather than re-encoding it in a form a linear probe
  misses.
- `peer_pos` is genuinely decodable (random 0.20 R² nonlinearly) from the
  reflection-only signal — so the failure is JEPA's, not an unlearnable target.

## Reproduce

```
tools\swarm_dump.exe 400 swarm_data400 6 1 3 8
.venv\Scripts\python.exe jepa\train_xview.py swarm_data400 --epochs 60 --batch 256 --out ckpt_xview400.pt
.venv\Scripts\python.exe jepa\probe_xview.py swarm_data400 --ckpt ckpt_xview400.pt
```
(Render carries ~1% run-to-run jitter; geometry/labels are exact — numbers reproduce
to ±0.01 R², not bit-identically.)
