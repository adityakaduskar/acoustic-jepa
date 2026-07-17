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
