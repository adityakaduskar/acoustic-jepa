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

## What this does and does not establish

- It is consistent with v1's 2000-room plateau (~0.1 R² below raw); scaling 400→2000
  would likely strengthen, not reverse, the negative result.
- The probe is **linear** by design (the whole comparison is about linear
  decodability vs raw). JEPA could encode geometry nonlinearly; that is a different
  question.
- `peer_pos` is only weakly decodable by anyone (~0.1 R²) from the reflection-only
  signal — a hard target, so its discriminating power is limited.

## Reproduce

```
tools\swarm_dump.exe 400 swarm_data400 6 1 3 8
.venv\Scripts\python.exe jepa\train_xview.py swarm_data400 --epochs 60 --batch 256 --out ckpt_xview400.pt
.venv\Scripts\python.exe jepa\probe_xview.py swarm_data400 --ckpt ckpt_xview400.pt
```
(Render carries ~1% run-to-run jitter; geometry/labels are exact — numbers reproduce
to ±0.01 R², not bit-identically.)
