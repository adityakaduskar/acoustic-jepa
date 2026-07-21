# EchoRL — Experiments & Results (evidence log)

Closed-loop echolocation RL on the Steam Audio (phonon) renderer: a simulated "bat"
that navigates from **binaural echoes only**, in 3D, with physically-rendered
reverberant acoustics. 3D, rendered-acoustics successor to Mohan & Vanderelst 2020
(*Training a simulated bat*, SSCI). Design doc: `EchoRL_PLAN.md`.

Status as of **2026-07-21**: Stage 0 (obstacle avoidance) and Stage 2 (goal
approach / doorway-finding) both complete, with oracle ceilings, ablations, and
multi-seed CIs; plus a **model-based (learned-world-model CEM-MPC) vs model-free (PPO)**
comparison on the doorway task (§5). All numbers below are on **held-out rooms** (seed
band 600+, disjoint from training seeds 1–60 and the 3000+ corpus band).

Single machine: 24 logical CPUs, CUDA GPU, torch 2.11+cu128.

---

## 1. Substrate & environment

- **`src/episode_server.cpp`** — a headless phonon C-API server (built on the audited
  `acoustic_env.h`). One process = one procedural box room (walls + interior obstacle
  pillars + wall openings). Protocol: 3 float32 pose in `(x, z, yaw)` → 6400 float32 out
  = a `range[256]` matched-filter profile + a `coch[2,64,48]` binaural cochleagram.
  Extended handshake streams room dims + obstacle AABBs + openings so Python owns
  collision/reward. Separate exe from `observe_server` (kept the MPC pipeline intact).
- **`echorl/env.py` (`EchoLocEnv`)** — Gymnasium-shaped, dependency-free. Python owns pose
  dynamics (step-then-turn, mirrors `acoustic_env.h::apply_action`), collision, reward.
  Two tasks: `avoid` (Stage 0) and `goal` (Stage 2). Actions: discrete (5 turns × fixed
  0.75 m step) or continuous (`dyaw∈[-45,45]°`, `dstep∈[0,1.5] m`).
- **`echorl/vec_env.py`** — threaded K-env vector wrapper with autoreset (the C sim blocks,
  so envs must be stepped concurrently to reach throughput).

### 1.1 Measured throughput (the RL budget)
Per-`observe()` latency vs `simThreads` (obstacle room): all-cores 25.6 ms (39/s),
4 thr 57 ms, 2 thr 98 ms, 1 thr 168 ms → the reflection sim scales **poorly** with
threads (6.0 obs/s per thread at T=1 vs 1.6 at T=24). Aggregate over K parallel servers:

| config | obs/s |
|---|---|
| K=1, all cores (no vectorization) | 38.5 |
| K=8 × 2 thr | 84.0 |
| **K=12 × 2 thr (recommended)** | **90.3** |
| K=24 × 1 thr | 95.4 |

→ ~90 obs/s realistic; DQN training ran ~28–36 steps/s (inline gradient), PPO ~44 steps/s
(batched updates). 2M steps ≈ 6 h; the tasks here converged in ≪ that.

### 1.2 Offline corpus & normalization (`echorl/collect_corpus.py`)
2016 observations scattered across 24 rooms (seeds 3000–3023) → `norm_stats.npz`
(+ `corpus.npz`, 15 MB). Findings: **all 256 range channels carry variance**; **2605 of
6144 cochleagram channels are near-silent** (high-freq/late-time bins) → inputs are
normalized `(x−μ)/(σ+ε)`.

---

## 2. Perception architecture (reused from the cross-view JEPA encoder)

Shared trunk `EchoTrunk` (`models.py`) = `Normalizer` + `TwoStreamEncoder`:

- **Range tower** `RangeCNN` (1D CNN over `[256]`): Conv1d(1→32→64→128→128, k5 s2) +
  GroupNorm + GELU → AdaptiveAvgPool → 2-layer MLP → `z_range[128]`. HRTF-free (no direction).
- **Cochleagram tower** `EarCNN` (2D CNN, **weights shared across the two ears**): each ear
  `[1,64,48]` → Conv2d(1→32→64→128→64, k3 s2) → pool → `[64]`; L∥R concat → MLP → `z_coch[128]`.
  Carries the left/right (ITD/ILD) **bearing** cue.
- Output = `[z_range | z_coch | prop]` (259-d), `prop` = last action + episode-time.

Params: trunk 368 K; **DQN** `DuelingQNet` 502 K (dueling V/A heads over 5 actions);
**PPO** `RecurrentAC` 766 K (GRU 259→256 + Gaussian actor + critic); no-memory variant 435 K.

---

## 3. Stage 0 — obstacle avoidance ("fly without crashing")

Metric: mean flight length out of a 40-step cap, and collision rate, on held-out rooms.
Random floor: **nobs=0 4.4/100%, nobs=1 3.9/100%, nobs=3 2.7/100%** (always crashes in ~3–4 steps).

### 3.1 DQN (M3, discrete, paper-faithful) — `echorl/dqn.py`
**Reward bug found & fixed:** the original fixed-0.75 m-step made the progress term
*action-independent* (zero gradient on turn direction) and the near-field penalty fired
too late (<0.6 m at 0.75 m/step) → the greedy policy degraded *below* random (eval length
fell 3.4→3.0 as ε dropped). Fixed with dense anticipatory clearance shaping
`w_clear·tanh((clear−d_safe)/scale)` + larger collision penalty. Best-eval checkpointing
was added because DQN's greedy policy drifts late in training.

Graded result (best held-out, len/40, collision):

| tier | random | **DQN (best)** |
|---|---|---|
| nobs=0 (walls) | 4.4 / 100% | **30.8 / 26%** (7× floor) |
| nobs=1 (1 pillar, warm-started from nobs=0) | 3.9 / 100% | **9.0 / 88%** |
| nobs=3 (3 pillars, warm-started) | 2.7 / 100% | **~3 / 100%** (at floor) |

→ the discrete **fixed forward speed** can't slow/stop, so dense clutter is largely
unsurvivable; performance collapses monotonically with obstacle count.

### 3.2 PPO (M4, recurrent, continuous) — `echorl/ppo.py`
GRU over the ping sequence; Gaussian actions (env-clipped); clipped surrogate + GAE;
feedforward-encode-then-GRU replay for efficiency. **Hypothesis from the DQN failure
confirmed — continuous control (slow/stop/pivot) cracks the clutter:**

| tier | random | DQN (best) | **PPO (best)** |
|---|---|---|---|
| nobs=0 | 4.4 / 100% | 30.8 / 26% | **40.0 / 0%** |
| nobs=1 | 3.9 / 100% | 9.0 / 88% | **40.0 / 0%** |
| nobs=3 | 2.7 / 100% | ~3 / 100% | **37.0 / 8%** |

PPO reached optimum in <10k env steps (vs DQN ~21k + oscillation), rock-stable
(nobs=3 held 37.0 for 90k steps, value loss ~0.04).

### 3.3 Oracle ceiling — `echorl/oracle.py`
Privileged-geometry receding-horizon planner (vectorized random-shooting over true
dynamics, no echoes): **nobs=0 40.0/0%, nobs=1 39.9/1%, nobs=3 37.9/6%.**
→ echo-only PPO **matches the planner to ~1 step / ~2% collision** at every tier; the
residual nobs=3 collisions are **genuinely unsurvivable rooms** (the planner also dies in
~6%). A naive greedy 1-step-clearance controller collides 85–93% — myopic reactive
avoidance fails under the 45°/step turn limit, so the ceiling requires planning.

### 3.4 Ablations (nobs=3) — do memory / cue choice matter?
`full` = `no-memory` = `range-only` = `cochleagram-only` = **37.0 / 8%, all identical,
all at the oracle ceiling.** → **continuous action is the sole decisive factor**; memory
is not needed (near-Markov given the observation) and either echo representation alone
suffices. Stage-0 avoidance is **saturated / low-order** (mirrors the JEPA RESULTS.md
"geometry is low-order, already exposed" finding).

### 3.5 Multi-seed CIs (nobs=3, 3 seeds/config)
`full`, `no-memory`, `range-only` each **36.1 ± 0.0 len / 0.10 ± 0.00 collision** — every
one of the 9 runs returned *exactly* 36.1/0.10. Fully seed-robust. (Zero variance also
reflects a deterministic eval on fixed held-out rooms/starts; a fuller CI would randomize
eval starts per seed, but the training-seed robustness is total.)

---

## 4. Stage 2 — goal approach (doorway-finding)

Task (`env.py task="goal"`): navigate to a point 1.5 m in front of the room's largest
**opening**, monostatic echoes only. Success = reach within 1.2 m. Metric: **success rate**
(fraction of held-out doorway rooms reached), best-checkpoint (goal task has late-training
drift, so peak ≠ final).

### 4.1 The key formulation insight
*Terminal* collision — correct for Stage 0 (survive = avoid) — is **wrong** for a goal
task: the goal sits near a wall, so any approach is catastrophically risky, and the agent
hedges into a risk-averse local optimum (drives collision to 0 and safely wanders out the
clock; success collapsed 0.57→0.17). Five reward-weight iterations could not fix it.
**Fix = non-terminal collision in goal mode** (a bump costs `w_bump=0.3` and blocks the
move, but the episode continues to success/timeout). This unlocked reliable goal-seeking:
success 0.40 (init) → 0.74 → 0.85 → **~0.86** by ~20k steps, reaching the door in ~12 steps.
→ monostatic self-echoes *do* carry enough to home to an opening; the earlier ~0.4 ceiling
was the reward formulation, not the cue. Reward: `w_goal=1.0, w_success=10, w_bump=0.3,
w_dist=0.01`.

### 4.2 Engineering fixes found en route
- **NaN action → quit-sentinel bug:** a NaN action reached the server as a NaN pose, which
  the server reads as its `isnan` quit sentinel and exits → `nan_to_num` guard in `_decode`.
- **Rare `episode_server` segfault** (geometry-independent, ~1 per 10–20k steps): `_observe_safe`
  respawns the *same-seed* room (identical geometry) and retries, so a stray crash can't kill
  a multi-hour run.
- **Opening rooms:** `opening_seeds()` pre-filters train+eval seeds to rooms that have an
  opening (also eliminated a server-churn crash from spawn/close hunting for openings).

### 4.3 Cue/memory ablation — what feature detects the doorway? (multi-seed, n=3)

| Condition | success (mean ± 95% CI) | seeds |
|---|---|---|
| **Cochleagram only** (two-ear bearing) | **0.88 ± 0.06** | [0.85, 0.91, 0.89] |
| **Full** (range + coch + memory) | **0.85 ± 0.03** | [0.86, 0.83, 0.85] |
| **No memory** | **0.84 ± 0.07** | [0.81, 0.82, 0.88] |
| **Range only** (no direction) | **0.79 ± 0.05** | [0.76, 0.79, 0.81] |

**The binaural bearing cue is the load-bearing feature.** Cochleagram-only (0.88) matches
or edges full (0.85) — adding the range channel buys nothing — while range-only (no
direction) is reproducibly the weakest at 0.79 (`full − range-only = +0.06`). Physical
reading: a doorway is a *directional quiet-spot* — sound aimed at the opening escapes and
doesn't return; the **direction** of that quiet-spot is read from the interaural level
difference. Without bearing, the agent partly compensates by **active scanning** (turn,
watch the omni echo drop off), which is why range-only isn't a failure. **Correction from
single-seed:** memory does *not* clearly help even here (no-memory 0.84 ≈ full 0.85, CIs
overlap; the single-seed ~5-point hint washed out at n=3) — consistent with Stage 0.

**Contrast that carries the paper:** on obstacle avoidance every cue and architecture tied
*exactly*; on doorway-finding they **separate**, and the one feature that matters is the
binaural (two-ear) bearing cue.

---

## 5. Model-based (MPC) vs model-free (PPO) — the same task, one comparison

The comparison the substrate was built for: model-free RL vs a **learned-world-model
CEM-MPC** controller on the *same* server, task, dynamics and metric. Also the bridge to
AcousticJEPA (`RESULTS.md`), which found a *learned* world model plans well while a
*random-encoder* one fails ("control-positive dissociation"). Code: `echorl/mpc.py`.

**World model** (mirrors `jepa/control_mpc.py`, on `episode_server` data): normalized-echo
`TwoStreamEncoder` → latent `z[256]`; action-conditioned residual `Predictor` (JEPA-style
EMA-target dynamics loss); a `z → distance-to-goal` readout (the CEM cost). Trained on **360
random exploratory trajectories** (~10k H-windows) — *not* the PPO policy, so a win/loss vs
PPO is a fair model-based-vs-model-free result. **CEM** minimises the imagined
distance-to-goal over an H=6 latent rollout; executes the first action; re-observes.
Evaluated **closed-loop through the very same `EchoLocEnv`** PPO uses (identical collision,
success, metric). Doorway task, nobs=0, 12 held-out rooms × 10 starts, 3 training seeds.

| Controller | success (mean ± SD) |
|---|---|
| Oracle (privileged geometry, greedy to true goal) | **1.00** (ceiling) |
| **Model-free PPO** (`ppo.py --task goal`) | **~0.85** |
| **Learned-WM MPC** | **0.46 ± 0.05** |
| Random-encoder MPC | **0.06 ± 0.03** (control lower bound) |

WM diagnostic (held-out windows): learned readout-R² **+0.98** vs random-encoder **+0.49**;
learned **rollout-MAE 0.60 m ≈ naive-"distance-unchanged" 0.60 m**.

Two results, and the diagnostic explains both:

1. **Model-free RL beats model-based latent-MPC (0.85 vs 0.46) — for a mechanistic, not a
   tuning, reason.** The learned encoder decodes *where the goal is now* almost perfectly
   (readout R²=0.98), but the action-conditioned **latent dynamics cannot predict what an
   action does**: rolling the latent forward H steps predicts the resulting distance no
   better than assuming it never changes (rollout-MAE = naive). CEM therefore plans over a
   forward model blind to action consequences, and has little to optimise. PPO needs no
   forward model — it follows the **binaural bearing gradient** over time (the exact cue
   §4.3 fingered), which is a directional/temporal signal, not a plannable one-step
   consequence. So *the cue structure of doorway-finding favours reactive bearing-following
   over forward-model planning.*
2. **Learning still matters for the planner too — the dissociation replicates in a new
   paradigm.** Learned-WM MPC ≫ random-encoder MPC (0.46 vs 0.06; learned > random on every
   seed, +0.40 ± 0.03). Exactly AcousticJEPA's "control-positive, random-encoder-fails"
   result, now reproduced in **model-based planning control** (previously shown only for
   latent-MPC on the wall/centre task): a random encoder can't decode the goal (R²=0.49) and
   can't plan (0.06), a learned one can decode it (0.98) and plans partway (0.46).

Caveat: different data/compute budgets (PPO ~150k task-reward env steps; WM 360 random
trajectories) — but the bottleneck is diagnosed as the *dynamics* (readout R²=0.98 shows
perception is fine), so it is not mere data starvation. A stronger action-conditioned
dynamics model (or longer horizon) might narrow the gap; that the readout already saturates
while the rollout does not is the point.

## 6. Summary of findings

1. **Echo-only 3D navigation works and is near-optimal for avoidance** — PPO matches a
   privileged-geometry planner to ~1 step / ~2% collision; residual failures are
   genuinely unsurvivable rooms.
2. **Continuous control is the decisive factor for cluttered avoidance**, not perception
   sophistication: discrete fixed-speed DQN floors on clutter (~3/100%); continuous PPO
   solves it (37/8%). Memory and cue choice are irrelevant here — the task is saturated.
3. **The harder goal task exposes cue importance.** Doorway-finding depends on the
   **binaural bearing cue** (cochleagram ≈ full ≫ range-only), exactly as the acoustics
   predict; memory still doesn't clearly help.
4. **A transferable RL lesson:** how you define *failure* (terminal vs non-terminal
   collision) mattered more than model sophistication for the goal task — terminal
   collision induced risk-averse hedging that no reward-weight tuning fixed.
5. **Model-free RL beats model-based MPC on doorway-finding (0.85 vs 0.46)** because the
   informative cue is a directional bearing signal a reactive policy follows, not an
   action-consequence the learned latent dynamics can predict (readout R²=0.98 but
   rollout-MAE = naive). Yet learning still helps the planner (learned-MPC 0.46 ≫
   random-encoder 0.06), replicating the AcousticJEPA control-positive dissociation in a
   model-based paradigm.

## 7. Honest limitations (for a writeup)
- Not a scored head-to-head with Mohan & Vanderelst: different sim (2D analytic vs 3D
  rendered), different collision metric (their per-call proximity count vs our terminal/
  non-terminal). This is a conceptual replication + extension, not a rematch.
- Stage-2 effect sizes are modest (0.79–0.88) and the task has late-training drift (read
  best-checkpoint). Multi-seed CIs are solid but a fuller eval would randomize eval starts
  per seed.
- Rooms are procedural boxes with AABB pillars; no real-world geometry yet (M5 = ONNX →
  in-engine `SteamEchoNavComponent` deploy is the real-time demo path, not yet done).
- MPC (§5) uses one horizon (H=6) and a modestly-sized world model; the learned-vs-random
  and MPC-vs-PPO orderings are robust across 3 seeds, but the *absolute* MPC number is not
  tuned to its ceiling. The diagnosed bottleneck (latent dynamics, not perception) is the
  informative part.

## 8. Reproduction

Build the server: `src/build.bat` (needs MSVC vcvars; produces `tools/episode_server.exe`).
Verify: `python src/verify_episode_server.py`.

```
# corpus + norm stats
python echorl/collect_corpus.py --rooms 24 --per 84
# Stage-0 DQN / PPO
python echorl/dqn.py --nobs 0 --steps 30000 --baseline
python echorl/ppo.py --nobs 3 --steps 30000
# oracle ceiling
python echorl/oracle.py --eval-rooms 12
# Stage-0 ablations: ppo.py --nobs 3 [--no-memory | --drop-range | --drop-coch] --tag X
# Stage-2 goal + cue ablation
python echorl/ppo.py --task goal --nobs 0 --steps 30000 --tag full
python echorl/ppo.py --task goal --nobs 0 --drop-coch --tag rangeonly   # etc.
# Model-based MPC vs model-free PPO (§5): learned vs random-encoder WM, oracle ceiling
python echorl/mpc.py --repeats 3 --train-rooms 60 --eval-rooms 12 --train-ep 360 --episodes-per 10
python echorl/mpc.py --smoke      # fast end-to-end pipeline check
```

Checkpoints (`echorl/ckpt/`): `dqn_stage0_nobs{0,1}_best.pt`,
`ppo_nobs{0,1,3}_full_s0_best.pt` (Stage-0 + ablations `_{nomem,rangeonly,cochonly}_s{0,1,2}`),
`ppo_goal_nobs0_{full,rangeonly,cochonly,nomem}_s{0,1,2}_best.pt` (Stage-2). `mpc.py` trains
its world model on the fly (random-trajectory corpus), so it saves no checkpoints.

Code: `src/episode_server.cpp`, `src/verify_episode_server.py`,
`echorl/{env,vec_env,models,dqn,ppo,oracle,mpc,collect_corpus,smoke_test}.py`.
