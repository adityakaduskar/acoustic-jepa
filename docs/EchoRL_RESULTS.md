# EchoRL — Experiments & Results (evidence log)

Closed-loop echolocation RL on the Steam Audio (phonon) renderer: a simulated "bat"
that navigates from **binaural echoes only**, in 3D, with physically-rendered
reverberant acoustics. 3D, rendered-acoustics successor to Mohan & Vanderelst 2020
(*Training a simulated bat*, SSCI). Design doc: `EchoRL_PLAN.md`.

Status as of **2026-07-25**: Stage 0 (obstacle avoidance) and Stage 2 (goal
approach / doorway-finding) both complete, with oracle ceilings, ablations, and
multi-seed CIs; a **model-based (CEM-MPC) vs model-free (PPO)** comparison on the doorway
task (§5), a Δd probe diagnosing the MPC gap (§5.1), an objective-mismatch result where a
near-perfect cost model plans *worse* (§5.2), a **cluttered-room showdown (§5.4)** whose
render exposed an **evaluation bug** that had inflated the model-free numbers — corrected
throughout, producing a clean **model-free/model-based crossover** (reactive PPO wins empty
rooms; it collapses and model-based wins when the task requires planning) — an honest
**negative (§5.5)**: the named on-policy (DAgger) fix does not rescue the planner; and a
**mapping experiment (§5.8)** that implements §5.7's "integrate pings over time" lever
explicitly — a known-pose occupancy map + planner ≈ **doubles** the latent-WM MPC on the
cluttered doorway task (0.49 ± 0.08 vs 0.24) but hits a genuine *information* ceiling (weak,
correlated per-ping geometry: walls mapped, pillars missed, doorway ~2.8 m); **§5.9**, which
breaks part of that ceiling *without touching either estimator's accuracy* — consuming the
binaural bearing **reactively** rather than as a fused coordinate is worth +0.063 ± 0.024, taking
the task to **0.52** against a 0.95 ceiling (its companion clearance-map fix measured nil once
re-run at 3× the episodes, and a claimed interaction between them is **retracted** — see the
correction in §5.9); **§5.10**, active sensing: a head-scan halves the doorway bearing error
(83.9° → 44.6°) and buys no measurable control; and **§5.11**, which builds the 4-receiver array
that was supposed to fix the doorway floor and finds it does not — the front-back ambiguity blamed
since §5.8 turns out to explain a minority of the error. All numbers
are on **held-out rooms** (seed band 600+, disjoint from training seeds 1–60 and the 3000+
corpus band). Trajectory maps for every condition: `results/echorl_{avoid,goal}_nobs{0,3}.png`
(task × clutter; columns = controllers, green = success, red = failure).

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
| nobs=0 | 4.4 / 100% | 30.8 / 26% | **39.4 / 2%** |
| nobs=1 | 3.9 / 100% | 9.0 / 88% | **40.0 / 0%** |
| nobs=3 | 2.7 / 100% | ~3 / 100% | **38.4 / 4%** |

PPO reached optimum in <10k env steps (vs DQN ~21k + oscillation), rock-stable
(nobs=3 held for 90k steps, value loss ~0.04). **Numbers are the corrected fair eval
(per-room macro-average).** The evaluation bug (§5.4) applied here too, but with the
*opposite* sign: on avoidance a **collision** ends an episode early, so the biased eval
*over*-sampled crash-prone rooms and thus **over-stated collisions / under-stated flight
length** — the mirror of the goal-task inflation. Recount: nobs=3 went **37.0 / 8% →
38.4 / 4%** (better, as predicted); nobs=0 40.0/0% → 39.4/2%. Conclusions unchanged.

### 3.3 Oracle ceiling — `echorl/oracle.py`
Privileged-geometry receding-horizon planner (vectorized random-shooting over true
dynamics, no echoes): **nobs=0 40.0/0%, nobs=1 39.9/1%, nobs=3 37.9/6%.**
→ echo-only PPO **matches the planner to ~1 step / ~2% collision** at every tier; the
residual nobs=3 collisions are **genuinely unsurvivable rooms** (the planner also dies in
~6%). A naive greedy 1-step-clearance controller collides 85–93% — myopic reactive
avoidance fails under the 45°/step turn limit, so the ceiling requires planning.

### 3.4 Ablations (nobs=3) — do memory / cue choice matter?
`full` = `no-memory` = `range-only` = `cochleagram-only` = **38.4 / 4% (fair eval),
all identical (±0.0 len over 3 seeds), all at the oracle ceiling.** → **continuous action
is the sole decisive factor**; memory is not needed (near-Markov given the observation) and
either echo representation alone suffices. Stage-0 avoidance is **saturated / low-order**
(mirrors the JEPA RESULTS.md "geometry is low-order, already exposed" finding). The tie is
exact under fair eval too, so the finding is robust to the §5.4 correction.

### 3.4.1 ⚠ Caveat: PPO "solves" avoidance by STOPPING (found via the render)
The trajectory maps (`results/echorl_avoid_nobs{0,3}.png`) show PPO's paths as tiny stubs —
the agent **hardly moves**. This is faithful, not a render bug: tracing the emitted actions,
PPO's mean forward step is **0.03 m (nobs=0) / 0.004 m (nobs=3)** out of a 1.5 m max — it
drifts for a few steps, finds open space, then **parks and only rotates** for the rest of the
episode. The reward explains it: per step ≈ `alive 0.05 + clearance≤0.15 + 0.02·dstep`, while
a crash is −5 and terminal — so *stop in open space* dominates (guaranteed ~0.20/step, zero
risk) and PPO converges to it. "Flight length" counts *steps survived*, and a parked agent
survives all 40, so **38.4/40 = "it stopped and nothing hit it."**

Consequences for how to read §3:
- It **deepens the saturation explanation**: the ablations tie not because the cues are
  redundant but because **the winning policy barely uses perception at all** ("stop" is
  nearly a constant action).
- It **sharpens the DQN-vs-PPO story**: the decisive advantage of continuous control is the
  ability to **brake**, not skilled weaving — the fixed-speed DQN (0.75 m every step) *cannot
  stop*, so in clutter it is forced to keep moving and eventually collides. True, but humbler
  than "PPO flies better."
- It is a **limitation of the avoidance task design** (the reward is winnable by not moving),
  not a property of echo-navigation. A task that forces sustained travel (patrol / coverage,
  or a hard per-step move requirement) would be needed to test *skilled* echo-guided flight.
- The **goal task has no such degeneracy** (stopping fails; the agent must travel to the
  doorway), which is why the cues there actually separate (§4.3) — PPO's goal-task forward
  step is ~1.39 m, i.e. it moves purposefully.

### 3.5 Multi-seed CIs (nobs=3, 3 seeds/config)
`full`, `no-memory`, `range-only` each **36.1 ± 0.0 len / 0.10 ± 0.00 collision** — every
one of the 9 runs returned *exactly* 36.1/0.10. Fully seed-robust. (These are the *old
biased-eval* numbers; the fair recount is 38.4/0.04, §3.2/§3.4. Zero variance also reflects a
deterministic eval on fixed held-out rooms/starts.)

### 3.6 Redesigned avoidance: the COVERAGE task (kills the parking exploit)
§3.4.1 showed the survive-only reward is winnable by stopping. The **explore** task
(`env.py task="explore"`) fixes this: the room is a 1 m coverage grid, and reward = **+1 per
newly-visited cell − 0.03 per step (time cost) + small clearance term**, with **terminal
collision** (−5). Parking now earns a *strictly negative* return (only the time cost), so the
agent must **keep moving into new ground while not crashing** — movement is forced and
perception is needed to route. Metric: **cells covered** (macro-avg over held-out rooms) in a
60-step episode. Best-checkpoint by coverage.

Empty rooms (nobs=0), PPO training curve (`results/echorl_explore_nobs0.png`):
- Init (random) covers 3.8 cells, crashes 100%.
- It **first relapses to parking** (cov 1.4–3.0, survives 60/60) — the §3.4.1 attractor is
  real and found first.
- Then the coverage reward pulls it out: cov climbs to **15.4** at 4% collision. The agent
  now **sweeps the room** (the render is unambiguous: old avoid-PPO = tiny stubs; new
  coverage-PPO = big space-filling loops), covering ~15× more ground than parking.

Cluttered rooms (nobs=3, `results/echorl_explore_nobs3.png`): best coverage **6.5 cells @
~20% collision** (vs 15 @ 4% empty). The task is now **genuinely hard and non-degenerate** —
there is a real **cover-vs-crash tradeoff** (higher coverage buys higher collision, and the
agent intermittently retreats toward a safer low-coverage mode), and the render shows it
**weaving its sweep around the pillars** — routing coverage around obstacles it senses only
acoustically. This is the *skilled* echo-guided flight the survive-only task could not
elicit.

So the redesign works in both tiers: the winning policy is now sustained flight that must use
perception (route around clutter), not stopping. Headline: `results/echorl_explore_nobs{0,3}.png`
put the old parked agent and the new sweeping agent side-by-side on the same rooms — the
behavioural change is unmistakable.

### 3.6.1 Cue/memory ablation on the coverage task — the cues finally SEPARATE
The payoff. On survive-only avoidance every ablation tied *exactly* (§3.4, 38.4/0.04) because
the winning policy — stop — needed no perception. Re-running the same four configs on the
**coverage** task (nobs=3, `explore`; clean eval = 12 held-out rooms × 20 eps; **3 training
seeds**; `echorl/explore_ablation_eval.py`):

| config | coverage (mean ± SD) | per-seed | collision | drop from full |
|---|---|---|---|---|
| **full** (range + coch + memory) | **5.9 ± 1.2** | [6.8, 4.5, 6.3] | 0.25 | — |
| coch-only (binaural bearing) | 4.5 ± 1.2 | [4.5, 3.3, 5.7] | 0.16 | −1.3 |
| no-memory | 4.3 ± 1.2 | [3.8, 3.6, 5.7] | 0.15 | −1.5 |
| **range-only** (no bearing) | **1.7 ± 0.3** | [1.8, 1.4, 1.9] | 0.06 | **−4.2** |

**A Δ4.2-cell spread** (vs Δ0 on survive-only), robust across 3 seeds. What multi-seed shows:
- **Binaural bearing is decisively the dominant cue.** Dropping it (range-only) collapses
  coverage to **1.7 ± 0.3** — a −4.2 hit, and the *only* tight config (±0.3): on every seed
  the bearing-less agent gives up sweeping and **reverts to parking** (coverage ~1.7, collision
  back down to 0.06). This is airtight — no CI overlap with anything.
- **Memory and range are a TIE at the second tier.** full − no-memory = −1.5 ≈ full −
  range-only(coch dropped… i.e. keep-range) — more precisely `drop-range −1.3 ≈ drop-memory
  −1.5`, with coch-only (4.5 ± 1.2) and no-memory (4.3 ± 1.2) **fully overlapping**. The
  single-seed hint that "memory > range" (−2.9 vs −1.9) **washed out** — at n=3 they each
  contribute a small, roughly-equal amount and are statistically indistinguishable.
- The three "engaged" configs (full / coch-only / no-memory) all carry ±1.2 variance because
  they trade coverage for collision risk (a seed-variable balance); only the parked
  range-only config is low-variance.

**Importance ordering (3-seed): binaural bearing ≫ (memory ≈ range).** Two conclusions:
1. The coverage redesign **un-saturates the avoidance task** — perception now clearly matters
   (Δ4.2 vs Δ0), exactly as intended.
2. The **binaural bearing cue is the load-bearing feature for *both* navigation tasks** —
   doorway-finding (§4.3) *and* coverage — a consistent, now multi-seed-confirmed through-line:
   two-ear direction is what echo-guided navigation runs on; range/loudness alone is weak
   (it can't even sustain a sweep). Memory helps modestly on coverage — the one task where it
   logically should — but no more than range does.

### 3.6.2 The bearing cue is CLUTTER-SPECIFIC — a clean monotonic gradient (nobs 0/1/3)
Running the ablation at **three** clutter levels (all **3 seeds**, clean 20-ep eval) turns the
empty-vs-cluttered contrast into a smooth gradient. The load-bearing quantity is **drop-bearing**
(full → range-only) and whether it triggers *collapse to parking*:

| clutter | full | range-only (bearing removed) | **drop-bearing** | range-only ÷ full | collapse? |
|---|---|---|---|---|---|
| nobs=0 (empty) | 14.0 ± 3.1 | 11.2 ± 1.3 | **−2.8** | 80 % | none (seeds 12.7/10.2/10.5) |
| nobs=1 (1 pillar) | 10.5 ± 3.2 | 6.7 ± 3.7 | **−3.8** | 64 % | partial (seeds 10.9/5.5/3.7) |
| nobs=3 (3 pillars) | 5.9 ± 1.2 | 1.7 ± 0.3 | **−4.2** | 29 % | full → parking (1.8/1.4/1.9) |

**Bearing's importance rises monotonically with clutter** (−2.8 → −3.8 → −4.2), and the
*collapse to parking* sharpens with it: with bearing removed the agent covers 80 % of full in
empty rooms, 64 % with one pillar, and just 29 % (parked) with three. The per-seed collapse
counts tell the same story as a probability — 0/3 seeds collapse in empty, ~2/3 at one pillar,
3/3 at three. So it is not a two-state on/off effect but a graded dependence: **the more
obstacles there are to route around, the more the binaural bearing cue is doing the work.**
(nobs=1 needed all 3 seeds — its single-seed run was uninterpretable noise, with `full` drawing
a parking-prone policy and landing *below* every ablation; coverage training variance is ±3
cells, so single seeds mislead on this task, as they did at nobs=0.)

**What multi-seed corrected** (vs the earlier single-seed nobs=0 read): the single seed had
range looking important in empty (coch-only 11.7, −3.1) and bearing looking least important
(−2.3). Both were draws — at 3 seeds, **range barely matters in empty** (coch-only 13.6 ≈ full
14.0, drop-range only −0.3) and bearing is the *second*-biggest factor (−2.8). Corrected
empty-room order: **memory (−3.4) ≈ bearing (−2.8) ≫ range (−0.3)**. A new wrinkle also
appears: **memory matters *more* in empty (−3.4) than in clutter (−1.5)** — sensible, since in
an open room efficient coverage is mostly a book-keeping problem (don't re-visit), whereas in
clutter bearing-driven obstacle-routing dominates and memory is secondary.

**Unifying reading (holds under multi-seed):** the binaural bearing cue matters — and becomes
*critical* — exactly when the task requires localizing a **direction to a spatial feature**:
the acoustic void of a doorway (§4.3), or obstacles to route around in clutter (§3.6.1).
Undirected sweeping of an *open* room does not need it (range is near-useless there; memory
does the work). "Bearing is load-bearing" is therefore **feature/clutter-specific**, not a
general property of navigation — and it scales *smoothly* with how much clutter there is to
route around (the nobs 0/1/3 gradient above). (Within-condition CIs are wide — ±1–4 cells,
coverage training is noisy — so rankings of the *second-tier* cues (memory vs range) overlap;
the solid, seed-robust claim is the monotonic drop-bearing gradient −2.8 → −3.8 → −4.2 and the
graded collapse 0/3 → ~2/3 → 3/3 seeds.)

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
success 0.40 (init) → 0.74 → 0.85 → **~0.86** by ~20k steps, reaching the door in ~12 steps
(these are the *biased* training-time evals; the fair per-room success is **~0.66**, §4.3/§5.4).
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

Numbers below are the **corrected fair eval** (per-room macro-average; see the §5.4 box for
why the earlier `evaluate` inflated these). The conclusion is **unchanged and sharper**.

| Condition | fair success (mean ± SD) | seeds | (old biased) |
|---|---|---|---|
| **Cochleagram only** (two-ear bearing) | **0.68 ± 0.03** | [0.67, 0.72, 0.67] | 0.88 |
| **Full** (range + coch + memory) | **0.66 ± 0.03** | [0.67, 0.69, 0.63] | 0.85 |
| **No memory** | **0.58 ± 0.10** | [0.49, 0.56, 0.68] | 0.84 |
| **Range only** (no direction) | **0.48 ± 0.05** | [0.43, 0.53, 0.49] | 0.79 |

**The binaural bearing cue is the load-bearing feature.** Cochleagram-only (0.68) matches
or edges full (0.66) — adding the range channel buys nothing — while range-only (no
direction) is reproducibly the weakest at 0.48 (`full − range-only = +0.18`, a *larger* gap
than the biased eval showed). Physical reading: a doorway is a *directional quiet-spot* —
sound aimed at the opening escapes and doesn't return; the **direction** of that quiet-spot
is read from the interaural level difference. Without bearing, the agent partly compensates
by **active scanning** (turn, watch the omni echo drop off), which is why range-only isn't a
total failure. Under fair eval **memory now shows a modest positive effect** (no-memory 0.58
< full 0.66, `+0.08`), reversing the earlier biased-eval read that memory didn't help
(still noisy at ±0.10 — suggestive, not conclusive).

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
| **Model-free PPO** (`ppo.py --task goal`) | **0.68** (fair eval; §5.4) |
| **Learned-WM MPC** | **0.46 ± 0.05** |
| Random-encoder MPC | **0.06 ± 0.03** (control lower bound) |

WM diagnostic (held-out windows): learned readout-R² **+0.98** vs random-encoder **+0.49**;
learned **rollout-MAE 0.60 m ≈ naive-"distance-unchanged" 0.60 m**.

Two results, and the diagnostic explains both:

1. **Model-free RL beats model-based latent-MPC (0.68 vs 0.46) — for a mechanistic, not a
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

### 5.1 Experiment A — the Δd probe: *why* the dynamics is blind (H0/H1/H2)

`echorl/probe_dd.py` isolates the cause of §5's "rollout = naive" result. Identical
small-MLP probes predict the one-step **Δd = d(t+1)−d(t)** (mean −0.01, sd 0.29 m over
12,171 held-out step-pairs) from different inputs; the WM's own predictor path is shown for
reference. Held-out R² (best over epochs, 3 probe seeds):

| probe input | R² | reads as |
|---|---|---|
| `raw ping + a` (H0 test) | **+0.43 ± 0.01** | the info **is** in one ping+action — but only ~43% determined (myopic cue) |
| `z + a` (trained latent + action) | **+0.42 ± 0.01** | encoder keeps ~all of it — **not** discarded (rejects H1) |
| `z` only (control) | +0.25 | state alone predicts typical Δd under the random policy |
| `a` only (control) | +0.01 | action alone is useless — Δd is a state×action interaction |
| **WM predictor path** `readout(pred(z,a))−readout(z)` | **+0.24** | **what CEM actually plans on — barely above z-only, the fixable defect** |

**Verdict: H2, with a partial-H0 ceiling.** The action-consequence information survives into
the latent (`z+a` = 0.42 ≈ raw ceiling 0.43), so neither the ping nor the encoder is the
problem. But the trained Predictor+readout path delivers only **0.24** — *below* even the
state-only 0.25, i.e. its action-conditioning contributes **nothing** to the goal-distance
estimate CEM optimises. Likely cause: the latent-MSE dynamics loss is spread over all 256
dims (the goal-relevant subspace is a tiny fraction), and `readout` was trained on *encoded*
latents, so it generalises poorly to *predicted* ones. Two stacked limits result:
- **accuracy-fixable (H2):** the predictor path (0.24) sits far below the achievable
  single-ping consequence R² (0.42) — a *direct* consequence head recovers the *accuracy*
  (Exp C hits R²=0.96 on the planning cost). **But recovering accuracy did NOT recover
  control — it made it worse (§5.2).** So accuracy was never the true bottleneck;
- **fundamental (partial H0):** even the single-ping ceiling is 0.42, so open-loop planning
  over one ping is intrinsically myopic — which is why model-free reactive control (many
  pings, active scanning) has the structural edge on this cue, and why the MPC gets 0.46
  only because it *re-plans closed-loop* each step rather than trusting a long rollout.

### 5.2 Experiment C — the informed rematch, and why "fix the model" backfired

A hovers between "the dynamics is inaccurate" (H2) and "the cue is myopic" (partial H0).
Exp C tests the H2 reading head-on. The env is **step-then-turn** (translation uses the
current yaw, `dyaw` only pays off later), so a 1-step head can't value turning; the fix is a
**multi-step outcome head** `g_H(z0, a₁..H) → Σ_{k} d(t+k)` — one network mapping the initial
latent + whole action sequence directly to the **cumulative distance CEM minimises**, trained
on the frozen learned encoder. This swaps *only* the cost estimator (compositional latent
rollout → direct outcome head); same encoder, corpus, CEM, and cost functional.
Code: `echorl/mpc_direct.py`. 12 held-out rooms × 10 starts, 3 head-training seeds:

| Controller | success | cost-model quality |
|---|---|---|
| Oracle (privileged) | 1.00 | — |
| Model-free PPO | 0.68 (fair; §5.4) | — |
| **Learned-WM MPC** (compositional rollout, §5) | **0.51** | rollout ≈ naive (R²≈0.24, §5.1) |
| **Direct-outcome MPC** (the "fix") | **0.29 ± 0.02** | predicts the cost at **R² 0.96 ± 0.01** |

**The near-perfect cost model is the worse controller (0.29 vs 0.51).** With everything else
held fixed, quadrupling+ the cost-model accuracy *halved* success. This is **objective
mismatch** (model accuracy ≠ control performance) demonstrated by a clean A/B: CEM does not
query the training distribution — it *optimises against the model*, proposing goal-directed
action sequences the random-exploration corpus never contained. The high-capacity direct head
is accurate on-distribution (0.96) but has sharp, exploitable minima off-distribution, so CEM
finds sequences it wrongly believes minimise distance. The compositional rollout, despite far
worse nominal accuracy, is a smoother, more constrained function (residual latent steps + a
shared readout) whose errors are *less exploitable* — so it plans better.

**This reframes §5 and §5.1.** The learned-WM MPC does not trail PPO because its dynamics is
inaccurate — making the dynamics accurate makes control *worse*. The real bottleneck is the
**planner–model interaction under distribution shift**, which points to on-policy data
aggregation (retrain the outcome model on the MPC's own optimised trajectories) or
uncertainty-penalised / trust-region planning — *not* to naive accuracy maximisation. Caveat:
this refutes the simplest fix, not MPC in general; the on-policy variant is the indicated
next step and is untested here.

### 5.3 Experiment B (gate) — when does the task *need* planning?

§5–§5.2 are all on nobs=0 (empty doorway rooms), where a greedy head-to-goal controller is
optimal — so model-free's win there is partly structural. `echorl/gate_b.py` checks, on
**privileged geometry (no echoes)**, whether cluttered rooms demand lookahead: greedy
head-to-goal vs a receding-horizon planner, both with the env's exact goal-mode rules
(non-terminal collision blocks translation). 12 rooms × 10 starts:

| tier | greedy oracle | planning oracle | gap |
|---|---|---|---|
| nobs=0 | 1.00 | 1.00 | +0.00 → **reactive-sufficient** (confirms §5's task needs no planning) |
| nobs=3 | **0.57** | **1.00** | **+0.43 → NEEDS PLANNING** (pillars block the doorway route) |

So the cluttered doorway task is genuinely discriminative: reactive control caps at 0.57
while planning reaches 1.00 — the regime where model-based control *should* earn its keep.

### 5.4 Experiment B (full) — the cluttered showdown, and a fixed evaluation bug

Running all controllers on **nobs=3 goal** (12 held-out doorway rooms × 10 starts) forced
a **methodology correction** that changes several earlier numbers (see box). All figures
below use the **fixed, per-room-macro-averaged** eval.

| controller (nobs=3 goal, fair eval) | success |
|---|---|
| planning oracle (privileged geometry) | **1.00** |
| greedy oracle (no obstacle avoidance) | **0.54** |
| learned-WM MPC (compositional rollout) | **0.25** |
| direct-outcome MPC | **0.24 ± 0.04** |
| **model-free PPO** | **0.08–0.18** |

**The result reverses the naive expectation.** When the task genuinely requires planning,
model-free PPO **collapses** — it essentially solves only the one trivial room whose goal is
~1 step from the start, and fails the rest (the render, `results/echorl_goal_nobs3.png`,
shows it spinning in place / looping without reaching). It is the **worst** method here,
below both MPC variants and even the greedy oracle. Model-based control, though far from the
privileged planning ceiling (1.00), does *better* than model-free in the regime that needs
lookahead — the mirror image of the nobs=0 result (§5), where reactive PPO won.
**Checkpoint-selection control (added):** retraining PPO with the *fixed* eval driving
best-checkpoint selection (`tag=showdown2`) gives a best checkpoint of **0.18** (early
training) that then **collapses to 0.08** and stays there (value loss → 0.02, a degenerate
low-entropy policy). So the generous best-checkpoint number is 0.18 and the converged number
is 0.08 — either way worst-tier, below the MPCs' 0.24–0.25. My earlier caveat that "0.08 may
under-sell a cleanly-retrained PPO" is thus **largely refuted**: fair selection recovers only
to 0.18, and the policy does not hold even that.

> **⚠ Evaluation bug found & fixed (affects earlier PPO numbers).** `ppo.evaluate` ran one
> env per room with autoreset and counted a fixed *total* of episodes. Because reaching the
> goal (or colliding) *ends an episode early*, rooms solved in ~1 step autoreset and flooded
> the sample while rooms that time out were barely counted → **goal success was inflated**.
> Corrected to a per-room macro-average (fixed episodes each). Impact (nobs=0 goal, seed 0):
> **PPO 0.86 → 0.68**; nobs=3 goal: **0.82 → 0.08**. The **MPC and oracle numbers are
> unaffected** (they always used fixed-episodes-per-room), so §5/§5.1/§5.2 conclusions hold
> with PPO's number lowered from ~0.85 to **0.68** (still > learned-WM MPC 0.51 > direct
> 0.29 at nobs=0). §4.3's cue-ablation ordering is **unchanged and sharper** under fair eval
> (below). Stage-0 avoidance used the same eval but the bias runs the *other* way there
> (collisions end episodes early → collision rate over-, flight-length under-estimated), so
> PPO avoidance is *better* than the old numbers: recount done (§3), nobs=3 **37.0/8% →
> 38.4/4%**, cue-ablation tie still exact — conclusions unchanged.

Corrected **§4.3 Stage-2 cue ablation** (fair eval, nobs=0, n=3 seeds):

| condition | fair success | (old biased) |
|---|---|---|
| cochleagram-only (bearing) | **0.68 ± 0.03** | 0.88 |
| full | **0.66 ± 0.03** | 0.85 |
| no-memory | **0.58 ± 0.10** | 0.84 |
| range-only (no bearing) | **0.48 ± 0.05** | 0.79 |

The **binaural-bearing conclusion is stronger, not weaker**: coch-only ≈ full ≫ range-only,
now with `full − range-only = +0.18` (was +0.06). And memory now shows a modest positive
effect (`full − no-memory = +0.08`, was ~0), overturning the earlier "memory doesn't help"
read for the goal task (still noisy at ±0.10).

### 5.5 Experiment D — does on-policy data fix the objective mismatch? (it doesn't)

§5.2 diagnosed the direct-outcome MPC's failure as the planner exploiting its model
*off the random-training distribution*, and named the indicated fix: **DAgger** — let the
model see the deliberate, goal-directed trajectories the planner actually produces. Round 0
trains the outcome head on the random corpus; each later round runs the current MPC, logs its
own trajectories, adds them to the training set, and retrains the head (frozen encoder
throughout). Code: `echorl/mpc_dagger.py`. nobs=0, 12 held-out rooms × 10 starts, **3 seeds**:

| seed | round 0 | 1 | 2 | 3 | 4 | net |
|---|---|---|---|---|---|---|
| 0 | 0.28 | 0.22 | 0.17 | 0.19 | 0.42 | +0.14 |
| 1 | 0.30 | 0.23 | 0.39 | 0.23 | 0.33 | +0.03 |
| 2 | 0.32 | 0.29 | 0.38 | 0.29 | 0.33 | +0.01 |

**Honest negative: on-policy aggregation does *not* reliably rescue the planner.** Round-0
mean 0.30 ± 0.02 → final 0.36 ± 0.05; the success bounces in 0.17–0.42 with **no monotonic
trend** and no consistent per-round pattern (the +0.14 in seed 0 was a lucky final draw —
seeds 1–2 show ~0). The head's held-out accuracy stays high (R² 0.91–0.96) throughout, so the
model isn't degrading; adding its own data just doesn't move control. The direct-outcome MPC
**stays stuck ~0.30–0.36, still below the compositional learned-WM MPC's 0.51** (§5.2).

So the §5.2 story sharpens: the failure is **not** a pure distribution-shift problem that
cheap on-policy data cures. It is the **planner–model interaction hitting the myopic-cue
ceiling** (§5.1: single-ping Δd caps at R²≈0.43) — CEM keeps finding actions the head scores
well but that don't reach, and better-covered data doesn't remove that exploit. The practical
lesson stands but is stronger: on this echo-navigation task you cannot cheaply plan your way
past a reactive policy; the fixes that would matter are a less-exploitable planner
(trust-region / uncertainty-penalised CEM) or a fundamentally less myopic observation, not
more data. (Refuting one's own proposed fix, across seeds, is the point of running it.)

### 5.6 Why the cluttered MPC is stuck at ~0.24 — and why a clearance cost doesn't fix it
The §5.4 MPC (0.24–0.25 on nobs=3 goal) sits **below even the privileged *greedy* oracle (0.54)**,
so two problems stack: **(A)** the cost is *distance-to-goal only* — no obstacle term, so it
drives at the goal and wedges behind pillars, inheriting the greedy objective's 0.54 ceiling;
and **(B)** weak dynamics + objective mismatch degrade it further, 0.54 → 0.24.

Attacking **A** directly (`echorl/mpc_clearance.py`): add an obstacle-**clearance** readout to
the world model and a clearance penalty to the CEM cost —
`Σ dist + w_pen·Σ relu(d_safe − clearance)` — so the planner can *see* pillars and route around
them. The clearance readout trains cleanly (**R²=0.95** on real observations — the encoder
*does* represent obstacle proximity). But the sweep (one WM, cost-only change, 12 rooms × 10):

| w_pen | 0 (= §5 MPC) | 2 | 5 | 10 |
|---|---|---|---|---|
| success | 0.22 | 0.19 | 0.17 | 0.20 |

**It doesn't help — it monotonically *hurts* (0→2→5).** Not a tuning miss (that would give an
inverted-U); the clearance term is an *actively misleading* signal. The reason is diagnostic:
the readout is accurate on **real** observations (0.95), but CEM evaluates it on **predicted
(rolled-out)** latents, where the weak dynamics (§5.1's predictor path) corrupts it — so more
weight on a corrupted signal costs more. **Problem A cannot be fixed independently of Problem
B**: cost-shaping is useless when the planner can only access the cost through a broken forward
model. This redirects to the real bottleneck — a **less-exploitable / uncertainty-penalised
planner** (ensemble-MPC that distrusts high-variance predictions), the same fix §5.2/§5.5 point
to. (Single-seed sweep; the monotone-hurt trend and the R²-real-vs-rollout gap are the signal.)

### 5.7 Uncertainty-penalised ensemble MPC also fails — and *why* closes the case
The fix §5.2/§5.5/§5.6 all pointed to: an **ensemble** of K=5 latent dynamics predictors
(shared encoder + readout, bootstrap-trained), with CEM penalising action sequences the
ensemble *disagrees* on — `mean_k Σdist + λ·std_k(Σdist)` — so the planner avoids the
off-distribution regions where the model is confidently wrong. `echorl/mpc_ensemble.py`, one
ensemble swept over λ (cost-only). nobs=3, 12 rooms × 10:

| λ | 0 (averaged model) | 1 | 3 | 6 |
|---|---|---|---|---|
| success | 0.16 | 0.14 | 0.10 | 0.09 |

**Also a monotone-hurt negative**, and even λ=0 (0.16) is *below* the §5 single model (0.25) —
averaging the ensemble made the forward model *worse* for planning. The diagnostic says why:
ensemble **disagreement is 0.13 m but rollout error is 0.54 m** — the members are wrong in a
*correlated* way (shared encoder → they fail together), so disagreement underestimates true
error ~4× and, worse, the disagreement that exists tracks the *goal-directed* maneuvers (which
are off-distribution) — so penalising it steers *away* from the plans that reach the goal.

**Synthesis — the MPC ceiling is perception-limited, not planner-limited.** Four independent
fixes now fail for one reason:

| fix (§) | idea | result | why it fails |
|---|---|---|---|
| accurate direct head (5.2) | make the cost model accurate (R²=0.96) | 0.29 < 0.51 | CEM exploits it off-distribution |
| DAgger (5.5) | aggregate the planner's own data | 0.30→0.36, no trend | doesn't remove the exploit |
| clearance cost (5.6) | add obstacle term to the cost | monotone hurt | evaluated through the broken rollout |
| ensemble uncertainty (5.7) | distrust high-variance predictions | monotone hurt | correlated errors → disagreement ≪ error |

They all route through the same **latent-dynamics rollout**, whose ceiling is set by §5.1's
finding: a *single ping* determines the action-consequence at only **R²≈0.43** (partial-H0).
No cost, data, or uncertainty trick fixes a rollout that is myopic at the source. The genuinely
different lever is a **less myopic observation** — a *temporal/recurrent* world model that
integrates pings over time to build state before predicting, rather than planning from one
ping. That is a perception redesign, not a planner tweak, and it is the honest next direction
for model-based control here. Model-free PPO sidesteps all of this by never rolling a model
forward — which is exactly why it wins when reaction suffices, and why *both* paradigms remain
far below the privileged planning oracle (1.00) that has ground-truth geometry.

### 5.8 The mapping experiment — testing §5.7's "integrate over time" lever explicitly
§5.7 concluded the cluttered-doorway MPC ceiling is **perception-limited, not planner-limited**
(a single ping fixes action-consequence at only R²≈0.43), and named the honest next lever: *a
temporal model that integrates pings over time to build state before planning*. This experiment
implements that lever in its most explicit, interpretable form — build an **occupancy MAP** from
the echo sequence, then run the already-known-good planner on it — and decomposes the result.
Code: `echorl/{mapping,mapper_collect,mapper,mapper_probe,mapper_fuse,mapper_nav,mapper_render}.py`.
Fixed room-centred grid (48×48, 0.25 m cells). **Scope caveat: KNOWN-POSE mapping** — the agent
knows its own (x,z,yaw) and infers only *geometry* from echoes, isolating the echo→map question
from localization. This is a new **privileged-pose, learned-geometry** rung, more privileged than
PPO/MPC (proprioception only) but far less than the full oracle (true geometry); dead-reckoned /
SLAM pose is the follow-up rung. All numbers on held-out doorway rooms (seeds 600+), per-room
macro-average (same protocol as §5.4).

**Planning side de-risked (M0).** A grid version of the gate_b planner, run on the *true*
rasterized map, scores **0.96** (nobs=3) — reproducing the privileged planning oracle (1.00) up to
grid discretization. So **0.96 is the "perfect-map" ceiling**; any shortfall of the learned
controller is *mapping* error, cleanly isolated.

**The per-ping signal is weak in clutter (`mapper_probe.py`).** A fresh encoder decoding geometry
from a *single* observation (+ yaw), held-out nobs=3: dist-to-goal **R²=0.25** (vs §5's 0.98 — but
that was the *empty*-room readout; interior pillars scatter the echoes), ego-bearing-to-goal
**37° MAE**, dist-to-pillar R²=0.52, ego-bearing-to-pillar 43°. A single cluttered-room ping
barely sees the geometry — the concrete mechanism behind §5.1's low ceiling.

**Known-pose fusion helps but SATURATES (`mapper_fuse.py`).** Because the poses are known, many
weak pings can be fused: a per-ping net predicts an ego (dist, bearing)-to-doorway with a learned
heteroscedastic confidence; each is transformed to a world point via the pose; the confidence-
weighted mean is a multilateration of the doorway. Held-out doorway-localization error falls
3.7 m (1 ping) → **2.8 m (≈20 pings), then flat** — extra pings stop helping. The per-ping errors
are **correlated across poses** (the same encoder mis-reads a given room consistently; likely the
binaural front-back bearing ambiguity a 2-mic array cannot resolve from one ping), so averaging
cannot drive them out. 2.8 m is still >2× the 1.2 m goal radius. (The coordinate transform was
unit-tested to 1e-16, so 2.8 m is a perception result, not a bug.)

**Occupancy: walls yes, pillars no (`mapper.py`, scatter-CNN).** A scatter-and-refine mapper
writes per-ping encoder features into the canvas at known poses and a U-Net refines them.
Held-out: **solid-IoU 0.78** (room extent, saturated from just 2 pings) but **pillar-IoU ≈ 0.005**
— the interior pillars are essentially never localized (the weak per-ping pillar signal does not
fuse into precise cells). The render (`results/echorl_mapping_nobs3.png`) shows it: a fuzzy but
correctly-sized room rectangle, no interior obstacles, doorway estimate within ~1–3 m.

**Closed-loop navigation — the headline (`mapper_nav.py`).** Each step the known-pose agent pings
the true server, re-estimates the map + doorway, and plans on them; physics use true geometry. The
value of *closed* loop is re-fusion from new vantage points, which recovers a lot the 2.8 m static
number would not predict. nobs=3, 12 rooms × 8, **3 model seeds**:

| controller (nobs=3 goal, held-out) | success |
|---|---|
| planning oracle (true geom + pose) | 1.00 |
| grid-planner on TRUE map (M0, perfect-map ceiling) | 0.96 |
| learned occ + **true** goal (obstacle-mapping only) | ~0.67 |
| **true** occ + fused goal (doorway-mapping only) | ~0.60 |
| greedy oracle (true geom, no avoidance) | 0.54 |
| **FULL learned map (occ + fused goal)** | **0.49 ± 0.08** (0.54/0.40/0.53) |
| learned-WM MPC (§5.4) | 0.24–0.25 |
| model-free PPO (§5.4) | 0.08–0.18 |

**Result: explicit known-pose mapping + planning ≈ doubles the latent-WM MPC (0.49 vs 0.24) and is
3–6× model-free PPO (0.08–0.18), matching the privileged *greedy* oracle (0.54).** §5.7's lever
*works* — but only in explicit-map form (the latent WM could not integrate over time; an explicit
metric map with known poses can), and only **partway**: the map stays coarse (walls recovered,
pillars missed, doorway ~2.8 m), so the controller lands at ~half the perfect-map ceiling (0.96).
The decomposition shows **both halves are imperfect and errors compound**: each half with the other
made true reaches only ~0.60–0.67, and together ~0.49. Doorway-localization (~0.60) is the slightly
*more* limiting half than obstacle-mapping (~0.67) — consistent with §4.3 (the binaural bearing to
the doorway is the hard, load-bearing cue). Caveats: single corpus, 3 model seeds; eval carries
~±0.04 render-jitter noise; per-room success ranges [0.0, 1.0] (room difficulty varies).

**The advantage survives removing the pose privilege (`mapper_nav_dr.py`).** The headline above is
*known-pose* — a rung above PPO/MPC, which get proprioception only. To make the comparison fair, the
agent instead **dead-reckons** its pose by integrating its own commanded actions and builds/fuses the
map in that estimated frame (echoes are rendered at the true pose but *attributed* to the estimated
one, so drift corrupts the map exactly like real odometry error). In this deterministic sim, dead
reckoning is exact except when a non-terminal collision blocks a commanded move — and the doorway
sits near a wall, so bumps (hence drift) concentrate right where precision matters. 3 model seeds,
nobs=3, 12 rooms × 8:

| pose source | success | vs known-pose |
|---|---|---|
| known pose (headline) | **0.49 ± 0.08** | — |
| dead-reckoned, bump-aware (robot feels a blocked move) | **0.41 ± 0.03** | −0.08 |
| dead-reckoned, pure (no bump sensing, hardest) | **0.37 ± 0.06** | −0.12 |

So the pose privilege was worth only ~0.08–0.12, and **even fully unprivileged (proprioception only,
the *same* information PPO/MPC get), explicit mapping (~0.37–0.41) still clearly beats the latent-WM
MPC (0.24) and is ~2–5× PPO (0.08–0.18).** Bump-aware DR recovers a third of the loss, confirming
bump-induced drift is the degradation mechanism (as predicted from the wall-adjacent doorway). §5.8's
conclusion is therefore robust to its main caveat: on the cluttered doorway task, an explicit metric
map + planner beats both the latent world model and model-free RL, privileged pose or not.

**A recurrent doorway estimator improves the STATIC metric but makes CONTROL worse — the
static≠control dissociation, a third time (`mapper_seq.py`).** The 2.8 m fusion saturation looked
like binaural front-back ambiguity (a 2-mic array can't disambiguate a bearing from one ping;
independent per-ping fusion can't fix it), so §5.7's literal prescription — *a temporal model that
integrates pings before predicting* — should exploit motion parallax to resolve it. A GRU over the
(echo, pose) sequence → direct world-frame doorway regression does lower the open-loop error: on the
same held-out set, doorway-loc falls 3.71→**2.04 m** (recurrent) vs 3.71→2.84 (fusion) — a real
~0.8 m gain, the saturation partially broken. **But in closed loop it navigates *worse*: 0.29 ± 0.06
(3 seeds) vs the fusion controller's 0.49**, and with high run-to-run variance (one seed swung
0.26↔0.41 across repeats). So the *more accurate* doorway estimator is the *worse* controller — the
same **static-decodability ≠ control** result as §5.2 (a near-perfect cost model planned worse) and
the AcousticJEPA `RESULTS.md` headline, now recurring at the *perception* layer. Likely mechanism
(as in §5.2/§5.5): the stateful GRU is trained on random-walk trajectories but deployed on the
policy's *goal-directed* distribution, where its hidden state generalizes poorly and injects chaotic
variance into the loop; the stateless per-ping fusion, processing each ping independently, is
distribution-shift-robust by construction. On-policy aggregation (DAgger) is the indicated fix but
§5.5 already showed it does not reliably help here. **So the stateless fusion remains the controller
of choice, and the honest close is that the ~2.8→2.0 m doorway floor is genuine echo information —
lowering it further did not buy control.** The gap to the 0.96 perfect-map ceiling is set by the
perception–control interaction, not by static estimator accuracy.

### 5.9 Changing the CONSUMER of each cue, not its accuracy — the reactive bearing + clearance map
§5.8 ended with two named causes and an explicit warning against the obvious fix. The doorway
half saturates at 2.8 m because a 2-mic array cannot resolve front-back from one ping, so the
per-ping bearing is *two-valued* and its errors are **correlated** across poses — averaging a
bimodal estimate lands confidently in the empty middle. The obstacle half fails because the
scatter-CNN gets pillar-IoU ≈0.005, while §5.6's **clearance** readout decodes surface proximity
at R²=0.95 on real observations and goes unused. And §5.8's recurrent estimator showed that
making the estimate *more accurate* (2.84→2.04 m) makes control *worse* (0.49→0.29). So this
section changes **who consumes each signal**, leaving both estimators' accuracy alone.
Code: `echorl/mapper_react.py`. Figure: `results/echorl_react_clearance.png`.

**A. Reactive bearing goal.** Stop forming a doorway coordinate. Each step, take the current
ping's ego-bearing, low-pass it *in the world frame* (so the agent's own turning does not smear
it), and plant a moving SUBGOAL a lookahead ahead of the **current** pose along it; the grid
planner routes to that subgoal around mapped pillars. Follow the gradient rather than compute
the address: a 2.8 m coordinate error puts you at the wrong place permanently, whereas a bearing
re-taken every step self-corrects and a front-back flip costs one step instead of contaminating
an average. No absolute doorway point is ever accumulated, so the 2.8 m floor cannot bind.

**The reactive form is free — the control that had to be run first.** With the **true** bearing
and *no range at all* (fixed 4 m lookahead) on the true map: **0.938**, against the true-goal
planner's **0.948** (12 rooms × 8; both use no learned model, so they are seed-independent).
Following a direction loses ~0.01. Therefore every point A fails to deliver is bearing-
*estimation* error, not the reactive scheme — the ceiling is A's to lose.

**B. Clearance-field occupancy.** Feed §5.6's readout to a classical sonar log-odds map instead
of to the planner's cost (where §5.6 showed it is actively misleading, because CEM reads it
through corrupted rollouts). At each known pose: carve the disc of radius *c* free (sound — a
clearance reading asserts nothing is nearer), and add occupied evidence at range *c*. One
finding from the label distribution changed the design: the corpus' median clearance is only
**0.41 m** (a random walk hugs walls; AGENT_R=0.3 is the floor), so the textbook **isotropic
ring** — *"something is 40 cm away, somewhere on this circle"* — is dominated by the wall behind
the agent and sprays phantom occupancy across the interior, exactly where the pillars are. So
the evidence is made **directional**: a `SurfNet` predicts the nearest surface *point*
(dist + bearing), and a Gaussian blob is placed there. Held-out per-ping quality: **dist R²=0.84,
MAE 8.1 cm, bearing 20.2° MAE** — far sharper than the *doorway* bearing (37–43°, §5.8), because
the nearest surface is close and dominates the echo.

Open-loop map quality (held-out nobs=3, 24 trajectories, pillar-IoU = occupied cells *inside* the
true room rectangle — the hard interior obstacles):

| #pings | scatter-CNN | isotropic ring | **directional** | scatter ⊕ directional |
|---|---|---|---|---|
| 2  | 0.004 | 0.016 | 0.018 | 0.004 |
| 10 | 0.005 | 0.022 | 0.040 | 0.008 |
| 40 | 0.005 | 0.023 | **0.049** | 0.011 |

The directional field is **10× the scatter-CNN on pillars and still climbing at 40 pings**
(the ring plateaus by ping 10 — the phantom-evidence prediction, confirmed). Its solid-IoU is
near zero because it only maps a thin trace along the path, whereas the scatter-CNN's 0.78
comes mostly from the easy exterior; the two are complementary, which is why they are summed in
log-odds for the controller.

**⚠ Statistical correction (2026-07-25).** The first version of this section reported the 2×2 at
12 rooms × **8** episodes — 96 binary episodes per cell, whose binomial standard error (~0.05 at
p≈0.5) is *the same size as the effects being claimed*. Nine independent runs of one identical
config spread over 0.500–0.625 (sd **0.038**), confirming it. The per-seed spread was therefore
reported as if it were the whole uncertainty when it was mostly single-run sampling noise, and
"positive in 3/3 seeds" is weak evidence when each seed is one noisy draw. Everything below is
re-measured at **12 rooms × 24 = 288 episodes per cell** (se ≈ 0.029). **One claim did not
survive: the super-additive interaction between A and B was noise.**

**Closed loop — the 2×2** (nobs=3 goal, 12 held-out rooms × **24**, **3 model seeds**,
macro-average per room; identical protocol to §5.4/§5.8):

| | fused doorway POINT (§5.8) | **reactive BEARING** (§5.9) |
|---|---|---|
| **scatter-CNN map** (§5.8) | 0.456 | **0.519** |
| **+ clearance field** (§5.9) | 0.462 | 0.514 |

Paired per-seed deltas: **A alone +0.063 ± 0.024** (+0.035/+0.073/+0.080, positive 3/3 — the one
effect this section establishes); **B alone +0.006** (+0.007/0.000/+0.011 — reliably *nothing*);
**B on top of A −0.005 ± 0.040** (+0.042/−0.021/−0.035 — **nothing, and not even consistently
signed**). At this sample size effects below ≈0.04 are not resolvable, so the honest reading is:

> **The reactive consumer (A) is the whole §5.9 result. The clearance field (B) does not help —
> alone or in combination.** Its 10× pillar-IoU gain (0.005 → 0.049) is real but far too small in
> absolute terms to change the planner's decisions; the earlier "+0.052 on top of A / interaction
> +0.045" was a 96-episode artifact and is retracted.

That B is *reliably* nil (+0.006, three seeds, tight) is still informative — it says the cluttered
doorway task is not obstacle-map-limited at this level of map quality, which the §5.8 decomposition
already hinted (perfect *obstacles* was the smaller of the two levers).

**Where the remaining gap lives** (3 seeds, at the older 8-episode resolution — treat as ±0.05):
giving the controller perfect obstacles measured **0.674**; giving it the perfect doorway
measured **0.785**, against ~0.52 for the learned version. The **doorway half is the bigger
limiter** (≈−0.27 vs ≈−0.15) — the load-bearing cue remains the hardest one, exactly as §4.3
said, and consistent with B (the obstacle half) buying nothing. The ladder: PPO 0.08–0.18 →
latent-WM MPC 0.24 → §5.8 map 0.46 → **§5.9 reactive 0.52** → perfect map 0.95.

**Ablation — a contradiction carve.** The clearance field marks surfaces ~0.4 m away and
config-space inflation is 0.5 m, so the map can swallow the agent's own cell, after which every
sampled move reads as blocked and the planner freezes. Letting proprioception override the map
*only there* ("I am demonstrably standing here") lifts A+B to **0.604 ± 0.064** and A to
0.580 ± 0.026, while costing the fused-point baseline ~0.03 (0.465 → 0.434) — an accurate map
does not want to be overridden. Both settings agree on the headline; the numbers above are the
**no-carve** ones because that is byte-identical to §5.8's protocol, and the carve is reported
as an extra mechanism rather than folded into the result.

**Honest scope.** Known-pose, like §5.8's headline (the dead-reckoning rung of §5.8 was not
re-run here). Single corpus, 3 model seeds. The reactive controller has three constants (EMA α,
lookahead bounds) chosen once and not swept. Effects below ≈0.04 are below this eval's resolution
even at 288 episodes — a lesson applied retroactively above.

### 5.10 Active sensing (fix C) — a large perceptual gain that buys no control
§5.9 left the doorway half as the binding limit and proved the *consumer* is not at fault (true
bearing, no range → 0.938 vs the 0.948 ceiling), so the residual is bearing **estimation** error.
The textbook remedy is motion: a 2-mic array resolves front-back ambiguity from a baseline, and
bats famously scan with head movements. Code: `echorl/mapper_active.py`.

**The gate (open loop): which kind of motion actually helps?** Six oracle-driven motion policies
in held-out rooms, matched pings and comparable distance-to-goal — `orbit` (baseline ⟂ to the
target = maximum parallax), `radial` (baseline ∥ = none), `spin` (rotate in place = zero
baseline, aspect only), `scan` (approach **and** sweep ±45°), `approach` (drive straight at it —
*what the §5.9 controller does*), and `random` (the corpus policy). 36 trajectories each, 40 pings:

| policy | fused doorway err | reactive **direction** err | mean dist |
|---|---|---|---|
| random (corpus / training distribution) | **3.11 m** | **40.4°** | 3.45 |
| scan (approach + ±45° sweep) | 3.70 m | 44.6° | 2.85 |
| orbit (max parallax) | 3.74 m | 57.9° | 2.72 |
| spin (aspect only) | 3.92 m | 62.3° | 2.78 |
| radial (no parallax) | 4.23 m | 80.6° | 2.28 |
| **approach (the §5.9 trajectory)** | 4.19 m | **83.9°** | 2.00 |

Three readings. (a) **It is ASPECT, not parallax.** Orbit beats radial by 23°, but plain scanning
beats orbit by another 13°: which way you are *facing* when you ping matters more than where you
stand. (b) **The §5.9 controller produces the worst possible input for its own estimator** —
driving at the goal scores 83.9°, bottom of the table, while being *closest* to it (2.00 m);
adding a head sweep to the same radial policy halves that to 44.6°. A closed-loop controller
degrading its own perception is a new, concrete mechanism for the doorway gap. (c) **The best
policy is the training distribution** — random walk wins outright, and that is exactly what
`EgoVecNet` was trained on, so part of what "active sensing" buys here is staying on-distribution
rather than new geometry (the §5.2/§5.5/§5.8 distribution-shift theme, now at the level of gait).

**Closed loop: it does not convert.** The subgoal is swept ±`scan_deg` off the estimated bearing,
annealed by the filter's circular concentration (scan while bearings disagree, commit once they
agree); `scan_deg=0` reproduces §5.9 exactly. At 288 episodes × 3 seeds, 30° scanning scores
**0.538 vs 0.514** — **+0.024 ± 0.013**, positive in 3/3 seeds but *below this eval's ≈0.04
resolution*. A 0/15/30/45° sweep at the lower resolution was flat (0.507/0.514/0.548/0.497).
**Verdict: not established.**

**Why the gain doesn't pay, and the fourth instance of the pattern.** The mechanism is a hard
threshold plus an immediate cost. Success requires arriving within 1.2 m; 44.6° of direction
error at 3 m is still 2.3 m of lateral miss, so scanning moves the estimate a long way *without
crossing the line that scores*, while the ±45° zig-zag costs ~cos 45° ≈ 30% of forward speed
inside a 40-step budget. Partial perceptual improvements do not pay when the task's reward has a
threshold. And so, a **fourth** time in this project, a large improvement in a static perceptual
metric fails to become control: §5.2 (cost model R²=0.96, halved success), §5.8 (recurrent
estimator 2.84→2.04 m, 0.49→0.29), the AcousticJEPA headline, and now −39° of doorway bearing for
+0.024 ± 0.013 of success. The corrected §5.9 above is arguably a fifth: B improved pillar-IoU
10× for nothing.

### 5.11 More microphones (fix D) — the front-back hypothesis, measured and mostly wrong
Since §5.8 this log has carried a *hypothesis* for the 2.8 m doorway-fusion floor: a 2-mic head
has ITD/ILD symmetric about its interaural axis, so a single ping cannot separate a source ahead
from its mirror image behind; the resulting errors are two-valued and therefore correlated across
poses, which is why averaging saturates. §5.9 and §5.10 both leaned on that story. It had never
been measured. This section builds the sensor that would fix it and measures it.
Code: `src/{acoustic_env.h,episode_server.cpp}` (`--ears 4`), `echorl/mapper_ears.py`.

**The array.** `episode_server --ears 4` decodes the *same* order-3 ambisonic field a second time
through a head yawed +90°, so ears 2-3 have an interaural axis perpendicular to ears 0-1 — two
crossed binaural heads, i.e. a non-collinear 4-receiver array. Cost: one extra ambisonics decode,
negligible against the reflection sim. `--ears 2` remains the default and is byte-identical to
every earlier §5 result (verified: 2-ear output differs from the 4-ear run's first pair only at
render-jitter scale). The comparison is exactly **paired**: the corpus policy is a pure RNG, so at
a fixed seed both sensors visit the same poses in the same rooms.

**Open loop — the array works, the hypothesis doesn't** (36 held-out trajectories, paired). A ping
counts as **mirrored** when the front-back-flipped hypothesis (β → π−β) fits the truth *better*
than the true bearing — a direct measurement of the cone of confusion, not the `err > 90°` proxy
that would conflate a flip with noise on a near-lateral target:

| | 2-ear head | 4-ear array | Δ |
|---|---|---|---|
| per-ping bearing MAE | 40.9° | 41.0° | +0.1 |
| **mirrored (cone of confusion)** | **22.8 %** | **22.1 %** | **−0.7** |
| residual MAE, non-mirrored pings | 28.1° | 26.8° | −1.3 |
| fused doorway floor (40 pings) | 3.03 m | 2.76 m | −0.27 |

Doubling the array did **not** reduce the flip rate, and moved the fusion floor by 0.27 m against
the ~1.6 m that separates it from the 1.2 m goal radius.

**Ruling out "the net ignored the new channels."** Nets trained on each pair *alone* both work
(fused error 3.05 m for ears 0-1, **2.85 m for ears 2-3** — the rotated head is the *better*
doorway sensor), so the second pair genuinely carries bearing. The discriminating test is whether
the two pairs — whose cones of confusion are **perpendicular** — flip on the same pings:

| | |
|---|---|
| P(pair 0-1 mirrored) | 24.7 % |
| P(pair 2-3 mirrored) | 17.2 % |
| P(both mirrored) | **10.6 %** |
| if independent | 4.3 % |
| **lift over independence** | **×2.50** |

Independent flips (lift ×1) would mean the ambiguity is geometric and a 4-mic array resolves it;
coincident flips (lift ×4.0 here, the fully-dependent bound) would mean array shape is irrelevant.
The measured ×2.50 sits about halfway: **roughly half the flip co-occurrence is common-cause**.
And pair 2-3 mirrors *less* (17.2 % vs 24.7 %) even though front-back is precisely the axis it
should be strong on — so the geometric effect is real, just small.

**Closed loop — no gain** (true occupancy isolates the doorway half; 288 episodes):
**2-ear 0.677 → 4-ear 0.635.** Slightly worse, within/near this eval's ≈0.04 resolution: **no
measurable conversion**, exactly as §5.10's threshold argument predicts (0.27 m of a 1.6 m gap
cannot cross a 1.2 m radius). A **fifth** static-decodability ≠ control instance.

**Verdict, and a partial retraction.** The 2.8 m doorway floor is **not a 2-mic hardware limit**.
The front-back ambiguity invoked since §5.8 is real but explains a minority of the error; the
dominant term is **scene-acoustic** — in a cluttered reverberant box the doorway's direction is
often simply not present in a single ping at that pose, and no array geometry recovers what was
never encoded. §5.8's "likely the binaural front-back bearing ambiguity" was a plausible untested
hypothesis and should be read as partially superseded here.

**Caveat — what was *not* tested.** All four receivers are decodes of the field at a **single
point**: crossed directivity patterns, not spatially separated microphones. This is the right test
for the cone of confusion (an HRTF/directivity property) but it does **not** test time-of-arrival
disambiguation across a physical baseline, which is how a real distributed array localizes. A
genuinely displaced 4-mic array remains the untested variant — though the pairs test above is
evidence about the *scene* rather than the sensor, so its prospects are modest. Also: one 4-ear
net, no seed repeats, so the closed-loop −0.042 confounds sensor with model seed.

## 6. Summary of findings

1. **Echo-only avoidance is "solved," but degenerately — PPO learns to STOP** (§3.4.1). It
   matches the privileged planner on the survival metric (~38/40), but by parking in open
   space (mean forward step 0.004–0.03 m of 1.5), not by skilled flight — the avoidance
   reward is winnable without moving. Faithful in the renders; caught *by* the renders.
2. **Continuous control is the decisive factor for cluttered avoidance** — but the mechanism
   is **braking**, not weaving: discrete fixed-speed DQN *cannot stop* so it floors on clutter
   (~3/100%); continuous PPO can stop and survives (38.4/4%). Memory and cue choice are
   irrelevant because the winning policy (stop in open space) barely needs perception — which
   is the real reason all ablations tie exactly.
3. **The binaural bearing cue is the cue for SPATIAL-FEATURE LOCALIZATION — it dominates
   exactly when the task needs to find a direction to something.** It carries doorway-finding
   (cochleagram 0.68 ≈ full 0.66 ≫ range-only 0.48, §4.3) and cluttered coverage (drop-bearing
   −4.2 cells over 3 seeds; range-only collapses to parking 1.7 ± 0.3, §3.6.1) — both require
   localizing a spatial feature (the doorway void; obstacles to route around). But on **empty**
   coverage (nobs=0, 3-seed, §3.6.2) removing bearing costs only −2.8 and the agent still covers
   fine (11.2, no collapse) — vs −4.2 *and collapse to parking* in clutter. So bearing goes from
   *helpful* to *critical* with clutter — clutter/feature-specific, not general. (In empty rooms
   range is near-useless, drop-range −0.3; memory does the book-keeping, mattering more there
   −3.4 than in clutter −1.5.) This also un-saturates
   avoidance: coverage cues SEPARATE (Δ4.2 cells) vs the degenerate survive-only task where
   they tied exactly.
4. **A transferable RL lesson:** how you define *failure* (terminal vs non-terminal
   collision) mattered more than model sophistication for the goal task — terminal
   collision induced risk-averse hedging that no reward-weight tuning fixed.
5. **Which paradigm wins depends on whether the task needs planning — a clean crossover.**
   On *empty* doorway rooms (reactive-sufficient) model-free PPO wins (0.68 vs learned-MPC
   0.46), because the cue is a directional bearing signal a reactive policy follows, not an
   action-consequence the latent dynamics can predict. On *cluttered* rooms that **require**
   lookahead (planning oracle 1.00 vs greedy 0.54), PPO **collapses to 0.08–0.18** — the worst
   method, below both MPCs (0.24–0.25) — while model-based control degrades more gracefully
   (§5.4). Yet learning still helps the planner throughout (learned-MPC ≫ random-encoder
   0.06), replicating the AcousticJEPA control-positive dissociation in a model-based paradigm.
6. **A better cost-model made a worse controller — objective mismatch (§5.2), and on-policy
   data doesn't fix it (§5.5).** Swapping the compositional latent rollout for a direct
   outcome head that predicts the planning cost at R²=0.96 (vs ≈0.24) *halved* MPC success
   (0.29 vs 0.51), everything else fixed — CEM exploits the accurate high-capacity model off
   the training distribution. The named fix, DAgger (aggregate the planner's own
   trajectories), was tested over 3 seeds and **failed to reliably help** (0.30→0.36, within
   noise, no trend). So the limit is not pure distribution shift but the planner–model
   exploit hitting the myopic-cue ceiling (§5.1); the real fixes are a less-exploitable
   planner or a less myopic observation, not more data or a more accurate model.
7. **Probe diagnosis (§5.1):** the action-consequence info is in a single ping (raw+a
   R²=0.43) and kept by the encoder (z+a R²=0.42), but the trained predictor path throws it
   away (0.24). Even so, the single-ping ceiling (0.43) is low — the cue is intrinsically
   myopic, favouring reactive control.
8. **An evaluation bug inflated the model-free numbers (§5.4).** Counting a fixed total of
   episodes across autoresetting per-room envs over-samples fast-finishing (early-*terminating*)
   rooms; corrected to a per-room macro-average. On the goal task success *ends* an episode so
   it was **inflated** (PPO 0.86→0.68 nobs=0, 0.82→0.08 nobs=3); on avoidance a *collision*
   ends an episode so it ran the other way and *understated* PPO (nobs=3 37/8%→38.4/4%). All
   numbers recomputed; every qualitative conclusion held. MPC/oracle numbers were never
   affected (fixed episodes per room), and the render figures were per-episode-correct
   throughout — the nobs=3 render is what exposed the bug.
9. **Explicit mapping partially rescues the cluttered task where the latent WM could not
   (§5.8) — §5.7's "integrate over time" lever, done right.** Building an occupancy MAP from
   the echo sequence at KNOWN POSES and planning on it ≈ **doubles** the latent-WM MPC (0.49 ±
   0.08 vs 0.24) and is 3–6× model-free PPO (0.08–0.18), matching the privileged greedy oracle
   (0.54). But it reaches only ~half the perfect-map ceiling (0.96): per-ping geometry in clutter
   is weak (dist-to-goal R²=0.25, bearings ~40° MAE) *and* correlated across poses, so even
   known-pose fusion saturates — walls are recovered (solid-IoU 0.78) but interior pillars are
   missed (pillar-IoU ≈0) and the doorway localizes to only ~2.8 m. So the lever works, but the
   ceiling is genuine information content, not architecture. **Not a pose-privilege artifact:**
   dead-reckoned proprioception (the same information PPO/MPC get) drops it only to 0.41 ± 0.03
   (bump-aware) / 0.37 ± 0.06 (pure) — still clearly above MPC (0.24), 2–5× PPO. And a **recurrent**
   doorway estimator (§5.7's literal "temporal model") *does* lower the open-loop doorway floor
   (2.8→2.0 m) but navigates **worse** in closed loop (0.29 vs fusion 0.49) — a *static-accuracy ≠
   control* dissociation (cf. §5.2), the stateful model fragile under the policy's own distribution.
10. **The fix was never "make the estimate better" — it was "give the signal to something that
    can use it" (§5.9).** Leaving both estimators' accuracy untouched and changing only their
    consumers moved the cluttered doorway task 0.47 → **0.59** (3 seeds, 0.95 ceiling): consume
    the binaural bearing **reactively** (a moving subgoal re-anchored at the current pose each
    step) instead of fusing it into a doorway coordinate, +0.069 ± 0.027; and feed §5.6's
    clearance readout to an **occupancy map** instead of the planner's cost, +0.052 ± 0.031 *on
    top of* the first. A control proves the reactive form is not the limit: the true bearing with
    **no range at all** scores 0.938 vs the true-goal planner's 0.948. Two structural results
    fall out. (a) **Super-additivity (interaction ≈+0.045):** the clearance map is worth *nil*
    alone (+0.007) because a better obstacle map only routes you accurately to a doorway estimate
    that is 2.8 m wrong — signal value is set by the consumer, the same lesson as §5.6 one layer
    up. (b) A **directional** surface readout (nearest-surface point, 20° bearing) is required —
    the textbook isotropic sonar ring fails here because the corpus' median clearance is 0.41 m,
    so the ring is dominated by the wall behind the agent; pillar-IoU 0.005 → 0.049 (10×) only
    with direction. The doorway half remains the bigger limiter (−0.198 vs −0.087 to ceiling).

    **⚠ SUPERSEDED — see the correction in §5.9.** Re-measured at 288 episodes/cell (the 96-episode
    protocol above carries ~0.05 of binomial noise, the size of the effects): **A survives at
    +0.063 ± 0.024** (0.456 → 0.519, 3/3 seeds), but **B is nil (+0.006 alone, −0.005 on top of
    A)** and the **super-additive interaction is retracted as an artifact**. The corrected reading:
    the reactive consumer is the whole result, the task is doorway-limited rather than
    obstacle-map-limited, and B's real 10× pillar-IoU gain is too small in absolute terms to change
    a planner decision.
12. **The doorway floor is NOT a 2-mic hardware limit, and §5.8's front-back story is mostly
    wrong (§5.11).** A non-collinear 4-receiver array (the same order-3 ambisonic field decoded
    through a second head yawed 90°) leaves the **front-back flip rate unchanged: 22.8 % → 22.1 %**,
    per-ping bearing MAE unchanged (40.9° → 41.0°), and moves the fused doorway floor only
    3.03 → 2.76 m — against the ~1.6 m that separates it from the 1.2 m goal radius. Closed loop:
    **0.677 → 0.635, no measurable gain** (a *fifth* static ≠ control instance). It is not that the
    net ignored the channels: nets trained on each pair alone both work, and the rotated pair is the
    *better* doorway sensor (2.85 m vs 3.05 m). The discriminating test is that the two pairs, whose
    cones of confusion are **perpendicular**, flip on the same pings at **×2.50 the independent
    rate** — about half the co-occurrence is common-cause. So the ambiguity is real but minor; the
    dominant term is **scene-acoustic**: in a cluttered reverberant box the doorway direction is
    often simply not in the ping, and no array geometry recovers what was never encoded. Caveat:
    all four receivers sit at one point (crossed directivity, not a spatial baseline), so
    time-of-arrival disambiguation across a displaced array remains untested.

11. **Active sensing improves the percept a lot and the control not at all (§5.10) — a fourth
    instance of static ≠ control.** An oracle-motion gate shows the doorway estimate responds to
    **aspect** diversity, not translational parallax: scanning beats maximum-parallax orbiting,
    and the §5.9 controller's own straight-at-the-goal trajectory is the **worst** input in the
    set (83.9° direction error while being *closest* to the target, 2.00 m) — a controller
    degrading its own perception. Adding a head sweep to the same radial policy halves that to
    44.6°. In closed loop it is worth **+0.024 ± 0.013**, below this eval's ≈0.04 resolution:
    **not established**. The mechanism is a threshold — 45° of bearing error at 3 m is still a
    2.3 m miss against a 1.2 m goal radius, so the estimate improves a long way without crossing
    the line that scores, while the zig-zag costs ~30% of forward speed in a fixed step budget.
    Also notable: the best motion policy is a **random walk** — the estimator's own training
    distribution — so part of what active sensing buys here is staying on-distribution rather
    than acquiring new geometry.

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
# cache a corpus + learned WM (for the probe / rematch), no closed-loop eval:
python echorl/mpc.py --train-rooms 60 --train-ep 360 --skip-eval \
    --corpus-out echorl/corpus_goal.pkl --save-wm echorl/ckpt/mpc_wm.pt
# Exp A (§5.1) -- Delta-d probe (H0/H1/H2 diagnosis):
python echorl/probe_dd.py --corpus echorl/corpus_goal.pkl --wm echorl/ckpt/mpc_wm.pt
# Exp C (§5.2) -- informed rematch (direct outcome head; objective-mismatch result):
python echorl/mpc_direct.py --corpus echorl/corpus_goal.pkl --wm echorl/ckpt/mpc_wm.pt --seeds 3
# Exp B (§5.3 gate / §5.4 showdown): planning-needed check + cluttered nobs=3 run
python echorl/gate_b.py
python echorl/mpc.py --nobs 3 --train-ep 360 --skip-eval --corpus-out echorl/corpus_goal_nobs3.pkl --save-wm echorl/ckpt/mpc_wm_nobs3.pt
python echorl/mpc_direct.py --corpus echorl/corpus_goal_nobs3.pkl --wm echorl/ckpt/mpc_wm_nobs3.pt --nobs 3 --seeds 3
# Exp D (§5.5): does on-policy (DAgger) data fix the direct-outcome MPC? (negative)
python echorl/mpc_dagger.py --corpus echorl/corpus_goal.pkl --wm echorl/ckpt/mpc_wm.pt --nobs 0 --rounds 4 --seed 0
# Mapping experiment (§5.8) -- explicit occupancy map + planning on the cluttered doorway task:
python echorl/mapping.py --eval-rooms 12 --episodes-per 10                    # M0 grid-planner on TRUE map (0.96 ceiling)
python echorl/mapper_collect.py --rooms 300 --ep-per-room 3 --out echorl/mapper_corpus_nobs3.pkl
python echorl/mapper_probe.py --corpus echorl/mapper_corpus_nobs3.pkl         # per-ping decodability in clutter (weak)
python echorl/mapper.py train --corpus echorl/mapper_corpus_nobs3.pkl --out echorl/ckpt/mapper_nobs3.pt   # occupancy mapper
python echorl/mapper_fuse.py train --target goal --out echorl/ckpt/egovec_goal_nobs3.pt                    # known-pose doorway fusion
python echorl/mapper.py evalq --ckpt echorl/ckpt/mapper_nobs3.pt              # M1 occ quality (solid-IoU 0.78, pillar ~0)
python echorl/mapper_fuse.py evalq --ckpt echorl/ckpt/egovec_goal_nobs3.pt    # M1 fused doorway err vs #pings (3.7->2.8 m)
python echorl/mapper_nav.py --eval-rooms 12 --episodes-per 8                  # M2 closed-loop ladder + decomposition (full 0.49)
python echorl/mapper_nav_dr.py --seeds 1 2 3                                  # unprivileged dead-reckoned pose (DR 0.37-0.41)
python echorl/mapper_seq.py train --out echorl/ckpt/seqgoal_nobs3.pt          # recurrent (GRU) doorway estimator
python echorl/mapper_seq.py evalq --ckpt echorl/ckpt/seqgoal_nobs3.pt         # open-loop 3.7->2.0 m (beats fusion) ...
python echorl/mapper_nav.py --modes scatter+seq                              # ... but closed-loop 0.29 < fusion 0.49 (static!=control)
python echorl/mapper_render.py                                                # results/echorl_mapping_nobs3.png
# Reactive bearing + clearance-field map (§5.9) -- change the CONSUMER, not the accuracy:
python echorl/mapper_react.py train-clear --directional                       # SurfNet: nearest-surface point (R2 0.84, 20 deg)
python echorl/mapper_react.py train-clear                                     # isotropic clearance net (the ring ablation)
python echorl/mapper_react.py mapq                                            # open-loop pillar-IoU 0.005 -> 0.049
python echorl/mapper_react.py nav --ego-carve 0 --episodes-per 24 \
  --modes true+true true+reacttrue scatter+fuse scatter+react both+fuse both+react true+react both+true
#   ^ the 2x2 + ceilings; repeat with --occ-ckpt/--goal-ckpt/--surf-ckpt *_s2,*_s3 for the 3 seeds.
#     USE --episodes-per 24: at 8 the binomial noise (~0.05) swamps the effects -- that is what
#     produced the retracted A-x-B interaction. See the correction at the top of §5.9.
#     true+reacttrue = the mechanism control (TRUE bearing, no range): 0.938 vs true-goal 0.948.
#     drop --ego-carve 0 for the contradiction-carve ablation.
# Active sensing (§5.10, fix C) -- aspect beats parallax open-loop, buys no control:
python echorl/mapper_active.py gate --eval-rooms 12 --ep-per-room 3   # the motion-policy gate
python echorl/mapper_react.py nav --ego-carve 0 --episodes-per 24 --scan-deg 30 --modes both+react
#     scan-deg 0 == §5.9 exactly; 30 deg scores +0.024 +/- 0.013 (below resolution).
# 4-receiver array (§5.11, fix D) -- needs the rebuilt server (src/build.bat):
python echorl/mapper_collect.py --rooms 300 --ep-per-room 3 --ears 4 --out echorl/mapper_corpus_nobs3_e4.pkl
python echorl/mapper_fuse.py train --corpus echorl/mapper_corpus_nobs3_e4.pkl --target goal   --out echorl/ckpt/egovec_goal_nobs3_e4.pt
python echorl/mapper_ears.py compare       # paired 2-ear vs 4-ear: flips 22.8% -> 22.1% (nil)
python echorl/mapper_fuse.py train --corpus echorl/mapper_corpus_nobs3_e4.pkl --ear-idx 0,1   --out echorl/ckpt/egovec_goal_nobs3_pair01.pt          # and --ear-idx 2,3 -> ..._pair23.pt
python echorl/mapper_ears.py pairs         # geometric or acoustic? lift x2.50 => mostly acoustic
python echorl/mapper_react.py nav --ears 4 --goal-ckpt echorl/ckpt/egovec_goal_nobs3_e4.pt   --episodes-per 24 --ego-carve 0 --modes true+react     # closed loop: 0.635 vs 2-ear 0.677
# Redesigned coverage/explore task (§3.6) + its cue ablation (§3.6.1):
python echorl/ppo.py --task explore --nobs 3 --steps 120000 --t-max 60 --tag cover
python echorl/ppo.py --task explore --nobs 3 --steps 120000 --t-max 60 --tag nomem --no-memory
python echorl/ppo.py --task explore --nobs 3 --steps 120000 --t-max 60 --tag rangeonly --drop-coch
python echorl/ppo.py --task explore --nobs 3 --steps 120000 --t-max 60 --tag cochonly --drop-range
python echorl/explore_ablation_eval.py --nobs 3 --episodes-per 20
# Trajectory maps -- all 4 conditions (task x clutter), columns = controllers:
python echorl/render_traj.py --task avoid --nobs 0 --dqn-ckpt echorl/ckpt/dqn_stage0_nobs0_best.pt --ppo-ckpt echorl/ckpt/ppo_stage0_nobs0_best.pt --out results/echorl_avoid_nobs0.png
python echorl/render_traj.py --task avoid --nobs 3 --dqn-ckpt echorl/ckpt/dqn_stage0_nobs1_best.pt --ppo-ckpt echorl/ckpt/ppo_stage0_nobs3_best.pt --out results/echorl_avoid_nobs3.png
python echorl/render_traj.py --task goal  --nobs 0 --ppo-ckpt echorl/ckpt/ppo_goal_nobs0_full_s0_best.pt --corpus echorl/corpus_goal.pkl --wm echorl/ckpt/mpc_wm.pt --out results/echorl_goal_nobs0.png
python echorl/render_traj.py --task goal  --nobs 3 --ppo-ckpt echorl/ckpt/ppo_goal_nobs3_showdown2_s0_best.pt --corpus echorl/corpus_goal_nobs3.pkl --wm echorl/ckpt/mpc_wm_nobs3.pt --out results/echorl_goal_nobs3.png
```

Checkpoints (`echorl/ckpt/`): `dqn_stage0_nobs{0,1}_best.pt`,
`ppo_nobs{0,1,3}_full_s0_best.pt` (Stage-0 + ablations `_{nomem,rangeonly,cochonly}_s{0,1,2}`),
`ppo_goal_nobs0_{full,rangeonly,cochonly,nomem}_s{0,1,2}_best.pt` (Stage-2). `mpc.py` trains
its world model on the fly (random-trajectory corpus), so it saves no checkpoints. Mapping
(§5.8): `mapper_nobs3{,_s2,_s3}.pt` (occupancy), `egovec_goal_nobs3{,_s2,_s3}.pt` (doorway
fusion); corpus `mapper_corpus_nobs3.pkl`; figure `results/echorl_mapping_nobs3.png`. §5.9:
`surfnet_nobs3{,_s2,_s3}.pt` (nearest-surface point), `clearnet_nobs3.pt` (isotropic ablation);
figure `results/echorl_react_clearance.png` (`echorl/plot_headlines.py`).

Code: `src/episode_server.cpp`, `src/verify_episode_server.py`,
`echorl/{env,vec_env,models,dqn,ppo,oracle,mpc,mpc_direct,mpc_dagger,probe_dd,gate_b,render_traj,collect_corpus,smoke_test}.py`,
mapping `echorl/{mapping,mapper_collect,mapper,mapper_probe,mapper_fuse,mapper_seq,mapper_nav,mapper_nav_dr,mapper_render,mapper_react}.py`.
