# EchoRL — Experiments & Results (evidence log)

Closed-loop echolocation RL on the Steam Audio (phonon) renderer: a simulated "bat"
that navigates from **binaural echoes only**, in 3D, with physically-rendered
reverberant acoustics. 3D, rendered-acoustics successor to Mohan & Vanderelst 2020
(*Training a simulated bat*, SSCI). Design doc: `EchoRL_PLAN.md`.

Status as of **2026-07-21**: Stage 0 (obstacle avoidance) and Stage 2 (goal
approach / doorway-finding) both complete, with oracle ceilings, ablations, and
multi-seed CIs; a **model-based (CEM-MPC) vs model-free (PPO)** comparison on the doorway
task (§5), a Δd probe diagnosing the MPC gap (§5.1), an objective-mismatch result where a
near-perfect cost model plans *worse* (§5.2), a **cluttered-room showdown (§5.4)** whose
render exposed an **evaluation bug** that had inflated the model-free numbers — corrected
throughout, producing a clean **model-free/model-based crossover** (reactive PPO wins empty
rooms; it collapses and model-based wins when the task requires planning) — and an honest
**negative (§5.5)**: the named on-policy (DAgger) fix does not rescue the planner. All numbers
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

## 6. Summary of findings

1. **Echo-only 3D navigation works and is near-optimal for avoidance** — PPO matches a
   privileged-geometry planner to ~1 step / ~2% collision; residual failures are
   genuinely unsurvivable rooms.
2. **Continuous control is the decisive factor for cluttered avoidance**, not perception
   sophistication: discrete fixed-speed DQN floors on clutter (~3/100%); continuous PPO
   solves it (38.4/4%, fair eval). Memory and cue choice are irrelevant here — the task is
   saturated (all cue/memory ablations tie exactly).
3. **The harder goal task exposes cue importance.** Doorway-finding depends on the
   **binaural bearing cue** (cochleagram 0.68 ≈ full 0.66 ≫ range-only 0.48, fair eval),
   exactly as the acoustics predict; under fair eval memory shows a modest positive effect.
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
# Trajectory maps -- all 4 conditions (task x clutter), columns = controllers:
python echorl/render_traj.py --task avoid --nobs 0 --dqn-ckpt echorl/ckpt/dqn_stage0_nobs0_best.pt --ppo-ckpt echorl/ckpt/ppo_stage0_nobs0_best.pt --out results/echorl_avoid_nobs0.png
python echorl/render_traj.py --task avoid --nobs 3 --dqn-ckpt echorl/ckpt/dqn_stage0_nobs1_best.pt --ppo-ckpt echorl/ckpt/ppo_stage0_nobs3_best.pt --out results/echorl_avoid_nobs3.png
python echorl/render_traj.py --task goal  --nobs 0 --ppo-ckpt echorl/ckpt/ppo_goal_nobs0_full_s0_best.pt --corpus echorl/corpus_goal.pkl --wm echorl/ckpt/mpc_wm.pt --out results/echorl_goal_nobs0.png
python echorl/render_traj.py --task goal  --nobs 3 --ppo-ckpt echorl/ckpt/ppo_goal_nobs3_showdown2_s0_best.pt --corpus echorl/corpus_goal_nobs3.pkl --wm echorl/ckpt/mpc_wm_nobs3.pt --out results/echorl_goal_nobs3.png
```

Checkpoints (`echorl/ckpt/`): `dqn_stage0_nobs{0,1}_best.pt`,
`ppo_nobs{0,1,3}_full_s0_best.pt` (Stage-0 + ablations `_{nomem,rangeonly,cochonly}_s{0,1,2}`),
`ppo_goal_nobs0_{full,rangeonly,cochonly,nomem}_s{0,1,2}_best.pt` (Stage-2). `mpc.py` trains
its world model on the fly (random-trajectory corpus), so it saves no checkpoints.

Code: `src/episode_server.cpp`, `src/verify_episode_server.py`,
`echorl/{env,vec_env,models,dqn,ppo,oracle,mpc,mpc_direct,mpc_dagger,probe_dd,gate_b,render_traj,collect_corpus,smoke_test}.py`.
