# EchoRL — a real-time closed-loop echolocation RL rig on Steam Audio

A DQN/PPO "bat" agent that navigates from **binaural echoes only**, in 3D, using the
Steam Audio (phonon) renderer as the world. This is the 3D, physically-rendered
successor to Mohan & Vanderelst 2020 (*Training a simulated bat*, SSCI), reusing the
acoustic machinery already built in this repo.

Status: **design**. Nothing here is built yet; build order is in §9.

---

## 1. Relationship to Mohan & Vanderelst (2020)

| | Mohan & Vanderelst 2020 | EchoRL (this plan) |
|---|---|---|
| World | 2D, analytic point-reflector echo model | 3D, full Steam Audio reflections (4 bounces, HRTF binaural) |
| Sensor | L/R echo envelope vs delay (IID image) | binaural cochleagram `[2,64,48]` **+** matched-filter range profile `[256]` |
| Task | obstacle avoidance / keep flying | obstacle avoidance **and** goal-approach / wall-following / exploration (curriculum) |
| Agent | DQN, discrete turn actions | **DQN (faithful baseline) → recurrent PPO (main)** |
| Reward | distance flown − collision | forward progress − collision − control cost (see §4) |
| "Dataset" | procedural obstacle fields | procedural rooms+obstacles (already in `acoustic_env.h`) + offline corpus (§6) |

What we keep from the paper: **echo-only perception, collision-avoidance reward, model-free
deep RL, a procedurally-generated obstacle distribution rather than a fixed map.** What we
change: the echo is *rendered*, not analytic, so the agent must cope with reverberation,
material-dependent absorption, HRTF coloration, and the ~10 ms reflection-histogram binning —
the exact reasons this repo built the matched-filter range tower.

---

## 2. Two-layer architecture

The RL loop needs 10⁵–10⁶ environment steps; the in-engine renderer is far too slow for that.
So the rig is split:

```
  TRAIN (headless, fast, reproducible)          DEPLOY (real-time, visual)
  ┌────────────────────────────────┐            ┌─────────────────────────────┐
  │ episode_server.exe (C-API)      │            │ Unreal + SteamEchoNav-style │
  │  procedural room + obstacles    │  policy    │ component runs the SAME      │
  │  pose→(coch,range) @ ~24 ms     │ ─weights─▶ │ obs op + the trained policy  │
  │  vectorized × K processes       │  (ONNX)    │ ticking at PingInterval      │
  │ Python Gymnasium env + PPO/DQN  │            │ (visual proof of transfer)   │
  └────────────────────────────────┘            └─────────────────────────────┘
```

The observation operator is **identical** on both sides — `AcousticEnv::observe()` in
`src/acoustic_env.h` is already the byte-for-byte render path used by the standalone tools and
mirrors the Unreal `SteamEchoNavComponent`. Training and deployment therefore share the same
sensor; "sim-to-real" here is only sim-to-engine (frame rate, tick scheduling, live geometry),
which is the point of the deployment layer.

### Measured cost (this machine, room seed 7)
`observe()` ≈ **24.4 ms** single-process, all CPU cores on reflections → **~41 obs/s**.
This is the throughput budget for everything below. Parallelism is by **process**, not batching:
each `iplSimulatorRunReflections` must re-run per pose (Steam Audio gotcha #1), so there is no
within-env batching win. To vectorize, cap `AcousticEnv::init(simThreads)` to a few threads each
and run K servers (see §5).

---

## 3. The MDP

**Observation** `o_t` (all already produced by `observe_server`):
- `range[256]` — matched-filter lag profile (sub-decimetre range structure, HRTF-free).
- `coch[2,64,48]` — binaural cochleagram (bearing/elevation/material via ITD/ILD).
- proprioception: previous action `a_{t-1}` and a normalized "steps-since-ping" scalar.

Partial observability is real (a single ping can't see behind the agent, obstacles occlude),
so the main agent is **recurrent** (GRU over the latent) and/or **frame-stacked** (last 2–3 pings).

**Action** — two regimes:
- *Discrete* (paper-faithful, DQN): {hard-left, left, straight, right, hard-right} × fixed step,
  i.e. `dyaw ∈ {−45,−22.5,0,22.5,45}`, `dstep = 0.75 m`. 5 actions.
- *Continuous* (PPO main): `dyaw ∈ [−45,45]°`, `dstep ∈ [0,1.5] m` — exactly the `Action` struct
  and `apply_action` clamp already in `acoustic_env.h`.

**Dynamics** are analytic and already written twice (`AcousticEnv::apply_action`,
`control_mpc.step_pose`) — Python owns pose; the C server is a **pure pose→obs sensor**. This keeps
the C++ change tiny (§5).

**Reward** — see §4.

**Termination**: collision (with wall or obstacle), goal reached, or `T_max` pings (e.g. 60).

**Episode**: sample a room seed from the train split, sample a valid start pose, ping-step until
termination. One "step" = one ping = one `observe()` ≈ 24 ms.

---

## 4. Reward design (curriculum of tasks)

Start with the paper's obstacle-avoidance reward, then layer goals. Per ping:

```
r_t =  w_prog · Δforward_clearance        # progress: reward staying in open space / advancing
     − w_coll · 1[collision]              # large terminal penalty (e.g. −1.0), ends episode
     − w_near · nearfield_penalty(o_t)    # soft penalty as nearest reflector < d_safe
     − w_ctrl · (|dyaw|/45 + dstep_penalty)
     + w_goal · Δ(−dist_to_goal)          # only in goal-approach curriculum stage
```

- **Stage 0 — free flight / avoid**: `w_goal=0`. Reward = distance travelled without collision
  (`w_prog·dstep`), minus collision. Reproduces Mohan & Vanderelst directly, in 3D.
- **Stage 1 — wall-following / corridor**: reward forward progress while holding nearest-wall
  distance in a band — tests that the agent *reads range from echoes*, not luck.
- **Stage 2 — goal approach**: add a target reflector (the `set_beacon` bistatic mode already in
  `acoustic_env.h`, or a target obstacle); reward closing `dist_to_goal`. This is the direct RL
  analogue of the existing CEM-MPC `goal="wall"` task, so we get a matched **MPC vs RL** comparison.
- **Stage 3 — exploration/coverage**: reward newly-visited area (grid cells) with a collision floor.

`nearfield_penalty` and `Δforward_clearance` are computed from **ground-truth geometry** (Python
has room dims + obstacle AABBs from the handshake, §5) — the reward may use privileged geometry;
the **policy may not** (echo-only). This is standard privileged-reward / restricted-observation RL
and keeps reward shaping honest without leaking the map into the observation.

---

## 5. Environment implementation

### 5a. C++ change: `observe_server` → `episode_server` (small)
`observe_server.cpp` today: generates a 0-obstacle room, handshakes `(Lx,Ly,Lz)`, answers
`(x,z,yaw) → (range,coch)`. Two additions:

1. **Obstacles on**: `make_box_room(seed, nObstacles)` with `nObstacles>0` (generator already
   supports it — floor-standing AABB pillars above ear height, non-overlapping).
2. **Extend the handshake** to also stream the obstacle list and openings so Python can do
   collision + reward geometry without re-deriving the RNG:
   `nObstacles:int32, then per obstacle (cx,cy,cz,hx,hy,hz,material):7×float32`.
   (Wall collision needs only `Lx,Lz`, already sent.)

That's ~30 lines. The observation protocol (`256 + 6144` float32 out, `3` float32 in) is unchanged,
so `control_mpc.ObserveServer` keeps working. Collision test = the `segment_hits_aabb` /
point-in-AABB logic already in `acoustic_env.h`, reimplemented in ~10 lines of Python.

### 5b. Python: Gymnasium env
```python
class EchoLocEnv(gym.Env):
    # obs: Dict(range=Box(256), coch=Box(2,64,48), prop=Box(3))
    # act: Discrete(5)  or  Box([-1,-1],[1,1])  (dyaw,dstep normalized)
    def reset(seed): pick room seed from split; spawn valid pose; return observe()
    def step(a):     pose = apply_action(pose,a); o = server.observe(pose)
                     r, done = reward_and_terminate(pose, o); return o,r,done
```
Wrap the existing `ObserveServer` subprocess. `reset` may need a **fresh room** → either one
long-lived server per room seed (reuse across episodes on that seed) or a small pool that
re-spawns; a server is one room per process, so a **vectorized env = K processes across K seeds**,
rotated each rollout. Cap `simThreads≈2–3` per server (add a CLI arg to `episode_server`) so K≈8
processes don't oversubscribe.

### 5c. Throughput math (MEASURED on this box — 24 logical cores)
Per-process latency vs `simThreads` (obstacle room, seed 42): all-cores 25.6 ms (39/s), 4 thr
57 ms, 2 thr 98 ms, 1 thr 168 ms. The reflection sim scales **poorly** with threads
(6.0 obs/s per thread at T=1 vs 1.6 at T=24) — so vectorize with **many thin processes**, not
few fat ones. Measured aggregate over K parallel servers:

| config | obs/s aggregate |
|---|---|
| K=1, all cores (no vectorization) | 38.5 |
| K=8 × T=3 | 84.0 |
| **K=12 × T=2 (recommended)** | **90.3** |
| K=24 × T=1 | 95.4 |

So realistic training throughput ≈ **90 obs/s (~2.4× single-process)** at K=12×T=2 (fewer
processes than K=24×T=1 for the same throughput → less pipe/memory overhead). This is lower than
first guessed: **1M steps ≈ 3 h, 2M ≈ 6 h, 5M ≈ 15 h.** 2M is an overnight run; budget accordingly
(shorter rollouts / ray count during dev, full rays for final eval).

---

## 6. "Develop your own dataset"

RL is online, but three offline artifacts are the "dataset" and each earns its keep:

1. **Environment distribution (the real dataset).** Disjoint room-seed splits, mirroring the
   existing convention (`control_mpc` uses seeds ≥600 as held-out): **train 1–2000, val
   2000–2200, test 2600–2800**, plus a **fixed 200 eval start poses** file so every controller is
   scored on identical starts (variance comes from training seed only). Curriculum tiers = obstacle
   count {0,1,3} and room-size bands.

2. **Offline transition corpus** (`jepa_dump`-style `.npz`), 500–2000 short random/scripted
   trajectories: used for (a) **observation normalization** (per-channel mean/std for range and
   coch — RL is sensitive to input scale), (b) optional **behavior-cloning / world-model warm
   start** of the encoder (reuse `TwoStreamEncoder`), (c) sanity plots. This is the same generator
   the JEPA work already dumps, so it's near-free.

3. **Held-out eval scenarios**: hand-designed rooms (narrow corridor, doorway, pillar slalom) as
   named seeds for qualitative + video eval and for the Unreal demo levels.

Note from prior work (memory): the render is **not bit-deterministic (~1% jitter)**; only
geometry/labels are exact. For RL this is a *feature* — free observation noise / domain
randomization — but **eval must average over multiple render seeds**, and the offline norm stats
should be computed over enough samples to be stable.

---

## 7. The agent

**Perception trunk (reuse `model_xview.TwoStreamEncoder`)**: `coch,range → [z_range|z_coch]`
(256-d). It's already a validated range+binaural encoder; drop the JEPA/predictor heads and attach
RL heads. This is the single biggest reuse win — the hard part (turning rendered echoes into a
geometry-bearing latent) is done.

- **PPO (main)**: trunk → GRU(256) over the ping sequence → actor (Gaussian for continuous, or
  categorical) + critic. Clip 0.2, GAE λ=0.95, γ=0.99, entropy bonus annealed, 8–16 parallel envs,
  rollout 128–256 steps. Recurrence handles partial observability; frame-stack is the fallback.
- **DQN (faithful baseline)**: trunk → dueling double-DQN over the 5 discrete actions, prioritized
  replay, ε-greedy. Directly comparable to Mohan & Vanderelst. Expect PPO to dominate in 3D but DQN
  anchors the reproduction claim.

Input normalization from §6.2. Encoder either trained from scratch end-to-end (default) or warm-started
and fine-tuned (ablation: does the JEPA-style pretraining help control? — ties into RESULTS.md's
"learned world model helps control even where it fails geometry probes").

---

## 8. Baselines & evaluation

Metrics on the held-out test rooms + fixed starts, averaged over ≥3 render seeds:
- **Collision-free flight time / distance** (Stage 0), **success rate & path efficiency** (Stage 2),
  **coverage** (Stage 3).
- **Time-to-collision** distribution; **min clearance** histograms.

Controllers compared (several already exist in `control_mpc.py`):
- **RL (PPO / DQN)** — this work.
- **Analytic oracle** — greedy toward-goal / away-from-nearest-wall (`analytic_action`, exists).
- **CEM-MPC on the learned world model** — exists (`eval_closed_loop`); the "model-based" foil to
  model-free RL, on the *same* task and server. This is a genuinely novel head-to-head the repo is
  one script away from.
- **Random-action** lower bound (exists); **random-encoder RL** (control analogue of the
  random-encoder probe) to show the learned perception matters.

---

## 9. Build order (milestones)

1. **M1 — episode_server** ✅ DONE (2026-07-19): `src/episode_server.cpp` (separate exe, obstacles
   on, extended handshake: dims+obstacle AABBs+openings, `simThreads` arg); in `build.bat`;
   `src/verify_episode_server.py` passes (geometry + 19% echo change vs 0-obstacle room). Measured
   throughput table in §5c.
2. **M2 — env + smoke test** ✅ DONE (2026-07-20): `echorl/env.py` (`EchoLocEnv`, gym-shaped &
   dependency-free, continuous+discrete actions, Stage-0 reward, Python-owned dynamics/collision
   from the handshake). `echorl/smoke_test.py` passes (both action modes, collisions fire, finite
   rewards). `echorl/collect_corpus.py` → `norm_stats.npz` + `corpus.npz` (2016 obs / 24 rooms,
   seeds 3000+). **Findings:** random agent survives ~3–4 steps (expected floor); all 256 range
   channels vary, but **2605/6144 coch channels are near-silent** → normalize with `(x−μ)/(σ+ε)`.
3. **M3 — DQN baseline, Stage 0** ✅ DONE (2026-07-20): dueling double-DQN (`echorl/dqn.py`,
   `models.py` reuses `TwoStreamEncoder` trunk + `Normalizer`; `vec_env.py` threaded K-env
   autoreset). **Result (nobs=0, wall-avoidance tier, held-out rooms 600+): mean flight length
   4.4→30.8 / 40 cap, collision 100%→26%** — ~7× the random floor. Bootstrapping masks collision
   (true terminal) but not time-limit truncation. **Reward bug found & fixed:** the original
   fixed-step progress term is action-independent for discrete actions (zero gradient on turn
   direction) and the near-field penalty fired too late (<0.6 m at 0.75 m/step) → greedy degraded
   *below* random. Fixed with dense anticipatory clearance shaping `w_clear·tanh((clear−d_safe)/scale)`
   + larger collision penalty, no turn cost. DQN greedy drifts late in training → keep best-eval
   checkpoint (`ckpt/dqn_stage0_best.pt`). **Graded curriculum result (best held-out mean flight
   length / 40 cap, collision rate):** nobs=0 **30.8 / 26%** (7× floor 4.4); nobs=1 (warm-started
   from nobs=0) **9.0 / 88%** (2.3× floor); nobs=3 **~3 / 100%** (at floor, even warm-started).
   Interpretation: the discrete **fixed 0.75 m step** can't slow/stop, so dense clutter (3 pillars
   in 3–12 m rooms) is largely unsurvivable → monotonic collapse with obstacle count. This directly
   motivates **M4 continuous PPO** (`dstep∈[0,1.5]`, can creep/pivot) and/or a discrete action set
   with a slow/pivot action. Checkpoints: `dqn_stage0_nobs{0,1}_best.pt`.
4. **M4 — recurrent PPO, continuous actions** ✅ Stage-0 DONE (2026-07-20): `echorl/ppo.py`
   (GRU over pings for partial observability; reuses `EchoTrunk`; feedforward-encode-then-GRU for
   efficient recurrent replay; clipped surrogate + GAE; Gaussian actions, env-clipped). **Hypothesis
   from M3 CONFIRMED — continuous control cracks the clutter DQN floored on.** Best held-out mean
   flight length / 40, collision:

   | tier | random | DQN (M3) | **PPO (M4)** |
   |---|---|---|---|
   | nobs=0 | 4.4 / 100% | 30.8 / 26% | **40.0 / 0%** |
   | nobs=1 | 3.9 / 100% | 9.0 / 88% | **40.0 / 0%** |
   | nobs=3 | 2.7 / 100% | ~3 / 100% | **37.0 / 8%** |

   PPO reached optimum in <10k steps (vs DQN ~21k + oscillation) and was rock-stable (nobs=3 held
   37.0 for 90k steps, value loss ~0.04). Slowing/stopping/pivoting (`dstep∈[0,1.5]`) is what makes
   the tight rooms survivable. Checkpoints `ppo_nobs{0,1,3}_full_s0_best.pt`.

   **Oracle ceiling (`echorl/oracle.py`, privileged-geometry receding-horizon planner, no echoes):**
   nobs=0 **40.0/0%**, nobs=1 **39.9/1%**, nobs=3 **37.9/6%**. The echo-only PPO agent matches it to
   ~1 step / ~2% collision at every tier — i.e. it is near-optimal, and the residual nobs=3 collisions
   are dominated by rooms that are **genuinely unsurvivable** (the planner also dies in ~6%), not by
   perception error. (A naive greedy 1-step-clearance controller collides 85–93% — myopic reactive
   avoidance fails under the 45°/step turn limit, so the ceiling requires planning.)

   **Ablations (nobs=3, best held-out len/40, coll):** full `37.0/8%`, no-memory (GRU→MLP) `37.0/8%`,
   range-tower-only `37.0/8%`, cochleagram-only `37.0/8%` — **all identical, all at the oracle
   ceiling.** Conclusions: (1) **continuous action is the sole decisive factor** — DQN's discrete
   fixed step floored, every continuous PPO variant hits the ceiling; the earlier "recurrence +
   continuous" framing was WRONG — **memory is not needed** (near-Markov given the observation).
   (2) Either echo representation alone suffices → Stage-0 avoidance is **saturated / low-order**
   (mirrors the JEPA RESULTS.md "geometry is low-order, already exposed" finding); the observation
   ablation cannot discriminate cues on a task this easy. **This motivates harder tasks** (Stages 1–2:
   wall-following, goal-approach/ranging) where fine range and memory should finally matter.
   **Multi-seed CIs (nobs=3, 3 seeds/config, final-policy 120-ep held-out eval):** full, no-memory,
   and range-only are **all 36.1 ± 0.0 len / 0.10 ± 0.00 collision** — every one of the 9 runs
   returned exactly 36.1/0.10 (coch-only seeds finishing; earlier single seed also 36.1). So the
   headline result and the ablation null are **fully seed-robust** (full − nomem = +0.00,
   full − rangeonly = +0.00). Caveat: the near-zero variance reflects reliable convergence *and* a
   deterministic eval (fixed held-out rooms/starts across training seeds); a fuller CI would also
   randomize eval starts per seed — but the training-seed robustness the CIs test is total.
   **Stage 2 — goal approach (doorway-finding)** ✅ SOLVED (2026-07-20): `env.py task="goal"` — home
   to a point 1.5 m in front of the room's largest opening, monostatic echoes only, success = reach
   within 1.2 m. **Result: ~0.80–0.90 held-out success, stable/monotonic**, reaching the doorway in
   ~12 steps (init ~0.40). **Key formulation insight:** *terminal* collision is wrong for a goal task
   — it makes any near-wall approach catastrophically risky, so the agent hedges into risk-averse
   safe-wandering (success collapsed 0.57→0.17; 5 reward-weight iterations couldn't fix it). Fix =
   **non-terminal collision** (a bump costs `w_bump` and blocks the move, episode continues to
   success/timeout). So monostatic self-echoes *do* carry enough to home to an opening; the earlier
   ~0.4 ceiling was the reward formulation, not the cue. Engineering fixes en route: NaN action →
   server quit-sentinel (`nan_to_num`); rare `episode_server` segfault → `_observe_safe` respawns the
   same-seed room; goal mode needs opening rooms → `opening_seeds()` pre-filter (also kills server
   churn). Remaining M4: **range-vs-coch + memory ablation on Stage 2** (the key test — does bearing
   finally matter here, unlike saturated Stage 0?), multi-seed CIs, MPC-vs-RL.
5. **M5 — Unreal deployment** (1–2 days): export policy to ONNX; run it inside a
   `SteamEchoNavComponent`-style tick (swap the hand-tuned bearing/step controller in
   `SteamEchoNavComponent.cpp` for the ONNX policy) on demo levels — the real-time closed loop.
6. **M6 — writeup** (`docs/EchoRL_RESULTS.md`): metrics, ablations, videos.

## 10. Risks & mitigations
- **24 ms/step bottleneck** → process-parallel vectorization (§5c); if still tight, drop `NUM_RAYS`
  during training and validate at full rays (accept a small sim-to-sim gap, measure it).
- **Reward hacking** (spinning in place for control-cost-free "safety") → require net displacement;
  cap episode return by distance; the `w_ctrl`/progress balance.
- **POMDP / aliasing** (different geometries, same echo) → recurrence + frame stack; the range tower
  breaks most aliasing already.
- **Non-determinism** → treat as noise, average eval over render seeds, never assert exact obs.
- **Unreal transfer gap** (tick jitter, live scene vs procedural box) → the obs op is identical;
  start the demo on box-like levels, then add real geometry via the plugin scene (README note).
```
