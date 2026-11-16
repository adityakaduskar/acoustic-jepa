# acoustic-jepa

Learning to perceive and navigate from echoes, in simulated rooms rendered with Steam Audio.

The repository holds two related studies that share one renderer and one encoder:

- **Acoustic-JEPA**: self-supervised world models trained on simulated binaural echoes. The question is whether
  a predictive (JEPA) objective learns room geometry better than fixed features, and whether it gives a usable
  model for planning.
- **EchoRL**: reinforcement learning agents (PPO, DQN) and model-based controllers that navigate from binaural
  echoes alone, with privileged oracle planners as ceilings. It is a 3D, rendered-acoustics follow-up to Mohan
  and Vanderelst (2020), *Training a simulated bat*.

Both studies run on the same C++ servers, which render each observation with the Steam Audio (phonon) C API in
procedural box rooms with pillars and openings. An observation is a 256-bin matched-filter range profile plus a
2 x 64 x 48 binaural cochleagram.

## Main results

### Acoustic-JEPA: geometry probes and control disagree

Full write-up: [`RESULTS.md`](RESULTS.md). Spec of the first version: [`docs/ACOUSTIC_JEPA_SPEC_v1.md`](docs/ACOUSTIC_JEPA_SPEC_v1.md).

- **Geometry probes.** In cross-view prediction, one agent emits and the model predicts what a second agent
  hears. A model trained this way decoded room geometry worse than raw pooled features or a random encoder.
  Nonlinear probe, 2000 rooms: room volume R² 0.848 (JEPA) against 0.917 (raw) and 0.895 (random). The result
  held under linear and nonlinear probes, across seeds, at 400 and 2000 rooms, and over a 10-config
  hyperparameter sweep. An autoencoder and a supervised encoder did no better than raw features. Three
  reverberation statistics alone recover R² 0.68 on volume, so the geometry is already exposed by pooling.
- **Control.** The ordering reverses when the model is used for planning. Model predictive control (CEM) with
  the learned world model reached the room centre (+0.86 ± 0.07 m, 5 seeds), while the same controller with a
  random encoder moved the wrong way (-0.34 ± 0.09 m) and random actions scored -0.55 m. Closed-loop MPC through
  the live renderer on 40 unseen rooms matched the open-loop result.
- **Reading.** Static decodability and dynamical predictability are different properties. Fixed features win
  the first. Only a learned predictive representation gives the second, which is what model-based control uses.

### EchoRL: navigating from echoes alone

Evidence log: [`docs/EchoRL_RESULTS.md`](docs/EchoRL_RESULTS.md). Design: [`docs/EchoRL_PLAN.md`](docs/EchoRL_PLAN.md).
All numbers are per-room averages on held-out rooms.

| Result | Number |
|---|---|
| Obstacle avoidance, 3 pillars, flight length out of 40 steps / collision rate | PPO 38.4 / 4%, oracle planner 37.9 / 6%, DQN about 3 / 100% |
| Doorway finding, success rate (3 seeds) | full input 0.66 ± 0.03, range only (no bearing) 0.48 ± 0.05 |
| Coverage in clutter, cells reached | full input 5.9 ± 1.2, without the bearing cue 1.7 ± 0.3 |
| Cluttered doorway task, success | learned-world-model MPC 0.24, occupancy map + planner 0.46, reactive bearing 0.52, perfect-map ceiling 0.95 |

- Continuous actions (slow down, stop, pivot) are what let PPO survive clutter. The fixed-speed discrete DQN
  could not brake.
- The binaural bearing cue carries doorway finding and cluttered coverage. Using bearing reactively, rather than
  fusing it into a map coordinate, gave a further gain without changing any estimator.
- Several perception gains bought no control. Head scanning halved the doorway bearing error (83.9° to 44.6°),
  and a 4-receiver array barely changed the front-back error, with no measurable change in success.
- Evaluate at 288 episodes per condition and 3 or more seeds. Smaller evaluations produced a result that later
  had to be retracted (see the corrections in section 5.9 of the evidence log).

## Layout

| Path | Contents |
|---|---|
| `src/` | C++ renderers on the Steam Audio C API: `acoustic_env.h` and `swarm_env.h` (single and multi-agent scenes), `jepa_dump` and `swarm_dump` (datasets), `observe_server` and `episode_server` (pose in, observation out, for closed-loop MPC and RL), `scene_dump`, `build.bat` |
| `jepa/` | Acoustic-JEPA: datasets, two-stream encoder, cross-view model, training, linear and nonlinear probes, non-JEPA controls, sweep, latent MPC (`control_mpc.py`) |
| `echorl/` | EchoRL: `EchoLocEnv` (Gymnasium-style), vector env, DQN, recurrent PPO, oracle planner, MPC variants, mapping experiments, trajectory renders and a first-person viewer |
| `results/` | Headline figures, and the SoundSpaces 2.0 port of the JEPA pipeline in `results/ss2_jepa/` |
| `docs/` | Results logs, plans, and `real_world_validation.md` (what a bench validation would need) |
| `run_all.ps1` | End-to-end JEPA pipeline: dataset, train, probe, plan |

## Requirements

- Windows with Visual Studio 2022 (MSVC) for the C++ tools.
- The Steam Audio SDK. `src/build.bat` and `run_all.ps1` expect it at `..\Plugins\SteamAudio\Source\SteamAudioSDK`,
  that is, this folder placed inside an Unreal project that has the Steam Audio plugin. Edit `SDK` in
  `build.bat` to point elsewhere. The later EchoRL reruns used a fork of phonon with finer reflection timing
  (not yet public), dropped into `tools/`. The stock SDK runs every experiment here.
- Python 3.11 or 3.12 with `numpy` and `torch` (`jepa/requirements.txt`). A CUDA build of PyTorch is
  recommended. The scripts fall back to the CPU.
- The SoundSpaces 2.0 port needs habitat-sim, which runs under Linux or WSL2.

Datasets, checkpoints and built executables are not tracked. They regenerate from seeds. Room geometry and
labels are exact, but the Steam Audio render carries about 1% run-to-run jitter in cluttered scenes, so numbers
reproduce to about ±0.01 R², not bit for bit.

## Quick start

Build the renderers (writes `tools\*.exe` and copies `phonon.dll`):

```
cd src
build.bat
```

Acoustic-JEPA, cross-view result at 400 rooms:

```
tools\swarm_dump.exe 400 swarm_data400 6 1 3 8
python jepa\train_xview.py swarm_data400 --epochs 60 --batch 256 --out ckpt_xview400.pt
python jepa\probe_xview.py swarm_data400 --ckpt ckpt_xview400.pt
```

or the whole first-version pipeline:

```
.\run_all.ps1 -SetupVenv -Episodes 2000 -Epochs 60 -Batch 256
```

EchoRL:

```
python src\verify_episode_server.py
python echorl\collect_corpus.py --rooms 24 --per 84
python echorl\ppo.py --nobs 3 --steps 30000
python echorl\ppo.py --task goal --nobs 0 --steps 30000 --tag full
python echorl\oracle.py --eval-rooms 12
python echorl\mpc.py --smoke
```

Section 8 of [`docs/EchoRL_RESULTS.md`](docs/EchoRL_RESULTS.md) lists the command for every experiment.

## Related repositories

- [audio-engine-echolocation](https://github.com/adityakaduskar/audio-engine-echolocation): validation of
  Steam Audio and SoundSpaces 2.0 as acoustic simulators, and the ranging, navigation and multi-agent
  experiments on the same renderer.

## Author

Aditya Kaduskar, Vanderelst Lab, University of Cincinnati.
