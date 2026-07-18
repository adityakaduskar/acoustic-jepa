# Real-world validation — guidance

Everything in `RESULTS.md` lives inside a simulator. This doc is how to test
whether the findings survive contact with real acoustics. It is written as a
tiered plan: the cheapest tests carry most of the risk, so do them first.

## What actually needs validating (and what doesn't)

Three distinct claims, in increasing cost to check:

1. **"The decodable geometry is in low-order echo statistics."** (The mechanism:
   3 reverberation numbers recover ~80% of room-volume decodability; raw pooling
   wins because pooling computes those statistics.) — *This is the load-bearing
   claim and the cheapest to test. Do it first.*
2. **"A sim-trained encoder transfers to real echoes."** (Sim-to-real for the
   representation.) — Medium cost; needs paired real recordings + labels.
3. **"The learned world model controls a real agent better than a random one."**
   (The control-positive, in the real world.) — Highest cost; needs a mobile
   rig and closed-loop. Aspirational.

Note what you *cannot* get in the real world: exact per-pose geometric ground
truth for arbitrary scenes. That absence is *why* the sim was used. Real-world
validation therefore substitutes coarser, measurable labels (tape-measured room
dimensions, marked source/receiver positions) and accepts smaller, noisier
datasets. Design around that, don't fight it.

## Tier 0 — reuse existing RIR databases (a weekend, no hardware)

The fastest, strongest first test of claim 1 needs **no recording at all**.
Several public databases ship real room impulse responses *with room dimensions
and source/receiver positions* as metadata. Candidates: the **ACE Challenge**
corpus (RIRs + room volumes + reverberation labels), **BUT Speech@FIT Reverb DB**
(rooms with measured dims), **dEchorate** (multichannel RIRs, known geometry),
and various **BRIR** sets (binaural, known rooms).

Protocol:
1. For each RIR, compute the same two streams the sim uses: the matched-filter
   range profile from the (mono/omni) RIR (port `MatchedFilter` from
   `src/acoustic_env.h`), and — where binaural RIRs exist — the gammatone
   cochleagram.
2. Compute the 3 hand statistics from `jepa/analyze_lowdim.py` (range-profile
   delay centroid, spread, late/early ratio).
3. Ridge-regress room volume (and, where available, source–receiver distance)
   from those 3 stats, room-level split.

**Pass criterion:** if 3 reverberation statistics predict real room volume with
R² in the same ballpark as the sim (~0.6–0.8), the core mechanism is real, not a
Steam Audio artifact. This single result de-risks the whole thesis and costs
only compute. If it *fails* on real RIRs, the sim result is suspect and
everything else waits on understanding why.

## Tier 1 — record a small set yourself (the controlled test)

Minimal rig: one **loudspeaker** (any full-range monitor) + one **omni measurement
microphone** (for the range stream) and, if you want the binaural stream, a
**dummy head or in-ear mics** (record the HRTF/rig you use). A laptop with an
audio interface. Total: consumer/hobbyist budget.

Protocol:
1. Emit a **10 s exponential sine sweep** (Farina) from the speaker; record at the
   mic. Deconvolve to the RIR (standard Farina inversion — you already have the
   inverse filter in `matched_range.py`). The long sweep buys huge noise immunity
   (pulse-compression gain), which is the whole reason to use ESS in a real,
   noisy room.
2. Record in **20–40 configurations**: several real rooms of tape-measured
   dimensions, several source–receiver distances each, positions marked on the
   floor. Log room volume, nearest-wall distance, and material where known.
3. **Calibrate out the rig:** record the speaker→mic response in an anechoic/
   free-field-ish setting (or a large open space) and deconvolve it, so you
   measure the *room*, not the loudspeaker's frequency response. This is the
   single most common real-acoustics mistake — skip it and your "material" and
   "spectrum" features are the speaker's coloration.
4. Run `analyze_lowdim.py` and the ridge/MLP probes (`probe_xview.py` machinery)
   on the real features vs. the sim baselines.

**What to look for:** the *ordering* from the sim — raw pooling and simple
statistics decoding volume well; whether a sim-trained encoder (frozen) decodes
real volume above chance (claim 2, the transfer test). Report R² with the same
room-level-split discipline.

## Tier 2 — closed-loop control on a real agent (aspirational)

To test the control-positive for real, you need an agent that can **emit, listen,
move, and re-observe** — i.e. a small robot (or a person carrying the rig) with a
speaker + mic, in a real room, running the trained world model + latent-MPC (the
`control_mpc.py --closed-loop` structure, with the observe-server replaced by the
real rig). The metric is the same: does it reach the goal (a wall / the room
centre) better than a random-encoder controller and near an analytic one?

This is a real robotics project, not a weekend. Scope it only after Tiers 0–1
confirm the representation transfers at all; a world model that can't decode real
echoes won't control from them.

## Pitfalls specific to this work

- **HRTF mismatch.** The sim uses clean SOFA HRTFs; your rig has its own. The
  *range* stream is HRTF-free (pre-binaural W channel) — lean on it. Treat the
  binaural cochleagram as rig-specific and, for transfer, expect the range
  stream to transfer better than the coch stream.
- **Real rooms aren't shoeboxes.** Furniture, non-parallel walls, and coupling
  break the clean reverberation↔volume relationship. Expect lower R² than sim
  and interpret it as "how low-order is real geometry," not as failure.
- **No direct-path convention.** Steam Audio renders no direct path; real
  recordings *do*. For anything peer/bistatic, window the direct arrival
  explicitly (as the SS2 manifest also flags).
- **Background noise & non-stationarity.** ESS + Farina handles stationary noise;
  it does not handle someone talking mid-sweep. Repeat sweeps and average; reject
  outliers.
- **Ground-truth pose error.** Tape-measured positions have ~cm error; that is
  fine for volume/nearest-wall labels, marginal for fine peer-position labels.
  Don't over-interpret a real peer-position probe at R²~0.1.

## The one-sentence recommendation

Start with **Tier 0** on an existing labelled RIR database this week: if three
reverberation numbers predict real room volume, the central mechanistic claim of
this project is validated on real acoustics for the price of some compute, and
every heavier experiment becomes worth doing. If they don't, that result — not a
new sim run — is the next thing to understand.
