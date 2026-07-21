"""EchoRL M2 smoke test + observation norm-stats collector.

Runs random-agent episodes through EchoLocEnv across several rooms in both action
modes, asserts the env contract holds (shapes, termination, collisions fire, finite
rewards), reports throughput, and accumulates per-channel mean/std for `range` and
`coch` -> saves echorl/norm_stats.npz (used to normalize RL inputs in M3/M4).
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import EchoLocEnv, RANGE_BINS, COCH_SHAPE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class RunningStats:
    """Welford-ish accumulation of per-channel mean/std (channel = flat index)."""
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, np.float64)
        self.M2 = np.zeros(shape, np.float64)

    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)

    def finalize(self):
        var = self.M2 / max(1, self.n - 1)
        return self.mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def run(action_mode, seeds, episodes, rng, rstat, cstat, fails):
    env = EchoLocEnv(seed_pool=seeds, action_mode=action_mode, nobs=3,
                     sim_threads=0, t_max=40, resample_room_every=2, base_seed=0)
    n_steps = n_coll = n_trunc = 0
    returns, lengths = [], []
    t0 = time.time()
    for ep in range(episodes):
        obs, info = env.reset()
        # obs contract
        assert obs["range"].shape == (RANGE_BINS,) and obs["range"].dtype == np.float32
        assert obs["coch"].shape == COCH_SHAPE and obs["coch"].dtype == np.float32
        assert obs["prop"].shape == (3,)
        ret, L = 0.0, 0
        done = False
        while not done:
            a = env.action_space.sample(rng)
            obs, r, term, trunc, info = env.step(a)
            assert np.isfinite(r), f"non-finite reward {r}"
            assert np.all(np.isfinite(obs["range"])) and np.all(np.isfinite(obs["coch"]))
            rstat.update(obs["range"].astype(np.float64))
            cstat.update(obs["coch"].astype(np.float64))
            ret += r; L += 1; n_steps += 1
            if term:
                n_coll += 1
            if term or trunc:
                if trunc:
                    n_trunc += 1
                done = True
        returns.append(ret); lengths.append(L)
    dt = time.time() - t0
    env.close()
    hz = n_steps / dt
    print(f"[{action_mode:10}] {episodes} eps, {n_steps} steps in {dt:.1f}s ({hz:.1f} steps/s) | "
          f"collisions {n_coll}, truncations {n_trunc} | "
          f"mean len {np.mean(lengths):.1f}, mean return {np.mean(returns):+.2f}")
    # smoke assertions: something terminated by collision, something ran to t_max
    if n_coll == 0:
        fails.append(f"{action_mode}: no collisions fired (avoidance task has no failure signal)")
    if n_steps == 0:
        fails.append(f"{action_mode}: zero steps")
    return hz


def main():
    rng = np.random.default_rng(0)
    fails = []
    rstat = RunningStats((RANGE_BINS,))
    cstat = RunningStats(COCH_SHAPE)

    seeds = [11, 22, 33, 44, 55, 66]
    print("=== M2 smoke test: random agent through EchoLocEnv ===")
    run("continuous", seeds, episodes=12, rng=rng, rstat=rstat, cstat=cstat, fails=fails)
    run("discrete", seeds, episodes=12, rng=rng, rstat=rstat, cstat=cstat, fails=fails)

    # NB: norm stats are owned by collect_corpus.py (a proper multi-room pose scatter);
    # here we only sanity-check that observations vary. Do NOT write norm_stats.npz.
    r_mean, r_std = rstat.finalize()
    c_mean, c_std = cstat.finalize()
    print(f"\nobs variability over {rstat.n} episode observations (sanity only, not saved):")
    print(f"  range std[{r_std.min():.2f},{r_std.max():.2f}]  coch std[{c_std.min():.3f},{c_std.max():.3f}]")
    if rstat.n < 20 or (r_std.max() < 1e-6):
        fails.append("observations show no variation across the rollout")

    print()
    if fails:
        print("FAIL:"); [print("  -", f) for f in fails]; sys.exit(1)
    print("M2 SMOKE TEST PASS")


if __name__ == "__main__":
    main()
