"""Threaded vector env for EchoLocEnv (EchoRL M3).

The C sim BLOCKS on each observe() (~50-170 ms at low thread counts), releasing the
GIL during the pipe read -- so K sub-envs must be stepped CONCURRENTLY via a thread
pool to reach the ~90 obs/s aggregate the benchmark showed. Sequential stepping
would give only ~10 obs/s. Autoreset on done (gym-vector convention): the returned
obs is the RESET obs for finished envs, with the true final obs in `terminal_obs`.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from env import EchoLocEnv


def _stack(obs_list):
    return dict(
        range=np.stack([o["range"] for o in obs_list]),
        coch=np.stack([o["coch"] for o in obs_list]),
        prop=np.stack([o["prop"] for o in obs_list]),
    )


class VecEchoLocEnv:
    def __init__(self, seed_shards, action_mode="discrete", nobs=3, sim_threads=2,
                 t_max=40, resample_room_every=3, start_margin=0.8, task="avoid", goal_radius=1.2):
        self.envs = [
            EchoLocEnv(seed_pool=shard, action_mode=action_mode, nobs=nobs,
                       sim_threads=sim_threads, t_max=t_max,
                       resample_room_every=resample_room_every,
                       start_margin=start_margin, base_seed=i,
                       task=task, goal_radius=goal_radius)
            for i, shard in enumerate(seed_shards)
        ]
        self.n = len(self.envs)
        self.pool = ThreadPoolExecutor(max_workers=self.n)

    def reset(self):
        res = list(self.pool.map(lambda e: e.reset(), self.envs))
        return _stack([r[0] for r in res])

    def step(self, actions):
        def _one(i):
            o, r, term, trunc, info = self.envs[i].step(actions[i])
            done = term or trunc
            term_obs = None
            if done:
                term_obs = o
                o, _ = self.envs[i].reset()      # autoreset
            return i, o, r, term, trunc, done, term_obs, info

        res = sorted(self.pool.map(_one, range(self.n)), key=lambda t: t[0])
        obs = _stack([t[1] for t in res])
        rew = np.array([t[2] for t in res], np.float32)
        term = np.array([t[3] for t in res], bool)
        trunc = np.array([t[4] for t in res], bool)
        done = np.array([t[5] for t in res], bool)
        term_obs = [t[6] for t in res]           # None where not done
        infos = [t[7] for t in res]              # per-env step info (collision/success/...)
        return obs, rew, term, trunc, done, term_obs, infos

    def close(self):
        for e in self.envs:
            e.close()
        self.pool.shutdown(wait=False)
