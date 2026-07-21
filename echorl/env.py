"""EchoLocEnv — closed-loop echolocation RL environment (EchoRL M2).

A Gymnasium-shaped (but dependency-free) environment wrapping episode_server.exe.
The C server is a pure pose->(range,coch) echo sensor; Python owns pose dynamics,
collision, and reward, using the collision geometry streamed in the handshake.

One env instance == one procedural room (one server process, seed baked at spawn).
Room diversity comes from running MANY envs at different seeds (that IS the
vectorization, and the per-process design the throughput benchmark recommends:
~12 envs x 2 threads ~= 90 obs/s aggregate on a 24-core box). A single env can also
rotate rooms via `resample_room_every`.

API (Gymnasium v0.26+ shape):
    obs, info                 = env.reset(seed=None)
    obs, r, term, trunc, info = env.step(action)
obs = {'range':(256,), 'coch':(2,64,48), 'prop':(3,)} float32.
action: continuous np.array([dyaw_norm, dstep_norm]) in [-1,1], OR discrete int in 0..4.
"""
from __future__ import annotations
import math
import os
import struct
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(_HERE, "..", "tools", "episode_server.exe")

RANGE_BINS, COCH_SHAPE = 256, (2, 64, 48)
COCH_N = int(np.prod(COCH_SHAPE))
NC = RANGE_BINS + COCH_N
EAR_Y = 1.6
AGENT_R = 0.3           # agent disc radius (matches apply_action wall margin)
DYAW_MAX, DSTEP_MAX = 45.0, 1.5
DISCRETE_DYAW = (-45.0, -22.5, 0.0, 22.5, 45.0)
DISCRETE_DSTEP = 0.75


# --------------------------------------------------------------------- server
class EpisodeServer:
    """One episode_server.exe process == one room. Parses the extended handshake."""

    def __init__(self, seed, nobs=3, threads=0):
        self.seed, self.nobs, self.threads = seed, nobs, threads
        self.p = subprocess.Popen([EXE, str(seed), str(nobs), str(threads)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        r = self.p.stdout
        self.Lx, self.Ly, self.Lz = struct.unpack("<3f", r.read(12))
        (n_ob,) = struct.unpack("<i", r.read(4))
        self.obstacles = [struct.unpack("<7f", r.read(28)) for _ in range(n_ob)]  # cx,cy,cz,hx,hy,hz,mat
        (n_op,) = struct.unpack("<i", r.read(4))
        self.openings = [struct.unpack("<6f", r.read(24)) for _ in range(n_op)]   # wall,cx,cy,cz,w,h

    def observe(self, x, z, yaw):
        self.p.stdin.write(struct.pack("<3f", float(x), float(z), float(yaw))); self.p.stdin.flush()
        buf = self.p.stdout.read(NC * 4)
        if len(buf) < NC * 4:                          # short read == server exited
            code = self.p.poll()
            err = b""
            try:
                err = self.p.stderr.read() if self.p.stderr else b""
            except Exception:
                pass
            raise RuntimeError(f"episode_server(seed={self.seed}) died: got {len(buf)}/{NC*4} bytes, "
                               f"exit={code}, req=({x:.3f},{z:.3f},{yaw:.3f}), stderr={err[:400]!r}")
        a = np.frombuffer(buf, dtype=np.float32)
        return a[:RANGE_BINS].copy(), a[RANGE_BINS:].reshape(COCH_SHAPE).copy()

    def close(self):
        try:
            self.p.stdin.write(struct.pack("<3f", float("nan"), 0, 0)); self.p.stdin.flush()
        except Exception:
            pass
        try:
            self.p.terminate()
        except Exception:
            pass


# ------------------------------------------------------------- collision geom
def _obstacle_clearance(x, z, obstacles):
    """Min gap (m) from point (x,z) to any ear-height obstacle surface; +inf if none.
    Negative if inside an obstacle. Uses 2D AABB distance."""
    best = math.inf
    for (cx, cy, cz, hx, hy, hz, _m) in obstacles:
        if not (cy - hy - 1e-3 <= EAR_Y <= cy + hy + 1e-3):
            continue
        dx = abs(x - cx) - hx
        dz = abs(z - cz) - hz
        if dx <= 0 and dz <= 0:
            d = max(dx, dz)                 # inside: negative, distance to nearest face
        else:
            d = math.hypot(max(dx, 0.0), max(dz, 0.0))
        best = min(best, d)
    return best


def _wall_clearance(x, z, Lx, Lz):
    return min(Lx / 2 - x, x + Lx / 2, Lz / 2 - z, z + Lz / 2)


def _hits(x, z, srv):
    """True if the agent disc at (x,z) intersects a wall or obstacle."""
    if _wall_clearance(x, z, srv.Lx, srv.Lz) <= AGENT_R:
        return True
    return _obstacle_clearance(x, z, srv.obstacles) <= AGENT_R


def _swept_collision(x0, z0, x1, z1, srv, n=8):
    """Sample the segment (x0,z0)->(x1,z1) so a step can't tunnel through a thin obstacle."""
    for k in range(1, n + 1):
        t = k / n
        if _hits(x0 + (x1 - x0) * t, z0 + (z1 - z0) * t, srv):
            return True
    return False


# ---------------------------------------------------------------- spaces (min)
class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low, self.high, self.shape, self.dtype = low, high, tuple(shape), dtype

    def sample(self, rng):
        return rng.uniform(self.low, self.high, size=self.shape).astype(self.dtype)


class Discrete:
    def __init__(self, n):
        self.n = n

    def sample(self, rng):
        return int(rng.integers(self.n))


# ------------------------------------------------------------------------ env
class EchoLocEnv:
    def __init__(self, seed_pool=(1,), action_mode="continuous", nobs=3, sim_threads=0,
                 t_max=60, start_margin=0.8, resample_room_every=0, reward_cfg=None, base_seed=0,
                 task="avoid", goal_radius=1.2):
        assert action_mode in ("continuous", "discrete")
        assert task in ("avoid", "goal")
        self.task, self.goal_radius = task, goal_radius   # "goal": home to the room's opening
        self.seed_pool = list(seed_pool)
        self.action_mode = action_mode
        self.nobs, self.sim_threads = nobs, sim_threads
        self.t_max, self.start_margin = t_max, start_margin
        self.resample_room_every = resample_room_every
        self.rng = np.random.default_rng(base_seed)
        self._room_ptr = 0
        self._ep_count = 0
        self.srv = None

        # Stage-0 reward: SURVIVE + stay clear. `w_clear*tanh((clear-d_safe)/scale)` is a
        # dense, anticipatory signal (rewards open space, penalizes approach over a WIDE band,
        # not just at the last 0.6 m) -> gives the discrete agent a real per-action gradient,
        # unlike the old fixed-step progress term which was action-independent. `w_prog` is kept
        # small (a mild fly-forward incentive that matters only for the continuous/PPO agent).
        self.reward = dict(w_prog=0.02, w_coll=5.0, alive=0.05, w_clear=0.15, d_safe=0.8, scale=0.5,
                           # Stage-2 goal-approach: dense progress toward the opening + a reach bonus,
                           # a small step cost for efficiency, and a light clearance term so it still
                           # avoids obstacles/walls en route (small, so it doesn't fight the final
                           # approach, which necessarily nears the wall the opening is in).
                           # w_success >> w_coll and a real step cost so "reach the goal" beats the
                           # risk-averse local optimum of safely wandering out the clock (seen at
                           # w_success=5/step_cost=0.02: success collapsed 0.57->0.17, collision->0).
                           # w_dist: a DENSE per-step penalty for distance-to-goal so "wander safely
                           # far away" is not reward-neutral (potential-based progress alone is
                           # zero-sum over a closed path, which let the agent hedge into risk-aversion).
                           # goal mode uses NON-TERMINAL collision: a bump costs w_bump and blocks
                           # the move (not episode-ending). w_dist is a small dense pull to the goal.
                           w_goal=1.0, w_success=10.0, w_bump=0.3, w_dist=0.01)
        if reward_cfg:
            self.reward.update(reward_cfg)

        self.observation_space = dict(
            range=Box(0.0, np.inf, (RANGE_BINS,)),
            coch=Box(0.0, np.inf, COCH_SHAPE),
            prop=Box(-1.0, 1.0, (3,)),
        )
        self.action_space = Box(-1.0, 1.0, (2,)) if action_mode == "continuous" else Discrete(5)

    # -- room lifecycle --
    def _spawn_room(self):
        if self.srv is not None:
            self.srv.close()
        # goal mode needs a room WITH an opening (the target); skip openingless seeds,
        # but always keep the LAST spawned server open (fallback goal = +x wall centre).
        tries = max(1, len(self.seed_pool) * 3) if self.task == "goal" else 1
        for k in range(tries):
            seed = self.seed_pool[self._room_ptr % len(self.seed_pool)]
            self._room_ptr += 1
            srv = EpisodeServer(seed, self.nobs, self.sim_threads)
            if self.task != "goal" or len(srv.openings) > 0 or k == tries - 1:
                self.srv = srv
                return
            srv.close()

    def _goal_pos(self):
        """A point ~1.5 m IN FRONT of the largest opening (interior side), so reaching it
        doesn't require touching the wall the doorway is in; the acoustic void toward the
        opening still points there. Fallback: room centre. Wall ids: 0:+x 1:-x 4:+z 5:-z."""
        ops = self.srv.openings                        # record = (wall, cx, cy, cz, w, h)
        if ops:
            o = max(ops, key=lambda r: r[4] * r[5])
            wall, cx, cz, off = int(o[0]), float(o[1]), float(o[3]), 1.5
            if wall == 0: return cx - off, cz
            if wall == 1: return cx + off, cz
            if wall == 4: return cx, cz - off
            if wall == 5: return cx, cz + off
        return 0.0, 0.0

    def _sample_start(self):
        s = self.srv
        for _ in range(200):
            x = self.rng.uniform(-s.Lx / 2 + self.start_margin, s.Lx / 2 - self.start_margin)
            z = self.rng.uniform(-s.Lz / 2 + self.start_margin, s.Lz / 2 - self.start_margin)
            if _obstacle_clearance(x, z, s.obstacles) > self.start_margin:
                return float(x), float(z), float(self.rng.uniform(0, 360))
        return 0.0, 0.0, float(self.rng.uniform(0, 360))   # fallback (dense room)

    # -- action decode --
    def _decode(self, a):
        if self.action_mode == "continuous":
            # nan_to_num FIRST: a NaN action (early policy instability) would otherwise reach the
            # server as a NaN pose, which the server reads as its quit sentinel and exits.
            a = np.nan_to_num(np.asarray(a, dtype=np.float32).reshape(2), nan=0.0)
            dyaw = float(np.clip(a[0], -1, 1)) * DYAW_MAX
            dstep = (float(np.clip(a[1], -1, 1)) + 1) * 0.5 * DSTEP_MAX
        else:
            dyaw = DISCRETE_DYAW[int(a)]
            dstep = DISCRETE_DSTEP
        return dyaw, dstep

    def _observe_safe(self, x, z, yaw):
        """episode_server occasionally segfaults (rare, geometry-independent). Recover by
        respawning the SAME room (deterministic seed -> identical geometry) and retrying, so a
        stray crash doesn't kill a multi-hour training run. ~1% render jitter makes this seamless."""
        for _ in range(3):
            try:
                return self.srv.observe(x, z, yaw)
            except RuntimeError:
                seed = self.srv.seed
                try:
                    self.srv.close()
                except Exception:
                    pass
                self.srv = EpisodeServer(seed, self.nobs, self.sim_threads)
        return (np.zeros(RANGE_BINS, np.float32), np.zeros(COCH_SHAPE, np.float32))

    def _obs(self, x, z, yaw, last_dyaw, last_dstep):
        rng, coch = self._observe_safe(x, z, yaw)
        prop = np.array([last_dyaw / DYAW_MAX,
                         last_dstep / DSTEP_MAX * 2 - 1,
                         self._t / self.t_max * 2 - 1], dtype=np.float32)
        return dict(range=rng, coch=coch, prop=prop)

    # -- gym API --
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.srv is None or (self.resample_room_every and
                                self._ep_count % self.resample_room_every == 0):
            self._spawn_room()
        self._ep_count += 1
        self._t = 0
        if self.task == "goal":
            self.goal = self._goal_pos()
        self.x, self.z, self.yaw = self._sample_start()
        if self.task == "goal":
            far = max(2.5, self.goal_radius + 1.5)      # start away from the goal
            for _ in range(30):
                if math.hypot(self.x - self.goal[0], self.z - self.goal[1]) > far:
                    break
                self.x, self.z, self.yaw = self._sample_start()
            self._prev_d = math.hypot(self.x - self.goal[0], self.z - self.goal[1])
        self._prev_clear = min(_wall_clearance(self.x, self.z, self.srv.Lx, self.srv.Lz),
                               _obstacle_clearance(self.x, self.z, self.srv.obstacles))
        obs = self._obs(self.x, self.z, self.yaw, 0.0, 0.0)
        info = dict(room=(self.srv.Lx, self.srv.Ly, self.srv.Lz),
                    seed=self.srv.seed, pose=(self.x, self.z, self.yaw))
        if self.task == "goal":
            info["goal"] = self.goal
        return obs, info

    def step(self, action):
        R = self.reward
        dyaw, dstep = self._decode(action)
        t = math.radians(self.yaw)
        nx = self.x + dstep * math.sin(t)
        nz = self.z + dstep * (-math.cos(t))

        collided = _swept_collision(self.x, self.z, nx, nz, self.srv)
        self._t += 1

        if self.task == "goal":
            # NON-TERMINAL collision: reaching is the objective, so a wall bump is a cost, not a
            # catastrophe. Terminal collision made any near-wall goal approach too risky, so the
            # agent hedged into safe wandering; here a bump just blocks the translation (agent still
            # turns) and costs w_bump, and the episode continues until success or timeout.
            if collided:
                self.yaw += dyaw
                bump_pen = R["w_bump"]
            else:
                self.x, self.z, self.yaw = nx, nz, self.yaw + dyaw
                bump_pen = 0.0
            d = math.hypot(self.x - self.goal[0], self.z - self.goal[1])
            success = d < self.goal_radius
            reward = R["w_goal"] * (self._prev_d - d) - R["w_dist"] * d - bump_pen
            if success:
                reward += R["w_success"]
            self._prev_d = d
            obs = self._obs(self.x, self.z, self.yaw, dyaw, dstep)
            info = dict(collision=collided, success=success, dist=d, pose=(self.x, self.z, self.yaw))
            return obs, reward, bool(success), self._t >= self.t_max, info

        # ---- avoid mode (Stage 0): TERMINAL collision ----
        if collided:
            reward = -R["w_coll"]
            obs = self._obs(self.x, self.z, self.yaw, dyaw, dstep)
            info = dict(collision=True, success=False, pose=(self.x, self.z, self.yaw))
            return obs, reward, True, False, info

        self.x, self.z, self.yaw = nx, nz, self.yaw + dyaw
        clear = min(_wall_clearance(self.x, self.z, self.srv.Lx, self.srv.Lz),
                    _obstacle_clearance(self.x, self.z, self.srv.obstacles))
        self._prev_clear = clear
        shaped = math.tanh((clear - R["d_safe"]) / R["scale"])   # + in open space, - near a surface
        reward = R["w_prog"] * dstep + R["alive"] + R["w_clear"] * shaped
        terminated = False
        truncated = self._t >= self.t_max
        obs = self._obs(self.x, self.z, self.yaw, dyaw, dstep)
        info = dict(collision=False, success=False, clearance=clear, pose=(self.x, self.z, self.yaw))
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.srv is not None:
            self.srv.close(); self.srv = None
