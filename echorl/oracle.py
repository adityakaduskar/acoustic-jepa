"""Planning oracle baseline (EchoRL) — the performance ceiling.

Privileged-geometry controller using ground-truth room dims + obstacle AABBs (from
the server handshake) and the analytic pose dynamics, so an entire episode runs
WITHOUT the sim (no echoes) — very fast. Receding-horizon random-shooting: sample K
action sequences of horizon H over the true dynamics (VECTORIZED over K with numpy),
keep the one that survives longest (tie-break: final clearance), execute its first
action. This is the near-optimal "if you could SEE the geometry" ceiling, and it
tests the 'some rooms are genuinely unsurvivable' claim: if the planner also dies in
a room, no echo-only policy could have done better there.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import EpisodeServer, AGENT_R, DYAW_MAX, DSTEP_MAX, EAR_Y   # noqa: E402


class Geom:
    def __init__(self, srv):
        self.Lx, self.Lz = srv.Lx, srv.Lz
        # obstacles spanning ear height, as arrays: centre (cx,cz), half (hx,hz)
        obs = [(o[0], o[2], o[3], o[5]) for o in srv.obstacles
               if o[1] - o[4] - 1e-3 <= EAR_Y <= o[1] + o[4] + 1e-3]
        a = np.array(obs, np.float64).reshape(-1, 4)
        self.ocx, self.ocz, self.ohx, self.ohz = a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def hits_vec(x, z, g):
    """Boolean (K,) — agent disc at (x,z) intersects a wall or obstacle."""
    hit = (x <= -g.Lx/2 + AGENT_R) | (x >= g.Lx/2 - AGENT_R) | \
          (z <= -g.Lz/2 + AGENT_R) | (z >= g.Lz/2 - AGENT_R)
    for k in range(len(g.ocx)):
        hit |= (np.abs(x - g.ocx[k]) <= g.ohx[k] + AGENT_R) & (np.abs(z - g.ocz[k]) <= g.ohz[k] + AGENT_R)
    return hit


def swept_vec(x0, z0, x1, z1, g, n=6):
    hit = np.zeros_like(x0, dtype=bool)
    for i in range(1, n + 1):
        t = i / n
        hit |= hits_vec(x0 + (x1 - x0) * t, z0 + (z1 - z0) * t, g)
    return hit


def clearance_vec(x, z, g):
    c = np.minimum(np.minimum(g.Lx/2 - x, x + g.Lx/2), np.minimum(g.Lz/2 - z, z + g.Lz/2))
    for k in range(len(g.ocx)):
        dx = np.abs(x - g.ocx[k]) - g.ohx[k]; dz = np.abs(z - g.ocz[k]) - g.ohz[k]
        inside = (dx <= 0) & (dz <= 0)
        d = np.where(inside, np.maximum(dx, dz), np.hypot(np.maximum(dx, 0), np.maximum(dz, 0)))
        c = np.minimum(c, d)
    return c


def plan_action(x, z, yaw, g, rng, K=256, H=10):
    dyaw = rng.uniform(-DYAW_MAX, DYAW_MAX, (K, H))
    dstep = rng.uniform(0, DSTEP_MAX, (K, H))
    px = np.full(K, x); pz = np.full(K, z); pyaw = np.full(K, yaw)
    alive = np.ones(K, bool); survived = np.zeros(K)
    for h in range(H):
        t = np.radians(pyaw)
        nx = px + dstep[:, h] * np.sin(t); nz = pz - dstep[:, h] * np.cos(t)
        coll = swept_vec(px, pz, nx, nz, g)
        step_ok = alive & ~coll
        survived += step_ok
        px = np.where(step_ok, nx, px); pz = np.where(step_ok, nz, pz)
        pyaw = np.where(step_ok, pyaw + dyaw[:, h], pyaw)
        alive &= ~coll
    score = survived + clearance_vec(px, pz, g) / 100.0
    b = int(score.argmax())
    return dyaw[b, 0], dstep[b, 0]


def run(nobs, seeds, rng, episodes_per=8, t_max=40, start_margin=0.8):
    lengths, colls = [], []
    for s in seeds:
        srv = EpisodeServer(s, nobs, threads=0)
        g = Geom(srv); srv.close()                    # geometry only; no echoes needed
        for _ in range(episodes_per):
            for _ in range(200):
                x = rng.uniform(-g.Lx/2 + start_margin, g.Lx/2 - start_margin)
                z = rng.uniform(-g.Lz/2 + start_margin, g.Lz/2 - start_margin)
                if clearance_vec(np.array([x]), np.array([z]), g)[0] > start_margin:
                    break
            yaw = rng.uniform(0, 360)
            L, crashed = 0, False
            for _ in range(t_max):
                dyaw, dstep = plan_action(x, z, yaw, g, rng)
                t = np.radians(yaw); nx = x + dstep*np.sin(t); nz = z - dstep*np.cos(t)
                L += 1
                if bool(swept_vec(np.array([x]), np.array([z]), np.array([nx]), np.array([nz]), g)[0]):
                    crashed = True; break
                x, z, yaw = nx, nz, yaw + dyaw
            lengths.append(L); colls.append(1.0 if crashed else 0.0)
    return float(np.mean(lengths)), float(np.mean(colls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--episodes-per", type=int, default=8)
    args = ap.parse_args()
    seeds = list(range(600, 600 + args.eval_rooms))
    rng = np.random.default_rng(1)
    print("Planning oracle (privileged geometry, receding-horizon) — held-out rooms 600+, "
          "mean flight len / 40, collision:")
    for nobs in (0, 1, 3):
        ml, cr = run(nobs, seeds, rng, args.episodes_per)
        print(f"  nobs={nobs}:  len {ml:5.1f}   collision {cr:.2f}")


if __name__ == "__main__":
    main()
