"""M1 verification for episode_server.exe (EchoRL).

Checks:
 1. extended handshake parses (dims + obstacle AABBs + openings);
 2. obstacle invariants match acoustic_env.h's generator (inside room, floor-standing,
    half-extents in range, span ear height, non-overlapping, valid material);
 3. Python collision geometry works: obstacle centres flag collision, an open point does not;
 4. obstacles are actually in the ACOUSTIC scene (range profile at a pose differs from the
    same room+seed generated with 0 obstacles).
"""
import subprocess, struct, os, sys
import numpy as np

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools") \
    if False else os.path.join("tools")
EXE = os.path.join("tools", "episode_server.exe")
RANGE_BINS, COCH = 256, 2 * 64 * 48
NC = RANGE_BINS + COCH
EAR_Y = 1.6
AGENT_R = 0.3          # matches apply_action wall margin


class Episode:
    def __init__(self, seed, nobs=3, threads=0):
        self.p = subprocess.Popen([EXE, str(seed), str(nobs), str(threads)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        r = self.p.stdout
        self.Lx, self.Ly, self.Lz = struct.unpack("<3f", r.read(12))
        (n_ob,) = struct.unpack("<i", r.read(4))
        self.obstacles = [struct.unpack("<7f", r.read(28)) for _ in range(n_ob)]  # cx,cy,cz,hx,hy,hz,mat
        (n_op,) = struct.unpack("<i", r.read(4))
        self.openings = [struct.unpack("<6f", r.read(24)) for _ in range(n_op)]   # wall,cx,cy,cz,w,h

    def observe(self, x, z, yaw):
        self.p.stdin.write(struct.pack("<3f", float(x), float(z), float(yaw))); self.p.stdin.flush()
        a = np.frombuffer(self.p.stdout.read(NC * 4), dtype=np.float32)
        return a[:RANGE_BINS].copy(), a[RANGE_BINS:].reshape(2, 64, 48).copy()

    def close(self):
        try:
            self.p.stdin.write(struct.pack("<3f", float("nan"), 0, 0)); self.p.stdin.flush()
        except Exception:
            pass
        self.p.terminate()


def hits_obstacle(x, z, obstacles, r=AGENT_R):
    """2D (x,z) AABB overlap with an agent disc of radius r, gated on the box
    spanning ear height. Obstacles are floor-standing pillars rising past 1.7 m."""
    for (cx, cy, cz, hx, hy, hz, _m) in obstacles:
        if not (cy - hy - 1e-3 <= EAR_Y <= cy + hy + 1e-3):
            continue
        if abs(x - cx) <= hx + r and abs(z - cz) <= hz + r:
            return True
    return False


def hits_wall(x, z, Lx, Lz, m=AGENT_R):
    return not (-Lx/2 + m <= x <= Lx/2 - m and -Lz/2 + m <= z <= Lz/2 - m)


def main():
    fails = []
    SEED = 42
    ep = Episode(SEED, nobs=3)
    print(f"room {ep.Lx:.2f} x {ep.Ly:.2f} x {ep.Lz:.2f} m | "
          f"{len(ep.obstacles)} obstacle(s), {len(ep.openings)} opening(s)")

    # --- (1)+(2) handshake + obstacle invariants ---
    assert len(ep.obstacles) >= 1, "expected obstacles with nobs=3 in a room this size"
    x0, x1 = -ep.Lx/2, ep.Lx/2
    z0, z1 = -ep.Lz/2, ep.Lz/2
    for i, (cx, cy, cz, hx, hy, hz, mat) in enumerate(ep.obstacles):
        top = cy + hy
        print(f"  obs[{i}] c=({cx:+.2f},{cy:.2f},{cz:+.2f}) half=({hx:.2f},{hy:.2f},{hz:.2f}) "
              f"top={top:.2f} mat={int(mat)}")
        if not (x0 + 0.6 - hx - 1e-2 <= cx <= x1 - 0.6 + hx + 1e-2): fails.append(f"obs{i} x out of room")
        if not (z0 + 0.6 - hz - 1e-2 <= cz <= z1 - 0.6 + hz + 1e-2): fails.append(f"obs{i} z out of room")
        if not (0.19 <= hx <= 0.61 and 0.19 <= hz <= 0.61): fails.append(f"obs{i} half-extent x/z out of [0.2,0.6]")
        if abs(cy - hy) > 1e-2: fails.append(f"obs{i} not floor-standing (cy-hy={cy-hy:.3f})")
        if not (1.69 <= top <= 2.41): fails.append(f"obs{i} top {top:.2f} not in [1.7,2.4]")
        if not (0 <= int(mat) <= 3): fails.append(f"obs{i} bad material {mat}")
    for i in range(len(ep.obstacles)):
        for j in range(i + 1, len(ep.obstacles)):
            a, b = ep.obstacles[i], ep.obstacles[j]
            if abs(a[0]-b[0]) < a[3]+b[3] and abs(a[2]-b[2]) < a[5]+b[5]:
                fails.append(f"obs{i},{j} overlap in x/z")

    # --- (3) collision geometry ---
    for i, ob in enumerate(ep.obstacles):
        if not hits_obstacle(ob[0], ob[2], ep.obstacles):
            fails.append(f"obs{i} centre not flagged as collision")
    # find a provably-open point: scan a grid, pick one clear of walls+obstacles
    open_pt = None
    for gx in np.linspace(x0 + 1, x1 - 1, 9):
        for gz in np.linspace(z0 + 1, z1 - 1, 9):
            if not hits_wall(gx, gz, ep.Lx, ep.Lz) and not hits_obstacle(gx, gz, ep.obstacles):
                open_pt = (gx, gz); break
        if open_pt: break
    assert open_pt, "no open point found (room fully blocked?)"
    if hits_obstacle(*open_pt, ep.obstacles): fails.append("open point wrongly flagged")
    print(f"  collision: all {len(ep.obstacles)} obstacle centres flag hit; "
          f"open point {tuple(round(v,2) for v in open_pt)} clear")

    # a point just outside vs inside an obstacle edge (radius-aware boundary)
    cx, cy, cz, hx, hy, hz, _ = ep.obstacles[0]
    inside_edge  = (cx + hx + AGENT_R - 0.05, cz)
    outside_edge = (cx + hx + AGENT_R + 0.10, cz)
    if not hits_obstacle(*inside_edge, ep.obstacles):  fails.append("inside-edge point missed")
    if hits_obstacle(*outside_edge, ep.obstacles):     fails.append("outside-edge point false-positive")

    # --- (4) obstacles are in the ACOUSTIC scene: diff vs 0-obstacle same seed ---
    rng_obs, _ = ep.observe(*open_pt, 0.0)
    ep.close()
    ep0 = Episode(SEED, nobs=0)
    assert len(ep0.obstacles) == 0, "nobs=0 should yield no obstacles"
    # observe at the SAME world point (open_pt is valid in both; room dims identical for same seed)
    assert abs(ep0.Lx - ep.Lx) < 1e-4 and abs(ep0.Lz - ep.Lz) < 1e-4, "same seed must give same room dims"
    rng_no, _ = ep0.observe(*open_pt, 0.0)
    ep0.close()
    d = float(np.linalg.norm(rng_obs - rng_no))
    rel = d / (np.linalg.norm(rng_no) + 1e-9)
    print(f"  range-profile L2 diff (3 obstacles vs 0, same seed/pose) = {d:.3f} (rel {rel:.3f})")
    if rel < 0.02:
        fails.append(f"obstacles barely change the echo (rel {rel:.3f}) -- not in acoustic scene?")

    print()
    if fails:
        print("FAIL:"); [print("  -", f) for f in fails]; sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
