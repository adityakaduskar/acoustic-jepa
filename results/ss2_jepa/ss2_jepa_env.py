#!/usr/bin/env python3
"""
ss2_jepa_env.py — SoundSpaces 2.0 observation operator for the Acoustic-JEPA
replication (see ss2_jepa_manifest.md).

Ports the engine-specific half of the Steam Audio rig to habitat-sim 0.2.2 +
RLR audio propagation:

  acoustic_env.h::make_box_room      -> make_box_room()
  acoustic_env.h::build_room_mesh    -> build_room_mesh()
  acoustic_env.h::analytic_label     -> analytic_label()
  acoustic_env.h::MatchedFilter      -> MatchedFilter      (verbatim algorithm)
  acoustic_env.h::Cochleagram        -> Cochleagram        (verbatim algorithm)
  acoustic_env.h::AcousticEnv        -> SS2Env             (RLR instead of Steam Audio)
  swarm_env.h::SwarmEnv::measure_tab -> SS2Env.measure_tab (real SS2 direct path)

Everything downstream (jepa/*.py) is engine-agnostic and consumes the .npz
schema unchanged.

Engine deviations from the Steam Audio rig, all deliberate (see NOTES in
SS2_JEPA_RESULTS.md):
  * SS2 renders a REAL direct path. Reflection-only observations therefore set
    the RLR `direct` flag OFF; a separate direct-only sensor supplies the
    arrival used for the measured transform T_meas.
  * Ambisonic order 1 (RLR's binaural decode is native, so the binaural stream
    comes straight from a Binaural-layout sensor rather than an order-3
    ambisonic decode).
  * RLR `transmission` flag OFF: the SS2 characterization battery (E11) showed
    the transmission path injects a late diffuse smear rather than a physical
    through-wall arrival. Steam Audio's transmission was inert for reflections,
    so OFF is the faithful match.
  * RLR `diffraction` flag OFF: it only bends the direct path, and only when
    meshSimplification is also on. Steam Audio rendered no edge diffraction.
"""
import os, sys, json, math

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np
import habitat_sim
from scipy.signal import lfilter, butter, filtfilt

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from ss2_collect import SceneBuilder, write_materials, curve  # noqa: E402

# --------------------------------------------------------------- render config
SR = 48000
IR_DUR = 0.5
IRN = int(IR_DUR * SR)          # 24000
FRAME = 1024                    # Steam Audio's block size; kept only so the
TOTAL = IRN + FRAME             # 25024  matched-filter index base is identical
C_NOM = 343.0                   # nominal c used for labels/geometry (as Steam Audio)
AMBI_ORDER = 1
RAYS = 5000
RAY_DEPTH = 200

# ------------------------------------------------------- observation dims (§0)
T_OBS = 0.2
RAW_N = int(T_OBS * SR)         # 9600 samples/ear
COCH_F, COCH_T = 64, 48
COCH_HOP = RAW_N // COCH_T      # 200
RANGE_BINS, RANGE_POOL, RANGE_GUARD = 256, 16, 96
EAR_H = 1.6

_ITD_LP_B, _ITD_LP_A = butter(4, 800.0 / (SR / 2), btype="low")


def _itd_lowpass(x):
    """Zero-phase 800 Hz low-pass, applied before interaural cross-correlation."""
    return filtfilt(_ITD_LP_B, _ITD_LP_A, x) if len(x) > 30 else x

# ===========================================================================
#  reproducible RNG — xorshift32, ported verbatim from acoustic_env.h::Rng so
#  a room is the same pure function of its seed on both engines.
# ===========================================================================
class Rng:
    def __init__(self, seed):
        self.s = np.uint32(seed if seed else 0x9E3779B9)

    def uf(self):
        s = np.uint32(self.s)
        s ^= np.uint32(s << np.uint32(13))
        s ^= np.uint32(s >> np.uint32(17))
        s ^= np.uint32(s << np.uint32(5))
        self.s = s
        return float(int(s) & 0xFFFFFF) / float(0x1000000)

    def rf(self, a, b):
        return a + (b - a) * self.uf()

    def ri(self, a, b):                      # inclusive
        return a + int(self.uf() * (b - a + 1))


# ===========================================================================
#  Materials — the 4-material table of acoustic_env.h::material_table(),
#  mapped onto RLR's (frequency, value) breakpoint curves.
#
#  Steam Audio absorption[3] uses nominal crossovers at 800 Hz and 8 kHz, so
#  each band value is held constant over its band. RLR binds a material by
#  matching the semantic CLASS NAME against the material's "name" field, so
#  every material is named after the surface label it is applied to.
# ===========================================================================
MAT_NAMES = ["plaster", "tile", "carpet", "mirror"]
MAT_ABS = [[0.10, 0.15, 0.20], [0.01, 0.01, 0.02], [0.30, 0.50, 0.60], [0.0, 0.0, 0.0]]
MAT_SCAT = [0.20, 0.10, 0.50, 0.00]
XOVER = (800.0, 8000.0)


def band3_curve(vals):
    """Piecewise-constant breakpoint curve over Steam Audio's 3 bands."""
    lo, hi = XOVER
    return [20.0, vals[0], lo - 1.0, vals[0], lo + 1.0, vals[1],
            hi - 1.0, vals[1], hi + 1.0, vals[2], 20000.0, vals[2]]


def material_specs():
    return [dict(name=MAT_NAMES[i], labels=[MAT_NAMES[i]],
                 absorption=band3_curve(MAT_ABS[i]),
                 scattering=curve(MAT_SCAT[i]),
                 transmission=curve(0.0))
            for i in range(4)]


# ===========================================================================
#  Procedural box room  (acoustic_env.h::make_box_room, verbatim port)
#  wall index convention: 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z
# ===========================================================================
class BoxRoom:
    __slots__ = ("Lx", "Ly", "Lz", "origin", "wall_material", "openings", "obstacles")


def make_box_room(seed, n_obstacles=0):
    r = Rng(seed)
    rm = BoxRoom()
    rm.Lx = r.rf(3.0, 12.0)
    rm.Ly = r.rf(2.5, 4.0)
    rm.Lz = r.rf(3.0, 12.0)
    rm.origin = np.array([-rm.Lx * 0.5, 0.0, -rm.Lz * 0.5])
    rm.wall_material = [r.ri(0, 3) for _ in range(6)]
    rm.openings = []          # dicts: wall, center(3), w, h
    rm.obstacles = []         # dicts: center(3), half(3), material

    x0, x1 = rm.origin[0], rm.origin[0] + rm.Lx
    y0, y1 = rm.origin[1], rm.origin[1] + rm.Ly
    z0, z1 = rm.origin[2], rm.origin[2] + rm.Lz
    MARGIN = 0.4

    side = [0, 1, 4, 5]
    for i in range(3, 0, -1):                              # Fisher-Yates
        j = r.ri(0, i)
        side[i], side[j] = side[j], side[i]
    n_open = r.ri(0, 3)
    for k in range(n_open):
        wall = side[k]
        wall_len_u = rm.Lz if wall in (0, 1) else rm.Lx
        w = min(r.rf(0.8, 2.0), wall_len_u - 2 * MARGIN)
        h = min(r.rf(1.0, 2.0), rm.Ly - 2 * MARGIN)
        if w <= 0.2 or h <= 0.2:
            continue
        cy = r.rf(y0 + MARGIN + h * 0.5, y1 - MARGIN - h * 0.5)
        if wall in (0, 1):
            lo, hi = z0 + MARGIN + w * 0.5, z1 - MARGIN - w * 0.5
        else:
            lo, hi = x0 + MARGIN + w * 0.5, x1 - MARGIN - w * 0.5
        cu = r.rf(lo, hi)
        if wall == 0:
            c = [x1, cy, cu]
        elif wall == 1:
            c = [x0, cy, cu]
        elif wall == 4:
            c = [cu, cy, z1]
        else:
            c = [cu, cy, z0]
        rm.openings.append(dict(wall=wall, center=np.array(c), w=w, h=h))

    OB_CLEAR = 0.6
    for _ in range(n_obstacles):
        placed = False
        for _try in range(60):
            if placed:
                break
            hx, hz = r.rf(0.2, 0.6), r.rf(0.2, 0.6)
            top = min(rm.Ly - 0.1, r.rf(1.7, 2.4))
            hy = top * 0.5
            cyv = y0 + hy
            cx = r.rf(x0 + OB_CLEAR + hx, x1 - OB_CLEAR - hx)
            cz = r.rf(z0 + OB_CLEAR + hz, z1 - OB_CLEAR - hz)
            if x1 - OB_CLEAR - hx <= x0 + OB_CLEAR + hx:
                break
            if z1 - OB_CLEAR - hz <= z0 + OB_CLEAR + hz:
                break
            c = np.array([cx, cyv, cz]); h = np.array([hx, hy, hz])
            overlaps = False
            for p in rm.obstacles:
                if (abs(c[0] - p["center"][0]) < h[0] + p["half"][0] + 0.3 and
                        abs(c[1] - p["center"][1]) < h[1] + p["half"][1] + 0.3 and
                        abs(c[2] - p["center"][2]) < h[2] + p["half"][2] + 0.3):
                    overlaps = True
                    break
            if overlaps:
                continue
            rm.obstacles.append(dict(center=c, half=h, material=r.ri(0, 3)))
            placed = True
    return rm


def segment_hits_aabb(a, b, c, h):
    tmin, tmax = 0.0, 1.0
    d = b - a
    lo, hi = c - h, c + h
    for k in range(3):
        if abs(d[k]) < 1e-9:
            if a[k] < lo[k] or a[k] > hi[k]:
                return False
            continue
        inv = 1.0 / d[k]
        t1, t2 = (lo[k] - a[k]) * inv, (hi[k] - a[k]) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return False
    return True


def room_occludes(rm, a, b):
    return any(segment_hits_aabb(np.asarray(a, float), np.asarray(b, float),
                                 ob["center"], ob["half"]) for ob in rm.obstacles)


def analytic_label(room, pose, yaw_deg):
    """acoustic_env.h::analytic_label — GT geometry labels, computed from the
    room, never from the audio."""
    pose = np.asarray(pose, float)
    x0, x1 = room.origin[0], room.origin[0] + room.Lx
    z0, z1 = room.origin[2], room.origin[2] + room.Lz

    dist_wall = np.empty(8, np.float32)
    for k in range(8):
        a = math.radians(yaw_deg + 45.0 * k)
        dx, dz = math.sin(a), -math.cos(a)
        tx = (x1 - pose[0]) / dx if dx > 1e-6 else ((x0 - pose[0]) / dx if dx < -1e-6 else 1e30)
        tz = (z1 - pose[2]) / dz if dz > 1e-6 else ((z0 - pose[2]) / dz if dz < -1e-6 else 1e30)
        dist_wall[k] = min(tx, tz)

    room_vol = room.Lx * room.Ly * room.Lz

    cand = [(x1 - pose[0], 0), (pose[0] - x0, 1), (z1 - pose[2], 4), (pose[2] - z0, 5)]
    best, wall = cand[0]
    for d, w in cand[1:]:
        if d < best:
            best, wall = d, w
    material_id = room.wall_material[wall]

    open_az = float("nan")
    best_area = 0.0
    for op in room.openings:
        area = op["w"] * op["h"]
        if area > best_area:
            best_area = area
            open_az = math.degrees(math.atan2(op["center"][0] - pose[0],
                                              -(op["center"][2] - pose[2])))
    return dict(dist_wall=dist_wall, open_az=np.float32(open_az),
                room_vol=np.float32(room_vol), material=np.int32(material_id))


# ===========================================================================
#  Room mesh (acoustic_env.h::build_room_mesh) -> per-material triangle soup
# ===========================================================================
def build_room_mesh(rm):
    """Returns {material_name: [ (3,3) triangles ]} — 6 walls with any sampled
    opening carved out, plus interior obstacle boxes."""
    groups = {n: [] for n in MAT_NAMES}
    x0, x1 = rm.origin[0], rm.origin[0] + rm.Lx
    y0, y1 = rm.origin[1], rm.origin[1] + rm.Ly
    z0, z1 = rm.origin[2], rm.origin[2] + rm.Lz

    def add_quad(a, b, c, d, mat):
        t = groups[MAT_NAMES[mat]]
        a, b, c, d = map(lambda v: np.asarray(v, float), (a, b, c, d))
        t.append([a, b, c]); t.append([a, c, d])

    def add_wall(pt, u0, u1, v0, v1, hole, hu0, hu1, hv0, hv1, mat):
        e = 1e-3
        if not hole:
            add_quad(pt(u0, v0), pt(u1, v0), pt(u1, v1), pt(u0, v1), mat); return
        if hv0 > v0 + e: add_quad(pt(u0, v0), pt(u1, v0), pt(u1, hv0), pt(u0, hv0), mat)
        if v1 > hv1 + e: add_quad(pt(u0, hv1), pt(u1, hv1), pt(u1, v1), pt(u0, v1), mat)
        if hu0 > u0 + e: add_quad(pt(u0, hv0), pt(hu0, hv0), pt(hu0, hv1), pt(u0, hv1), mat)
        if u1 > hu1 + e: add_quad(pt(hu1, hv0), pt(u1, hv0), pt(u1, hv1), pt(hu1, hv1), mat)

    wop = [None] * 6
    for op in rm.openings:
        wop[op["wall"]] = op

    def hole_uv(o, x_wall):
        cu = o["center"][2] if x_wall else o["center"][0]
        return (cu - o["w"] * 0.5, cu + o["w"] * 0.5,
                o["center"][1] - o["h"] * 0.5, o["center"][1] + o["h"] * 0.5)

    z4 = (0.0, 0.0, 0.0, 0.0)
    add_wall(lambda u, v: (u, y0, v), x0, x1, z0, z1, False, *z4, rm.wall_material[3])
    add_wall(lambda u, v: (u, y1, v), x0, x1, z0, z1, False, *z4, rm.wall_material[2])
    h = hole_uv(wop[0], True) if wop[0] else z4
    add_wall(lambda u, v: (x1, v, u), z0, z1, y0, y1, wop[0] is not None, *h, rm.wall_material[0])
    h = hole_uv(wop[1], True) if wop[1] else z4
    add_wall(lambda u, v: (x0, v, u), z0, z1, y0, y1, wop[1] is not None, *h, rm.wall_material[1])
    h = hole_uv(wop[4], False) if wop[4] else z4
    add_wall(lambda u, v: (u, v, z1), x0, x1, y0, y1, wop[4] is not None, *h, rm.wall_material[4])
    h = hole_uv(wop[5], False) if wop[5] else z4
    add_wall(lambda u, v: (u, v, z0), x0, x1, y0, y1, wop[5] is not None, *h, rm.wall_material[5])

    for ob in rm.obstacles:
        ax, ay, az = ob["center"] - ob["half"]
        bx, by, bz = ob["center"] + ob["half"]
        m = ob["material"]
        add_quad((ax, ay, az), (bx, ay, az), (bx, by, az), (ax, by, az), m)
        add_quad((ax, ay, bz), (bx, ay, bz), (bx, by, bz), (ax, by, bz), m)
        add_quad((ax, ay, az), (ax, ay, bz), (ax, by, bz), (ax, by, az), m)
        add_quad((bx, ay, az), (bx, ay, bz), (bx, by, bz), (bx, by, az), m)
        add_quad((ax, ay, az), (bx, ay, az), (bx, ay, bz), (ax, ay, bz), m)
        add_quad((ax, by, az), (bx, by, az), (bx, by, bz), (ax, by, bz), m)
    return {k: v for k, v in groups.items() if v}


def export_room_scene(rm, scene_dir):
    """Write the 6-file habitat/RLR scene set for a BoxRoom. Semantic class
    names equal material names so RLR's name-matching binds them."""
    sb = SceneBuilder()
    for label, tris in build_room_mesh(rm).items():
        sb.objects.append((label, tris))
    sb.export(scene_dir)
    write_materials(scene_dir, "materials.json", material_specs())
    return scene_dir


# ===========================================================================
#  Matched-filter range tower (acoustic_env.h::MatchedFilter, verbatim)
# ===========================================================================
def _fftconv(a, b):
    n = len(a) + len(b) - 1
    nf = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, nf) * np.fft.rfft(b, nf), nf)[:n]


class MatchedFilter:
    def __init__(self, f1=100.0, f2=10000.0, T=0.050, sr=SR):
        self.N = int(T * sr)
        t = np.arange(self.N) / float(sr)
        w1, w2 = 2 * np.pi * f1, 2 * np.pi * f2
        L = T / np.log(w2 / w1)
        K = w1 * L
        self.ess = np.sin(K * (np.exp(t / L) - 1.0))
        inv = self.ess[::-1] * np.exp(-t / L)
        mx = np.abs(_fftconv(self.ess, inv)).max()
        self.inv = inv / mx if mx > 1e-12 else inv

    def profile(self, W):
        """W = the engine's W-channel (omni) IR. Returns the pulse-compressed
        |z| profile starting RANGE_GUARD samples past the zero-delay reference."""
        y = _fftconv(self.ess, np.asarray(W, float))
        n = min(self.N, len(y))
        y[:n] += self.ess[:n]                     # self-heard emission
        y[:n] -= 1.01 * self.ess[:n]              # self-cancel, 1% residual
        z = np.abs(_fftconv(y, self.inv))
        start = (self.N - 1) + RANGE_GUARD
        want = RANGE_BINS * RANGE_POOL
        out = np.zeros(want)
        seg = z[start:start + want]
        out[:len(seg)] = seg
        return out

    @staticmethod
    def pool_log(raw):
        e = (raw[:RANGE_BINS * RANGE_POOL] ** 2).reshape(RANGE_BINS, RANGE_POOL).mean(1)
        return np.log1p(e * 1e4).astype(np.float32)


# ===========================================================================
#  Gammatone cochleagram (acoustic_env.h::Cochleagram, verbatim)
#  ERB-spaced RBJ constant-0dB band-pass bank, transposed-DF2 == lfilter.
# ===========================================================================
class Cochleagram:
    def __init__(self):
        erb_rate = lambda f: 21.4 * np.log10(0.00437 * f + 1.0)
        inv_erb = lambda e: (10.0 ** (e / 21.4) - 1.0) / 0.00437
        e_lo, e_hi = erb_rate(50.0), erb_rate(20000.0)
        fc = inv_erb(e_lo + (e_hi - e_lo) * np.arange(COCH_F) / (COCH_F - 1))
        erb = 24.7 * (0.00437 * fc + 1.0)
        Q = fc / erb
        w0 = 2 * np.pi * fc / SR
        al = np.sin(w0) / (2 * Q)
        a0 = 1 + al
        self.b = np.stack([al / a0, np.zeros(COCH_F), -al / a0], 1)
        self.a = np.stack([np.ones(COCH_F), -2 * np.cos(w0) / a0, (1 - al) / a0], 1)

    def apply(self, x):
        """one ear (RAW_N samples) -> [COCH_F, COCH_T] log-energy frames"""
        out = np.empty((COCH_F, COCH_T), np.float32)
        for k in range(COCH_F):
            band = lfilter(self.b[k], self.a[k], x)
            e = (band[:COCH_T * COCH_HOP] ** 2).reshape(COCH_T, COCH_HOP).mean(1)
            out[k] = np.log1p(e * 1e4)
        return out


# ===========================================================================
#  SoundSpaces 2.0 environment
# ===========================================================================
CT = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType


def _audio_spec(uuid, channel, direct, indirect, order=AMBI_ORDER, max_ir=IR_DUR,
                rays=RAYS, threads=1):
    spec = habitat_sim.AudioSensorSpec()
    spec.uuid = uuid
    spec.enableMaterials = True
    spec.position = [0.0, 0.0, 0.0]          # listener exactly at the agent pose
    if channel == "ambisonics":
        spec.channelLayout.type = CT.Ambisonics
        spec.channelLayout.channelCount = (order + 1) ** 2
    else:
        spec.channelLayout.type = CT.Binaural
        spec.channelLayout.channelCount = 2
    ac = spec.acousticsConfig
    ac.sampleRate = SR
    ac.maxIRLength = float(max_ir)
    ac.direct = bool(direct)
    ac.indirect = bool(indirect)
    ac.diffraction = False        # direct-path-only in RLR; Steam Audio had none
    ac.transmission = False       # see module docstring (E11)
    ac.meshSimplification = False
    ac.indirectRayCount = int(rays)
    ac.indirectRayDepth = RAY_DEPTH
    ac.sourceRayCount = 200
    ac.sourceRayDepth = 10
    ac.directRayCount = 500
    ac.indirectSHOrder = order if channel == "ambisonics" else 1
    ac.directSHOrder = order if channel == "ambisonics" else 3
    ac.threadCount = int(threads)
    return spec


# Sensor roles. `ambi`/`bin` are the reflection-only observation streams that
# mirror Steam Audio (which renders no direct path at all); `dir` isolates SS2's
# real direct arrival, which is what makes a genuinely measured T_meas possible.
SENSORS = (("ambi", "ambisonics", False, True, IR_DUR),
           ("bin",  "binaural",   False, True, IR_DUR),
           ("dir",  "binaural",   True,  False, 0.2))


def yaw_to_quat(yaw_deg):
    """Steam Audio head convention: ahead = (sin th, 0, -cos th). Habitat's
    agent faces -z and rotates about +y, so the habitat angle is -yaw."""
    y = math.radians(-yaw_deg)
    return np.quaternion(math.cos(y / 2), 0, math.sin(y / 2), 0)


def to_local(d, yaw_deg):
    """world direction -> listener-local (+x right, -z ahead), == swarm_env.h::toLocal"""
    t = math.radians(yaw_deg)
    ah = np.array([math.sin(t), 0.0, -math.cos(t)])
    ri = np.array([math.cos(t), 0.0, math.sin(t)])
    return np.array([float(np.dot(d, ri)), float(d[1]), -float(np.dot(d, ah))])


class SS2Env:
    """pose -> (coch, range) observation operator, plus the measured transform."""

    def __init__(self, scene_root, threads=1, rays=RAYS):
        self.scene_root = scene_root
        self.threads = threads
        self.rays = rays
        self.sim = None
        self.room = None
        self.mf = MatchedFilter()
        self.coch = Cochleagram()
        self.cal_itd = None
        self.cal_az = None
        self._range_cal = (0.0, C_NOM)   # (onset offset in samples, effective c)
        os.makedirs(scene_root, exist_ok=True)

    # ---------------------------------------------------------------- scene
    def _open_scene(self, scene_dir):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        b = habitat_sim.SimulatorConfiguration()
        b.scene_id = os.path.join(scene_dir, "s.glb")
        b.scene_dataset_config_file = os.path.join(scene_dir, "s.scene_dataset_config.json")
        b.load_semantic_mesh = True
        specs = [_audio_spec(u, ch, d, i, max_ir=m, rays=self.rays, threads=self.threads)
                 for (u, ch, d, i, m) in SENSORS]
        agc = habitat_sim.agent.AgentConfiguration()
        agc.sensor_specifications = specs
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(b, [agc]))
        mats = os.path.join(scene_dir, "materials.json")
        for u, *_ in SENSORS:
            self.sim.get_agent(0)._sensors[u].setAudioMaterialsJSON(mats)

    def set_room(self, rm, tag):
        sd = os.path.join(self.scene_root, tag)
        if not os.path.exists(os.path.join(sd, "s.glb")):
            export_room_scene(rm, sd)
        else:
            write_materials(sd, "materials.json", material_specs())
        self._open_scene(sd)
        self.room = rm

    def set_scene_dir(self, scene_dir):
        self._open_scene(scene_dir)

    # --------------------------------------------------------------- render
    def render(self, uuid, listener, yaw_deg, source, n=None):
        ag = self.sim.get_agent(0)
        st = ag.get_state()
        st.position = np.asarray(listener, np.float32)
        st.rotation = yaw_to_quat(yaw_deg)
        ag.set_state(st)
        sen = self.sim.get_agent(0)._sensors[uuid]
        rot = ag.get_state().rotation
        sen.setAudioListenerTransform(sen.node.absolute_translation,
                                      np.array([rot.w, rot.x, rot.y, rot.z]))
        sen.setAudioSourceTransform(np.asarray(source, np.float32))
        sen.runSimulation(self.sim)
        ir = np.array(sen.getIR(), dtype=np.float64)
        if ir.ndim == 1:
            ir = ir[None, :]
        if n is not None:
            out = np.zeros((ir.shape[0], n))
            k = min(n, ir.shape[1])
            out[:, :k] = ir[:, :k]
            return out
        return ir

    # ---------------------------------------------------------- observation
    def observe(self, listener, yaw_deg, source=None):
        """The observation operator. source=None => monostatic (co-located)."""
        src = listener if source is None else source
        ambi = self.render("ambi", listener, yaw_deg, src, n=TOTAL)
        binr = self.render("bin", listener, yaw_deg, src, n=RAW_N)
        coch = np.stack([self.coch.apply(binr[0]), self.coch.apply(binr[1])], 0)
        rng = MatchedFilter.pool_log(self.mf.profile(ambi[0]))
        return coch.astype(np.float16), rng.astype(np.float16)

    # ------------------------------------------------- measured transform
    @staticmethod
    def itd_samples(l, r, maxlag=48):
        """POSITIVE = right ear later = source on the LEFT (swarm_env.h).

        Deviation from the C++ original: the interaural cross-correlation is
        low-passed to 800 Hz first. Steam Audio's HRIR cross-correlates cleanly
        broadband, but RLR's does not — broadband it cycle-skips and the ITD
        saturates at ~+-165 us (vs the ~+-680 us the SS2 battery's E7 measured
        directly), which folds the calibration curve past +-60 deg. Below one
        half-wavelength of head width the correlation is unambiguous.
        """
        l = _itd_lowpass(np.asarray(l, float))
        r = _itd_lowpass(np.asarray(r, float))
        n = min(len(l), len(r))
        cc = np.empty(2 * maxlag + 1)
        for gi in range(2 * maxlag + 1):
            g = gi - maxlag
            i0, i1 = max(0, -g), n - max(0, g)
            cc[gi] = float(np.dot(l[i0:i1], r[i0 + g:i1 + g])) if i1 > i0 else 0.0
        bl = int(np.argmax(cc))
        sub = 0.0
        if 0 < bl < 2 * maxlag:
            y0, y1, y2 = cc[bl - 1], cc[bl], cc[bl + 1]
            den = y0 - 2 * y1 + y2
            if den != 0:
                sub = 0.5 * (y0 - y2) / den
        return (bl - maxlag) + sub

    @staticmethod
    def onset(x, thresh=1e-3):
        pk = np.abs(x).max()
        if pk <= 0:
            return -1
        idx = np.flatnonzero(np.abs(x) > thresh * pk)
        return int(idx[0]) if len(idx) else -1

    def calibrate(self, free_scene_dir, azimuths=range(-90, 91, 15), dists=(1., 2., 4., 8.)):
        """Self-calibration in an (effectively) empty scene, exactly the
        information an agent could obtain about its own sensor:
          * ITD-vs-azimuth curve from the direct arrival (swarm_env.h::build_calibration)
          * onset-vs-distance line -> (fixed offset, effective sound speed)
        """
        self.set_scene_dir(free_scene_dir)
        pts = []
        for az in azimuths:
            a = math.radians(az)
            src = [2 * math.sin(a), EAR_H, -2 * math.cos(a)]     # az>0 = to the RIGHT
            ir = self.render("dir", [0, EAR_H, 0], 0.0, src)
            pts.append((self.itd_samples(ir[0], ir[1]), float(az)))
        pts.sort()
        self.cal_itd = np.array([p[0] for p in pts])
        self.cal_az = np.array([p[1] for p in pts])

        ons, dd = [], []
        for d in dists:
            ir = self.render("dir", [0, EAR_H, 0], 0.0, [0, EAR_H, -float(d)])
            o = self.onset(np.abs(ir).max(0))
            if o >= 0:
                ons.append(o); dd.append(d)
        slope, icept = np.polyfit(dd, ons, 1)
        self._range_cal = (float(icept), float(SR / slope))
        return dict(itd=self.cal_itd.tolist(), az=self.cal_az.tolist(),
                    onset_offset=self._range_cal[0], c_eff=self._range_cal[1])

    def itd_to_az(self, itd):
        return float(np.interp(itd, self.cal_itd, self.cal_az))

    def measure_tab(self, l_pos, l_yaw, e_pos):
        """swarm_env.h::SwarmEnv::measure_tab, but every quantity is read off
        SS2's REAL direct arrival instead of an injected HRIR:
          range   <- direct-arrival ToA through the onset calibration
          bearing <- binaural ITD through the agent's own ITD-vs-azimuth curve
          front/back <- +20 deg head-turn re-measurement (Wallach)
        """
        off, c_eff = self._range_cal
        d = np.asarray(e_pos, float) - np.asarray(l_pos, float)
        R_true = float(np.linalg.norm(d))

        def look(head_yaw):
            ir = self.render("dir", l_pos, head_yaw, e_pos)
            return ir, self.itd_to_az(self.itd_samples(ir[0], ir[1]))

        ir0, lat0 = look(l_yaw)
        _, lat20 = look(l_yaw + 20.0)

        o = self.onset(np.abs(ir0).max(0))
        R = (o - off) / SR * c_eff if o >= 0 else R_true
        if not np.isfinite(R) or R <= 0:
            R = R_true

        wrap180 = lambda a: math.fmod(a + 540.0, 360.0) - 180.0
        pred_lat = lambda az: math.degrees(math.asin(math.sin(math.radians(az - 20.0))))
        hyp_front = lat0
        hyp_rear = 180.0 - lat0 if lat0 >= 0 else -180.0 - lat0
        err_f = abs(wrap180(pred_lat(hyp_front) - lat20))
        err_r = abs(wrap180(pred_lat(hyp_rear) - lat20))
        chose_front = err_f <= err_r
        az = hyp_front if chose_front else hyp_rear

        t = math.radians(l_yaw)
        ahead = d[0] * math.sin(t) + d[2] * (-math.cos(t))
        fb_ok = int(chose_front == (ahead >= 0))

        ar = math.radians(az)
        fwd = np.array([math.sin(t), 0.0, -math.cos(t)])
        rgt = np.array([math.cos(t), 0.0, math.sin(t)])
        v = fwd * (R * math.cos(ar)) + rgt * (R * math.sin(ar))
        return dict(dx=float(v[0]), dy=0.0, dz=float(v[2]),
                    range=float(R), bearing=float(az), fb_ok=fb_ok)

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None


def cross_meta(rm, l_pos, l_yaw, e_pos, e_yaw):
    """swarm_env.h::cross_meta — the ORACLE transform (scoring only)."""
    d = np.asarray(e_pos, float) - np.asarray(l_pos, float)
    R = float(np.linalg.norm(d))
    t = math.radians(l_yaw)
    ahead = d[0] * math.sin(t) + d[2] * (-math.cos(t))
    right = d[0] * math.cos(t) + d[2] * math.sin(t)
    return dict(dx=float(d[0]), dy=float(d[1]), dz=float(d[2]), range=R,
                rel_yaw=float(e_yaw - l_yaw),
                bearing=math.degrees(math.atan2(right, ahead)),
                occluded=int(room_occludes(rm, l_pos, e_pos)))


def place_agents(rm, seed, n, min_sep=1.5, wall_clear=0.8):
    """swarm_env.h::place_agents — scatter N agents, clear of walls/obstacles."""
    r = Rng((np.uint32(seed) * np.uint32(40503) + np.uint32(7)))
    x0, x1 = rm.origin[0] + wall_clear, rm.origin[0] + rm.Lx - wall_clear
    z0, z1 = rm.origin[2] + wall_clear, rm.origin[2] + rm.Lz - wall_clear
    OB_PAD = 0.4
    pos, yaw = [], []
    for i in range(n):
        p = None
        for _ in range(400):
            p = np.array([r.rf(x0, x1), EAR_H, r.rf(z0, z1)])
            ok = all(math.hypot(p[0] - q[0], p[2] - q[2]) >= min_sep for q in pos)
            if ok:
                for ob in rm.obstacles:
                    c, h = ob["center"], ob["half"]
                    if (c[0] - h[0] - OB_PAD < p[0] < c[0] + h[0] + OB_PAD and
                            c[2] - h[2] - OB_PAD < p[2] < c[2] + h[2] + OB_PAD and
                            c[1] - h[1] < p[1] < c[1] + h[1]):
                        ok = False
                        break
            if ok:
                break
        pos.append(p)
        yaw.append(r.rf(0.0, 360.0))
    return np.array(pos), np.array(yaw)


def free_scene(scene_root):
    """Effectively free space (habitat needs a stage): one tiny slab 400 m below.

    Unlike the per-episode room scenes this path is SHARED by every worker, so
    it is written once and then only read. Rewriting it per worker races: a
    second process can read a half-written materials.json and the material bind
    fails ("Error parsing audio material data"). Harmless here (calibration uses
    the direct path only, which no material touches) but avoided regardless.
    """
    sd = os.path.join(scene_root, "_free")
    if not os.path.exists(os.path.join(sd, "materials.json")):
        sb = SceneBuilder()
        sb.add_slab("mirror", -0.05, 0.05, -400.1, -400.0, -0.05, 0.05)
        sb.export(sd)
        write_materials(sd, "_materials.tmp.json", material_specs())
        os.replace(os.path.join(sd, "_materials.tmp.json"),
                   os.path.join(sd, "materials.json"))
    return sd
