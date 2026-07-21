#!/usr/bin/env python3
"""
ss2_collect.py — SoundSpaces 2.0 (habitat-sim 0.2.2 + RLR audio propagation)
acoustic characterization battery collector, mirroring the Steam Audio 4.8.1
`char_rig` instrumentation. See ss2_experiment_manifest.md.

Builds synthetic geometry programmatically (trimesh -> GLB), binds RLR audio
materials through the semantic-mesh chain (semantic PLY authored in the
engine's MP3D frame: verts written as (x, -z, y) — see
data/ss_test_3/results/headon_toa_validation_20260614/headon_toa.py, the
validated reference implementation), renders RIRs through the AudioSensor,
and writes 32-bit-float WAVs + manifest.csv in the paper_data layout.

Usage:  python ss2_collect.py <driver> [<driver> ...]
Drivers: smoke e1_coarse e1_sr44100 e1_fine:<lo>:<hi>:<step>:<name>
         e2 e2_fresh:<rays> e3 e3_near e4 e5 e6 e6_q e7_ambi e7_binaural
         e8 e9 e10 e10_fine
Output:  ~/ss2_results/ss2_<name>/ (copied to the /mnt/d paper_data dir at the
         end of the battery).
"""
import os, sys, json, math, time, shutil, subprocess

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import numpy as np
import trimesh
import habitat_sim
from scipy.io import wavfile

REPO = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.expanduser("~/ss2_results")
SCENE_ROOT = os.path.join(OUT_ROOT, "scenes")
BASE_MATS = os.path.join(REPO, "data/ss_test_3/single_reflector_materials.json")
EAR_Y = 1.5
# Band edges used to author per-band material sweeps (RLR materials are
# (freq, value) breakpoint curves, not fixed bands; these 4 bands are our
# choice, mirroring acousticsConfig.frequencyBands = 4).
BAND_EDGES = [20.0, 500.0, 2000.0, 8000.0, 20000.0]
MANIFEST_HDR = ("file,geo,dist_m,repeat,sr,frame,dur_s,order,rays,bounces,"
                "diffuse,imd,material,onset_smp,onset_ms,peak,air")
ONSET_THRESH = 1e-6

# ----------------------------------------------------------------- materials
def curve(v):
    return [20.0, float(v), 20000.0, float(v)]

def band_curve(vals):
    """Piecewise-constant curve over the 4 BAND_EDGES bands (values vals[0..3])."""
    pts = [20.0, float(vals[0])]
    for i in range(1, 4):
        e = BAND_EDGES[i]
        if vals[i - 1] != vals[i]:
            pts += [e - 1.0, float(vals[i - 1]), e + 1.0, float(vals[i])]
    pts += [20000.0, float(vals[3])]
    return pts

def write_materials(scene_dir, fname, mats):
    """mats: list of dicts {name, labels, absorption, scattering, transmission}
    (each acoustic field a breakpoint list). Appended to the stock MP3D
    material table so the loader's required damping/density/speed fields and
    the 'default' material are present."""
    data = json.load(open(BASE_MATS))
    donor = next(m for m in data["materials"] if "damping" in m)
    for m in data["materials"]:
        for fld in ("damping", "density", "speed"):
            m.setdefault(fld, donor[fld])
    for spec in mats:
        m = {"name": spec["name"], "labels": spec["labels"],
             "absorption": spec["absorption"], "scattering": spec["scattering"],
             "transmission": spec["transmission"]}
        for fld in ("damping", "density", "speed"):
            m[fld] = donor[fld]
        data["materials"].append(m)
    path = os.path.join(scene_dir, fname)
    json.dump(data, open(path, "w"))
    return path

# RLR binds a material by matching the semantic CLASS NAME (logged as
# "Material: <class>") against the material's "name" field (case-sensitive),
# NOT "labels". So every material's name MUST equal the slab's semantic label
# ("mirror", "boxwall"). Naming it anything else silently falls back to the
# stock "Default" material (abs=0.1) — the bug that made the first battery's
# material sweeps no-ops.
MIRROR = dict(name="mirror", labels=["mirror"], absorption=curve(0),
              scattering=curve(0), transmission=curve(0))

# ------------------------------------------------------------------ geometry
class SceneBuilder:
    """Triangle-soup scenes with per-object semantic labels. Exports the
    6-file set habitat-sim + RLR need (GLB, semantic PLY in MP3D frame,
    ssd.json, stage config, dataset config; materials.json written separately)."""
    def __init__(self):
        self.objects = []   # (label, tris list[(3,3) float])

    def add_slab(self, label, xmin, xmax, ymin, ymax, zmin, zmax):
        c = [[xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymin,zmax],[xmin,ymin,zmax],
             [xmin,ymax,zmin],[xmax,ymax,zmin],[xmax,ymax,zmax],[xmin,ymax,zmax]]
        quads = [(0,1,2,3),(4,7,6,5),(0,4,5,1),(3,2,6,7),(0,3,7,4),(1,5,6,2)]
        tris = []
        for q in quads:
            p = [np.array(c[i], float) for i in q]
            tris.append([p[0], p[1], p[2]]); tris.append([p[0], p[2], p[3]])
        self.objects.append((label, tris))

    def add_trimesh(self, label, tm):
        self.objects.append((label, [t.copy() for t in np.asarray(tm.triangles, float)]))

    def export(self, scene_dir):
        os.makedirs(scene_dir, exist_ok=True)
        all_v, all_f = [], []
        sem_v, sem_f, sem_oid = [], [], []
        classes, objects = [], []
        for oid, (label, tris) in enumerate(self.objects, start=1):
            pts = np.array(tris, float).reshape(-1, 3)
            for t in tris:
                b = len(all_v); all_v.extend(t); all_f.append([b, b+1, b+2])
                bs = len(sem_v); sem_v.extend(t); sem_f.append([bs, bs+1, bs+2])
                sem_oid.append(oid)
            classes.append({"id": oid, "name": label})
            lo, hi = pts.min(0), pts.max(0)
            objects.append({"id": oid, "class_id": oid, "oriented_bbox": {"abb": {
                "center": ((lo+hi)/2).tolist(), "sizes": (hi-lo+0.1).tolist()},
                "orientation": {"translation": [0,0,0], "rotation": [0,0,0,1]}}})
        v = np.array(all_v, float); f = np.array(all_f)
        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        tm.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                name="geo", metallicFactor=1.0, roughnessFactor=0.0, doubleSided=True))
        trimesh.Scene(geometry={"geo": tm}).export(os.path.join(scene_dir, "s.glb"))
        # semantic PLY in the engine's MP3D frame: (x, -z, y)
        sv = np.array(sem_v, np.float32)
        sv = np.stack([sv[:, 0], -sv[:, 2], sv[:, 1]], 1)
        hdr = ("ply\nformat binary_little_endian 1.0\n"
               f"element vertex {len(sv)}\n"
               "property float x\nproperty float y\nproperty float z\n"
               "property uchar red\nproperty uchar green\nproperty uchar blue\n"
               "property ushort object_id\n"
               f"element face {len(sem_f)}\n"
               "property list uchar int vertex_indices\nend_header\n")
        with open(os.path.join(scene_dir, "s_semantic.ply"), "wb") as fh:
            fh.write(hdr.encode())
            for i, p in enumerate(sv):
                fh.write(p.astype("<f4").tobytes())
                fh.write(np.array([60, 90, 200], "<u1").tobytes())
                fh.write(np.uint16(sem_oid[i // 3]).astype("<u2").tobytes())
            for tri in sem_f:
                fh.write(np.uint8(3).tobytes()); fh.write(np.array(tri, "<i4").tobytes())
        json.dump({"classes": classes, "objects": objects},
                  open(os.path.join(scene_dir, "ssd.json"), "w"))
        json.dump({"render_asset": "s.glb", "collision_asset": "s.glb",
                   "semantic_asset": "s_semantic.ply",
                   "semantic_descriptor_filename": "ssd.json",
                   "up": [0,1,0], "front": [0,0,-1], "origin": [0,0,0],
                   "requires_lighting": False, "shader_type": "flat"},
                  open(os.path.join(scene_dir, "s.stage_config.json"), "w"))
        json.dump({"stages": {"default_attributes": {"up": [0,1,0], "front": [0,0,-1]},
                   "paths": {".json": ["s.stage_config.json"]}}},
                  open(os.path.join(scene_dir, "s.scene_dataset_config.json"), "w"))
        return scene_dir

def scene_wall(name, w, h, mats=(MIRROR,)):
    """Reflective slab, +z face in the z=0 plane, centered on (x=0, y=EAR_Y)."""
    sd = os.path.join(SCENE_ROOT, name)
    if not os.path.exists(os.path.join(sd, "s.glb")):
        sb = SceneBuilder()
        sb.add_slab("mirror", -w/2, w/2, EAR_Y - h/2, EAR_Y + h/2, -0.2, 0.0)
        sb.export(sd)
    write_materials(sd, "materials.json", list(mats))
    return sd

def scene_screen():
    """Half-plane: panel x in [0,30], 40 m tall, +z face at z=0, edge at x=0."""
    sd = os.path.join(SCENE_ROOT, "screen")
    if not os.path.exists(os.path.join(sd, "s.glb")):
        sb = SceneBuilder()
        sb.add_slab("mirror", 0.0, 30.0, EAR_Y - 20.0, EAR_Y + 20.0, -0.2, 0.0)
        sb.export(sd)
    return sd

KNIFE = dict(name="knifebar", labels=["knifebar"], absorption=curve(0.95),
             scattering=curve(0.05), transmission=curve(0.0))
KNIFE_SHELL = dict(name="knifeshell", labels=["knifeshell"], absorption=curve(0.90),
                   scattering=curve(0.50), transmission=curve(0.0))

def scene_knife():
    """E5B occlusion knife edge: sealed absorptive room (interior x 0..30, y 0..6,
    z 0..20) with a FULL-HEIGHT barrier at x=15 spanning z in [0, 7.5], so the only
    path from source to listener is around the vertical edge at z=7.5.

    This differs from scene_screen() in the one way that matters: here the barrier
    OCCLUDES the source->listener path, so there is a true acoustic shadow to diffract
    into. In scene_screen() the panel is fully visible from every listener position and
    x<0 only removes the SPECULAR point -- a specular shadow, not an occlusion shadow.
    """
    sd = os.path.join(SCENE_ROOT, "knife")
    if not os.path.exists(os.path.join(sd, "s.glb")):
        sb = SceneBuilder()
        T = 0.2
        sb.add_slab("knifebar", 14.9, 15.1, 0.0, 6.0, 0.0, 7.5)      # occluder, edge at z=7.5
        sb.add_slab("knifeshell", -T, 30 + T, -T, 0.0, -T, 20 + T)   # floor
        sb.add_slab("knifeshell", -T, 30 + T, 6.0, 6 + T, -T, 20 + T)  # ceiling
        sb.add_slab("knifeshell", -T, 0.0, 0.0, 6.0, -T, 20 + T)     # wall x0
        sb.add_slab("knifeshell", 30.0, 30 + T, 0.0, 6.0, -T, 20 + T)  # wall x1
        sb.add_slab("knifeshell", 0.0, 30.0, 0.0, 6.0, -T, 0.0)      # wall z0
        sb.add_slab("knifeshell", 0.0, 30.0, 0.0, 6.0, 20.0, 20 + T)  # wall z1
        sb.export(sd)
    write_materials(sd, "m_knife.json", [dict(KNIFE), dict(KNIFE_SHELL)])
    return sd

def scene_box():
    """Closed shoebox 8 x 5 x 3 m (habitat: x 0..8, y 0..3 up, z 0..5)."""
    sd = os.path.join(SCENE_ROOT, "box853")
    if not os.path.exists(os.path.join(sd, "s.glb")):
        sb = SceneBuilder()
        sb.add_slab("boxwall", 0, 8, 0, 3, 0, 5)   # single closed slab: listener inside
        sb.export(sd)
    return sd

def scene_corridor():
    """Two parallel 20x20 walls, inner faces at z=0 (facing +z) and z=10."""
    sd = os.path.join(SCENE_ROOT, "corridor")
    if not os.path.exists(os.path.join(sd, "s.glb")):
        sb = SceneBuilder()
        sb.add_slab("mirror", -10, 10, EAR_Y - 10, EAR_Y + 10, -0.2, 0.0)
        sb.add_slab("mirror", -10, 10, EAR_Y - 10, EAR_Y + 10, 10.0, 10.2)
        sb.export(sd)
    write_materials(sd, "materials.json", [MIRROR])
    return sd

def scene_free():
    """Effectively free space: only a tiny slab 400 m below (habitat needs a stage)."""
    sd = os.path.join(SCENE_ROOT, "free")
    if not os.path.exists(os.path.join(sd, "s.glb")):
        sb = SceneBuilder()
        sb.add_slab("mirror", -0.05, 0.05, -400.1, -400.0, -0.05, 0.05)
        sb.export(sd)
    write_materials(sd, "materials.json", [MIRROR])
    return sd

def scene_level4():
    """Shared real-game geometry level4.obj (31.9 x 32 x 0.2 m slab, y-up)."""
    sd = os.path.join(SCENE_ROOT, "level4")
    if not os.path.exists(os.path.join(sd, "s.glb")):
        tm = trimesh.load(os.path.join(REPO, "level4.obj"), force="mesh")
        print("level4 bbox extents:", tm.extents, "bounds:", tm.bounds)
        sb = SceneBuilder()
        sb.add_trimesh("mirror", tm)
        sb.export(sd)
    write_materials(sd, "materials.json", [MIRROR])
    return sd

# ----------------------------------------------------------------- simulator
CT = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType

def build_sim(scene_dir, sr=48000, max_ir=0.4, channel="ambisonics", order=1,
              direct=True, indirect=True, diffraction=True, transmission=True,
              rays=5000, depth=200, source_rays=200, source_depth=10,
              direct_rays=500, threads=1, hrtf=None, mats="materials.json",
              mesh_simp=False):
    b = habitat_sim.SimulatorConfiguration()
    b.scene_id = os.path.join(scene_dir, "s.glb")
    b.scene_dataset_config_file = os.path.join(scene_dir, "s.scene_dataset_config.json")
    b.load_semantic_mesh = True
    spec = habitat_sim.AudioSensorSpec(); spec.uuid = "audio_sensor"
    spec.enableMaterials = True
    spec.position = [0.0, 0.0, 0.0]        # listener exactly at agent position
    if channel == "ambisonics":
        spec.channelLayout.type = CT.Ambisonics
        spec.channelLayout.channelCount = (order + 1) ** 2
    elif channel == "mono":
        spec.channelLayout.type = CT.Mono; spec.channelLayout.channelCount = 1
    else:
        spec.channelLayout.type = CT.Binaural; spec.channelLayout.channelCount = 2
    ac = spec.acousticsConfig
    ac.sampleRate = sr; ac.maxIRLength = float(max_ir)
    ac.direct = bool(direct); ac.indirect = bool(indirect)
    ac.diffraction = bool(diffraction); ac.transmission = bool(transmission)
    # meshSimplification applies VERTEX WELDING, which is what RLR needs to build the
    # math::ConnectedMesh that DiffractionEdge::extractMeshEdges pulls diffracting edges
    # from. Habitat's semantic-mesh path (getJoinedSemanticMesh -> RLRA_AddMeshVertices)
    # delivers disconnected triangles, so with this OFF the diffraction graph is empty and
    # the `diffraction` flag is inert -- which is why E5 measured a 0.0 dB toggle.
    # Default False = engine default = exactly what E5 ran. See ss2_e5b_diffraction_correction.md.
    ac.meshSimplification = bool(mesh_simp)
    ac.indirectRayCount = int(rays); ac.indirectRayDepth = int(depth)
    ac.sourceRayCount = int(source_rays); ac.sourceRayDepth = int(source_depth)
    ac.directRayCount = int(direct_rays)
    ac.indirectSHOrder = order if channel == "ambisonics" else 1
    ac.directSHOrder = order if channel == "ambisonics" else 3
    ac.threadCount = int(threads)
    agc = habitat_sim.agent.AgentConfiguration(); agc.sensor_specifications = [spec]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(b, [agc]))
    sen = sim.get_agent(0)._sensors["audio_sensor"]
    if hrtf:
        sen.setListenerHRTF(hrtf)
    sen.setAudioMaterialsJSON(os.path.join(scene_dir, mats))
    cfg = dict(scene=scene_dir, sr=sr, max_ir=max_ir, channel=channel, order=order,
               direct=direct, indirect=indirect, diffraction=diffraction,
               transmission=transmission,
               indirectRayCount=int(rays), indirectRayDepth=int(depth),
               sourceRayCount=int(source_rays), sourceRayDepth=int(source_depth),
               directRayCount=int(direct_rays), threadCount=int(threads),
               hrtf=hrtf or "engine-default", materials=mats,
               frequencyBands=ac.frequencyBands, globalVolume=ac.globalVolume,
               unitScale=ac.unitScale, temporalCoherence=ac.temporalCoherence,
               meshSimplification=ac.meshSimplification,
               maxDiffractionOrder=ac.maxDiffractionOrder)
    return sim, sen, cfg

def render(sim, sen, rcv, src, yaw_deg=0.0):
    ag = sim.get_agent(0); st = ag.get_state()
    st.position = np.array(rcv, float)
    y = math.radians(yaw_deg)
    st.rotation = np.quaternion(math.cos(y/2), 0, math.sin(y/2), 0)
    ag.set_state(st)
    sen.setAudioSourceTransform(np.array(src, np.float32))
    return np.array(sim.get_sensor_observations()["audio_sensor"], dtype=np.float64)

# ----------------------------------------------------------------- recording
def onset_of(x):
    pk = np.max(np.abs(x))
    if pk <= 0: return -1, pk
    idx = np.flatnonzero(np.abs(x) > ONSET_THRESH * pk)
    return (int(idx[0]) if len(idx) else -1), pk

def exp_dir(name):
    d = os.path.join(OUT_ROOT, f"ss2_{name}")
    os.makedirs(d, exist_ok=True)
    mp = os.path.join(d, "manifest.csv")
    if not os.path.exists(mp):
        open(mp, "w").write(MANIFEST_HDR + "\n")
    return d

def save_case(outdir, tag, ir, cfg, geo, dist_m, repeat, material, extra_note=None):
    fs = int(cfg["sr"])
    w = ir[0] if ir.ndim > 1 else ir
    wname = f"W_{tag}.wav"
    wavfile.write(os.path.join(outdir, wname), fs, w.astype(np.float32))
    if cfg["channel"] == "ambisonics" and ir.ndim > 1 and ir.shape[0] > 1:
        wavfile.write(os.path.join(outdir, f"ambi_{tag}.wav"), fs,
                      ir.T.astype(np.float32))
    elif cfg["channel"] == "binaural":
        wavfile.write(os.path.join(outdir, f"bin_{tag}.wav"), fs,
                      ir.T.astype(np.float32))
    on, pk = onset_of(w)
    mat = f'"{material}"' if "," in material else material
    row = (f"{wname},{geo},{dist_m:.4f},{repeat},{fs},-1,{cfg['max_ir']},"
           f"{cfg['order'] if cfg['channel']=='ambisonics' else 'na'},"
           f"{cfg['indirectRayCount']},{cfg['indirectRayDepth']},"
           f"{cfg.get('diffuse','na')},na,{mat},{on},"
           f"{on/fs*1000 if on >= 0 else -1:.5f},{pk:.6e},default")
    open(os.path.join(outdir, "manifest.csv"), "a").write(row + "\n")
    return on, pk

def write_meta(outdir, cfg, notes=None, extra=None):
    c = dict(cfg)
    if extra: c.update(extra)
    json.dump(c, open(os.path.join(outdir, "config.json"), "w"), indent=2)
    vsrc = os.path.join(OUT_ROOT, "VERSION.txt")
    if os.path.exists(vsrc):
        shutil.copy(vsrc, os.path.join(outdir, "VERSION.txt"))
    if notes:
        open(os.path.join(outdir, "NOTES.md"), "w").write(notes)

# ------------------------------------------------------------------- drivers
C_NOM = 343.0

def drv_smoke():
    """20x20 mirror wall, co-located src/rcv at d=2,5,10 m + listener-position probe."""
    sd = scene_wall("wall20", 20, 20)
    sim, sen, cfg = build_sim(sd, max_ir=0.4)
    out = exp_dir("smoke")
    ok = True; prev_e = None
    for d in (2.0, 5.0, 10.0):
        p = [0.0, EAR_Y, d]
        t0 = time.time(); ir = render(sim, sen, p, p); dt = time.time() - t0
        w = ir[0]; fs = cfg["sr"]
        n0 = int(0.7 * 2*d/C_NOM * fs)
        seg = w[n0:]
        pk_i = n0 + int(np.argmax(np.abs(seg)))
        e = float(np.sum(seg**2))
        pred_ms = 2*d/C_NOM*1000
        got_ms = pk_i/fs*1000
        print(f"smoke d={d:4.1f}  echo peak {got_ms:7.3f} ms (pred {pred_ms:7.3f}) "
              f"energy {e:.3e}  ir{ir.shape}  {dt:.1f}s", flush=True)
        if not (abs(got_ms - pred_ms) < 0.05*pred_ms + 1.0): ok = False
        if prev_e is not None and e >= prev_e: ok = False
        prev_e = e
        save_case(out, f"d{int(d*100):05d}_r00", ir, cfg, "wall 20x20x0.2m", d, 0,
                  "mirror abs=0 scat=0 trans=0")
    # listener-position probe: direct path source 2 m in front -> onset must be ~2/c
    ir = render(sim, sen, [0, EAR_Y, 10], [0, EAR_Y, 8])
    on, _ = onset_of(ir[0])
    print(f"probe: direct onset {on/cfg['sr']*1000:.3f} ms (expect ~{2/C_NOM*1000:.3f} "
          f"if listener==agent pos)", flush=True)
    sim.close()
    write_meta(out, cfg)
    print("SMOKE", "PASS" if ok else "FAIL", flush=True)
    return ok

def _sweep_wall(name, scene, distances, cfg_kw, geo, material="mirror abs=0 scat=0 trans=0",
                repeats=1):
    sim, sen, cfg = build_sim(scene, **cfg_kw)
    out = exp_dir(name)
    for d in distances:
        p = [0.0, EAR_Y, float(d)]
        for r in range(repeats):
            ir = render(sim, sen, p, p)
            on, pk = save_case(out, f"d{int(round(d*100)):05d}_r{r:02d}", ir, cfg,
                               geo, d, r, material)
        print(f"{name}: d={d:7.3f} onset={on} smp ({on/cfg['sr']*1000:.4f} ms) "
              f"peak={pk:.3e}", flush=True)
    sim.close()
    write_meta(out, cfg, extra=dict(distances=list(map(float, distances))))
    return out

def drv_e1_coarse():
    sd = scene_wall("wall20", 20, 20)
    ds = np.round(np.arange(1.0, 20.0 + 1e-9, 0.25), 4)
    _sweep_wall("e1_coarse", sd, ds, dict(max_ir=0.4, direct=False), "wall 20x20x0.2m")

def drv_e1_sr44100():
    sd = scene_wall("wall20", 20, 20)
    ds = np.round(np.arange(1.0, 20.0 + 1e-9, 0.25), 4)
    _sweep_wall("e1_sr44100", sd, ds, dict(max_ir=0.4, sr=44100, direct=False), "wall 20x20x0.2m")

def drv_e1_fine(lo, hi, step, name):
    sd = scene_wall("wall20", 20, 20)
    ds = np.round(np.arange(lo, hi + 1e-9, step), 4)
    _sweep_wall(name, sd, ds, dict(max_ir=0.4, direct=False), "wall 20x20x0.2m")

def drv_e2():
    sd = scene_wall("wall20", 20, 20)
    for rays in (1024, 4096, 32768, 131072):
        _sweep_wall(f"e2_rays{rays}", sd, [10.0], dict(max_ir=0.4, rays=rays, direct=False),
                    "wall 20x20x0.2m", repeats=20)

def drv_e2_fresh(rays):
    """One render in a fresh process (cross-process determinism probe); repeat 90+."""
    sd = scene_wall("wall20", 20, 20)
    out = exp_dir(f"e2_rays{rays}")
    existing = [f for f in os.listdir(out) if f.startswith("W_d01000_r9")]
    r = 90 + len(existing)
    sim, sen, cfg = build_sim(sd, max_ir=0.4, rays=rays, direct=False)
    ir = render(sim, sen, [0, EAR_Y, 10.0], [0, EAR_Y, 10.0])
    save_case(out, f"d01000_r{r}", ir, cfg, "wall 20x20x0.2m", 10.0, r,
              "mirror abs=0 scat=0 trans=0")
    sim.close()
    print(f"e2_fresh rays={rays} wrote repeat {r}", flush=True)

E3_DIST = [1, 1.5, 2, 3, 4, 5, 7, 10, 14, 20, 28, 35, 42, 50]

def drv_e3():
    sd = scene_wall("wall400", 400, 400)
    _sweep_wall("e3_air_on", sd, E3_DIST, dict(max_ir=0.5, direct=False), "wall 400x400x0.2m")
    # no exposed air-absorption toggle in this build -> no e3_air_off (see NOTES)

def drv_e3_near():
    sd = scene_wall("wall400", 400, 400)
    ds = np.round(np.arange(1.2, 4.6 + 1e-9, 0.2), 4)
    _sweep_wall("e3_near", sd, ds, dict(max_ir=0.5, direct=False), "wall 400x400x0.2m")

def drv_e4():
    d = 10.0
    base = dict(name="mirror", labels=["mirror"], absorption=curve(0),
                scattering=curve(0), transmission=curve(0))
    jobs = [("e4_base", base, "mirror abs=0 scat=0 trans=0")]
    for b in range(4):
        for a in (0.25, 0.5, 0.75, 1.0):
            vals = [0.0]*4; vals[b] = a
            m = dict(base, absorption=band_curve(vals))
            jobs.append((f"e4_abs_b{b}_a{int(a*100)}", m,
                         f"abs band{b} [{BAND_EDGES[b]:.0f}-{BAND_EDGES[b+1]:.0f}Hz]={a} others=0 scat=0 trans=0"))
    for s in (0.25, 0.5, 0.75, 1.0):
        m = dict(base, scattering=curve(s))
        jobs.append((f"e4_scat_s{int(s*100)}", m, f"abs=0 scat={s} trans=0"))
    jobs.append(("e4_trans90", dict(base, transmission=curve(0.9)),
                 "abs=0 scat=0 trans=0.9"))
    sd = scene_wall("wall20", 20, 20)
    for name, mat, desc in jobs:
        write_materials(sd, f"m_{name}.json", [mat])
        sim, sen, cfg = build_sim(sd, max_ir=0.4, direct=False, mats=f"m_{name}.json")
        cfg["diffuse"] = mat["scattering"][1]
        out = exp_dir(name)
        ir = render(sim, sen, [0, EAR_Y, d], [0, EAR_Y, d])
        on, pk = save_case(out, "d01000_r00", ir, cfg, "wall 20x20x0.2m", d, 0, desc)
        sim.close()
        write_meta(out, cfg, extra=dict(material_curves=mat))
        print(f"{name}: onset={on} peak={pk:.3e}", flush=True)

E5_X = [4, 3, 2, 1, 0.5, 0.25, 0, -0.25, -0.5, -1, -2, -3, -4]

def drv_e5():
    sd = scene_screen()
    write_materials(sd, "m_s0.json", [dict(MIRROR)])
    write_materials(sd, "m_s1.json", [dict(MIRROR, scattering=curve(1.0))])
    for diff in (True, False):
        for s in (0, 1):
            name = f"e5_diff{'on' if diff else 'off'}_s{s}"
            sim, sen, cfg = build_sim(sd, max_ir=0.4, direct=False, diffraction=diff,
                                      mats=f"m_s{s}.json")
            cfg["diffuse"] = float(s)
            out = exp_dir(name)
            for x in E5_X:
                p = [float(x), EAR_Y, 5.0]
                ir = render(sim, sen, p, p)
                tag = f"x{'p' if x >= 0 else 'm'}{int(round(abs(x)*100)):04d}_r00"
                on, pk = save_case(out, tag, ir, cfg, "halfplane 30x40x0.2m edge@x0",
                                   x, 0, f"mirror abs=0 scat={s} trans=0")
                print(f"{name}: x={x:+5.2f} onset={on} peak={pk:.3e}", flush=True)
            sim.close()
            write_meta(out, cfg, extra=dict(x_positions=E5_X, layout=
                "one dir per condition; files W_x<p|m><XXXX>_r00.wav, XXXX=|x| in cm; manifest dist_m=x (signed)"))

E5C_REPEATS = 3
KNIFE_ZL = [4.0, 5.0, 6.0, 6.5, 7.0, 7.25, 7.4, 7.5, 7.6, 7.75, 8.0, 8.5, 9.5, 11.0]
KNIFE_SRC = [8.0, EAR_Y, 7.5]

def drv_e5c():
    """E5C — E5's half-plane screen re-run with meshSimplification ON and OFF.

    E5 concluded SS2.0's diffraction flag is a no-op. That measurement is CONFIRMED here,
    but its interpretation was wrong: E5 rendered with direct=False, and RLR's diffraction
    acts on the DIRECT path (it 'makes occlusion smoother'), not on the reflected field.
    E5 also left meshSimplification=0. Both are necessary; E5 had neither. This driver
    isolates the meshSimplification variable in E5's own geometry -- expected: still null,
    because the co-located src/rcv geometry has no occluded direct path to diffract.
    See drv_e5b for the geometry where diffraction does appear."""
    sd = scene_screen()
    write_materials(sd, "m_s0.json", [dict(MIRROR)])
    write_materials(sd, "m_s1.json", [dict(MIRROR, scattering=curve(1.0))])
    for simp in (True, False):
        for diff in (True, False):
            for s in (0, 1):
                name = f"e5c_diff{'on' if diff else 'off'}_s{s}_simp{int(simp)}"
                sim, sen, cfg = build_sim(sd, max_ir=0.4, direct=False, diffraction=diff,
                                          mats=f"m_s{s}.json", mesh_simp=simp)
                cfg["diffuse"] = float(s)
                out = exp_dir(name)
                for x in E5_X:
                    p = [float(x), EAR_Y, 5.0]
                    render(sim, sen, p, p)               # warm-up, discarded
                    for r in range(E5C_REPEATS):
                        ir = render(sim, sen, p, p)
                        tag = f"x{'p' if x >= 0 else 'm'}{int(round(abs(x)*100)):04d}_r{r:02d}"
                        on, pk = save_case(out, tag, ir, cfg, "halfplane 30x40x0.2m edge@x0",
                                           x, r, f"mirror abs=0 scat={s} trans=0")
                    print(f"{name}: x={x:+5.2f} onset={on} peak={pk:.3e}", flush=True)
                sim.close()
                write_meta(out, cfg, extra=dict(x_positions=E5_X, repeats=E5C_REPEATS,
                    warmup_discarded=True, layout=
                    "one dir per condition; files W_x<p|m><XXXX>_r<RR>.wav, XXXX=|x| in cm; manifest dist_m=x (signed)"))

def drv_e5b():
    """E5B — occlusion knife edge: the geometry where RLR's diffraction DOES appear.

    Sealed absorptive room, full-height barrier at x=15 spanning z in [0,7.5]; the only
    path from source to listener is around the vertical edge at z=7.5. Unlike scene_screen,
    the barrier OCCLUDES the direct path, so there is a true acoustic shadow.
    Sweeps direct x meshSimplification x diffraction: diffraction requires BOTH direct=True
    AND meshSimplification=True. Same config family as E5 (ambisonics o1, threads=1,
    rays=5000) so the result cannot be blamed on the rig."""
    sd = scene_knife()
    for direct in (True, False):
        for simp in (True, False):
            for diff in (True, False):
                name = f"e5b_knife_d{int(direct)}_simp{int(simp)}_diff{int(diff)}"
                sim, sen, cfg = build_sim(sd, max_ir=0.5, direct=direct, indirect=True,
                                          diffraction=diff, mats="m_knife.json",
                                          mesh_simp=simp, rays=5000, threads=1,
                                          channel="ambisonics", order=1)
                out = exp_dir(name)
                for z in KNIFE_ZL:
                    p = [22.0, EAR_Y, float(z)]
                    render(sim, sen, p, KNIFE_SRC)       # warm-up, discarded
                    for r in range(E5C_REPEATS):
                        ir = render(sim, sen, p, KNIFE_SRC)
                        tag = f"z{int(round(z*100)):04d}_r{r:02d}"
                        on, pk = save_case(out, tag, ir, cfg,
                                           "sealed room 30x6x20 + full-height barrier x=15 z=[0,7.5], edge z=7.5",
                                           z, r, "knifebar abs=0.95 scat=0.05 trans=0")
                    print(f"{name}: z_L={z:5.2f} onset={on} peak={pk:.3e}", flush=True)
                sim.close()
                write_meta(out, cfg, extra=dict(z_positions=KNIFE_ZL, repeats=E5C_REPEATS,
                    warmup_discarded=True, source=KNIFE_SRC, shadow_boundary_zL=7.5,
                    layout="one dir per condition; files W_z<ZZZZ>_r<RR>.wav, ZZZZ=z_L in cm; manifest dist_m=z_L"))

def drv_e6():
    sd = scene_box()
    src = [2.0, 0.83, 1.7]; rcv = [2.2, 0.83, 1.7]   # 20 cm x-offset receiver
    for a in (0.1, 0.3, 0.6):
        name = f"e6_box_a{int(a*100)}"
        write_materials(sd, f"m_a{int(a*100)}.json",
                        [dict(name="boxwall", labels=["boxwall"], absorption=curve(a),
                              scattering=curve(0), transmission=curve(0))])
        sim, sen, cfg = build_sim(sd, max_ir=2.5, mats=f"m_a{int(a*100)}.json")
        out = exp_dir(name)
        ir = render(sim, sen, rcv, src)
        on, pk = save_case(out, "d00020_r00", ir, cfg, "box 8x5x3m",
                           0.2, 0, f"uniform abs={a} scat=0 trans=0")
        sim.close()
        write_meta(out, cfg, extra=dict(src=src, rcv=rcv))
        print(f"{name}: onset={on} peak={pk:.3e}", flush=True)
    # corridor
    sim, sen, cfg = build_sim(scene_corridor(), max_ir=0.6)
    out = exp_dir("e6_corridor")
    p = [0.0, EAR_Y, 3.0]
    ir = render(sim, sen, p, p)
    save_case(out, "d00300_r00", ir, cfg, "corridor 2x(20x20) 10m apart",
              3.0, 0, "mirror abs=0 scat=0 trans=0")
    sim.close()
    write_meta(out, cfg, extra=dict(note="agent 3 m from wall A; image series 6,14,20,26,34,40 m"))
    print("e6_corridor done", flush=True)

def drv_e6_q():
    sd = scene_box()
    src = [2.0, 0.83, 1.7]; rcv = [2.2, 0.83, 1.7]
    write_materials(sd, "m_a10.json",
                    [dict(name="boxwall", labels=["boxwall"], absorption=curve(0.1),
                          scattering=curve(0), transmission=curve(0))])
    for rays in (512, 2048, 8192, 32768):
        name = f"e6_box_a10_qrays{rays}"
        sim, sen, cfg = build_sim(sd, max_ir=2.5, rays=rays, mats="m_a10.json")
        out = exp_dir(name)
        ir = render(sim, sen, rcv, src)
        save_case(out, "d00020_r00", ir, cfg, "box 8x5x3m", 0.2, 0,
                  "uniform abs=0.1 scat=0 trans=0")
        sim.close(); write_meta(out, cfg, extra=dict(src=src, rcv=rcv))
        print(f"{name} done", flush=True)
    for depth in (2, 5, 10, 50, 200):
        name = f"e6_box_a10_qdepth{depth}"
        sim, sen, cfg = build_sim(sd, max_ir=2.5, depth=depth, mats="m_a10.json")
        out = exp_dir(name)
        ir = render(sim, sen, rcv, src)
        save_case(out, "d00020_r00", ir, cfg, "box 8x5x3m", 0.2, 0,
                  "uniform abs=0.1 scat=0 trans=0")
        sim.close(); write_meta(out, cfg, extra=dict(src=src, rcv=rcv))
        print(f"{name} done", flush=True)

def drv_e7_ambi():
    sd = scene_wall("wall40", 40, 40)
    cases = [("e7_normal",  [0, EAR_Y, 10], [0, EAR_Y, 10]),
             ("e7_obl17R", [-3, EAR_Y, 10], [3, EAR_Y, 10]),
             ("e7_obl17L", [3, EAR_Y, 10], [-3, EAR_Y, 10]),
             ("e7_obl31R", [-6, EAR_Y, 10], [6, EAR_Y, 10])]
    for name, rcv, src in cases:
        sim, sen, cfg = build_sim(sd, max_ir=0.4, order=1)
        out = exp_dir(name)
        ir = render(sim, sen, rcv, src)
        d = float(np.linalg.norm(np.array(rcv) - np.array(src)))
        save_case(out, "d%05d_r00" % int(round(d*100)), ir, cfg,
                  "wall 40x40x0.2m", d, 0, "mirror abs=0 scat=0 trans=0")
        sim.close()
        write_meta(out, cfg, extra=dict(rcv=rcv, src=src,
                   note="agent yaw=0 faces -z (toward wall)"))
        print(f"{name} done", flush=True)

def drv_e7_binaural():
    sd = scene_free()
    out = exp_dir("e7_binaural")
    sim, sen, cfg = build_sim(sd, max_ir=0.2, channel="binaural")
    for az in (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90):
        a = math.radians(az)
        src = [2*math.sin(a), EAR_Y, -2*math.cos(a)]   # az>0 = source to the RIGHT
        ir = render(sim, sen, [0, EAR_Y, 0], src, yaw_deg=0.0)
        fs = int(cfg["sr"])
        fn = f"dir_az{'+' if az >= 0 else '-'}{abs(az):03d}.wav"
        wavfile.write(os.path.join(out, fn), fs, ir.T.astype(np.float32))
        on, pk = onset_of(np.max(np.abs(ir), axis=0))
        open(os.path.join(out, "manifest.csv"), "a").write(
            f"{fn},freespace,2.0000,{az},{fs},-1,{cfg['max_ir']},na,"
            f"{cfg['indirectRayCount']},{cfg['indirectRayDepth']},na,na,"
            f"binaural engine-default HRTF,{on},{on/fs*1000:.5f},{pk:.6e},default\n")
        print(f"e7_binaural az={az:+d} onset={on} peak={pk:.3e}", flush=True)
    sim.close()
    write_meta(out, cfg, extra=dict(
        note="repeat column carries azimuth (deg); az>0 = source right of nose; src at 2 m"))

def drv_e8():
    sd = scene_wall("wall20", 20, 20)
    for d, tag in ((10.00, "d1000"), (10.01, "d1001"), (10.10, "d1010"), (12.00, "d1200")):
        _sweep_wall(f"e8_{tag}", sd, [d], dict(max_ir=0.4, direct=False), "wall 20x20x0.2m")
    sdp = scene_wall("panel05", 0.5, 0.5)
    _sweep_wall("e8_panel05", sdp, [10.0], dict(max_ir=0.4, direct=False), "panel 0.5x0.5x0.2m")

def drv_e9():
    sd = scene_free()
    for d in (1, 2, 4, 8):
        name = f"e9_direct_s{d}m"
        sim, sen, cfg = build_sim(sd, max_ir=0.2)
        out = exp_dir(name)
        ir = render(sim, sen, [0, EAR_Y, 0], [0, EAR_Y, -float(d)])
        on, pk = save_case(out, f"d{int(d*100):05d}_r00", ir, cfg, "freespace",
                           float(d), 0, "n/a (direct path)")
        sim.close(); write_meta(out, cfg)
        print(f"{name}: onset={on} ({on/cfg['sr']*1000:.4f} ms, d/c={d/C_NOM*1000:.4f})",
              flush=True)
    # extra near points for the d->0 fixed-latency extrapolation
    sim, sen, cfg = build_sim(sd, max_ir=0.2)
    out = exp_dir("e9_direct_near")
    for d in (0.0625, 0.125, 0.25, 0.5):
        ir = render(sim, sen, [0, EAR_Y, 0], [0, EAR_Y, -float(d)])
        on, pk = save_case(out, f"d{int(round(d*10000)):05d}_r00", ir, cfg, "freespace",
                           float(d), 0, "n/a (direct path)")
        print(f"e9_near d={d}: onset={on} ({on/cfg['sr']*1000:.4f} ms)", flush=True)
    sim.close()
    write_meta(out, cfg, notes="Extra sub-metre direct-path points (not in manifest "
               "spec) to extrapolate onset at d->0. Filename distance field here is "
               "in 0.1 mm units (d00625 = 0.0625 m).\n")

def drv_e10():
    sd = scene_level4()
    sim, sen, cfg = build_sim(sd, max_ir=0.4, direct=False)
    out = exp_dir("level4_echo")
    for d in (10.0, 15.0, 20.0):
        p = [0.75, 1.7, -19.9 + d]
        ir = render(sim, sen, p, p)
        on, pk = save_case(out, f"d{int(d*100):05d}_r00", ir, cfg,
                           "level4 slab 31.9x32x0.2m face z=-19.9", d, 0,
                           "mirror abs=0 scat=0 trans=0")
        print(f"e10 d={d}: onset={on} ({on/cfg['sr']*1000:.3f} ms, "
              f"2d/c={2*d/C_NOM*1000:.3f})", flush=True)
    sim.close()
    write_meta(out, cfg, extra=dict(agent_xy=[0.75, 1.7]))

def drv_e10_fine():
    sd = scene_level4()
    sim, sen, cfg = build_sim(sd, max_ir=0.4, direct=False)
    out = exp_dir("level4_fine")
    for d in np.round(np.arange(10.0, 10.2 + 1e-9, 0.01), 4):
        p = [0.75, 1.7, -19.9 + float(d)]
        ir = render(sim, sen, p, p)
        on, pk = save_case(out, f"d{int(round(d*100)):05d}_r00", ir, cfg,
                           "level4 slab 31.9x32x0.2m face z=-19.9", float(d), 0,
                           "mirror abs=0 scat=0 trans=0")
        print(f"e10_fine d={d}: onset={on}", flush=True)
    sim.close()
    write_meta(out, cfg, extra=dict(agent_xy=[0.75, 1.7]))

# E11 — transmission probe. Question: does RLR CONTINUE transmitted rays into
# reflect-and-return paths, or only attenuate paths it already constructs?
# Prior evidence: e4_trans90 (abs=0 trans=0.9) cut the wall's own echo peak to
# 0.56x base, so the coefficient IS bound and drains the reflected path. The
# range-resolution experiment (2026-06-15) saw NO echo from a reflector behind a
# transmissive wall — observed once, mechanism unpinned. Two geometries decide it:
#   e11_direct: source BEHIND a large wall, listener in front. The through-wall
#     direct arrival at d/c vs material t tests whether transmission acts on the
#     direct path (and the config-flag toggle tests reachability).
#   e11_echo: co-located src/rcv, mirror reflector BEHIND the transmissive wall.
#     A t^2-scaled arrival at the reflector's round-trip ToA = rays continue and
#     re-transmit; silence at every t (even t=1, abs=0) = rays terminate at the
#     transmissive surface.
E11_T = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
E11_RCV = [0.0, EAR_Y, 3.0]          # echo geometry: wall echo @2*3/c=17.49 ms,
E11_MIRROR_D = 6.0                   # reflector face z=-3 -> echo @2*6/c=34.99 ms
E11_WIN_MS = 1.5

def scene_transpair():
    """Transmissive wall (z in [-0.2,0]) + mirror reflector (z in [-3.2,-3]), both 20x20."""
    sd = os.path.join(SCENE_ROOT, "transpair")
    if not os.path.exists(os.path.join(sd, "s.glb")):
        sb = SceneBuilder()
        sb.add_slab("transwall", -10, 10, EAR_Y - 10, EAR_Y + 10, -0.2, 0.0)
        sb.add_slab("mirror", -10, 10, EAR_Y - 10, EAR_Y + 10, -3.2, -3.0)
        sb.export(sd)
    return sd

def scene_transwall():
    """Transmissive wall alone (z in [-0.2,0], 20x20) for the through-wall direct test."""
    sd = os.path.join(SCENE_ROOT, "transwall20")
    if not os.path.exists(os.path.join(sd, "s.glb")):
        sb = SceneBuilder()
        sb.add_slab("transwall", -10, 10, EAR_Y - 10, EAR_Y + 10, -0.2, 0.0)
        sb.export(sd)
    return sd

def trans_mat(t, fam):
    """fam A: abs=0 (wall keeps its reflection budget); fam B: abs=1-t (non-reflective)."""
    a = 0.0 if fam == "A" else 1.0 - t
    return dict(name="transwall", labels=["transwall"], absorption=curve(a),
                scattering=curve(0), transmission=curve(t))

def _win_stats(w, fs, center_ms, win_ms=E11_WIN_MS):
    lo = max(0, int((center_ms - win_ms) / 1000 * fs))
    hi = min(len(w), int((center_ms + win_ms) / 1000 * fs))
    seg = w[lo:hi]
    if len(seg) == 0: return 0.0, 0.0, -1.0
    pk_i = int(np.argmax(np.abs(seg)))
    return float(np.max(np.abs(seg))), float(np.sum(seg**2)), (lo + pk_i) / fs * 1000

def _e11_case(name, scene, mats_file, rcv, src, cfg_kw, geo, dist, material, centers):
    sim, sen, cfg = build_sim(scene, **cfg_kw)
    out = exp_dir(name)
    fs = int(cfg["sr"])
    render(sim, sen, rcv, src)                       # warm-up, discarded
    rows = []
    for r in range(E5C_REPEATS):
        ir = render(sim, sen, rcv, src)
        save_case(out, f"d{int(round(dist*100)):05d}_r{r:02d}", ir, cfg, geo, dist, r, material)
        w = ir[0]
        rows.append({k: _win_stats(w, fs, ms) for k, ms in centers.items()})
    sim.close()
    write_meta(out, cfg, extra=dict(rcv=rcv, src=src, repeats=E5C_REPEATS,
               warmup_discarded=True, windows_ms={k: [v - E11_WIN_MS, v + E11_WIN_MS]
                                                  for k, v in centers.items()}))
    agg = {}
    for k in centers:
        agg[k] = dict(peak=float(np.mean([r[k][0] for r in rows])),
                      energy=float(np.mean([r[k][1] for r in rows])),
                      peak_ms=float(np.mean([r[k][2] for r in rows])))
    print(f"{name}: " + "  ".join(f"{k} pk={agg[k]['peak']:.3e} E={agg[k]['energy']:.3e} "
          f"@{agg[k]['peak_ms']:.2f}ms" for k in centers), flush=True)
    return agg

def drv_e11():
    summary = {}
    # --- e11_echo: reflector behind transmissive wall, co-located src/rcv (direct off)
    sd = scene_transpair()
    wall_ms = 2 * E11_RCV[2] / C_NOM * 1000
    mirr_ms = 2 * E11_MIRROR_D / C_NOM * 1000
    centers = dict(wall=wall_ms, mirror=mirr_ms, noise=26.0)
    for fam in ("A", "B"):
        for t in E11_T:
            mf = f"m_e11_{fam}_t{int(t*100):03d}.json"
            write_materials(sd, mf, [trans_mat(t, fam), dict(MIRROR)])
            name = f"e11_echo_{fam}_t{int(t*100):03d}"
            summary[name] = _e11_case(name, sd, mf, E11_RCV, E11_RCV,
                dict(max_ir=0.4, direct=False, mats=mf), "transwall+mirror 20x20, faces z=0/z=-3",
                E11_MIRROR_D, f"transwall fam{fam} t={t}; mirror behind", centers)
    # config-flag toggle at t=0.9 (both families)
    for fam in ("A", "B"):
        mf = f"m_e11_{fam}_t090.json"
        name = f"e11_echo_{fam}_t090_floff"
        summary[name] = _e11_case(name, sd, mf, E11_RCV, E11_RCV,
            dict(max_ir=0.4, direct=False, transmission=False, mats=mf),
            "transwall+mirror 20x20, faces z=0/z=-3", E11_MIRROR_D,
            f"transwall fam{fam} t=0.9; transmission flag OFF", centers)
    # amplitude reference: bare mirror wall at 6 m (existing wall20 scene)
    sdw = scene_wall("wall20", 20, 20)
    summary["e11_echo_ctrl6m"] = _e11_case("e11_echo_ctrl6m", sdw, "materials.json",
        [0.0, EAR_Y, 6.0], [0.0, EAR_Y, 6.0], dict(max_ir=0.4, direct=False),
        "wall 20x20x0.2m", 6.0, "mirror abs=0 scat=0 trans=0",
        dict(mirror=mirr_ms, noise=26.0))
    # --- e11_direct: source behind the wall, listener in front (direct on)
    sdt = scene_transwall()
    src = [0.0, EAR_Y, -4.0]; rcv = [0.0, EAR_Y, 2.0]
    d_ms = 6.0 / C_NOM * 1000
    dcent = dict(direct=d_ms, noise=30.0)
    for t in (0.0, 0.5, 0.9, 1.0):
        mf = f"m_e11d_B_t{int(t*100):03d}.json"
        write_materials(sdt, mf, [trans_mat(t, "B")])
        name = f"e11_direct_B_t{int(t*100):03d}"
        summary[name] = _e11_case(name, sdt, mf, rcv, src,
            dict(max_ir=0.4, direct=True, diffraction=False, mats=mf),
            "transwall 20x20x0.2m face z=0", 6.0, f"transwall famB t={t}", dcent)
    name = "e11_direct_B_t090_floff"
    summary[name] = _e11_case(name, sdt, "m_e11d_B_t090.json", rcv, src,
        dict(max_ir=0.4, direct=True, diffraction=False, transmission=False,
             mats="m_e11d_B_t090.json"),
        "transwall 20x20x0.2m face z=0", 6.0, "transwall famB t=0.9; flag OFF", dcent)
    summary["e11_direct_free"] = _e11_case("e11_direct_free", scene_free(), "materials.json",
        [0.0, EAR_Y, 0.0], [0.0, EAR_Y, -6.0], dict(max_ir=0.4, direct=True, diffraction=False),
        "freespace", 6.0, "n/a (unobstructed direct)", dcent)
    json.dump(summary, open(os.path.join(OUT_ROOT, "ss2_e11_stats.json"), "w"), indent=2)
    # verdict table
    ctrl = summary["e11_echo_ctrl6m"]["mirror"]["energy"]
    free = summary["e11_direct_free"]["direct"]["peak"]
    print("\n== E11 verdict ==")
    print(f"control mirror-echo energy @6m: {ctrl:.3e}   free direct peak: {free:.3e}")
    for fam in ("A", "B"):
        for t in E11_T:
            s = summary[f"e11_echo_{fam}_t{int(t*100):03d}"]
            print(f"echo fam{fam} t={t:4.2f}: mirror E/ctrl={s['mirror']['energy']/ctrl:9.4f} "
                  f"(t^2={t*t:5.3f})  wall E={s['wall']['energy']:.3e}  "
                  f"noise E={s['noise']['energy']:.3e}", flush=True)
    for t in (0.0, 0.5, 0.9, 1.0):
        s = summary[f"e11_direct_B_t{int(t*100):03d}"]
        print(f"direct famB t={t:4.2f}: pk/free={s['direct']['peak']/free:9.4f} (t={t})")

DRIVERS = {"smoke": drv_smoke, "e1_coarse": drv_e1_coarse, "e1_sr44100": drv_e1_sr44100,
           "e2": drv_e2, "e3": drv_e3, "e3_near": drv_e3_near, "e4": drv_e4,
           "e5": drv_e5, "e5b": drv_e5b, "e5c": drv_e5c,
           "e6": drv_e6, "e6_q": drv_e6_q, "e7_ambi": drv_e7_ambi,
           "e7_binaural": drv_e7_binaural, "e8": drv_e8, "e9": drv_e9,
           "e10": drv_e10, "e10_fine": drv_e10_fine, "e11": drv_e11}

if __name__ == "__main__":
    os.makedirs(SCENE_ROOT, exist_ok=True)
    for arg in sys.argv[1:]:
        t0 = time.time()
        if arg.startswith("e1_fine:"):
            _, lo, hi, step, name = arg.split(":")
            drv_e1_fine(float(lo), float(hi), float(step), name)
        elif arg.startswith("e2_fresh:"):
            drv_e2_fresh(int(arg.split(":")[1]))
        else:
            DRIVERS[arg]()
        print(f"== {arg} finished in {time.time()-t0:.1f} s ==", flush=True)
