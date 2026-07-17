// ---------------------------------------------------------------------------
//  acoustic_env.h  (header-only)
//
//  The Acoustic-JEPA environment: the pose -> binaural-echo observation
//  operator, refactored out of echo_nav.cpp::sense(), plus a procedural
//  box-room generator + analytic geometry labeller. See ACOUSTIC_JEPA_SPEC.md.
//
//  The phonon render sequence in AcousticEnv::observe() is byte-for-byte the
//  reflection path from sense() (re-run reflections -> ambisonic IR from a unit
//  impulse -> binaural decode). Engine gotchas 1/4/5 are preserved; see the
//  steam-audio-ir-tool notes.
//
//  Header-only so a dumper .cpp (jepa_dump.cpp) can just #include it. Meant to
//  live in ONE translation unit per exe (like the other SteamAudioIR tools);
//  the members are implicitly inline.
//
//  Self-test:  cl /EHsc /O2 /DACOUSTIC_ENV_TEST acoustic_env.h ... phonon.lib
//  (renames to .cpp or use /Tp). Renders one room and prints a sanity line.
// ---------------------------------------------------------------------------
#pragma once
#define _USE_MATH_DEFINES
#include <phonon.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace acoustic_env {

// ---- render config (mirrors echo_nav.cpp) ---------------------------------
static const int   SR = 48000, FRAME = 1024, ORDER = 3, AMBI = (ORDER + 1) * (ORDER + 1); // 16
static const float IR_DUR = 0.5f, C = 343.0f;
static const int   IRN = (int)(IR_DUR * SR);      // 24000
static const int   TOTAL = IRN + FRAME;           // 25024
static const int   NUM_RAYS = 131072, NUM_BOUNCES = 4, NUM_DIFFUSE = 2048;

// ---- observation dims (ACOUSTIC_JEPA_SPEC.md §0; v2 adds the range tower) -
static const float T_OBS = 0.2f;                  // crop window
static const int   RAW_N = (int)(T_OBS * SR);     // 9600 samples/ear
static const int   COCH_F = 64;                   // gammatone bands
static const int   COCH_T = 48;                   // time frames
static const int   COCH_HOP = RAW_N / COCH_T;     // 200 samples/frame

// v2: matched-filter (ESS + Farina inverse) range/reflectivity profile, built
// from the pre-binaural W channel -- see MatchedFilter below. Range resolution
// here is set by the matched-filter leading-edge walk (sub-decimetre), NOT the
// engine's native 10 ms / 1.7 m reflection-histogram bin (see steam-audio-ir-
// tool notes, Gotcha 5): this is what makes the range tower worth adding.
static const int   RANGE_BINS = 256;              // output lag-profile length
static const int   RANGE_POOL = 16;               // samples/bin (@48kHz -> ~5.7cm/bin one-way)
static const int   RANGE_GUARD = 96;               // 2ms: clears the self-cancel residual skirt

struct Obs {
    float coch [2 * COCH_F * COCH_T];              // [ear, band, frame], log energy
    float raw  [2 * RAW_N];                         // [ear, sample], cropped binaural
    float range[RANGE_BINS];                        // v2: matched-filter lag-domain profile (log energy)
};
struct Label {
    float dist_wall[8];   // GT range to a wall in 8 yaw sectors (rel. heading), m
    float open_azim;      // allocentric azimuth of largest opening, deg (NAN if none)
    float room_vol;       // m^3
    int   material_id;    // material of nearest wall
    float pose[3];        // eval-only
    float yaw;            // eval-only, deg
};
struct Action { float dyaw;   /* deg, [-45,45] */ float dstep; /* m, [0,1.5] */ };

// ---- small helpers ---------------------------------------------------------
static inline IPLVector3 V(float x, float y, float z) { IPLVector3 v{ x, y, z }; return v; }

// agent heading: ahead = (sin th, 0, -cos th); right = (cos th, 0, sin th)  [as echo_nav]
static inline IPLCoordinateSpace3 headFrame(IPLVector3 o, float thDeg) {
    float t = thDeg * (float)M_PI / 180.f; IPLCoordinateSpace3 c;
    c.ahead = V(std::sin(t), 0, -std::cos(t)); c.up = V(0, 1, 0);
    c.right = V(std::cos(t), 0, std::sin(t)); c.origin = o; return c;
}
static inline void chk(IPLerror e, const char* w) {
    if (e) { std::fprintf(stderr, "FATAL %s=%d\n", w, e); std::exit(1); }
}

// reproducible RNG (xorshift32) so a room is a pure function of its seed
struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed ? seed : 0x9E3779B9u) {}
    float uf() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (s & 0xFFFFFF) / (float)0x1000000; }
    float rf(float a, float b) { return a + (b - a) * uf(); }
    int   ri(int a, int b) { return a + (int)(uf() * (b - a + 1)); } // inclusive
};

// ===========================================================================
//  Materials
// ===========================================================================
static const int NUM_MATERIALS = 4;
// IPLMaterial = { float absorption[3]; float scattering; float transmission[3]; }
static inline const IPLMaterial* material_table() {
    static IPLMaterial m[NUM_MATERIALS] = {
        {{0.10f, 0.15f, 0.20f}, 0.20f, {0.05f, 0.05f, 0.05f}}, // 0 plaster
        {{0.01f, 0.01f, 0.02f}, 0.10f, {0.01f, 0.01f, 0.01f}}, // 1 hard / tile
        {{0.30f, 0.50f, 0.60f}, 0.50f, {0.10f, 0.10f, 0.10f}}, // 2 absorptive / carpet
        {{0.00f, 0.00f, 0.00f}, 0.00f, {0.00f, 0.00f, 0.00f}}, // 3 perfect mirror
    };
    return m;
}

// ===========================================================================
//  Procedural box room  (STUB: closed box, openings sampled but not yet carved)
// ===========================================================================
// wall index convention: 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z
struct Opening { int wall; IPLVector3 center; float w, h; };  // world-space center
// Interior obstacle: an axis-aligned box (center + half-extents), its own
// material. Obstacles are what stop the cross-view task from being solvable by
// regressing the ~10 box-room parameters -- they add interior structure only a
// truly allocentric representation can predict across viewpoints, and they let
// agent pairs occlude one another. Default rooms have none (nObstacles=0), so
// the monostatic jepa_dump dataset is byte-identical to before.
struct Obstacle { IPLVector3 center, half; int material; };
struct BoxRoom {
    float Lx, Ly, Lz;              // interior dims (m)
    IPLVector3 origin;             // min corner; room spans origin .. origin+(Lx,Ly,Lz)
    int  wall_material[6];
    std::vector<Opening>  openings;  // recorded for labels; carved into the mesh
    std::vector<Obstacle> obstacles; // interior AABB reflectors (v2 cross-view)
};

// segment (a->b) vs AABB (slab method); used for agent-agent occlusion tests
inline bool segment_hits_aabb(IPLVector3 a, IPLVector3 b, IPLVector3 c, IPLVector3 h) {
    float tmin = 0.f, tmax = 1.f;
    float o[3] = { a.x, a.y, a.z }, d[3] = { b.x - a.x, b.y - a.y, b.z - a.z };
    float lo[3] = { c.x - h.x, c.y - h.y, c.z - h.z }, hi[3] = { c.x + h.x, c.y + h.y, c.z + h.z };
    for (int k = 0; k < 3; ++k) {
        if (std::fabs(d[k]) < 1e-9f) { if (o[k] < lo[k] || o[k] > hi[k]) return false; continue; }
        float inv = 1.f / d[k];
        float t1 = (lo[k] - o[k]) * inv, t2 = (hi[k] - o[k]) * inv;
        if (t1 > t2) std::swap(t1, t2);
        tmin = std::max(tmin, t1); tmax = std::min(tmax, t2);
        if (tmin > tmax) return false;
    }
    return true;
}

inline BoxRoom make_box_room(uint32_t seed, int nObstacles = 0) {
    Rng r(seed);
    BoxRoom rm{};
    rm.Lx = r.rf(3.f, 12.f);
    rm.Ly = r.rf(2.5f, 4.f);
    rm.Lz = r.rf(3.f, 12.f);
    rm.origin = V(-rm.Lx * 0.5f, 0.f, -rm.Lz * 0.5f);   // centred on x/z, floor at y=0
    for (int i = 0; i < 6; ++i) rm.wall_material[i] = r.ri(0, NUM_MATERIALS - 1);

    // Openings go on DISTINCT side walls (at most one per wall) so build_room_mesh
    // only ever carves a single rectangular hole per wall. Placement keeps a MARGIN
    // clear of every wall edge so the hole is fully interior and the surrounding
    // frame quads stay non-degenerate.
    const float x0 = rm.origin.x, x1 = x0 + rm.Lx;
    const float y0 = rm.origin.y, y1 = y0 + rm.Ly;
    const float z0 = rm.origin.z, z1 = z0 + rm.Lz;
    const float MARGIN = 0.4f;

    int sideWalls[4] = { 0, 1, 4, 5 };
    for (int i = 3; i > 0; --i) { int j = r.ri(0, i); std::swap(sideWalls[i], sideWalls[j]); } // Fisher-Yates
    int nOpen = r.ri(0, 3);
    for (int k = 0; k < nOpen; ++k) {
        Opening o{}; o.wall = sideWalls[k];
        float wallLenU = (o.wall == 0 || o.wall == 1) ? rm.Lz : rm.Lx;   // horizontal extent of this wall
        o.w = std::min(r.rf(0.8f, 2.0f), wallLenU - 2.f * MARGIN);
        o.h = std::min(r.rf(1.0f, 2.0f), rm.Ly - 2.f * MARGIN);
        if (o.w <= 0.2f || o.h <= 0.2f) continue;                        // wall too small to hole
        float cy = r.rf(y0 + MARGIN + o.h * 0.5f, y1 - MARGIN - o.h * 0.5f);
        float lo, hi;
        if (o.wall == 0 || o.wall == 1) { lo = z0 + MARGIN + o.w * 0.5f; hi = z1 - MARGIN - o.w * 0.5f; }
        else                            { lo = x0 + MARGIN + o.w * 0.5f; hi = x1 - MARGIN - o.w * 0.5f; }
        float cu = r.rf(lo, hi);
        if (o.wall == 0)      o.center = V(x1, cy, cu); // +x
        else if (o.wall == 1) o.center = V(x0, cy, cu); // -x
        else if (o.wall == 4) o.center = V(cu, cy, z1); // +z
        else                  o.center = V(cu, cy, z0); // -z
        rm.openings.push_back(o);
    }

    // Interior obstacles (v2). Floor-standing AABB "pillars/furniture", kept
    // clear of the walls and non-overlapping, a random material each. Sampled
    // AFTER openings so obstacle presence never perturbs the wall/opening RNG
    // stream -- make_box_room(seed, 0) is unchanged. Pillars stand ON the floor
    // and rise ABOVE ear height (1.6 m) so they actually occlude agent-to-agent
    // sightlines (the point of having them for the cross-view task); their tops
    // vary so not every pair is blocked.
    const float OB_CLEAR = 0.6f;   // min gap from any wall
    for (int k = 0; k < nObstacles; ++k) {
        Obstacle ob{};
        bool placed = false;
        for (int tries = 0; tries < 60 && !placed; ++tries) {
            float hx = r.rf(0.2f, 0.6f), hz = r.rf(0.2f, 0.6f);
            float top = std::min(rm.Ly - 0.1f, r.rf(1.7f, 2.4f));   // rises past ear height
            float hy  = top * 0.5f;
            float cyv = y0 + hy;                                    // bottom sits on the floor
            float cx  = r.rf(x0 + OB_CLEAR + hx, x1 - OB_CLEAR - hx);
            float cz  = r.rf(z0 + OB_CLEAR + hz, z1 - OB_CLEAR - hz);
            if (x1 - OB_CLEAR - hx <= x0 + OB_CLEAR + hx) break;   // room too small
            if (z1 - OB_CLEAR - hz <= z0 + OB_CLEAR + hz) break;
            IPLVector3 c = V(cx, cyv, cz), h = V(hx, hy, hz);
            bool overlaps = false;
            for (const auto& p : rm.obstacles) {
                if (std::fabs(c.x - p.center.x) < h.x + p.half.x + 0.3f &&
                    std::fabs(c.y - p.center.y) < h.y + p.half.y + 0.3f &&
                    std::fabs(c.z - p.center.z) < h.z + p.half.z + 0.3f) { overlaps = true; break; }
            }
            if (overlaps) continue;
            ob.center = c; ob.half = h; ob.material = r.ri(0, NUM_MATERIALS - 1);
            rm.obstacles.push_back(ob); placed = true;
        }
    }
    return rm;
}

// is the straight segment a->b blocked by any interior obstacle? (agent
// occlusion test for cross-view pair sampling)
inline bool room_occludes(const BoxRoom& rm, IPLVector3 a, IPLVector3 b) {
    for (const auto& ob : rm.obstacles)
        if (segment_hits_aabb(a, b, ob.center, ob.half)) return true;
    return false;
}

// ---- analytic GT geometry labels for a pose in a box room -----------------
// Shared by AcousticEnv (monostatic) and SwarmEnv (cross-view) so both label
// identically. Wall distances are for the empty box; interior obstacles are NOT
// reflected in dist_wall (they show up acoustically, and obstacle-aware targets
// are computed separately where needed).
inline Label analytic_label(const BoxRoom& room, IPLVector3 pose, float yawDeg) {
    Label lb{}; lb.pose[0] = pose.x; lb.pose[1] = pose.y; lb.pose[2] = pose.z; lb.yaw = yawDeg;
    const float x0 = room.origin.x, x1 = x0 + room.Lx;
    const float z0 = room.origin.z, z1 = z0 + room.Lz;

    for (int k = 0; k < 8; ++k) {
        float a = (yawDeg + 45.f * k) * (float)M_PI / 180.f;
        float dx = std::sin(a), dz = -std::cos(a);          // ahead convention
        float tx = dx > 1e-6f ? (x1 - pose.x) / dx : (dx < -1e-6f ? (x0 - pose.x) / dx : 1e30f);
        float tz = dz > 1e-6f ? (z1 - pose.z) / dz : (dz < -1e-6f ? (z0 - pose.z) / dz : 1e30f);
        lb.dist_wall[k] = std::min(tx, tz);
    }

    lb.room_vol = room.Lx * room.Ly * room.Lz;

    float dpx = x1 - pose.x, dnx = pose.x - x0, dpz = z1 - pose.z, dnz = pose.z - z0;
    float best = dpx; int wall = 0;
    if (dnx < best) { best = dnx; wall = 1; }
    if (dpz < best) { best = dpz; wall = 4; }
    if (dnz < best) { best = dnz; wall = 5; }
    lb.material_id = room.wall_material[wall];

    lb.open_azim = NAN;
    float bestArea = 0;
    for (const auto& op : room.openings) {
        float area = op.w * op.h;
        if (area > bestArea) {
            bestArea = area;
            lb.open_azim = std::atan2(op.center.x - pose.x, -(op.center.z - pose.z)) * 180.f / (float)M_PI;
        }
    }
    return lb;
}

// Build the room mesh: 6 walls, with any sampled opening cut out as a rectangular
// hole. A holed wall becomes a frame of up to 4 quads (bottom/top/left/right)
// around the hole; a solid wall is a single quad. Floor/ceiling never hole.
// Openings make the box acoustically open (rays through the hole don't return),
// which is what makes opening direction a learnable cue.
inline void build_room_mesh(const BoxRoom& rm,
                            std::vector<IPLVector3>& verts,
                            std::vector<IPLTriangle>& tris,
                            std::vector<IPLint32>&   matIdx) {
    verts.clear(); tris.clear(); matIdx.clear();
    const float x0 = rm.origin.x, x1 = x0 + rm.Lx;
    const float y0 = rm.origin.y, y1 = y0 + rm.Ly;
    const float z0 = rm.origin.z, z1 = z0 + rm.Lz;

    auto addQuad = [&](IPLVector3 a, IPLVector3 b, IPLVector3 c, IPLVector3 d, int mat) {
        int base = (int)verts.size();
        verts.push_back(a); verts.push_back(b); verts.push_back(c); verts.push_back(d);
        tris.push_back(IPLTriangle{ { base, base + 1, base + 2 } }); matIdx.push_back(mat);
        tris.push_back(IPLTriangle{ { base, base + 2, base + 3 } }); matIdx.push_back(mat);
    };

    // Emit a wall over [uMin,uMax]x[vMin,vMax] via pt(u,v)->3D. If it has a hole
    // [hu0,hu1]x[hv0,hv1], emit the 4 surrounding strips (skipping degenerate ones);
    // reflections are side-agnostic so winding is not critical.
    auto addWall = [&](auto pt, float uMin, float uMax, float vMin, float vMax,
                       bool hole, float hu0, float hu1, float hv0, float hv1, int mat) {
        const float e = 1e-3f;
        if (!hole) { addQuad(pt(uMin, vMin), pt(uMax, vMin), pt(uMax, vMax), pt(uMin, vMax), mat); return; }
        if (hv0 > vMin + e) addQuad(pt(uMin, vMin), pt(uMax, vMin), pt(uMax, hv0), pt(uMin, hv0), mat); // bottom
        if (vMax > hv1 + e) addQuad(pt(uMin, hv1), pt(uMax, hv1), pt(uMax, vMax), pt(uMin, vMax), mat); // top
        if (hu0 > uMin + e) addQuad(pt(uMin, hv0), pt(hu0, hv0), pt(hu0, hv1), pt(uMin, hv1), mat);     // left
        if (uMax > hu1 + e) addQuad(pt(hu1, hv0), pt(uMax, hv0), pt(uMax, hv1), pt(hu1, hv1), mat);     // right
    };

    const Opening* wop[6] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };
    for (const auto& op : rm.openings) wop[op.wall] = &op;
    // hole rect in (u,v): horizontal axis u is z for x-walls, x for z-walls; v is y
    auto holeUV = [&](const Opening* o, bool xWall, float& hu0, float& hu1, float& hv0, float& hv1) {
        float cu = xWall ? o->center.z : o->center.x;
        hu0 = cu - o->w * 0.5f; hu1 = cu + o->w * 0.5f;
        hv0 = o->center.y - o->h * 0.5f; hv1 = o->center.y + o->h * 0.5f;
    };
    float a0 = 0, a1 = 0, b0 = 0, b1 = 0;

    // floor (-y) and ceiling (+y): solid
    addWall([&](float u, float v) { return V(u, y0, v); }, x0, x1, z0, z1, false, 0, 0, 0, 0, rm.wall_material[3]);
    addWall([&](float u, float v) { return V(u, y1, v); }, x0, x1, z0, z1, false, 0, 0, 0, 0, rm.wall_material[2]);
    // +x (x=x1): u=z, v=y
    if (wop[0]) holeUV(wop[0], true, a0, a1, b0, b1);
    addWall([&](float u, float v) { return V(x1, v, u); }, z0, z1, y0, y1, wop[0] != nullptr, a0, a1, b0, b1, rm.wall_material[0]);
    // -x (x=x0)
    if (wop[1]) holeUV(wop[1], true, a0, a1, b0, b1);
    addWall([&](float u, float v) { return V(x0, v, u); }, z0, z1, y0, y1, wop[1] != nullptr, a0, a1, b0, b1, rm.wall_material[1]);
    // +z (z=z1): u=x, v=y
    if (wop[4]) holeUV(wop[4], false, a0, a1, b0, b1);
    addWall([&](float u, float v) { return V(u, v, z1); }, x0, x1, y0, y1, wop[4] != nullptr, a0, a1, b0, b1, rm.wall_material[4]);
    // -z (z=z0)
    if (wop[5]) holeUV(wop[5], false, a0, a1, b0, b1);
    addWall([&](float u, float v) { return V(u, v, z0); }, x0, x1, y0, y1, wop[5] != nullptr, a0, a1, b0, b1, rm.wall_material[5]);

    // interior obstacles: each an axis-aligned box (6 solid quads, closed so the
    // agent gets echoes from every visible face; winding is not critical for
    // reflections)
    for (const auto& ob : rm.obstacles) {
        float ax = ob.center.x - ob.half.x, bx = ob.center.x + ob.half.x;
        float ay = ob.center.y - ob.half.y, by = ob.center.y + ob.half.y;
        float az = ob.center.z - ob.half.z, bz = ob.center.z + ob.half.z;
        int mt = ob.material;
        addQuad(V(ax, ay, az), V(bx, ay, az), V(bx, by, az), V(ax, by, az), mt); // -z
        addQuad(V(ax, ay, bz), V(bx, ay, bz), V(bx, by, bz), V(ax, by, bz), mt); // +z
        addQuad(V(ax, ay, az), V(ax, ay, bz), V(ax, by, bz), V(ax, by, az), mt); // -x
        addQuad(V(bx, ay, az), V(bx, ay, bz), V(bx, by, bz), V(bx, by, az), mt); // +x
        addQuad(V(ax, ay, az), V(bx, ay, az), V(bx, ay, bz), V(ax, ay, bz), mt); // -y
        addQuad(V(ax, by, az), V(bx, by, az), V(bx, by, bz), V(ax, by, bz), mt); // +y
    }
}

// ===========================================================================
//  Matched-filter (ESS + Farina inverse) range tower  (v2)
//
//  Ports matched_range.py's make_ess / received / est_mf pipeline to C++ so it
//  can run inline in observe(). Same algorithm, verified there against the
//  engine (Report_MatchedFilter_Ranging.docx): a 100Hz-10kHz/50ms exponential
//  sweep is convolved with the engine's own W-channel echo IR to synthesise
//  what a real ping would sound like; the self-heard emission is subtracted
//  (1% amplitude miscalibration, matching the validated experiment) and the
//  result is convolved with the Farina inverse filter, giving a pulse-
//  compressed reflectivity profile whose zero-delay reference is sample
//  z0 = len(ess)-1. Unlike matched_range.py this returns the WHOLE profile
//  (a lag-domain "sonar image"), not a single scalar range estimate -- the
//  point of the v2 range tower is to hand the encoder structure to discover,
//  not to hand it a pre-decided range number.
// ===========================================================================
using cplx = std::complex<double>;

// iterative radix-2 Cooley-Tukey (standard form; n must be a power of 2)
static void fft_inplace(std::vector<cplx>& a, bool invert) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len * (invert ? -1.0 : 1.0);
        cplx wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            cplx w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                cplx u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) for (auto& x : a) x /= (double)n;
}

// linear convolution via zero-padded FFT; result length = len(a)+len(b)-1
static std::vector<float> fftconv(const std::vector<float>& a, const std::vector<float>& b) {
    int rn = (int)a.size() + (int)b.size() - 1;
    int n = 1; while (n < rn) n <<= 1;
    std::vector<cplx> fa(n, cplx(0, 0)), fb(n, cplx(0, 0));
    for (size_t i = 0; i < a.size(); ++i) fa[i] = cplx(a[i], 0.0);
    for (size_t i = 0; i < b.size(); ++i) fb[i] = cplx(b[i], 0.0);
    fft_inplace(fa, false); fft_inplace(fb, false);
    for (int i = 0; i < n; ++i) fa[i] *= fb[i];
    fft_inplace(fa, true);
    std::vector<float> res(rn);
    for (int i = 0; i < rn; ++i) res[i] = (float)fa[i].real();
    return res;
}

struct MatchedFilter {
    std::vector<float> ess, inv;   // sweep + Farina inverse (unit-peak self-compression)
    int N = 0;                     // ess length (samples)

    void init(float f1 = 100.f, float f2 = 10000.f, float T = 0.050f, int sr = SR) {
        N = (int)(T * sr);
        std::vector<double> t(N);
        for (int i = 0; i < N; ++i) t[i] = i / (double)sr;
        double w1 = 2.0 * M_PI * f1, w2 = 2.0 * M_PI * f2;
        double L = T / std::log(w2 / w1), K = w1 * L;
        ess.resize(N); inv.resize(N);
        for (int i = 0; i < N; ++i) ess[i] = (float)std::sin(K * (std::exp(t[i] / L) - 1.0));
        for (int i = 0; i < N; ++i) inv[i] = (float)(ess[N - 1 - i] * std::exp(-t[i] / L));
        std::vector<float> self = fftconv(ess, inv);
        float mx = 0.f; for (float v : self) mx = std::max(mx, std::fabs(v));
        if (mx > 1e-12f) for (float& v : inv) v /= mx;
    }

    // W = engine's materialised W-channel echo IR (any length). Returns the
    // pulse-compressed |z| profile starting RANGE_GUARD samples after the
    // zero-delay reference (clears the self-cancel residual skirt), length
    // >= RANGE_BINS*RANGE_POOL + a small margin; caller pools/crops it.
    std::vector<float> profile(const std::vector<float>& W) const {
        std::vector<float> y = fftconv(ess, W);                 // room response to the ping
        for (int i = 0; i < N && i < (int)y.size(); ++i) y[i] += ess[i];             // self-heard emission
        for (int i = 0; i < N && i < (int)y.size(); ++i) y[i] -= 1.01f * ess[i];     // self-cancel (1% resid)
        std::vector<float> z = fftconv(y, inv);
        for (float& v : z) v = std::fabs(v);
        int z0 = N - 1;
        int start = z0 + RANGE_GUARD;
        int want = RANGE_BINS * RANGE_POOL;
        std::vector<float> out(want, 0.f);
        for (int i = 0; i < want && start + i < (int)z.size(); ++i) out[i] = z[start + i];
        return out;
    }

    // pool the raw profile into RANGE_BINS log-energy bins (same log1p style
    // as Cochleagram::apply, so both towers feed the encoder on a similar scale)
    void pool_log(const std::vector<float>& raw, float* out) const {
        for (int b = 0; b < RANGE_BINS; ++b) {
            double e = 0; int s0 = b * RANGE_POOL, s1 = std::min((int)raw.size(), s0 + RANGE_POOL);
            for (int i = s0; i < s1; ++i) e += (double)raw[i] * raw[i];
            e /= std::max(1, s1 - s0);
            out[b] = std::log1p((float)e * 1e4f);
        }
    }
};

// ===========================================================================
//  Gammatone cochleagram  (approx: ERB-spaced RBJ band-pass bank; a proper
//  Slaney 4th-order gammatone can drop in later without changing the interface)
// ===========================================================================
struct Cochleagram {
    // biquad coeffs per band (RBJ constant-0dB band-pass), computed once
    float b0[COCH_F], b2[COCH_F], a1[COCH_F], a2[COCH_F];
    Cochleagram() {
        auto erbRate = [](float f) { return 21.4f * std::log10(0.00437f * f + 1.f); };
        auto invErb  = [](float e) { return (std::pow(10.f, e / 21.4f) - 1.f) / 0.00437f; };
        float e_lo = erbRate(50.f), e_hi = erbRate(20000.f);
        for (int k = 0; k < COCH_F; ++k) {
            float fc  = invErb(e_lo + (e_hi - e_lo) * k / (float)(COCH_F - 1));
            float erb = 24.7f * (0.00437f * fc + 1.f);           // Glasberg-Moore ERB(fc)
            float Q   = fc / erb;
            float w0  = 2.f * (float)M_PI * fc / SR;
            float al  = std::sin(w0) / (2.f * Q);
            float a0  = 1.f + al;
            b0[k] =  al / a0;  b2[k] = -al / a0;                 // b1 = 0
            a1[k] = -2.f * std::cos(w0) / a0;  a2[k] = (1.f - al) / a0;
        }
    }
    // one ear -> COCH_F x COCH_T log-energy frames
    void apply(const float* x, int n, float* out) const {
        std::vector<float> band(n);
        for (int k = 0; k < COCH_F; ++k) {
            float w1 = 0, w2 = 0;                                // transposed-DF2 state
            for (int i = 0; i < n; ++i) {
                float in = x[i];
                float y  = b0[k] * in + w1;
                w1 = w2 - a1[k] * y;
                w2 = -a2[k] * y + b2[k] * in;                    // (b1=0)
                band[i] = y;
            }
            for (int t = 0; t < COCH_T; ++t) {
                double e = 0; int s0 = t * COCH_HOP, s1 = std::min(n, s0 + COCH_HOP);
                for (int i = s0; i < s1; ++i) e += (double)band[i] * band[i];
                e /= std::max(1, s1 - s0);
                out[k * COCH_T + t] = std::log1p((float)e * 1e4f);
            }
        }
    }
};

// ===========================================================================
//  Environment
// ===========================================================================
struct AcousticEnv {
    IPLContext ctx = nullptr;
    IPLHRTF    hrtf = nullptr;
    IPLSimulator sim = nullptr;
    IPLSource    src = nullptr;
    IPLReflectionEffect      refl = nullptr;
    IPLAmbisonicsDecodeEffect dec = nullptr;
    IPLScene      scene = nullptr;
    IPLStaticMesh mesh = nullptr;
    BoxRoom room{};
    bool    has_room = false;
    Cochleagram  coch;   // filterbank coeffs, built once
    MatchedFilter mf;    // ESS + Farina inverse, built once (v2 range tower)

    // Displaced emitter (bistatic sonar). When set, the sound source sits at a
    // fixed `beacon` in the room rather than co-located with the listener, so the
    // received echo depends on emitter+wall+agent geometry (parallax) -- this is
    // the pretext-task pressure that forces the latent to encode where things are.
    // Unset => monostatic (source == listener), the original behaviour.
    IPLVector3 beacon{ 0, 0, 0 };
    bool       has_beacon = false;
    void set_beacon(IPLVector3 p) { beacon = p; has_beacon = true; }
    void clear_beacon() { has_beacon = false; }

    // ---- one-time setup (scene-independent handles), mirrors echo_nav main() ----
    // simThreads = phonon's internal reflection thread count; pass 0 for all cores,
    // or a small value when several AcousticEnv run in parallel (avoid oversubscription).
    void init(int simThreads = 0) {
        IPLContextSettings cs{}; cs.version = STEAMAUDIO_VERSION; cs.simdLevel = IPL_SIMDLEVEL_AVX2;
        chk(iplContextCreate(&cs, &ctx), "ctx");
        IPLAudioSettings au{ SR, FRAME };
        IPLHRTFSettings hs{}; hs.type = IPL_HRTFTYPE_DEFAULT; hs.volume = 1.f; hs.normType = IPL_HRTFNORMTYPE_NONE;
        chk(iplHRTFCreate(ctx, &au, &hs, &hrtf), "hrtf");

        IPLSimulationSettings ss{}; ss.flags = IPL_SIMULATIONFLAGS_REFLECTIONS; ss.sceneType = IPL_SCENETYPE_DEFAULT;
        ss.reflectionType = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION; ss.maxNumRays = NUM_RAYS; ss.numDiffuseSamples = NUM_DIFFUSE;
        ss.maxDuration = IR_DUR; ss.maxOrder = ORDER; ss.maxNumSources = 1;
        ss.numThreads = (IPLint32)(simThreads > 0 ? simThreads : std::max(1u, std::thread::hardware_concurrency()));
        ss.samplingRate = SR; ss.frameSize = FRAME;
        chk(iplSimulatorCreate(ctx, &ss, &sim), "sim");
        IPLSourceSettings se{}; se.flags = IPL_SIMULATIONFLAGS_REFLECTIONS;
        chk(iplSourceCreate(sim, &se, &src), "src"); iplSourceAdd(src, sim);
        // (commit happens in set_room once a scene exists)

        IPLReflectionEffectSettings rs{}; rs.type = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION; rs.irSize = IRN; rs.numChannels = AMBI;
        chk(iplReflectionEffectCreate(ctx, &au, &rs, &refl), "refl");
        IPLAmbisonicsDecodeEffectSettings ds{}; ds.speakerLayout.type = IPL_SPEAKERLAYOUTTYPE_STEREO;
        ds.hrtf = hrtf; ds.maxOrder = ORDER;
        chk(iplAmbisonicsDecodeEffectCreate(ctx, &au, &ds, &dec), "dec");
        mf.init();   // v2: ESS + Farina inverse for the range tower
    }

    // ---- swap in a procedural room; releases the previous scene/mesh ----
    void set_room(const BoxRoom& rm) {
        std::vector<IPLVector3> verts; std::vector<IPLTriangle> tris; std::vector<IPLint32> matIdx;
        build_room_mesh(rm, verts, tris, matIdx);

        IPLSceneSettings ss{}; ss.type = IPL_SCENETYPE_DEFAULT;
        IPLScene newScene = nullptr; chk(iplSceneCreate(ctx, &ss, &newScene), "scene");
        IPLStaticMeshSettings ms{}; ms.numVertices = (int)verts.size(); ms.numTriangles = (int)tris.size();
        ms.numMaterials = NUM_MATERIALS; ms.vertices = verts.data(); ms.triangles = tris.data();
        ms.materialIndices = matIdx.data(); ms.materials = const_cast<IPLMaterial*>(material_table());
        IPLStaticMesh newMesh = nullptr; chk(iplStaticMeshCreate(newScene, &ms, &newMesh), "mesh");
        iplStaticMeshAdd(newMesh, newScene); iplSceneCommit(newScene);

        iplSimulatorSetScene(sim, newScene); iplSimulatorCommit(sim);

        if (mesh)  iplStaticMeshRelease(&mesh);
        if (scene) iplSceneRelease(&scene);
        scene = newScene; mesh = newMesh; room = rm; has_room = true;
    }

    // ---- THE observation operator (== echo_nav.cpp::sense() render path) ----
    Obs observe(IPLVector3 pose, float yawDeg) {
        IPLCoordinateSpace3 hf = headFrame(pose, yawDeg);
        // Emitter frame: the beacon if displaced (bistatic), else the head (monostatic).
        // Only the origin matters for an omni source; orientation is arbitrary.
        IPLCoordinateSpace3 sf = has_beacon ? headFrame(beacon, 0.f) : hf;

        // reflection sim: listener = agent's head, source = emitter frame `sf`
        IPLSimulationSharedInputs sh{}; sh.listener = hf; sh.numRays = NUM_RAYS; sh.numBounces = NUM_BOUNCES;
        sh.duration = IR_DUR; sh.order = ORDER; sh.irradianceMinDistance = 0.1f;   // Gotcha 4
        iplSimulatorSetSharedInputs(sim, IPL_SIMULATIONFLAGS_REFLECTIONS, &sh);
        IPLSimulationInputs in{}; in.flags = IPL_SIMULATIONFLAGS_REFLECTIONS; in.source = sf;
        in.distanceAttenuationModel.type = IPL_DISTANCEATTENUATIONTYPE_DEFAULT;
        in.airAbsorptionModel.type = IPL_AIRABSORPTIONTYPE_DEFAULT;
        in.reverbScale[0] = in.reverbScale[1] = in.reverbScale[2] = 1.0f;
        iplSourceSetInputs(src, IPL_SIMULATIONFLAGS_REFLECTIONS, &in);
        iplSimulatorRunReflections(sim);                                            // Gotcha 1: re-run every time
        IPLSimulationOutputs out{}; iplSourceGetOutputs(src, IPL_SIMULATIONFLAGS_REFLECTIONS, &out);
        IPLReflectionEffectParams rp = out.reflections;
        rp.type = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION; rp.numChannels = AMBI; rp.irSize = IRN;

        // ambisonic IR from a unit impulse
        iplReflectionEffectReset(refl);
        std::vector<std::vector<float>> ambi(AMBI, std::vector<float>(TOTAL, 0.f));
        std::vector<float> inF(FRAME, 0.f); inF[0] = 1.f; float* inP = inF.data(); IPLAudioBuffer inB{ 1, FRAME, &inP };
        std::vector<std::vector<float>> oF(AMBI, std::vector<float>(FRAME)); std::vector<float*> oP(AMBI);
        for (int c = 0; c < AMBI; ++c) oP[c] = oF[c].data(); IPLAudioBuffer oB{ AMBI, FRAME, oP.data() };
        for (int p = 0; p < TOTAL; p += FRAME) {
            iplReflectionEffectApply(refl, &rp, &inB, &oB, nullptr);
            int cp = std::min(FRAME, TOTAL - p);
            for (int c = 0; c < AMBI; ++c) std::memcpy(&ambi[c][p], oF[c].data(), cp * sizeof(float));
            inF[0] = 0.f;
        }

        // binaural decode with the head orientation
        iplAmbisonicsDecodeEffectReset(dec);
        IPLAmbisonicsDecodeEffectParams dp{}; dp.order = ORDER; dp.hrtf = hrtf; dp.orientation = hf; dp.binaural = IPL_TRUE;
        std::vector<std::vector<float>> iF(AMBI, std::vector<float>(FRAME)); std::vector<float*> iP(AMBI);
        for (int c = 0; c < AMBI; ++c) iP[c] = iF[c].data(); IPLAudioBuffer iB{ AMBI, FRAME, iP.data() };
        std::vector<float> l(FRAME), r(FRAME); float* bP[2] = { l.data(), r.data() }; IPLAudioBuffer bB{ 2, FRAME, bP };
        std::vector<float> binL(TOTAL, 0.f), binR(TOTAL, 0.f);
        for (int p = 0; p < TOTAL; p += FRAME) {
            for (int c = 0; c < AMBI; ++c) for (int i = 0; i < FRAME; ++i) { int idx = p + i; iF[c][i] = idx < TOTAL ? ambi[c][idx] : 0.f; }
            iplAmbisonicsDecodeEffectApply(dec, &dp, &iB, &bB);
            int cp = std::min(FRAME, TOTAL - p);
            std::memcpy(&binL[p], l.data(), cp * sizeof(float));
            std::memcpy(&binR[p], r.data(), cp * sizeof(float));
        }

        // pack observation: crop first RAW_N (0.2 s) + cochleagram + range tower.
        // Range tower reads ambi[0] (the W / omni ambisonic channel) BEFORE the
        // binaural decode -- geometry must come off this pre-binaural channel
        // (see acoustic_env_test below and matched_range.*); the binaural path
        // carries bearing/material/elevation but not fine range.
        Obs o{};
        for (int i = 0; i < RAW_N; ++i) { o.raw[i] = binL[i]; o.raw[RAW_N + i] = binR[i]; }
        coch.apply(&o.raw[0],     RAW_N, &o.coch[0]);
        coch.apply(&o.raw[RAW_N], RAW_N, &o.coch[COCH_F * COCH_T]);
        std::vector<float> rawRange = mf.profile(ambi[0]);
        mf.pool_log(rawRange, o.range);
        return o;
    }

    // ---- analytic GT labels for the current box room ----
    Label label(IPLVector3 pose, float yawDeg) const { return analytic_label(room, pose, yawDeg); }

    // convenience: advance pose by an action (step then turn); clamps off walls
    void apply_action(IPLVector3& pose, float& yawDeg, const Action& a) const {
        float dy = std::max(-45.f, std::min(45.f, a.dyaw));
        float ds = std::max(0.f, std::min(1.5f, a.dstep));
        float t = yawDeg * (float)M_PI / 180.f;
        pose.x += ds * std::sin(t); pose.z += ds * -std::cos(t);
        const float m = 0.3f;
        pose.x = std::max(room.origin.x + m, std::min(room.origin.x + room.Lx - m, pose.x));
        pose.z = std::max(room.origin.z + m, std::min(room.origin.z + room.Lz - m, pose.z));
        yawDeg += dy;
    }

    void shutdown() {
        if (dec)   iplAmbisonicsDecodeEffectRelease(&dec);
        if (refl)  iplReflectionEffectRelease(&refl);
        if (mesh)  iplStaticMeshRelease(&mesh);
        if (scene) iplSceneRelease(&scene);
        if (src)   iplSourceRelease(&src);
        if (sim)   iplSimulatorRelease(&sim);
        if (hrtf)  iplHRTFRelease(&hrtf);
        if (ctx)   iplContextRelease(&ctx);
    }
};

} // namespace acoustic_env

// ---------------------------------------------------------------------------
//  Self-test:  -DACOUSTIC_ENV_TEST
// ---------------------------------------------------------------------------
#ifdef ACOUSTIC_ENV_TEST
int main() {
    using namespace acoustic_env;
    AcousticEnv env; env.init();
    BoxRoom rm = make_box_room(12345);
    env.set_room(rm);
    std::printf("room %.1f x %.1f x %.1f m, %zu opening(s), mats[+x-x+y-y+z-z]=%d%d%d%d%d%d\n",
                rm.Lx, rm.Ly, rm.Lz, rm.openings.size(),
                rm.wall_material[0], rm.wall_material[1], rm.wall_material[2],
                rm.wall_material[3], rm.wall_material[4], rm.wall_material[5]);

    IPLVector3 pose = V(0.f, 1.6f, 0.f); float yaw = 0.f;   // room centre, facing -z
    Obs o = env.observe(pose, yaw);
    Label lb = env.label(pose, yaw);

    float pk = 0, ce = 0; for (float v : o.raw) pk = std::max(pk, std::fabs(v));
    for (float v : o.coch) ce += v; ce /= (2 * COCH_F * COCH_T);
    float nearest = lb.dist_wall[0];
    for (float d : lb.dist_wall) nearest = std::min(nearest, d);

    // Where does binaural echo energy peak? Informational only: the binaural path
    // has no clean first-echo onset (HRTF ~6 ms latency + pre-ringing + dense room
    // tail smear the arrival), so this does NOT equal wall distance -- geometry
    // must be read off the pre-binaural W channel (see matched_range.*). The
    // geometry check below is the labels, which are analytic and exact.
    float epk = 0; int pi = 0;
    for (int i = 0; i < RAW_N; ++i) { float e = std::fabs(o.raw[i]) + std::fabs(o.raw[RAW_N + i]); if (e > epk) { epk = e; pi = i; } }

    // PASS criterion: at room centre all 4 axis directions must equal the half-dims.
    bool geo_ok = std::fabs(lb.dist_wall[0] - rm.Lz * 0.5f) < 0.05f &&   // -z
                  std::fabs(lb.dist_wall[2] - rm.Lx * 0.5f) < 0.05f;     // +x
    std::printf("obs: raw peak %.4g, coch mean %.3f, binaural energy peak @ %.1f ms (info only)\n",
                pk, ce, pi * 1000.f / SR);
    std::printf("label: nearest-wall %.2f m, vol %.1f m^3, nearest-mat %d, open_azim %.1f deg\n",
                nearest, lb.room_vol, lb.material_id, lb.open_azim);
    std::printf("geometry check (labels vs room dims): %s\n", geo_ok ? "PASS" : "FAIL");
    for (int k = 0; k < 8; ++k) std::printf("  dist_wall[%d]=%.2f m\n", k, lb.dist_wall[k]);

    // v2: does the range tower's peak land near the analytic nearest-wall
    // distance? Unlike the binaural energy peak above, this DOES read range
    // (it's built from the pre-binaural W channel via matched filtering, the
    // validated matched_range.* pipeline) -- loose tolerance because a box
    // room's several near-equidistant walls can blur the single-reflector
    // leading-edge criterion the original experiment was tuned on.
    int rb = 0; float rpk = o.range[0];
    for (int b = 1; b < RANGE_BINS; ++b) if (o.range[b] > rpk) { rpk = o.range[b]; rb = b; }
    float rangeEstM = C * ((rb * RANGE_POOL + RANGE_POOL * 0.5f) + RANGE_GUARD) / SR / 2.0f;
    bool range_ok = std::fabs(rangeEstM - nearest) < 1.5f;
    std::printf("range tower: peak bin %d (energy %.3f) -> est one-way range %.2f m "
                "(nearest-wall label %.2f m) %s\n",
                rb, rpk, rangeEstM, nearest, range_ok ? "OK" : "CHECK");

    env.shutdown();
    return 0;
}
#endif
