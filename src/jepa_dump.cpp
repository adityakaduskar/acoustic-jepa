// ---------------------------------------------------------------------------
//  jepa_dump.cpp
//
//  Offline dataset generator for the Acoustic-JEPA (ACOUSTIC_JEPA_SPEC.md §2, §8).
//  Walks explore-and-avoid trajectories through procedural box rooms and writes
//  one .npz per episode of (obs, action, label) tuples for an action-conditioned
//  JEPA. Uses AcousticEnv from acoustic_env.h as the pose -> binaural-echo operator.
//
//  PRETEXT-TASK REDESIGN (spec §8): LONGER EPISODES (default 48 steps) with a
//  room-covering explore-and-avoid policy, so training can use a long prediction
//  horizon integrated by a temporal (sequence) encoder. Emitter is MONOSTATIC by
//  default (source co-located with agent -- biologically faithful, as bats/humans
//  do); the last arg toggles a displaced fixed beacon (bistatic) as an ablation.
//
//  Usage:  jepa_dump [episodes] [out_dir] [steps] [seed0] [threads] [beacon]
//          defaults: 8  jepa_data  48  1  1  0     (beacon: 0=monostatic, 1=displaced)
//
//  Each ep%05u.npz contains (np.load(f) -> keys without the .npy suffix):
//     coch        float16 [K, 2, 64, 48]   binaural gammatone cochleagram
//     range       float16 [K, 256]         v2: matched-filter lag-domain range/
//                                          reflectivity profile (see acoustic_env.h)
//     raw         float32 [K, 2, 9600]     cropped binaural echo (0.2 s)
//     actions     float32 [K, 2]           (dyaw deg, dstep m); actions[t] maps
//                                          stored obs t -> obs t+1
//     labels_dist float32 [K, 8]           GT wall distance per yaw sector
//     open_az     float32 [K]              GT azimuth of largest opening (NaN=none)
//     room_vol    float32 [K]              GT room volume (m^3)
//     material    int32   [K]              GT nearest-wall material id
//     pose        float32 [K, 4]           (x, y, z, yaw) eval-only
//     room_meta   float32 [10]             (Lx, Ly, Lz, nOpen, mat0..mat5)
//     beacon      float32 [3]              displaced-emitter position (x, y, z)
//
//  The .npz writer here is a minimal stored (uncompressed) ZIP of .npy members;
//  no external libraries. Build via build.bat (adds the jepa_dump target).
// ---------------------------------------------------------------------------
#include "acoustic_env.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace acoustic_env;

// ---- half-precision (IEEE 754 binary16), round-to-nearest -----------------
static uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man  = x & 0x7fffffu;
    if (exp <= 0) {                                  // subnormal / zero
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u; int shift = 14 - exp;
        uint32_t h = man >> shift;
        if ((man >> (shift - 1)) & 1u) h += 1;       // round
        return (uint16_t)(sign | h);
    } else if (exp >= 31) {                          // overflow -> inf
        return (uint16_t)(sign | 0x7c00u);
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    if ((man >> 12) & 1u) h += 1;                    // round
    return h;
}

// ---- CRC-32 (zip) ----------------------------------------------------------
static uint32_t crc32_buf(const uint8_t* p, size_t n) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) {
        crc ^= p[i];
        for (int k = 0; k < 8; ++k) crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(crc & 1u)));
    }
    return ~crc;
}

// ---- one .npy member (header + raw little-endian data) --------------------
struct NpyArray { std::string name; std::vector<uint8_t> bytes; };

static NpyArray make_npy(const std::string& name, const char* descr,
                         const std::vector<int>& shape, const uint8_t* data, size_t nbytes) {
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i) { sh += std::to_string(shape[i]); if (i + 1 < shape.size()) sh += ", "; }
    if (shape.size() == 1) sh += ",";
    sh += ")";
    std::string hdr = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': " + sh + ", }";
    size_t base = 10 + hdr.size() + 1;                       // magic(6)+ver(2)+len(2)+hdr+'\n'
    hdr.append((64 - (base % 64)) % 64, ' '); hdr += "\n";
    uint16_t hlen = (uint16_t)hdr.size();

    NpyArray a; a.name = name;
    const uint8_t magic[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    a.bytes.insert(a.bytes.end(), magic, magic + 8);
    a.bytes.push_back(hlen & 0xff); a.bytes.push_back((hlen >> 8) & 0xff);
    a.bytes.insert(a.bytes.end(), hdr.begin(), hdr.end());
    a.bytes.insert(a.bytes.end(), data, data + nbytes);
    return a;
}

// ---- write a set of .npy members as a stored .npz --------------------------
static void write_npz(const std::string& path, const std::vector<NpyArray>& arrays) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    struct CD { uint32_t crc, size, off; std::string name; };
    std::vector<CD> cds; uint32_t offset = 0;
    auto w16 = [&](uint16_t v) { std::fputc(v & 0xff, f); std::fputc((v >> 8) & 0xff, f); offset += 2; };
    auto w32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) std::fputc((v >> (8 * i)) & 0xff, f); offset += 4; };
    auto wr  = [&](const void* p, size_t n) { std::fwrite(p, 1, n, f); offset += (uint32_t)n; };

    for (const auto& a : arrays) {
        uint32_t crc = crc32_buf(a.bytes.data(), a.bytes.size());
        uint32_t loc = offset;
        w32(0x04034b50); w16(20); w16(0); w16(0);            // sig, ver, flags, method=stored
        w16(0); w16(0);                                      // time, date
        w32(crc); w32((uint32_t)a.bytes.size()); w32((uint32_t)a.bytes.size());
        w16((uint16_t)a.name.size()); w16(0);                // name len, extra len
        wr(a.name.data(), a.name.size());
        wr(a.bytes.data(), a.bytes.size());
        cds.push_back({ crc, (uint32_t)a.bytes.size(), loc, a.name });
    }
    uint32_t cdstart = offset;
    for (const auto& c : cds) {
        w32(0x02014b50); w16(20); w16(20); w16(0); w16(0);   // sig, vermade, verneed, flags, method
        w16(0); w16(0);                                      // time, date
        w32(c.crc); w32(c.size); w32(c.size);
        w16((uint16_t)c.name.size()); w16(0); w16(0);        // name, extra, comment len
        w16(0); w16(0); w32(0);                              // disk, internal attr, external attr
        w32(c.off);                                          // local header offset
        wr(c.name.data(), c.name.size());
    }
    uint32_t cdsize = offset - cdstart;
    w32(0x06054b50); w16(0); w16(0);
    w16((uint16_t)cds.size()); w16((uint16_t)cds.size());
    w32(cdsize); w32(cdstart); w16(0);
    std::fclose(f);
}

// helpers to turn typed vectors into .npy members
static NpyArray npy_f32(const std::string& n, const std::vector<int>& shape, const std::vector<float>& v) {
    return make_npy(n, "<f4", shape, (const uint8_t*)v.data(), v.size() * sizeof(float));
}
static NpyArray npy_i32(const std::string& n, const std::vector<int>& shape, const std::vector<int32_t>& v) {
    return make_npy(n, "<i4", shape, (const uint8_t*)v.data(), v.size() * sizeof(int32_t));
}
static NpyArray npy_f16(const std::string& n, const std::vector<int>& shape, const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = f2h(v[i]);
    return make_npy(n, "<f2", shape, (const uint8_t*)h.data(), h.size() * sizeof(uint16_t));
}

// ---- one episode: random room + trajectory -> one .npz ---------------------
static void run_episode(AcousticEnv& env, uint32_t seed, int K, const std::string& path, bool useBeacon) {
    BoxRoom rm = make_box_room(seed);
    env.set_room(rm);

    Rng pol(seed * 2654435761u + 1u);                        // policy stream, decoupled from room
    const float m = 0.6f;
    IPLVector3 pose = V(rm.origin.x + m + pol.uf() * (rm.Lx - 2 * m), 1.6f,
                        rm.origin.z + m + pol.uf() * (rm.Lz - 2 * m));
    float yaw = pol.rf(0.f, 360.f);

    // Emitter placement. DEFAULT is monostatic (source co-located with the agent),
    // which is what real bat/human echolocation uses and is the biologically
    // faithful setup. `useBeacon` enables a displaced fixed beacon (bistatic) as an
    // ABLATION only -- earlier framing that treated bistatic as the fix was wrong.
    const float bm = 0.8f;
    IPLVector3 beacon = V(rm.origin.x + bm + pol.uf() * (rm.Lx - 2 * bm),
                          rm.origin.y + 0.5f + pol.uf() * (rm.Ly - 1.0f),
                          rm.origin.z + bm + pol.uf() * (rm.Lz - 2 * bm));
    if (useBeacon) env.set_beacon(beacon);
    else { env.clear_beacon(); beacon = V(NAN, NAN, NAN); }

    std::vector<float>   coch; coch.reserve((size_t)K * 2 * COCH_F * COCH_T);
    std::vector<float>   range; range.reserve((size_t)K * RANGE_BINS);
    std::vector<float>   raw;  raw.reserve((size_t)K * 2 * RAW_N);
    std::vector<float>   act, ldist, oaz, vol, pos;
    std::vector<int32_t> mat;

    for (int t = 0; t < K; ++t) {
        Obs   o  = env.observe(pose, yaw);
        Label lb = env.label(pose, yaw);

        coch.insert(coch.end(), o.coch, o.coch + 2 * COCH_F * COCH_T);
        range.insert(range.end(), o.range, o.range + RANGE_BINS);
        raw.insert(raw.end(), o.raw, o.raw + 2 * RAW_N);
        for (int k = 0; k < 8; ++k) ldist.push_back(lb.dist_wall[k]);
        oaz.push_back(lb.open_azim); vol.push_back(lb.room_vol); mat.push_back(lb.material_id);
        pos.push_back(pose.x); pos.push_back(pose.y); pos.push_back(pose.z); pos.push_back(yaw);

        // explore-and-avoid policy: wander open space, steer AWAY from a near wall.
        // Covers room geometry so a long-horizon prediction spans real spatial
        // change (the old policy hugged the nearest wall -> degenerate long rollouts).
        int ko = 0; float dmax = -1.f, dmin = 1e9f;
        for (int k = 0; k < 8; ++k) { if (lb.dist_wall[k] > dmax) { dmax = lb.dist_wall[k]; ko = k; } dmin = std::min(dmin, lb.dist_wall[k]); }
        float dyaw;
        if (dmin < 1.5f) { float rel = 45.f * ko; if (rel > 180.f) rel -= 360.f; dyaw = std::max(-45.f, std::min(45.f, rel)); } // turn toward the openest sector
        else dyaw = pol.rf(-25.f, 25.f);                          // gentle wander
        float dstep = pol.rf(0.5f, 1.5f);                        // move decisively
        act.push_back(dyaw); act.push_back(dstep);

        env.apply_action(pose, yaw, Action{ dyaw, dstep });
    }

    std::vector<float> meta = { rm.Lx, rm.Ly, rm.Lz, (float)rm.openings.size() };
    for (int i = 0; i < 6; ++i) meta.push_back((float)rm.wall_material[i]);
    std::vector<float> bpos = { beacon.x, beacon.y, beacon.z };

    std::vector<NpyArray> arrays = {
        npy_f16("coch.npy",        { K, 2, COCH_F, COCH_T }, coch),
        npy_f16("range.npy",       { K, RANGE_BINS },        range),
        npy_f32("raw.npy",         { K, 2, RAW_N },          raw),
        npy_f32("actions.npy",     { K, 2 },                 act),
        npy_f32("labels_dist.npy", { K, 8 },                 ldist),
        npy_f32("open_az.npy",     { K },                    oaz),
        npy_f32("room_vol.npy",    { K },                    vol),
        npy_i32("material.npy",    { K },                    mat),
        npy_f32("pose.npy",        { K, 4 },                 pos),
        npy_f32("room_meta.npy",   { 10 },                   meta),
        npy_f32("beacon.npy",      { 3 },                    bpos),   // displaced-emitter position
    };
    write_npz(path, arrays);
}

int main(int argc, char** argv) {
    int         episodes = argc > 1 ? std::atoi(argv[1]) : 8;
    std::string outdir   = argc > 2 ? argv[2] : "jepa_data";
    int         steps    = argc > 3 ? std::atoi(argv[3]) : 48;   // longer episodes -> longer training horizons
    uint32_t    seed0    = argc > 4 ? (uint32_t)std::strtoul(argv[4], nullptr, 10) : 1u;
    int         threads  = argc > 5 ? std::max(1, std::atoi(argv[5])) : 1;
    bool        useBeacon = argc > 6 ? std::atoi(argv[6]) != 0 : false;  // 0=monostatic (default), 1=displaced beacon

    std::filesystem::create_directories(outdir);
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int simThreads = std::max(1, (int)hw / threads);         // avoid oversubscription
    std::printf("jepa_dump: %d episodes x %d steps [%s] -> %s (%d worker(s), %d sim thread(s) each)\n",
                episodes, steps, useBeacon ? "displaced beacon" : "monostatic",
                outdir.c_str(), threads, simThreads);

    std::mutex io;
    int done = 0;
    auto worker = [&](int tid) {
        AcousticEnv env; env.init(simThreads);
        for (int ep = tid; ep < episodes; ep += threads) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/ep%05d.npz", outdir.c_str(), ep);
            run_episode(env, seed0 + (uint32_t)ep, steps, path, useBeacon);
            std::lock_guard<std::mutex> lk(io);
            std::printf("  [%3d/%3d] %s\n", ++done, episodes, path);
        }
        env.shutdown();
    };

    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& th : pool) th.join();

    std::printf("done: %d episodes written to %s\n", done, outdir.c_str());
    return 0;
}
