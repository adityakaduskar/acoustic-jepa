// ---------------------------------------------------------------------------
//  swarm_dump.cpp
//
//  Cross-view dataset generator for the v2 Acoustic-JEPA pretext task. Per
//  episode: one procedural room (optionally with interior obstacles) + N agents
//  scattered in it; writes each agent's MONOSTATIC self-observation (context
//  candidates) and every ordered PAIR's cross-view observation (listener L's
//  reflection-only reception of emitter E's ping), plus the true relative
//  transform T and analytic per-agent geometry labels.
//
//  The cross-view JEPA task (memory acoustic-jepa-spec): predict listener B's
//  received latent from emitter A's own observation + T_AB. Egocentric
//  interpolation cannot cross viewpoints, so a good latent must be allocentric.
//  Targets are reflection-only (Steam Audio renders no direct-path delay, so no
//  gating is needed -- see swarm_env.h).
//
//  Usage:  swarm_dump [episodes] [out_dir] [agents] [seed0] [obstacles] [threads]
//          defaults: 8  swarm_data  4  1  2  1
//
//  Each ep%05u.npz (np.load(f) -> keys without the .npy suffix):
//    mono_coch   f16 [N, 2, 64, 48]  agent i's own monostatic cochleagram
//    mono_range  f16 [N, 256]        agent i's own monostatic range profile
//    cross_coch  f16 [P, 2, 64, 48]  listener L's binaural obs of emitter E (L's HRTF)
//    cross_range f16 [P, 256]        listener L's range profile of E's ping (HRTF-free)
//    pair        i32 [P, 2]          (listener L, emitter E) for each cross row, P=N*(N-1)
//    T_true      f32 [P, 6]          (dx, dy, dz, range_m, rel_yaw_deg, bearing_local_deg)
//    occluded    i32 [P]             1 if the E->L straight sightline is blocked
//    agent_pose  f32 [N, 4]          (x, y, z, yaw_deg)
//    dist_wall   f32 [N, 8]          per-agent GT wall distance per yaw sector
//    room_vol    f32 [N]             GT room volume (m^3)
//    material    i32 [N]             GT nearest-wall material id
//    open_az     f32 [N]             GT azimuth of largest opening (NaN=none)
//    room_meta   f32 [10]            (Lx, Ly, Lz, nOpen, mat0..mat5)
//    obstacles   f32 [nOb, 7]        (cx,cy,cz, hx,hy,hz, material) per obstacle
//
//  NOTE: T_true is the ORACLE transform. The v2 plan calls for a MEASURED T_AB
//  (MF-ToA range + calibrated ITD bearing, per SteamAudioIR multi_swarm Stage 3)
//  so the predictor's conditioning is self-supervised; that estimator is the
//  next step and will add a T_meas key alongside T_true.
// ---------------------------------------------------------------------------
#include "swarm_env.h"
#include "npz_writer.h"
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace acoustic_env;
using npz::NpyArray; using npz::npy_f32; using npz::npy_i32; using npz::npy_f16; using npz::write_npz;

static void run_episode(SwarmEnv& env, uint32_t seed, int N, int nOb, const std::string& path) {
    BoxRoom rm = make_box_room(seed, nOb);
    env.set_room(rm);
    env.place_agents(seed);

    const int P = N * (N - 1);
    std::vector<float> mono_coch, mono_range, cross_coch, cross_range, Tt, apose, dwall, vol, oaz;
    std::vector<int32_t> pair, occl, mat;
    mono_coch.reserve((size_t)N * 2 * COCH_F * COCH_T);
    cross_coch.reserve((size_t)P * 2 * COCH_F * COCH_T);

    // per-agent monostatic self-observation + analytic label
    for (int i = 0; i < N; ++i) {
        Obs o = env.cross_observe(i, i);       // source == listener == agent i => monostatic
        mono_coch.insert(mono_coch.end(), o.coch, o.coch + 2 * COCH_F * COCH_T);
        mono_range.insert(mono_range.end(), o.range, o.range + RANGE_BINS);

        Label lb = analytic_label(rm, env.pos[i], env.yaw[i]);
        apose.push_back(env.pos[i].x); apose.push_back(env.pos[i].y); apose.push_back(env.pos[i].z); apose.push_back(env.yaw[i]);
        for (int k = 0; k < 8; ++k) dwall.push_back(lb.dist_wall[k]);
        vol.push_back(lb.room_vol); mat.push_back(lb.material_id); oaz.push_back(lb.open_azim);
    }

    // every ordered cross-view pair (listener L, emitter E)
    for (int l = 0; l < N; ++l) for (int e = 0; e < N; ++e) {
        if (e == l) continue;
        Obs o = env.cross_observe(l, e);
        CrossMeta m = env.cross_meta(l, e);
        cross_coch.insert(cross_coch.end(), o.coch, o.coch + 2 * COCH_F * COCH_T);
        cross_range.insert(cross_range.end(), o.range, o.range + RANGE_BINS);
        pair.push_back(l); pair.push_back(e);
        Tt.push_back(m.dx); Tt.push_back(m.dy); Tt.push_back(m.dz);
        Tt.push_back(m.range); Tt.push_back(m.rel_yaw); Tt.push_back(m.bearing_local);
        occl.push_back(m.occluded ? 1 : 0);
    }

    std::vector<float> meta = { rm.Lx, rm.Ly, rm.Lz, (float)rm.openings.size() };
    for (int i = 0; i < 6; ++i) meta.push_back((float)rm.wall_material[i]);
    int nObAct = (int)rm.obstacles.size();
    std::vector<float> obs; obs.reserve((size_t)nObAct * 7);
    for (const auto& ob : rm.obstacles) {
        obs.push_back(ob.center.x); obs.push_back(ob.center.y); obs.push_back(ob.center.z);
        obs.push_back(ob.half.x); obs.push_back(ob.half.y); obs.push_back(ob.half.z);
        obs.push_back((float)ob.material);
    }
    if (nObAct == 0) { obs.push_back(0.f); nObAct = 0; }   // keep a valid (0-or-1 x 7) shape

    std::vector<NpyArray> arrays = {
        npy_f16("mono_coch.npy",   { N, 2, COCH_F, COCH_T }, mono_coch),
        npy_f16("mono_range.npy",  { N, RANGE_BINS },        mono_range),
        npy_f16("cross_coch.npy",  { P, 2, COCH_F, COCH_T }, cross_coch),
        npy_f16("cross_range.npy", { P, RANGE_BINS },        cross_range),
        npy_i32("pair.npy",        { P, 2 },                 pair),
        npy_f32("T_true.npy",      { P, 6 },                 Tt),
        npy_i32("occluded.npy",    { P },                    occl),
        npy_f32("agent_pose.npy",  { N, 4 },                 apose),
        npy_f32("dist_wall.npy",   { N, 8 },                 dwall),
        npy_f32("room_vol.npy",    { N },                    vol),
        npy_i32("material.npy",    { N },                    mat),
        npy_f32("open_az.npy",     { N },                    oaz),
        npy_f32("room_meta.npy",   { 10 },                   meta),
        npy_f32("obstacles.npy",   { std::max(1, nObAct), 7 }, obs),
    };
    write_npz(path, arrays);
}

int main(int argc, char** argv) {
    int         episodes = argc > 1 ? std::atoi(argv[1]) : 8;
    std::string outdir   = argc > 2 ? argv[2] : "swarm_data";
    int         N        = argc > 3 ? std::max(2, std::atoi(argv[3])) : 4;
    uint32_t    seed0    = argc > 4 ? (uint32_t)std::strtoul(argv[4], nullptr, 10) : 1u;
    int         nOb      = argc > 5 ? std::max(0, std::atoi(argv[5])) : 2;
    int         threads  = argc > 6 ? std::max(1, std::atoi(argv[6])) : 1;

    std::filesystem::create_directories(outdir);
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    int simThreads = std::max(1, (int)hw / threads);
    std::printf("swarm_dump: %d episodes, %d agents, %d obstacle(s) -> %s (%d worker(s), %d sim thread(s) each)\n",
                episodes, N, nOb, outdir.c_str(), threads, simThreads);

    std::mutex io; int done = 0;
    auto worker = [&](int tid) {
        SwarmEnv env; env.init(N, simThreads);
        for (int ep = tid; ep < episodes; ep += threads) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/ep%05d.npz", outdir.c_str(), ep);
            run_episode(env, seed0 + (uint32_t)ep, N, nOb, path);
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
