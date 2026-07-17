// ---------------------------------------------------------------------------
//  swarm_env.h  (header-only)
//
//  The v2 Acoustic-JEPA CROSS-VIEW environment: N agents share one procedural
//  room; agent E emits, agent L listens, and cross_observe(L, E) returns L's
//  received observation of E's ping. This is the pretext-task machinery for the
//  v2 hypothesis (ACOUSTIC_JEPA_SPEC / memory acoustic-jepa-spec): predicting a
//  PEER's received signal forces an allocentric room representation that
//  self-prediction of your own echo does not.
//
//  Built on acoustic_env.h -- reuses make_box_room / build_room_mesh /
//  material_table / Cochleagram / MatchedFilter / headFrame / Rng / Obs, and the
//  multi-source render pattern verified in SteamAudioIR/src/multi_swarm.cpp
//  (Stage 3). One IPLSimulator with maxNumSources = N, N IPLSources, per-agent
//  HRTF (built-in or SOFA), listener pose time-multiplexed.
//
//  KEY MODELLING FACT (multi-agent-echolocation memory, multi_swarm.cpp): Steam
//  Audio renders NO direct-path propagation delay -- the reflection-sim output
//  is ALREADY reflection-only (direct delay subtracted, Gotcha 3). So the
//  bistatic IR that cross_observe materialises is exactly the reflection-only
//  cross-view target the v2 plan wants; there is nothing to "gate out". The
//  direct path (if a future estimator needs it, e.g. to MEASURE T_AB) must be
//  injected analytically (delay |EL|/c, gain 1/r) by the caller, as multi_swarm
//  does -- it is not in this IR.
//
//  Self-test:  cl /EHsc /O2 /std:c++17 /DSWARM_ENV_TEST /Tp swarm_env.h ...
//              phonon.lib   -> renders one room of agents and prints sanity lines.
// ---------------------------------------------------------------------------
#pragma once
#include "acoustic_env.h"
#include <string>

namespace acoustic_env {

// ---- relative transform listener <- emitter -------------------------------
// Phase 1a fills the TRUE (oracle) fields. A measured T_AB (MF-ToA range +
// calibrated ITD bearing, per multi_swarm Stage 3) is added later so the
// conditioning the predictor sees is what an agent could actually measure.
struct CrossMeta {
    int   listener = -1, emitter = -1;
    float dx = 0, dy = 0, dz = 0;   // (emitter - listener), world coords (m)
    float range = 0;                // |emitter - listener| (m)
    float rel_yaw = 0;              // emitter_yaw - listener_yaw (deg)
    float bearing_local = 0;        // azimuth of emitter in listener's head frame (deg)
    bool  occluded = false;         // true if the E->L segment is blocked (phase 1a: always false)
};

struct SwarmEnv {
    IPLContext ctx = nullptr;
    IPLSimulator sim = nullptr;
    IPLReflectionEffect refl = nullptr;
    IPLScene      scene = nullptr;
    IPLStaticMesh mesh  = nullptr;
    BoxRoom room{};
    bool    has_room = false;
    int     N = 0;

    std::vector<IPLSource> srcs;                 // one per agent (emitter role)
    std::vector<IPLHRTF>   hrtfs;                // one per agent (listener role)
    std::vector<IPLAmbisonicsDecodeEffect> decs; // one per agent HRTF
    std::vector<IPLVector3> pos;                 // agent positions (world, ear height)
    std::vector<float>      yaw;                 // agent head yaw (deg)

    Cochleagram   coch;
    MatchedFilter mf;

    // ---- one-time setup ---------------------------------------------------
    // sofa: optional per-agent SOFA path; empty string (or short vector) => the
    // built-in default HRTF for that agent. Per-agent HRTF makes HRTF identity a
    // nuisance variable in binaural targets -- which is exactly why the v2 plan
    // makes the HRTF-FREE range stream the primary cross-view target.
    void init(int nAgents, int simThreads = 0, const std::vector<std::string>& sofa = {}) {
        N = nAgents;
        IPLContextSettings cs{}; cs.version = STEAMAUDIO_VERSION; cs.simdLevel = IPL_SIMDLEVEL_AVX2;
        chk(iplContextCreate(&cs, &ctx), "ctx");
        IPLAudioSettings au{ SR, FRAME };

        hrtfs.resize(N, nullptr); decs.resize(N, nullptr);
        for (int i = 0; i < N; ++i) {
            IPLHRTFSettings hs{}; hs.volume = 1.f; hs.normType = IPL_HRTFNORMTYPE_NONE;
            if (i < (int)sofa.size() && !sofa[i].empty()) { hs.type = IPL_HRTFTYPE_SOFA; hs.sofaFileName = sofa[i].c_str(); }
            else                                          { hs.type = IPL_HRTFTYPE_DEFAULT; }
            chk(iplHRTFCreate(ctx, &au, &hs, &hrtfs[i]), "hrtf");
        }

        IPLSimulationSettings ss{}; ss.flags = IPL_SIMULATIONFLAGS_REFLECTIONS; ss.sceneType = IPL_SCENETYPE_DEFAULT;
        ss.reflectionType = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION; ss.maxNumRays = NUM_RAYS; ss.numDiffuseSamples = NUM_DIFFUSE;
        ss.maxDuration = IR_DUR; ss.maxOrder = ORDER; ss.maxNumSources = N;
        ss.numThreads = (IPLint32)(simThreads > 0 ? simThreads : std::max(1u, std::thread::hardware_concurrency()));
        ss.samplingRate = SR; ss.frameSize = FRAME;
        chk(iplSimulatorCreate(ctx, &ss, &sim), "sim");

        srcs.resize(N, nullptr);
        for (int i = 0; i < N; ++i) {
            IPLSourceSettings se{}; se.flags = IPL_SIMULATIONFLAGS_REFLECTIONS;
            chk(iplSourceCreate(sim, &se, &srcs[i]), "src"); iplSourceAdd(srcs[i], sim);
        }
        // (commit once a scene exists, in set_room)

        IPLReflectionEffectSettings rs{}; rs.type = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION; rs.irSize = IRN; rs.numChannels = AMBI;
        chk(iplReflectionEffectCreate(ctx, &au, &rs, &refl), "refl");
        for (int i = 0; i < N; ++i) {
            IPLAmbisonicsDecodeEffectSettings ds{}; ds.speakerLayout.type = IPL_SPEAKERLAYOUTTYPE_STEREO;
            ds.hrtf = hrtfs[i]; ds.maxOrder = ORDER;
            chk(iplAmbisonicsDecodeEffectCreate(ctx, &au, &ds, &decs[i]), "dec");
        }
        mf.init();
        pos.assign(N, V(0, EAR_H, 0)); yaw.assign(N, 0.f);
    }

    static constexpr float EAR_H = 1.6f;

    // ---- swap in a procedural room (same mesh path as AcousticEnv) --------
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

    // ---- scatter N agents in the room (min separation + wall clearance) ---
    void place_agents(uint32_t seed, float minSep = 1.5f, float wallClear = 0.8f) {
        Rng r(seed * 40503u + 7u);
        const float x0 = room.origin.x + wallClear, x1 = room.origin.x + room.Lx - wallClear;
        const float z0 = room.origin.z + wallClear, z1 = room.origin.z + room.Lz - wallClear;
        for (int i = 0; i < N; ++i) {
            IPLVector3 p{}; bool ok = false;
            for (int tries = 0; tries < 200 && !ok; ++tries) {
                p = V(r.rf(x0, x1), EAR_H, r.rf(z0, z1));
                ok = true;
                for (int j = 0; j < i; ++j) {
                    float dx = p.x - pos[j].x, dz = p.z - pos[j].z;
                    if (std::sqrt(dx * dx + dz * dz) < minSep) { ok = false; break; }
                }
            }
            pos[i] = p; yaw[i] = r.rf(0.f, 360.f);
        }
    }

    // ---- true relative transform listener <- emitter ----------------------
    CrossMeta cross_meta(int l, int e) const {
        CrossMeta m; m.listener = l; m.emitter = e;
        m.dx = pos[e].x - pos[l].x; m.dy = pos[e].y - pos[l].y; m.dz = pos[e].z - pos[l].z;
        m.range = std::sqrt(m.dx * m.dx + m.dy * m.dy + m.dz * m.dz);
        m.rel_yaw = yaw[e] - yaw[l];
        // azimuth of emitter in listener's head frame: 0 = ahead, +90 = right.
        // listener head frame: ahead=(sin th,0,-cos th), right=(cos th,0,sin th).
        float t = yaw[l] * (float)M_PI / 180.f;
        float fx = std::sin(t), fz = -std::cos(t);    // ahead
        float rx = std::cos(t), rz = std::sin(t);     // right
        float ahead = m.dx * fx + m.dz * fz;
        float right = m.dx * rx + m.dz * rz;
        m.bearing_local = std::atan2(right, ahead) * 180.f / (float)M_PI;
        return m;
    }

    // ---- THE cross-view observation: L's received obs of E's ping ---------
    // Reflection-only (bistatic), decoded with L's own HRTF. Range tower reads
    // the pre-binaural W channel, same as AcousticEnv::observe. Re-runs the
    // reflection sim per call (simple + correct; the dumper batches by listener
    // to amortise -- one run per listener materialises every emitter once, per
    // the multi_swarm materialise-once-per-source-per-run gotcha).
    Obs cross_observe(int l, int e) {
        IPLCoordinateSpace3 lf = headFrame(pos[l], yaw[l]);
        IPLCoordinateSpace3 ef = headFrame(pos[e], 0.f);   // omni source: only origin matters

        IPLSimulationSharedInputs sh{}; sh.listener = lf; sh.numRays = NUM_RAYS; sh.numBounces = NUM_BOUNCES;
        sh.duration = IR_DUR; sh.order = ORDER; sh.irradianceMinDistance = 0.1f;   // Gotcha 4
        iplSimulatorSetSharedInputs(sim, IPL_SIMULATIONFLAGS_REFLECTIONS, &sh);

        IPLSimulationInputs in{}; in.flags = IPL_SIMULATIONFLAGS_REFLECTIONS; in.source = ef;
        in.distanceAttenuationModel.type = IPL_DISTANCEATTENUATIONTYPE_DEFAULT;
        in.airAbsorptionModel.type = IPL_AIRABSORPTIONTYPE_DEFAULT;
        in.reverbScale[0] = in.reverbScale[1] = in.reverbScale[2] = 1.0f;
        iplSourceSetInputs(srcs[e], IPL_SIMULATIONFLAGS_REFLECTIONS, &in);

        iplSimulatorRunReflections(sim);                                            // Gotcha 1
        IPLSimulationOutputs out{}; iplSourceGetOutputs(srcs[e], IPL_SIMULATIONFLAGS_REFLECTIONS, &out);
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

        // binaural decode with the LISTENER's HRTF and head orientation
        iplAmbisonicsDecodeEffectReset(decs[l]);
        IPLAmbisonicsDecodeEffectParams dp{}; dp.order = ORDER; dp.hrtf = hrtfs[l]; dp.orientation = lf; dp.binaural = IPL_TRUE;
        std::vector<std::vector<float>> iF(AMBI, std::vector<float>(FRAME)); std::vector<float*> iP(AMBI);
        for (int c = 0; c < AMBI; ++c) iP[c] = iF[c].data(); IPLAudioBuffer iB{ AMBI, FRAME, iP.data() };
        std::vector<float> l_(FRAME), r_(FRAME); float* bP[2] = { l_.data(), r_.data() }; IPLAudioBuffer bB{ 2, FRAME, bP };
        std::vector<float> binL(TOTAL, 0.f), binR(TOTAL, 0.f);
        for (int p = 0; p < TOTAL; p += FRAME) {
            for (int c = 0; c < AMBI; ++c) for (int i = 0; i < FRAME; ++i) { int idx = p + i; iF[c][i] = idx < TOTAL ? ambi[c][idx] : 0.f; }
            iplAmbisonicsDecodeEffectApply(decs[l], &dp, &iB, &bB);
            int cp = std::min(FRAME, TOTAL - p);
            std::memcpy(&binL[p], l_.data(), cp * sizeof(float));
            std::memcpy(&binR[p], r_.data(), cp * sizeof(float));
        }

        Obs o{};
        for (int i = 0; i < RAW_N; ++i) { o.raw[i] = binL[i]; o.raw[RAW_N + i] = binR[i]; }
        coch.apply(&o.raw[0],     RAW_N, &o.coch[0]);
        coch.apply(&o.raw[RAW_N], RAW_N, &o.coch[COCH_F * COCH_T]);
        std::vector<float> rawRange = mf.profile(ambi[0]);
        mf.pool_log(rawRange, o.range);
        return o;
    }

    void shutdown() {
        for (auto& d : decs) if (d) iplAmbisonicsDecodeEffectRelease(&d);
        if (refl) iplReflectionEffectRelease(&refl);
        if (mesh) iplStaticMeshRelease(&mesh);
        if (scene) iplSceneRelease(&scene);
        for (auto& s : srcs) if (s) iplSourceRelease(&s);
        if (sim) iplSimulatorRelease(&sim);
        for (auto& h : hrtfs) if (h) iplHRTFRelease(&h);
        if (ctx) iplContextRelease(&ctx);
    }
};

} // namespace acoustic_env

// ---------------------------------------------------------------------------
//  Self-test:  -DSWARM_ENV_TEST
// ---------------------------------------------------------------------------
#ifdef SWARM_ENV_TEST
int main() {
    using namespace acoustic_env;
    const int N = 3;
    SwarmEnv env; env.init(N);
    BoxRoom rm = make_box_room(2024);
    env.set_room(rm);
    env.place_agents(2024);

    std::printf("room %.1f x %.1f x %.1f m, %d agents\n", rm.Lx, rm.Ly, rm.Lz, N);
    for (int i = 0; i < N; ++i)
        std::printf("  agent %d @ (%.2f, %.2f, %.2f) yaw %.0f\n", i, env.pos[i].x, env.pos[i].y, env.pos[i].z, env.yaw[i]);

    // every ordered pair (listener l, emitter e), e != l
    bool all_nonsilent = true;
    for (int l = 0; l < N; ++l) for (int e = 0; e < N; ++e) {
        if (e == l) continue;
        Obs o = env.cross_observe(l, e);
        CrossMeta m = env.cross_meta(l, e);
        float rawpk = 0, ce = 0, rpk = 0; int rb = 0;
        for (float v : o.raw) rawpk = std::max(rawpk, std::fabs(v));
        for (float v : o.coch) ce += v; ce /= (2 * COCH_F * COCH_T);
        for (int b = 0; b < RANGE_BINS; ++b) if (o.range[b] > rpk) { rpk = o.range[b]; rb = b; }
        if (rawpk < 1e-9f) all_nonsilent = false;
        std::printf("  L%d<-E%d: range %.2f m, bearing %+.0f deg | raw peak %.3g, coch mean %.3f, "
                    "range-tower peak bin %d (%.3f)\n",
                    l, e, m.range, m.bearing_local, rawpk, ce, rb, rpk);
    }

    // reciprocity sanity: L0<-E1 and L1<-E0 traverse the same wall geometry; their
    // W-channel range profiles should be similar in gross energy (not identical --
    // different head frames). Just report, don't gate.
    Obs a = env.cross_observe(0, 1), b = env.cross_observe(1, 0);
    double ea = 0, eb = 0; for (int i = 0; i < RANGE_BINS; ++i) { ea += a.range[i]; eb += b.range[i]; }
    std::printf("reciprocity check: sum|range| L0<-E1=%.2f  L1<-E0=%.2f  ratio %.2f\n",
                ea, eb, ea / std::max(1e-9, eb));

    std::printf("cross-view render: %s\n", all_nonsilent ? "PASS (all pairs non-silent)" : "FAIL (a pair was silent)");
    env.shutdown();
    return 0;
}
#endif
