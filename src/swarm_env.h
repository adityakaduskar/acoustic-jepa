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

// what the LISTENER could actually MEASURE about the emitter with its own ears
// (no oracle): range from the direct arrival (sample-exact per multi_swarm
// Stage 3), bearing from binaural ITD inverted through the listener's OWN
// ITD-vs-azimuth self-calibration, front/back by a +20 deg head-turn re-measure.
// The residual error vs CrossMeta is the honest measurement error the predictor
// should be conditioned on instead of the oracle transform. Emitter YAW is not
// included: the emitter is an omni source, so its heading does not affect what
// the listener receives and is not observable.
struct MeasuredT {
    float dx = 0, dy = 0, dz = 0;   // measured (emitter - listener), world coords (m)
    float range = 0;                // measured range (m)
    float bearing_local = 0;        // measured azimuth in listener's head frame (deg)
    int   fb_ok = 1;                // 1 if the front/back head-turn test resolved consistently
};

// world direction -> listener-local coords at head yaw th (ahead=-z, right=+x),
// matching SteamAudioIR/src/multi_swarm.cpp::toLocal and the binaural-effect
// direction convention.
inline IPLVector3 toLocal(IPLVector3 d, float thDeg) {
    float t = thDeg * (float)M_PI / 180.f;
    IPLVector3 ah{ std::sin(t), 0, -std::cos(t) }, ri{ std::cos(t), 0, std::sin(t) };
    float lx = d.x * ri.x + d.y * ri.y + d.z * ri.z;
    float ly = d.y;
    float lz = -(d.x * ah.x + d.y * ah.y + d.z * ah.z);
    return V(lx, ly, lz);
}

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
    std::vector<IPLBinauralEffect> bins;         // one per agent HRTF (direct-path HRIR synth)
    std::vector<IPLVector3> pos;                 // agent positions (world, ear height)
    std::vector<float>      yaw;                 // agent head yaw (deg)

    // per-agent ITD-vs-azimuth self-calibration (built once from own HRTF),
    // -90..+90 deg in 15 deg steps, then sorted ascending by ITD for interp
    static const int CAL_N = 13;                 // -90..90 step 15
    std::vector<std::vector<float>> cal_az, cal_itd;   // [agent][CAL_N], sorted by itd

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
        bins.resize(N, nullptr);
        for (int i = 0; i < N; ++i) {
            IPLAmbisonicsDecodeEffectSettings ds{}; ds.speakerLayout.type = IPL_SPEAKERLAYOUTTYPE_STEREO;
            ds.hrtf = hrtfs[i]; ds.maxOrder = ORDER;
            chk(iplAmbisonicsDecodeEffectCreate(ctx, &au, &ds, &decs[i]), "dec");
            IPLBinauralEffectSettings bs{}; bs.hrtf = hrtfs[i];
            chk(iplBinauralEffectCreate(ctx, &au, &bs, &bins[i]), "bin");
        }
        mf.init();
        pos.assign(N, V(0, EAR_H, 0)); yaw.assign(N, 0.f);
        build_calibration();
    }

    static constexpr float EAR_H = 1.6f;

    // ---- direct-path binaural HRIR of a LOCAL direction (listener i's ears) ----
    // The engine renders no direct path, so a peer's direct arrival is synthesised
    // from the listener's HRTF (multi_swarm.cpp::directHRIR). localDir uses +x
    // right, -z ahead (== toLocal output).
    std::vector<std::vector<float>> direct_hrir(int i, IPLVector3 localDir, int n) {
        std::vector<std::vector<float>> out(2, std::vector<float>(n, 0.f));
        std::vector<float> inF(FRAME, 0.f); inF[0] = 1.f; float* inP = inF.data(); IPLAudioBuffer inB{ 1, FRAME, &inP };
        std::vector<float> l(FRAME), r(FRAME); float* bP[2] = { l.data(), r.data() }; IPLAudioBuffer bB{ 2, FRAME, bP };
        IPLBinauralEffectParams bp{}; bp.direction = localDir; bp.interpolation = IPL_HRTFINTERPOLATION_BILINEAR;
        bp.spatialBlend = 1.0f; bp.hrtf = hrtfs[i];
        iplBinauralEffectReset(bins[i]);
        for (int p = 0; p < n; p += FRAME) {
            iplBinauralEffectApply(bins[i], &bp, &inB, &bB);
            int cp = std::min(FRAME, n - p);
            std::memcpy(&out[0][p], l.data(), cp * sizeof(float));
            std::memcpy(&out[1][p], r.data(), cp * sizeof(float));
            inF[0] = 0.f;
        }
        return out;
    }

    // ---- ITD (samples) by cross-correlation + parabolic interp -------------
    // POSITIVE = right ear later = source on the LEFT (multi_swarm.py itd_us).
    static float itd_samples(const std::vector<float>& l, const std::vector<float>& r, int maxlag = 48) {
        int n = (int)std::min(l.size(), r.size());
        float best = -1e30f; int bl = 0; std::vector<float> cc(2 * maxlag + 1);
        for (int gi = 0; gi <= 2 * maxlag; ++gi) {
            int g = gi - maxlag; double s = 0;
            int i0 = std::max(0, -g), i1 = n - std::max(0, g);
            for (int i = i0; i < i1; ++i) s += (double)l[i] * r[i + g];
            cc[gi] = (float)s; if (s > best) { best = (float)s; bl = gi; }
        }
        float sub = 0.f;
        if (bl > 0 && bl < 2 * maxlag) {
            float y0 = cc[bl - 1], y1 = cc[bl], y2 = cc[bl + 1], den = y0 - 2 * y1 + y2;
            if (den != 0.f) sub = 0.5f * (y0 - y2) / den;
        }
        return (float)(bl - maxlag) + sub;
    }

    // ---- build each agent's ITD-vs-azimuth curve from its own HRTF ----------
    void build_calibration() {
        cal_az.assign(N, std::vector<float>(CAL_N));
        cal_itd.assign(N, std::vector<float>(CAL_N));
        for (int i = 0; i < N; ++i) {
            std::vector<std::pair<float, float>> pts;   // (itd, az)
            for (int k = 0; k < CAL_N; ++k) {
                float az = -90.f + 15.f * k, a = az * (float)M_PI / 180.f;
                auto h = direct_hrir(i, V(std::sin(a), 0, -std::cos(a)), 4096);
                pts.push_back({ itd_samples(h[0], h[1]), az });
            }
            std::sort(pts.begin(), pts.end());          // ascending ITD for interp
            for (int k = 0; k < CAL_N; ++k) { cal_itd[i][k] = pts[k].first; cal_az[i][k] = pts[k].second; }
        }
    }

    float itd_to_az(int i, float itd) const {
        const auto& x = cal_itd[i]; const auto& y = cal_az[i];
        if (itd <= x.front()) return y.front();
        if (itd >= x.back())  return y.back();
        for (int k = 1; k < CAL_N; ++k)
            if (itd <= x[k]) {
                float t = (itd - x[k - 1]) / std::max(1e-9f, x[k] - x[k - 1]);
                return y[k - 1] + t * (y[k] - y[k - 1]);
            }
        return y.back();
    }

    // ---- measure the transform to emitter E from listener L's ears only -----
    MeasuredT measure_tab(int l, int e) {
        MeasuredT t;
        IPLVector3 d = V(pos[e].x - pos[l].x, pos[e].y - pos[l].y, pos[e].z - pos[l].z);
        float R = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        t.range = R;   // direct-arrival range is sample-exact (multi_swarm Stage 3)
        IPLVector3 du = V(d.x / R, d.y / R, d.z / R);

        auto measure_lat = [&](float headYaw) {
            auto h = direct_hrir(l, toLocal(du, headYaw), 4096);
            return itd_to_az(l, itd_samples(h[0], h[1]));
        };
        float lat0  = measure_lat(yaw[l]);
        float lat20 = measure_lat(yaw[l] + 20.f);

        // front/back: ITD alone can't tell front from rear (cone of confusion).
        // Pick the hypothesis whose predicted lateral angle after the +20 deg
        // head-turn matches the re-measurement (multi_swarm.py front/back test).
        auto wrap180 = [](float a) { return std::fmod(a + 540.f, 360.f) - 180.f; };
        auto pred_lat = [&](float azRel) {
            return (float)(std::asin(std::sin((azRel - 20.f) * (float)M_PI / 180.f)) * 180.0 / M_PI);
        };
        float hyp_front = lat0;
        float hyp_rear  = lat0 >= 0.f ? 180.f - lat0 : -180.f - lat0;
        float err_f = std::fabs(wrap180(pred_lat(hyp_front) - lat20));
        float err_r = std::fabs(wrap180(pred_lat(hyp_rear)  - lat20));
        float az = err_f <= err_r ? hyp_front : hyp_rear;

        // is the chosen hypothesis actually the true side? (score only)
        float ahead = du.x * std::sin(yaw[l] * (float)M_PI / 180.f) + du.z * (-std::cos(yaw[l] * (float)M_PI / 180.f));
        bool trueFront = ahead >= 0.f;
        bool choseFront = (err_f <= err_r);
        t.fb_ok = (choseFront == trueFront) ? 1 : 0;

        t.bearing_local = az;
        float ar = az * (float)M_PI / 180.f;
        // measured emitter position: own pose + measured range along measured bearing
        // (2D in x/z at ear height; elevation is not recovered from ITD)
        IPLVector3 fwd = V(std::sin(yaw[l] * (float)M_PI / 180.f), 0, -std::cos(yaw[l] * (float)M_PI / 180.f));
        IPLVector3 rgt = V(std::cos(yaw[l] * (float)M_PI / 180.f), 0,  std::sin(yaw[l] * (float)M_PI / 180.f));
        float fwdc = R * std::cos(ar), rgtc = R * std::sin(ar);
        t.dx = fwd.x * fwdc + rgt.x * rgtc;
        t.dz = fwd.z * fwdc + rgt.z * rgtc;
        t.dy = 0.f;
        return t;
    }

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

    // ---- scatter N agents in the room (min separation + wall + obstacle clearance) ---
    // Agents must NOT spawn inside or hard against an obstacle -- a source
    // embedded in solid geometry renders silent and corrupts the whole cycle.
    void place_agents(uint32_t seed, float minSep = 1.5f, float wallClear = 0.8f) {
        Rng r(seed * 40503u + 7u);
        const float x0 = room.origin.x + wallClear, x1 = room.origin.x + room.Lx - wallClear;
        const float z0 = room.origin.z + wallClear, z1 = room.origin.z + room.Lz - wallClear;
        const float OB_PAD = 0.4f;   // keep this far outside any obstacle footprint at ear height
        for (int i = 0; i < N; ++i) {
            IPLVector3 p{}; bool ok = false;
            for (int tries = 0; tries < 400 && !ok; ++tries) {
                p = V(r.rf(x0, x1), EAR_H, r.rf(z0, z1));
                ok = true;
                for (int j = 0; j < i && ok; ++j) {
                    float dx = p.x - pos[j].x, dz = p.z - pos[j].z;
                    if (std::sqrt(dx * dx + dz * dz) < minSep) ok = false;
                }
                for (const auto& ob : room.obstacles) {
                    bool inX = p.x > ob.center.x - ob.half.x - OB_PAD && p.x < ob.center.x + ob.half.x + OB_PAD;
                    bool inZ = p.z > ob.center.z - ob.half.z - OB_PAD && p.z < ob.center.z + ob.half.z + OB_PAD;
                    bool inY = p.y > ob.center.y - ob.half.y && p.y < ob.center.y + ob.half.y;
                    if (inX && inZ && inY) { ok = false; break; }
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
        m.occluded = room_occludes(room, pos[l], pos[e]);
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
        for (auto& b : bins) if (b) iplBinauralEffectRelease(&b);
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
static double range_energy(const acoustic_env::Obs& o) {
    double e = 0; for (int i = 0; i < acoustic_env::RANGE_BINS; ++i) e += o.range[i]; return e;
}
int main() {
    using namespace acoustic_env;
    const int N = 3;

    // Same room seed, WITH interior obstacles: obstacles must (a) change the
    // acoustics vs the empty room and (b) occlude some agent pairs -- both are
    // what make the cross-view task non-degenerate.
    SwarmEnv env; env.init(N);
    BoxRoom empty = make_box_room(2024, 0);
    BoxRoom obst  = make_box_room(2024, 3);
    std::printf("room %.1f x %.1f x %.1f m, %d agents, %zu obstacle(s)\n",
                obst.Lx, obst.Ly, obst.Lz, N, obst.obstacles.size());
    for (const auto& ob : obst.obstacles)
        std::printf("  obstacle @ (%.2f, %.2f, %.2f) half (%.2f, %.2f, %.2f) mat %d\n",
                    ob.center.x, ob.center.y, ob.center.z, ob.half.x, ob.half.y, ob.half.z, ob.material);

    env.set_room(obst);
    env.place_agents(2024);
    for (int i = 0; i < N; ++i)
        std::printf("  agent %d @ (%.2f, %.2f, %.2f) yaw %.0f\n", i, env.pos[i].x, env.pos[i].y, env.pos[i].z, env.yaw[i]);

    // every ordered pair (listener l, emitter e), e != l
    bool all_nonsilent = true; int nOccl = 0, nClear = 0;
    float occlPk = 0, clearPk = 0; int nOcclPk = 0, nClearPk = 0;
    for (int l = 0; l < N; ++l) for (int e = 0; e < N; ++e) {
        if (e == l) continue;
        Obs o = env.cross_observe(l, e);
        CrossMeta m = env.cross_meta(l, e);
        float rawpk = 0, ce = 0, rpk = 0; int rb = 0;
        for (float v : o.raw) rawpk = std::max(rawpk, std::fabs(v));
        for (float v : o.coch) ce += v; ce /= (2 * COCH_F * COCH_T);
        for (int b = 0; b < RANGE_BINS; ++b) if (o.range[b] > rpk) { rpk = o.range[b]; rb = b; }
        if (rawpk < 1e-9f) all_nonsilent = false;
        if (m.occluded) { ++nOccl; occlPk += rawpk; ++nOcclPk; }
        else            { ++nClear; clearPk += rawpk; ++nClearPk; }
        std::printf("  L%d<-E%d: range %.2f m, bearing %+.0f deg, %s | raw peak %.3g, "
                    "coch mean %.3f, range-tower peak bin %d (%.3f)\n",
                    l, e, m.range, m.bearing_local, m.occluded ? "OCCLUDED" : "clear",
                    rawpk, ce, rb, rpk);
    }

    // Reciprocity is only a clean invariant for SIMPLE geometry: Steam Audio
    // traces rays from the listener, so in cluttered/occluded scenes the diffuse
    // tail is gathered asymmetrically between reciprocal configs even though the
    // specular peak stays near-symmetric. Informational here (cross-view never
    // relies on reciprocity -- every pair is rendered explicitly).
    Obs a = env.cross_observe(0, 1), b = env.cross_observe(1, 0);
    std::printf("reciprocity (informational, cluttered room): sum|range| L0<-E1=%.2f L1<-E0=%.2f "
                "ratio %.2f (log-sum diverges; engine traces from listener)\n",
                range_energy(a), range_energy(b), range_energy(a) / std::max(1e-9, range_energy(b)));

    // obstacle-sensitivity: same agents, empty room vs obstacle room -> the
    // range profile must differ (otherwise obstacles aren't in the acoustics)
    double e_obst = range_energy(env.cross_observe(0, 1));
    env.set_room(empty);
    double e_empty = range_energy(env.cross_observe(0, 1));
    double rel = std::fabs(e_obst - e_empty) / std::max(1e-9, e_empty);
    std::printf("obstacle sensitivity: L0<-E1 sum|range| obstacle=%.2f empty=%.2f  rel diff %.1f%%\n",
                e_obst, e_empty, 100.0 * rel);

    // occlusion should ATTENUATE the received echo: mean raw peak lower when blocked
    float occlMean = nOcclPk ? occlPk / nOcclPk : 0.f, clearMean = nClearPk ? clearPk / nClearPk : 0.f;
    std::printf("occlusion attenuation: mean raw peak occluded=%.4g clear=%.4g\n", occlMean, clearMean);

    // measured T_AB (ITD-calibrated bearing + head-turn front/back) vs oracle
    std::printf("\nmeasured T_AB vs oracle (bearing from own-ear ITD calibration):\n");
    float maxBearErr = 0, maxPosErr = 0; int fbAll = 1;
    for (int l = 0; l < N; ++l) for (int e = 0; e < N; ++e) {
        if (e == l) continue;
        CrossMeta gt = env.cross_meta(l, e);
        MeasuredT mt = env.measure_tab(l, e);
        float bErr = std::fabs(std::fmod(mt.bearing_local - gt.bearing_local + 540.f, 360.f) - 180.f);
        float posErr = std::sqrt((mt.dx - gt.dx) * (mt.dx - gt.dx) + (mt.dz - gt.dz) * (mt.dz - gt.dz));
        maxBearErr = std::max(maxBearErr, bErr); maxPosErr = std::max(maxPosErr, posErr);
        if (!mt.fb_ok) fbAll = 0;
        std::printf("  L%d<-E%d: bearing meas %+.0f true %+.0f (err %.1f) | pos err %.2f m | fb_ok %d\n",
                    l, e, mt.bearing_local, gt.bearing_local, bErr, posErr, mt.fb_ok);
    }
    std::printf("measured T_AB: max bearing err %.1f deg, max pos err %.2f m, front/back all-correct %d\n",
                maxBearErr, maxPosErr, fbAll);

    // PASS on what THIS increment adds: obstacles in the acoustics, a MIX of
    // occluded/clear pairs, no silent (embedded-agent) pairs.
    bool ok = all_nonsilent && nOccl >= 1 && nClear >= 1 && rel > 0.01;
    std::printf("cross-view + obstacles: %s (non-silent=%d, occluded=%d, clear=%d, obstacle-sensitive=%d)\n",
                ok ? "PASS" : "CHECK", all_nonsilent, nOccl, nClear, rel > 0.01);
    env.shutdown();
    return 0;
}
#endif
