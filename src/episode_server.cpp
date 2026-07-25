// ---------------------------------------------------------------------------
//  episode_server.cpp   (EchoRL M1)
//
//  Persistent observation server for the CLOSED-LOOP RL rig (see
//  docs/EchoRL_PLAN.md). Same pose->observation protocol as observe_server, but
//  the room carries interior OBSTACLES and the handshake streams the full
//  collision geometry (room dims + obstacle AABBs + wall openings) so the Python
//  Gymnasium env can do collision / reward without re-deriving the RNG.
//
//  This is a SEPARATE exe from observe_server on purpose: observe_server's 12-byte
//  handshake is consumed verbatim by control_mpc.py::ObserveServer, so extending
//  that handshake in place would desync the MPC pipeline. The per-observation
//  protocol below is byte-identical to observe_server, so the same observe() reader
//  works once the (longer) handshake has been consumed.
//
//  Protocol (little-endian binary; stdio in binary mode):
//    HANDSHAKE (emitted once at startup, in order):
//      3  x float32   room dims (Lx, Ly, Lz)
//      1  x int32     nObstacles
//      nObstacles x (7 x float32)  = (cx, cy, cz, hx, hy, hz, material)   // AABB: centre + half-extents
//      1  x int32     nOpenings
//      nOpenings  x (6 x float32)  = (wall, cx, cy, cz, w, h)             // wall id as float
//    REQUEST  : 3 x float32 = (x, z, yaw_deg)   [y fixed at ear height 1.6]
//               x = NaN  => quit.
//    RESPONSE : RANGE_BINS float32 (matched-filter range profile)
//               then nEars*COCH_F*COCH_T float32 (cochleagram).
//               nEars=2 (default): 256 + 6144 = 6400  -- byte-identical to v2, so every
//               existing pipeline is untouched. nEars=4 (§5.11): 256 + 12288 = 12544,
//               ears 0-1 = head at yaw, ears 2-3 = head at yaw+90 (non-collinear array).
//
//  Usage:  episode_server <seed> [nObstacles=3] [simThreads=0(all cores)] [nEars=2]
// ---------------------------------------------------------------------------
#include "acoustic_env.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#endif

using namespace acoustic_env;

int main(int argc, char** argv) {
    uint32_t seed       = argc > 1 ? (uint32_t)std::strtoul(argv[1], nullptr, 10) : 1u;
    int      nObstacles = argc > 2 ? (int)std::strtol(argv[2], nullptr, 10) : 3;
    int      simThreads = argc > 3 ? (int)std::strtol(argv[3], nullptr, 10) : 0;   // 0 => all cores
    int      nEars      = argc > 4 ? (int)std::strtol(argv[4], nullptr, 10) : 2;   // 2 or 4
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    AcousticEnv env; env.init(simThreads); env.set_ears(nEars);
    BoxRoom rm = make_box_room(seed, nObstacles);
    env.set_room(rm);

    // ---- handshake: dims, then obstacle AABBs, then wall openings ----
    float dims[3] = { rm.Lx, rm.Ly, rm.Lz };
    std::fwrite(dims, sizeof(float), 3, stdout);

    IPLint32 nOb = (IPLint32)rm.obstacles.size();
    std::fwrite(&nOb, sizeof(IPLint32), 1, stdout);
    for (const auto& ob : rm.obstacles) {
        float rec[7] = { ob.center.x, ob.center.y, ob.center.z,
                         ob.half.x,   ob.half.y,   ob.half.z, (float)ob.material };
        std::fwrite(rec, sizeof(float), 7, stdout);
    }

    IPLint32 nOp = (IPLint32)rm.openings.size();
    std::fwrite(&nOp, sizeof(IPLint32), 1, stdout);
    for (const auto& op : rm.openings) {
        float rec[6] = { (float)op.wall, op.center.x, op.center.y, op.center.z, op.w, op.h };
        std::fwrite(rec, sizeof(float), 6, stdout);
    }
    std::fflush(stdout);

    // ---- observation loop (identical to observe_server) ----
    const int NE = (nEars == 4) ? 4 : 2;
    const int NC = RANGE_BINS + NE * COCH_F * COCH_T; // 6400 (2 ears) / 12544 (4 ears)
    std::vector<float> outbuf(NC);
    float req[3];
    while (std::fread(req, sizeof(float), 3, stdin) == 3) {
        if (std::isnan(req[0])) break;               // quit sentinel
        Obs o = env.observe(V(req[0], 1.6f, req[1]), req[2]);
        std::memcpy(&outbuf[0], o.range, RANGE_BINS * sizeof(float));
        std::memcpy(&outbuf[RANGE_BINS], o.coch, (size_t)NE * COCH_F * COCH_T * sizeof(float));
        std::fwrite(outbuf.data(), sizeof(float), NC, stdout);
        std::fflush(stdout);
    }
    env.shutdown();
    return 0;
}
