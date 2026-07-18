// ---------------------------------------------------------------------------
//  observe_server.cpp
//
//  Persistent observation server for CLOSED-LOOP MPC. Builds one procedural room
//  (same generator as jepa_dump: make_box_room(seed, 0), monostatic AcousticEnv)
//  and then answers pose queries on stdin with the real observation on stdout, so
//  a Python controller can re-observe against the actual simulator after each
//  move instead of planning open-loop in imagination.
//
//  Protocol (little-endian binary; stdio set to binary mode):
//    request  : 3 x float32  = (x, z, yaw_deg)   [y is fixed at ear height 1.6]
//               a request with x = NaN means "quit".
//    response : RANGE_BINS float32 (matched-filter range profile)
//               followed by 2*COCH_F*COCH_T float32 (binaural cochleagram),
//               i.e. 256 + 6144 = 6400 float32.
//
//  Usage:  observe_server <seed>          (one room per process)
// ---------------------------------------------------------------------------
#include "acoustic_env.h"
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#endif

using namespace acoustic_env;

int main(int argc, char** argv) {
    uint32_t seed = argc > 1 ? (uint32_t)std::strtoul(argv[1], nullptr, 10) : 1u;
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    AcousticEnv env; env.init(0);
    BoxRoom rm = make_box_room(seed, 0);            // 0 obstacles: matches jepa_dump traj rooms
    env.set_room(rm);

    // handshake: emit room dims first so the driver need not re-derive the RNG
    float dims[3] = { rm.Lx, rm.Ly, rm.Lz };
    std::fwrite(dims, sizeof(float), 3, stdout); std::fflush(stdout);

    const int NC = RANGE_BINS + 2 * COCH_F * COCH_T; // 6400
    std::vector<float> outbuf(NC);
    float req[3];
    while (std::fread(req, sizeof(float), 3, stdin) == 3) {
        if (std::isnan(req[0])) break;               // quit sentinel
        Obs o = env.observe(V(req[0], 1.6f, req[1]), req[2]);
        std::memcpy(&outbuf[0], o.range, RANGE_BINS * sizeof(float));
        std::memcpy(&outbuf[RANGE_BINS], o.coch, 2 * COCH_F * COCH_T * sizeof(float));
        std::fwrite(outbuf.data(), sizeof(float), NC, stdout);
        std::fflush(stdout);
    }
    env.shutdown();
    return 0;
}
