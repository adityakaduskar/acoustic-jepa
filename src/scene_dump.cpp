// ---------------------------------------------------------------------------
//  scene_dump.cpp
//
//  Dumps the ACTUAL procedural scene geometry (the exact triangles the simulator
//  uses) for a given seed, so a sample room can be rendered faithfully -- no
//  stylising, just the real mesh + object parameters. Uses the same
//  make_box_room / build_room_mesh from acoustic_env.h and the same agent
//  placement as swarm_env.h::place_agents, so `scene_dump <seed> <N> <nOb>`
//  reproduces exactly the scene swarm_dump wrote for episode (seed - seed0).
//
//  Writes to results/scenes/:
//    scene_<seed>.obj   -- room walls (with openings carved) + obstacle boxes
//    scene_<seed>.txt   -- room dims, openings, obstacles, agent poses
//
//  No phonon context is created (only POD geometry types are used), so this
//  links without needing the DLL at runtime.
// ---------------------------------------------------------------------------
#include "acoustic_env.h"
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace acoustic_env;

// mirror swarm_env.h::place_agents exactly (so agents match the real episode)
static void place_agents(const BoxRoom& room, int N, uint32_t seed,
                         std::vector<IPLVector3>& pos, std::vector<float>& yaw,
                         float minSep = 1.5f, float wallClear = 0.8f) {
    const float EAR_H = 1.6f, OB_PAD = 0.4f;
    Rng r(seed * 40503u + 7u);
    const float x0 = room.origin.x + wallClear, x1 = room.origin.x + room.Lx - wallClear;
    const float z0 = room.origin.z + wallClear, z1 = room.origin.z + room.Lz - wallClear;
    pos.assign(N, V(0, EAR_H, 0)); yaw.assign(N, 0.f);
    for (int i = 0; i < N; ++i) {
        IPLVector3 p{}; bool ok = false;
        for (int t = 0; t < 400 && !ok; ++t) {
            p = V(r.rf(x0, x1), EAR_H, r.rf(z0, z1)); ok = true;
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

int main(int argc, char** argv) {
    uint32_t seed = argc > 1 ? (uint32_t)std::strtoul(argv[1], nullptr, 10) : 1u;
    int N   = argc > 2 ? std::atoi(argv[2]) : 6;
    int nOb = argc > 3 ? std::atoi(argv[3]) : 3;

    BoxRoom rm = make_box_room(seed, nOb);
    std::vector<IPLVector3> verts; std::vector<IPLTriangle> tris; std::vector<IPLint32> matIdx;
    build_room_mesh(rm, verts, tris, matIdx);
    std::vector<IPLVector3> apos; std::vector<float> ayaw;
    place_agents(rm, N, seed, apos, ayaw);

    std::filesystem::create_directories("results/scenes");
    char pobj[256], ptxt[256];
    std::snprintf(pobj, sizeof(pobj), "results/scenes/scene_%u.obj", seed);
    std::snprintf(ptxt, sizeof(ptxt), "results/scenes/scene_%u.txt", seed);

    FILE* o = std::fopen(pobj, "w");
    std::fprintf(o, "# procedural acoustic scene, seed %u\n", seed);
    for (auto& v : verts) std::fprintf(o, "v %.4f %.4f %.4f\n", v.x, v.y, v.z);
    for (size_t i = 0; i < tris.size(); ++i)
        std::fprintf(o, "f %d %d %d\n", tris[i].indices[0] + 1, tris[i].indices[1] + 1, tris[i].indices[2] + 1);
    std::fclose(o);

    FILE* t = std::fopen(ptxt, "w");
    std::fprintf(t, "seed %u\nroom %.4f %.4f %.4f\norigin %.4f %.4f %.4f\n",
                 seed, rm.Lx, rm.Ly, rm.Lz, rm.origin.x, rm.origin.y, rm.origin.z);
    std::fprintf(t, "materials %d %d %d %d %d %d\n", rm.wall_material[0], rm.wall_material[1],
                 rm.wall_material[2], rm.wall_material[3], rm.wall_material[4], rm.wall_material[5]);
    std::fprintf(t, "nopenings %zu\n", rm.openings.size());
    for (auto& op : rm.openings)
        std::fprintf(t, "opening wall %d center %.4f %.4f %.4f w %.4f h %.4f\n",
                     op.wall, op.center.x, op.center.y, op.center.z, op.w, op.h);
    std::fprintf(t, "nobstacles %zu\n", rm.obstacles.size());
    for (auto& ob : rm.obstacles)
        std::fprintf(t, "obstacle center %.4f %.4f %.4f half %.4f %.4f %.4f mat %d\n",
                     ob.center.x, ob.center.y, ob.center.z, ob.half.x, ob.half.y, ob.half.z, ob.material);
    std::fprintf(t, "nagents %d\n", N);
    for (int i = 0; i < N; ++i)
        std::fprintf(t, "agent %.4f %.4f %.4f yaw %.2f\n", apos[i].x, apos[i].y, apos[i].z, ayaw[i]);
    std::fclose(t);

    std::printf("wrote %s (%zu verts, %zu tris) and %s (%d agents, %zu obstacles, %zu openings)\n",
                pobj, verts.size(), tris.size(), ptxt, N, rm.obstacles.size(), rm.openings.size());
    return 0;
}
