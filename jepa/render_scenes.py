"""Render the ACTUAL procedural scene geometry dumped by scene_dump.exe.

Reads results/scenes/scene_<seed>.obj (the real wall+obstacle triangles) and
scene_<seed>.txt (room/opening/obstacle/agent parameters), and renders a 3D
perspective view + a top-down map per scene. Faithful to the geometry the
simulator uses -- nothing is hand-drawn.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

MAT_COL = {0: "#c8b6a6", 1: "#9db4c0", 2: "#a7c4a0", 3: "#d6a2ad"}  # plaster/tile/carpet/mirror
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "scenes")


def load_obj(path):
    V, F = [], []
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in ln.split()[1:4]])
    return np.array(V), np.array(F)


def load_meta(path):
    m = {"openings": [], "obstacles": [], "agents": []}
    for ln in open(path):
        p = ln.split()
        if p[0] == "room": m["room"] = list(map(float, p[1:4]))
        elif p[0] == "origin": m["origin"] = list(map(float, p[1:4]))
        elif p[0] == "opening":
            m["openings"].append(dict(wall=int(p[2]), c=list(map(float, p[4:7])), w=float(p[8]), h=float(p[10])))
        elif p[0] == "obstacle":
            m["obstacles"].append(dict(c=list(map(float, p[2:5])), half=list(map(float, p[6:9])), mat=int(p[10])))
        elif p[0] == "agent":
            m["agents"].append(dict(pos=list(map(float, p[1:4])), yaw=float(p[5])))
    return m


def render(seed, out):
    V, F = load_obj(os.path.join(HERE, f"scene_{seed}.obj"))
    m = load_meta(os.path.join(HERE, f"scene_{seed}.txt"))
    Lx, Ly, Lz = m["room"]; ox, oy, oz = m["origin"]
    x0, x1, z0, z1 = ox, ox + Lx, oz, oz + Lz

    fig = plt.figure(figsize=(13, 5.6))

    # ---- 3D perspective (the real mesh) ----
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    tris = V[F]
    ax.add_collection3d(Poly3DCollection(tris, facecolor="#8aa0b4", edgecolor="#33475a",
                                         linewidths=0.25, alpha=0.28))
    for ob in m["obstacles"]:
        c, h = np.array(ob["c"]), np.array(ob["half"])
        cor = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
        faces = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]]
        ax.add_collection3d(Poly3DCollection([cor[f] for f in faces], facecolor=MAT_COL[ob["mat"]],
                                             edgecolor="#2a2a2a", linewidths=0.4, alpha=0.9))
    for a in m["agents"]:
        px, py, pz = a["pos"]; t = np.radians(a["yaw"])
        ax.scatter([px], [py], [pz], color="#e8420a", s=45, depthshade=False)
        ax.quiver(px, py, pz, np.sin(t) * 1.2, 0, -np.cos(t) * 1.2, color="#e8420a", linewidth=1.6)
    ax.set_xlim(x0, x1); ax.set_zlim(0, Ly); ax.set_ylim(z0, z1)
    ax.set_box_aspect((Lx, Lz, Ly)); ax.view_init(elev=32, azim=-58)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_zlabel("y (m)")
    ax.set_title(f"seed {seed}: real mesh (walls + pillars + agents)", fontsize=10)

    # ---- top-down map ----
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.add_patch(plt.Rectangle((x0, z0), Lx, Lz, fill=False, edgecolor="#33475a", linewidth=2))
    for op in m["openings"]:                       # mark carved openings in green
        cx, cy, cz = op["c"]; w = op["w"]
        if op["wall"] in (0, 1):                   # x-wall: opening spans z
            ax2.plot([cx, cx], [cz - w/2, cz + w/2], color="#12805c", linewidth=4, solid_capstyle="butt")
        else:                                      # z-wall: opening spans x
            ax2.plot([cx - w/2, cx + w/2], [cz, cz], color="#12805c", linewidth=4, solid_capstyle="butt")
    for ob in m["obstacles"]:
        c, h = ob["c"], ob["half"]
        ax2.add_patch(plt.Rectangle((c[0]-h[0], c[2]-h[2]), 2*h[0], 2*h[2],
                                    facecolor=MAT_COL[ob["mat"]], edgecolor="#2a2a2a", linewidth=0.8))
    for i, a in enumerate(m["agents"]):
        px, _, pz = a["pos"]; t = np.radians(a["yaw"])
        ax2.arrow(px, pz, np.sin(t)*0.8, -np.cos(t)*0.8, head_width=0.28, color="#e8420a",
                  length_includes_head=True, zorder=5)
        ax2.text(px+0.15, pz+0.15, str(i), color="#e8420a", fontsize=9, fontweight="bold")
    ax2.set_xlim(x0-0.5, x1+0.5); ax2.set_ylim(z0-0.5, z1+0.5); ax2.set_aspect("equal")
    ax2.invert_yaxis(); ax2.grid(alpha=0.25, lw=0.5)
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("z (m)")
    ax2.set_title(f"top-down: {Lx:.1f}x{Lz:.1f} m, {len(m['obstacles'])} pillars, "
                  f"{len(m['openings'])} opening(s), {len(m['agents'])} agents", fontsize=10)
    from matplotlib.lines import Line2D
    ax2.legend(handles=[Line2D([],[],color="#12805c",lw=4,label="opening"),
                        Line2D([],[],marker=">",color="#e8420a",ls="",label="agent+heading"),
                        plt.Rectangle((0,0),1,1,fc="#a7c4a0",ec="#2a2a2a",label="pillar")],
               loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 5, 10])
    args = ap.parse_args()
    for s in args.seeds:
        render(s, os.path.join(HERE, f"scene_{s}.png"))


if __name__ == "__main__":
    main()
