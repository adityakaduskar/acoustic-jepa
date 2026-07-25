"""Three headline figures for EchoRL:
  1. The bearing gradient    -> results/echorl_bearing_gradient.png
  2. The MPC-vs-RL crossover -> results/echorl_mpc_crossover.png
  3. The §5.9 2x2 + ladder   -> results/echorl_react_clearance.png
Numbers are the final 3-seed / fair-eval results from docs/EchoRL_RESULTS.md.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)

INK, GRID = "#1a1a1a", "#d8d8d8"
C_FULL, C_RANGE, C_BEAR = "#1f4e79", "#c23a2f", "#1a9850"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": "#555", "axes.linewidth": 0.8})


# ------------------------------------------------------------------ Figure 1
def bearing_gradient():
    nobs = [0, 1, 3]
    x = [0, 1, 2]                                    # even spacing, labelled by nobs
    full   = [14.0, 10.5, 5.9]; full_sd  = [3.1, 3.2, 1.2]
    rangeo = [11.2,  6.7, 1.7]; rangeo_sd = [1.3, 3.7, 0.3]
    drop   = [2.8, 3.8, 4.2]                          # full - range-only
    frac   = [0.80, 0.64, 0.29]                       # range-only / full

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # -- left: coverage of full vs bearing-removed, with the shrinking gap --
    axL.errorbar(x, full, yerr=full_sd, marker="o", ms=8, lw=2.2, color=C_FULL,
                 capsize=4, label="full (all cues)", zorder=3)
    axL.errorbar(x, rangeo, yerr=rangeo_sd, marker="s", ms=8, lw=2.2, color=C_RANGE,
                 capsize=4, label="bearing removed (range-only)", zorder=3)
    for xi, f, r, d in zip(x, full, rangeo, drop):
        axL.annotate("", xy=(xi, f), xytext=(xi, r),
                     arrowprops=dict(arrowstyle="<->", color="#888", lw=1.2))
        axL.text(xi + 0.06, (f + r) / 2, f"−{d:.1f}", color="#555",
                 va="center", fontsize=10, fontweight="bold")
    axL.axhspan(0, 2.2, color=C_RANGE, alpha=0.06)
    axL.text(2.0, 1.0, "parked\n(collapse)", color=C_RANGE, ha="center", va="center",
             fontsize=9, style="italic")
    axL.set_xticks(x); axL.set_xticklabels([f"{n} pillar{'s' if n!=1 else ''}\n(nobs={n})" for n in nobs])
    axL.set_ylabel("cells covered  (higher = more of the room swept)")
    axL.set_xlabel("clutter")
    axL.set_ylim(0, 18); axL.set_xlim(-0.35, 2.35)
    axL.grid(axis="y", color=GRID, lw=0.7); axL.set_axisbelow(True)
    axL.legend(loc="upper right", frameon=False, fontsize=10)
    axL.set_title("Bearing removed → less coverage, worsening with clutter",
                  fontsize=10.5, fontweight="bold", loc="left")

    # -- right: the drop-bearing gradient itself + collapse fraction --
    ax2 = axR.twinx()                                # collapse fraction on secondary axis (draw first, behind)
    coll_frac = [0/3, 2/3, 3/3]
    ax2.bar(x, coll_frac, width=0.34, color=C_RANGE, alpha=0.16, zorder=1)
    for xi, c in zip(x, coll_frac):
        ax2.text(xi, 0.03, f"{int(round(c*3))}/3", ha="center", va="bottom",
                 color=C_RANGE, fontsize=9)       # labels at the BASE of the bars
    ax2.set_ylabel("seeds that collapse to parking", color=C_RANGE)
    ax2.tick_params(axis="y", labelcolor=C_RANGE)
    ax2.set_ylim(0, 1.15)

    axR.plot(x, drop, marker="o", ms=9, lw=2.4, color=C_BEAR, zorder=3)
    for xi, d in zip(x, drop):
        axR.annotate(f"−{d:.1f}", (xi, d), textcoords="offset points", xytext=(-6, 8),
                     ha="right", color=C_BEAR, fontweight="bold")
    axR.set_xticks(x); axR.set_xticklabels([f"{n}\n(nobs={n})" for n in nobs])
    axR.set_xlabel("number of pillars")
    axR.set_ylabel("drop-bearing effect  (cells lost)", color=C_BEAR)
    axR.tick_params(axis="y", labelcolor=C_BEAR)
    axR.set_ylim(2.2, 4.7); axR.set_xlim(-0.35, 2.35)
    axR.grid(axis="y", color=GRID, lw=0.7); axR.set_axisbelow(True)
    axR.set_zorder(ax2.get_zorder() + 1); axR.patch.set_visible(False)
    axR.set_title("Monotonic: bearing matters more with more obstacles",
                  fontsize=10.5, fontweight="bold", loc="left")

    fig.suptitle("EchoRL — the binaural bearing cue is clutter-specific "
                 "(coverage task, 3 seeds, held-out rooms)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "echorl_bearing_gradient.png")
    fig.savefig(p, dpi=150); print("saved", p)


# ------------------------------------------------------------------ Figure 2
def mpc_crossover():
    # doorway task, success rate; two regimes
    labels = ["model-free\nPPO (reflex)", "learned-WM\nMPC (plan)", "direct-outcome\nMPC (plan)"]
    empty  = [0.68, 0.46, 0.29]
    clutter = [0.13, 0.25, 0.24]        # PPO 0.08-0.18 -> plot midpoint 0.13 w/ range bar
    ppo_lo, ppo_hi = 0.08, 0.18

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    xs = np.arange(3); w = 0.36
    b1 = ax.bar(xs - w/2, empty, w, color="#5b8dd6", label="empty rooms (reacting suffices)", zorder=3)
    b2 = ax.bar(xs + w/2, clutter, w, color="#d98a3d", label="cluttered rooms (planning required)", zorder=3)

    # oracle ceilings + PPO uncertainty
    ax.axhline(1.00, ls="--", lw=1, color="#888")
    ax.text(2.46, 1.005, "oracle ceiling = 1.00", ha="right", va="bottom", color="#666", fontsize=9)
    ax.errorbar(0 + w/2, 0.13, yerr=[[0.13-ppo_lo],[ppo_hi-0.13]], fmt="none",
                ecolor=INK, capsize=4, lw=1.4, zorder=4)

    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{b.get_height():.2f}",
                ha="center", fontsize=10, fontweight="bold", color="#1f4e79")
    for i, b in enumerate(b2):
        lab = "0.08–0.18" if i == 0 else f"{b.get_height():.2f}"
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, lab,
                ha="center", fontsize=10, fontweight="bold", color="#9c5a1e")

    # crossover annotation on the reflex column
    ax.annotate("reflex agent COLLAPSES\nwhen planning is required",
                xy=(0 + w/2, 0.18), xytext=(0.55, 0.62),
                arrowprops=dict(arrowstyle="->", color=C_RANGE, lw=1.5),
                color=C_RANGE, fontsize=10, fontweight="bold")

    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("doorway-finding success rate")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=False, fontsize=9.5, ncol=1)
    ax.set_title("EchoRL — model-free vs model-based crossover\n"
                 "reactive wins in the open; model-based wins when lookahead is needed",
                 fontsize=11.5, fontweight="bold", loc="left")
    fig.tight_layout()
    p = os.path.join(OUT, "echorl_mpc_crossover.png")
    fig.savefig(p, dpi=150); print("saved", p)


# ------------------------------------------------------------------ Figure 3
# Categorical slots 1-2 of the documented palette (blue, orange). Validated:
# OKLCH L 0.575/0.671 in band, C 0.163/0.175 over floor, CVD dE min(protan,deutan)
# 24.7 (target 8), normal dE 33.6 (floor 15), contrast 4.30:1 / 3.12:1 vs surface.
C_FUSE, C_REACT = "#2a78d6", "#eb6834"


def react_clearance():
    # per-seed success, nobs=3 goal, 12 held-out rooms x 24 = 288 episodes/cell.
    # The first version used 8 eps/room; 96 binary episodes carry ~0.05 of binomial noise,
    # which manufactured an A-x-B interaction that did not survive. See the §5.9 correction.
    runs = {("scatter", "fuse"):  [0.479, 0.417, 0.472],
            ("scatter", "react"): [0.514, 0.490, 0.552],
            ("both", "fuse"):     [0.486, 0.417, 0.483],
            ("both", "react"):    [0.556, 0.469, 0.517]}
    mean = {k: float(np.mean(v)) for k, v in runs.items()}
    sd = {k: float(np.std(v, ddof=1)) for k, v in runs.items()}
    CEIL = 0.948                                   # planner on the TRUE map

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # -- left: the 2x2. Colour = which consumer reads the doorway cue --
    x = np.arange(2); w = 0.34
    for j, (goal, colour, lab) in enumerate(
            [("fuse", C_FUSE, "fused doorway POINT (§5.8)"),
             ("react", C_REACT, "reactive BEARING (§5.9)")]):
        m = [mean[(o, goal)] for o in ("scatter", "both")]
        e = [sd[(o, goal)] for o in ("scatter", "both")]
        bars = axL.bar(x + (j - 0.5) * (w + 0.02), m, w, color=colour, label=lab, zorder=3)
        axL.errorbar(x + (j - 0.5) * (w + 0.02), m, yerr=e, fmt="none", ecolor=INK,
                     capsize=4, lw=1.2, zorder=4)
        for b, mi, ei in zip(bars, m, e):
            axL.text(b.get_x() + b.get_width() / 2, mi + ei + 0.018, f"{mi:.2f}",
                     ha="center", fontsize=10, fontweight="bold", color=INK)
    axL.axhline(CEIL, ls="--", lw=1, color="#888")
    axL.text(1.52, CEIL + 0.012, "perfect-map ceiling 0.95", ha="right", va="bottom",
             color="#666", fontsize=9)
    # what the re-powered data says: the map half is flat, the consumer half is not
    axL.annotate("", xy=(1 - (w + 0.02) / 2, mean[("both", "fuse")] + 0.06),
                 xytext=(0 - (w + 0.02) / 2, mean[("scatter", "fuse")] + 0.06),
                 arrowprops=dict(arrowstyle="->", color="#7a7a7a", lw=1.3, ls=(0, (4, 2))))
    axL.text(0.5, 0.80, "the clearance map buys nothing\neither way (+0.01 alone, −0.01 on top)",
             ha="center", color="#5a5a5a", fontsize=9, fontweight="bold")
    axL.set_xticks(x)
    axL.set_xticklabels(["scatter-CNN map\n(§5.8)", "+ clearance field\n(§5.9)"])
    axL.set_xlabel("obstacle map")
    axL.set_ylabel("doorway-finding success  (3 seeds, held-out rooms)")
    axL.set_ylim(0, 1.26); axL.set_xlim(-0.55, 1.55)
    axL.grid(axis="y", color=GRID, lw=0.7); axL.set_axisbelow(True)
    axL.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=9.5)
    axL.set_title("The CONSUMER of the bearing is the whole effect",
                  fontsize=10.5, fontweight="bold", loc="left")

    # -- right: where §5.9 sits on the whole §5 ladder --
    rows = [("model-free PPO", 0.13, "#9a9a9a"),
            ("latent-WM MPC", 0.245, "#9a9a9a"),
            ("§5.8 map + fused point", mean[("scatter", "fuse")], C_FUSE),
            ("§5.9 + reactive bearing", mean[("scatter", "react")], C_REACT),
            ("    → given perfect obstacles*", 0.674, "#f0a480"),
            ("    → given perfect doorway*", 0.785, "#f0a480"),
            ("planner on the TRUE map", CEIL, "#5a5a5a")]
    ys = np.arange(len(rows))[::-1]
    for y, (lab, v, colour) in zip(ys, rows):
        axR.barh(y, v, height=0.6, color=colour, zorder=3)
        off = 0.075 if lab.startswith("model-free") else 0.012   # clear the PPO range bar
        axR.text(v + off, y, f"{v:.2f}", va="center", fontsize=10,
                 fontweight="bold", color=INK)
    axR.errorbar(0.13, ys[0], xerr=[[0.05], [0.05]], fmt="none", ecolor=INK,
                 capsize=4, lw=1.2, zorder=4)
    axR.set_yticks(ys); axR.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    axR.set_xlabel("doorway-finding success (cluttered, nobs=3)")
    axR.set_xlim(0, 1.12)
    axR.grid(axis="x", color=GRID, lw=0.7); axR.set_axisbelow(True)
    axR.annotate("following a DIRECTION costs nothing: true bearing, NO range = 0.94\n"
                 "* half-ceilings measured at the older 8-episode resolution (±0.05)",
                 xy=(CEIL, ys[-1] - 0.34), xytext=(0.52, ys[-1] - 1.25),
                 arrowprops=dict(arrowstyle="->", color="#666", lw=1.2),
                 color="#444", fontsize=8.5, style="italic", ha="center")
    axR.set_ylim(-1.9, len(rows) - 0.4)
    axR.set_title("Doorway-limited, not obstacle-map-limited",
                  fontsize=10.5, fontweight="bold", loc="left")

    fig.suptitle("EchoRL §5.9 — a bearing is a gradient to follow, not a coordinate to compute\n"
                 "288 episodes/cell, 3 seeds; effects below ~0.04 are not resolvable",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "echorl_react_clearance.png")
    fig.savefig(p, dpi=150); print("saved", p)


if __name__ == "__main__":
    bearing_gradient()
    mpc_crossover()
    react_clearance()
