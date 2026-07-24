"""Two headline figures for EchoRL:
  1. The bearing gradient   -> results/echorl_bearing_gradient.png
  2. The MPC-vs-RL crossover -> results/echorl_mpc_crossover.png
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


if __name__ == "__main__":
    bearing_gradient()
    mpc_crossover()
