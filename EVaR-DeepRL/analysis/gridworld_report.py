"""Figures for the lottery gridworld: the environment, C1, and the C3 collapse.

Palette and conventions follow analysis/report.py -- categorical slots in fixed
order, thin marks, recessive axes, direct labels rather than a number on every
point. Writes PNG (to look at) and PDF (vector, for LaTeX).

    python analysis/gridworld_report.py --trained-glob '/tmp/gws_*'
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from evar_deeprl.envs.lottery_gridworld import (
    DEFAULT_SEGMENTS, LotteryGridWorld, evar_exact, lane_sequence, optimal_policy,
    return_distribution)
from analysis.c3_attribution import fixed_point, table_policy

# Categorical slots 1-3 of the reference palette, fixed order. Only three are used:
# those three validate on the all-pairs list, and colour never encodes rank here.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
ALPHAS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)


def save(fig, out, stem):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"), dpi=200,
                    facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}/{stem}.png / .pdf")


# ------------------------------------------------------------------ figure 1 --
def _draw_route(ax, env, lanes, CW, CH, GAP, colour, dashed, halo, zorder):
    def route_y(c):
        return (1 - lanes[c]) * (CH + GAP) + CH * (0.17 if not dashed else 0.34)

    pts = [(-GAP * 0.8, route_y(0))]
    for c in range(env.n_segments):
        pts.append((c * (CW + GAP), route_y(c)))
        pts.append((c * (CW + GAP) + CW, route_y(c)))
    pts.append((env.n_segments * (CW + GAP) - GAP * 0.2, pts[-1][1]))
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=colour, linewidth=2.0, zorder=zorder,
            solid_capstyle="round", linestyle="--" if dashed else "-",
            path_effects=halo)
    ax.plot(xs[-1], ys[-1], marker=">", color=colour, markersize=8, zorder=zorder + 1,
            path_effects=halo)
    ax.plot(xs[0], ys[0], marker="o", color=colour, markersize=6, zorder=zorder + 1)


def _draw_grid(ax, env, lanes, title, subtitle, learned=None):
    """One gridworld with the route drawn through it."""
    CW, CH, GAP = 1.22, 0.62, 0.26   # wide enough that payoff text fits inside
    for c, seg in enumerate(env.segments):
        for row in (0, 1):
            on = (lanes[c] == row)
            colour = AQUA if row == 0 else ORANGE
            y = (1 - row) * (CH + GAP)
            ax.add_patch(FancyBboxPatch(
                (c * (CW + GAP), y), CW, CH,
                boxstyle="round,pad=0.01,rounding_size=0.05",
                linewidth=1.6 if on else 0,
                edgecolor=colour if on else "none",
                facecolor=colour, alpha=0.30 if on else 0.07, zorder=1))
            txt = (f"{seg.safe:.0f}" if row == 0
                   else f"{seg.hi:.0f}  p={seg.p:g}\nelse {seg.lo:g}")
            ax.text(c * (CW + GAP) + CW / 2, y + CH / 2, txt, ha="center", va="center",
                    fontsize=9.5 if row == 0 else 7.9,
                    color=INK if on else INK_SOFT, zorder=3,
                    weight="bold" if on else "normal")

    # The route: enter left, hop between lanes, exit right. It runs through the
    # lower third of each cell rather than the middle so it does not strike through
    # the payoff text, and carries a surface-coloured halo where it crosses a border.
    def route_y(c):
        return (1 - lanes[c]) * (CH + GAP) + CH * 0.17

    pts = [(-GAP * 0.8, route_y(0))]
    for c in range(env.n_segments):
        pts.append((c * (CW + GAP), route_y(c)))
        pts.append((c * (CW + GAP) + CW, route_y(c)))
    pts.append((env.n_segments * (CW + GAP) - GAP * 0.2, pts[-1][1]))
    xs, ys = zip(*pts)
    halo = [pe.Stroke(linewidth=4.6, foreground=SURFACE), pe.Normal()]
    ax.plot(xs, ys, color=INK, linewidth=2.0, zorder=5, solid_capstyle="round",
            path_effects=halo)
    ax.plot(xs[-1], ys[-1], marker=">", color=INK, markersize=8, zorder=6,
            path_effects=halo)
    ax.plot(xs[0], ys[0], marker="o", color=INK, markersize=6, zorder=6)

    ax.set_xlim(-0.75, env.n_segments * (CW + GAP) + 0.1)
    ax.set_ylim(-0.45, 2 * CH + GAP + 0.42)
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=9)
    ax.text(-0.7, -0.30, subtitle, fontsize=8.8, color=INK_SOFT, ha="left", va="top")


def fig_paths(env, out):
    """The routes themselves: which lanes each risk appetite actually walks."""
    # Group alphas by the policy they induce -- the policy only takes a few values,
    # so one panel per distinct route says more than one panel per alpha.
    groups, order = {}, []
    for a in ALPHAS:
        _, _, table, _, _ = optimal_policy(env, a)
        seq = lane_sequence(env, table)
        lanes = tuple(0 if s == "SAFE" else 1 for s in seq.split("-"))
        if lanes not in groups:
            groups[lanes] = []
            order.append(lanes)
        groups[lanes].append(a)

    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(4.15 * n, 3.6))
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(top=0.74, bottom=0.12, wspace=0.16)
    if n == 1:
        axes = [axes]
    for ax, lanes in zip(axes, order):
        ax.set_facecolor(SURFACE)
        alphas = groups[lanes]
        rng = (f"alpha = {alphas[0]:g}" if len(alphas) == 1
               else f"alpha {alphas[0]:g} to {alphas[-1]:g}")
        def pol(r, c, _l=lanes):
            return (1.0, 0.0) if _l[c] == 0 else (0.0, 1.0)
        v, p = return_distribution(env, pol)
        ev, _ = evar_exact(v, p, alphas[0])
        mean, best = float((v * p).sum()), float(v.max())
        n_risky = sum(lanes)
        sub = (f"mean {mean:.1f}    best case {best:.0f}\n"
               f"{n_risky} of 3 lotteries taken")
        _draw_grid(ax, env, lanes, f"{rng}", sub)
    fig.suptitle("The route each risk appetite walks: safe lane on top, lottery below\n"
                 "as alpha falls the agent takes more lotteries, and the best case grows",
                 fontsize=12.5, color=INK, x=0.006, ha="left", y=0.99, va="top")
    for ext in ("png", "pdf"):
        os.makedirs(out, exist_ok=True)
        fig.savefig(os.path.join(out, f"gridworld_paths.{ext}"), dpi=200,
                    facecolor=SURFACE)          # no bbox_inches: keep the reserved top
    plt.close(fig)
    print(f"  wrote {out}/gridworld_paths.png / .pdf")


def fig_environment(env, out):
    """Payoffs and the mean/upside trade, without any policy drawn on it."""
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.set_facecolor(SURFACE)
    for c, seg in enumerate(env.segments):
        for row, (color, label) in enumerate(
                ((AQUA, f"{seg.safe:.0f}"),
                 (ORANGE, f"{seg.hi:.0f} w.p. {seg.p:g}\nelse {seg.lo:g}"))):
            y = 1.0 if row == 0 else 0.0
            ax.add_patch(FancyBboxPatch(
                (c * 1.25, y), 1.0, 0.72, boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=0, facecolor=color, alpha=0.16, zorder=1))
            ax.text(c * 1.25 + 0.5, y + 0.46, label, ha="center", va="center",
                    fontsize=10.5, color=INK, zorder=3)
            ax.text(c * 1.25 + 0.5, y + 0.10,
                    "safe" if row == 0 else f"mean {seg.risky_mean:.2f}",
                    ha="center", va="center", fontsize=8.5, color=INK_SOFT, zorder=3)
        ax.text(c * 1.25 + 0.5, 1.95, f"segment {c}", ha="center", fontsize=9,
                color=INK_SOFT)
        ax.text(c * 1.25 + 0.5, -0.35, f"pays {seg.hi/seg.safe:.1f}x", ha="center",
                fontsize=9, color=INK_SOFT)
    ax.text(-0.55, 1.36, "SAFE", ha="right", va="center", fontsize=11, color=AQUA,
            weight="bold")
    ax.text(-0.55, 0.36, "RISKY", ha="right", va="center", fontsize=11, color=ORANGE,
            weight="bold")
    ax.text(3.9, 0.86, "GOAL", ha="left", va="center", fontsize=11, color=INK_SOFT)
    ax.set_xlim(-2.0, 4.9)
    ax.set_ylim(-0.6, 2.2)
    ax.axis("off")
    ax.set_title("Each lottery costs a little mean and pays a lot in the tail\n"
                 "all-risky: 6% less mean than all-safe, 9x the best case",
                 fontsize=11.5, color=INK, loc="left")
    save(fig, out, "gridworld_environment")


# ------------------------------------------------------------------ figure 2 --
def fig_c1(env, trained, out):
    """How many lotteries does each method take, against ground truth."""
    gt, fp = [], []
    for a in ALPHAS:
        _, _, table_opt, _, _ = optimal_policy(env, a)
        gt.append(lane_sequence(env, table_opt).count("RISKY"))
        best = None
        for init in (0, 1):
            start = {(r, c): init for c in range(env.n_segments) for r in (0, 1)}
            t, _ = fixed_point(env, a, start)
            v, p = return_distribution(env, table_policy(t))
            ev, _ = evar_exact(v, p, a)
            if best is None or ev > best[0]:
                best = (ev, lane_sequence(env, t).count("RISKY"))
        fp.append(best[1])

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    style(ax)
    x = np.arange(len(ALPHAS))
    ax.step(x, gt, where="mid", color=INK, linewidth=2.0, zorder=4)
    ax.plot(x, gt, "o", color=INK, markersize=8, zorder=5, label="optimal (exact)")
    if trained:
        mean = [np.mean(trained[a]) if a in trained else np.nan for a in ALPHAS]
        sd = [np.std(trained[a]) if a in trained else np.nan for a in ALPHAS]
        ax.errorbar(x, mean, yerr=sd, color=BLUE, linewidth=2.0, marker="o",
                    markersize=8, capsize=4, zorder=6, label="EVaR-AC, 5 seeds")
    ax.plot(x, fp, color=ORANGE, linewidth=2.0, marker="s", markersize=8, zorder=6,
            label="per-state operator, perfect critic")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("alpha  (smaller = more risk-seeking)", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("lotteries taken", color=INK_SOFT, fontsize=10)
    ax.set_ylim(-0.35, 3.4)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_SOFT, loc="upper right")
    ax.set_title("The trained agent tracks the optimum; the per-state operator does not\n"
                 "with a perfect critic it stays risk-neutral at every alpha",
                 fontsize=11.5, color=INK, loc="left")
    save(fig, out, "gridworld_c1_staircase")


# ------------------------------------------------------------------ figure 3 --
def fig_c3_mechanism(env, out):
    """Why it collapses: at the last decision there is no future left to tilt."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    style(ax)
    x = np.arange(len(ALPHAS))
    seg = env.segments[-1]
    ax.plot(x, [seg.safe] * len(ALPHAS), color=AQUA, linewidth=2.0, marker="o",
            markersize=8, zorder=5, label="safe lane")
    ax.plot(x, [seg.risky_mean] * len(ALPHAS), color=ORANGE, linewidth=2.0,
            marker="s", markersize=8, zorder=5, label="lottery, as the operator sees it")
    ev = []
    for a in ALPHAS:
        v = np.array([seg.hi, seg.lo])
        p = np.array([seg.p, 1 - seg.p])
        ev.append(evar_exact(v, p, a)[0])
    ax.plot(x, ev, color=BLUE, linewidth=2.0, marker="^", markersize=8, zorder=6,
            linestyle="--", label="lottery, EVaR of the reward itself")
    ax.annotate("the operator compares these two,\nand neither depends on alpha",
                xy=(3.0, 6.0), fontsize=9.5, color=INK_SOFT, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in ALPHAS])
    ax.set_xlabel("alpha", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("value of the final decision", color=INK_SOFT, fontsize=10)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_SOFT, loc="upper right")
    ax.set_title("At the last decision V(terminal) = 0, so alpha cannot enter\n"
                 "the immediate reward is compared by its mean, not its tail",
                 fontsize=11.5, color=INK, loc="left")
    save(fig, out, "gridworld_c3_mechanism")


# ------------------------------------------------------------------ figure 4 --
def fig_returns(env, out):
    """Return distributions of the two extreme policies, with EVaR marked."""
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    style(ax)
    for name, act, color in (("all safe", 0, AQUA), ("all risky", 1, ORANGE)):
        def pol(r, c, _a=act):
            return (1.0, 0.0) if _a == 0 else (0.0, 1.0)
        v, p = return_distribution(env, pol)
        ax.vlines(v, 0, p, color=color, linewidth=2.0, zorder=4)
        ax.plot(v, p, "o", color=color, markersize=7, zorder=5, label=name)
        ev, _ = evar_exact(v, p, 0.1)
        ax.axvline(ev, color=color, linestyle=":", linewidth=1.6, zorder=3)
        ax.text(ev, 0.62, f"  EVaR$_{{0.1}}$ = {ev:.0f}", color=color, fontsize=9.5,
                rotation=90, va="top")
    ax.set_xlabel("trajectory return", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("probability", color=INK_SOFT, fontsize=10)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK_SOFT, loc="upper right")
    ax.set_title("All-risky has the lower mean and the far heavier upper tail\n"
                 "which is exactly the trade alpha is supposed to govern",
                 fontsize=11.5, color=INK, loc="left")
    save(fig, out, "gridworld_return_distributions")


def load_trained(pattern):
    out = {}
    for d in glob.glob(pattern):
        for f in glob.glob(os.path.join(d, "*", "c1_summary.txt")):
            rec = dict(line.split("\t") for line in open(f).read().strip().split("\n"))
            a = float(rec["alpha"])
            out.setdefault(a, []).append(rec["learned_lanes"].count("RISKY"))
    return out


SWITCH_X = []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trained-glob", default="/tmp/gws_*")
    ap.add_argument("--out", default="results/figures_gridworld")
    args = ap.parse_args()

    env = LotteryGridWorld(DEFAULT_SEGMENTS)
    for seg in env.segments:
        xs = np.geomspace(0.05, 500.0, 20000)
        m = max(seg.hi, seg.lo)
        ce = np.array([x * (m / x + np.log(seg.p * np.exp(seg.hi / x - m / x)
                                           + (1 - seg.p) * np.exp(seg.lo / x - m / x)))
                       for x in xs])
        above = ce > seg.safe
        SWITCH_X.append(xs[np.argmax(~above)] if not above.all() else float("inf"))

    print(env.__class__.__name__, "figures ->", args.out)
    from evar_deeprl.envs.lottery_gridworld import render
    print(render(env))
    print()
    trained = load_trained(args.trained_glob)
    fig_paths(env, args.out)
    fig_environment(env, args.out)
    fig_c1(env, trained, args.out)
    fig_c3_mechanism(env, args.out)
    fig_returns(env, args.out)


if __name__ == "__main__":
    main()
