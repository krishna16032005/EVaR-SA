"""Pull runs from wandb and render paper-ready figures + a summary table.

Aggregates across seeds rather than plotting them raw: for each wandb group, the
per-seed curves are interpolated onto a common x-grid and reduced to a mean with
a 95% confidence band. A single seed of an RL curve says very little; the band is
the honest object, and it is what a reviewer will ask for.

    python analysis/wandb_report.py                       # every group in the project
    python analysis/wandb_report.py --groups cartpole-c51-alpha0.1 cartpole-iqn-alpha0.1
    python analysis/wandb_report.py --out results/figures

Figures land as both PNG (to look at) and PDF (vector, for LaTeX).
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless server: no display, write files only
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1-3 of the reference palette, in fixed order. Fixed order
# matters: a series keeps its colour when the filter changes which runs are
# present, so "the blue one" means the same method across every figure.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

# (wandb key, axis label, figure title, filename stem)
PANELS = [
    ("episode/return_avg_window", "Episode return (moving average)",
     "Learning curves", "learning_curves"),
    ("update/risk_premium_mean", "EVaR - E[Z]  (risk premium)",
     "How much the entropic tilt inflates the critic", "risk_premium"),
    ("update/evar_dual_x_mean", "x* = 1/beta*",
     "Solved EVaR dual variable", "evar_dual"),
]


def style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    """Recessive grid and axes; the data is the only prominent thing."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)


def fetch(entity: str, project: str, groups: list[str] | None) -> dict[str, list]:
    import wandb
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    by_group: dict[str, list] = defaultdict(list)
    for run in runs:
        if run.state == "crashed":
            continue
        group = run.group or run.name
        if groups and group not in groups:
            continue
        by_group[group].append(run)
    return dict(by_group)


def series_for(runs: list, key: str, points: int = 400):
    """Per-seed (x, y) curves on a shared grid -> (grid, mean, lo, hi, n_seeds).

    Seeds are interpolated onto a common grid because each run logs at its own
    step boundaries; without that the mean would be taken over ragged x values.
    """
    curves = []
    for run in runs:
        xs, ys = [], []
        for row in run.scan_history(keys=["_step", key], page_size=10_000):
            if row.get(key) is None:
                continue
            xs.append(row["_step"])
            ys.append(row[key])
        if len(xs) > 2:
            curves.append((np.asarray(xs, float), np.asarray(ys, float)))
    if not curves:
        return None

    hi_x = min(c[0].max() for c in curves)  # only the span every seed reached
    lo_x = max(c[0].min() for c in curves)
    if hi_x <= lo_x:
        return None
    grid = np.linspace(lo_x, hi_x, points)
    stack = np.vstack([np.interp(grid, x, y) for x, y in curves])

    mean = stack.mean(axis=0)
    n = stack.shape[0]
    if n > 1:
        # 95% CI of the mean across seeds (normal approx; n is small, so this is
        # indicative rather than exact -- the seed count is printed alongside).
        half = 1.96 * stack.std(axis=0, ddof=1) / np.sqrt(n)
        return grid, mean, mean - half, mean + half, n
    return grid, mean, mean, mean, n


def render(by_group: dict[str, list], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    order = sorted(by_group)

    for key, ylabel, title, stem in PANELS:
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
        fig.patch.set_facecolor(SURFACE)
        plotted = 0
        for i, group in enumerate(order):
            data = series_for(by_group[group], key)
            if data is None:
                continue
            grid, mean, lo, hi, n = data
            colour = SERIES[i % len(SERIES)]
            ax.fill_between(grid, lo, hi, color=colour, alpha=0.15, linewidth=0, zorder=2)
            ax.plot(grid, mean, color=colour, linewidth=2.0, zorder=3,
                    label=f"{group}  (n={n})")
            # Direct label at the line end: identity without a legend round-trip.
            ax.annotate(f" {group}", (grid[-1], mean[-1]), color=colour,
                        fontsize=9, va="center", ha="left", zorder=4)
            plotted += 1

        if not plotted:
            plt.close(fig)
            continue

        style_axes(ax, "Environment steps", ylabel, title)
        if plotted > 1:
            leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
            for text in leg.get_texts():
                text.set_color(INK_SOFT)      # text wears ink, not the series colour
        ax.margins(x=0.18)                     # room for the direct labels
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(out_dir, f"{stem}.{ext}")
            fig.savefig(path, facecolor=SURFACE)
            written.append(path)
        plt.close(fig)
    return written


def summary_table(by_group: dict[str, list], out_dir: str) -> str:
    """Final-performance table: mean +/- std across seeds, plus per-seed values."""
    path = os.path.join(out_dir, "summary.csv")
    rows = []
    for group in sorted(by_group):
        finals = []
        for run in by_group[group]:
            value = run.summary.get("episode/return_avg_window")
            if value is not None:
                finals.append(float(value))
        if finals:
            arr = np.asarray(finals)
            rows.append({
                "group": group,
                "seeds": len(arr),
                "final_return_mean": round(float(arr.mean()), 2),
                "final_return_std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 2),
                "final_return_min": round(float(arr.min()), 2),
                "final_return_max": round(float(arr.max()), 2),
            })
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["group", "seeds", "final_return_mean",
                                               "final_return_std", "final_return_min",
                                               "final_return_max"])
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        print(f"\n{'GROUP':<34} {'SEEDS':>5} {'FINAL RETURN':>18}")
        for r in rows:
            print(f"{r['group']:<34} {r['seeds']:>5} "
                  f"{r['final_return_mean']:>10.1f} +/- {r['final_return_std']:<5.1f}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get(
        "WANDB_ENTITY", "deepg98-technical-university-of-munich"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "evar-deeprl"))
    parser.add_argument("--groups", nargs="*", default=None,
                        help="wandb groups to include (default: all in the project)")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()

    by_group = fetch(args.entity, args.project, args.groups)
    if not by_group:
        print("no runs matched -- check --entity/--project/--groups")
        return
    for group, runs in sorted(by_group.items()):
        print(f"{group}: {len(runs)} run(s) [{', '.join(sorted({r.state for r in runs}))}]")

    written = render(by_group, args.out)
    written.append(summary_table(by_group, args.out))
    print("\nwrote:")
    for path in written:
        print(" ", path)


if __name__ == "__main__":
    main()
