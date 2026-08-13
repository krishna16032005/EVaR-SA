"""Render paper-ready figures + a summary table from wandb or from local CSVs.

Two sources, one set of figures:

    python analysis/report.py                    # from wandb (needs the api key)
    python analysis/report.py --source csv       # from results/ on disk

The CSV path exists because the run CSVs live on the bind mount, so results can
be plotted from the *host* -- no container, no wandb credentials, no network.

Aggregates across seeds rather than plotting them raw: per-seed curves are
interpolated onto a common x-grid and reduced to a mean with a 95% band. A single
seed of an RL curve says very little; the band is what a reviewer will ask for.

Figures land as both PNG (to look at) and PDF (vector, for LaTeX).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless: write files, never try to open a window
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1-3 of the reference palette, in fixed order. Fixed order
# matters: a series keeps its colour when a filter changes which runs are
# present, so "the blue one" means the same method in every figure.
X_LABEL = ["Episode"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

FINAL_KEY = "episode/return_avg_window"

# (key, axis label, title, filename stem, smoothing window)
# Update-level metrics are logged every few env steps and are unreadable raw --
# the spikes hide the trend the panel exists to show. Episode return is already a
# moving average, so it gets none.
PANELS = [
    (FINAL_KEY, "Episode return (moving average)", "Learning curves", "learning_curves", 0),
    ("update/risk_premium_mean", "EVaR - E[Z]  (risk premium)",
     "How much the entropic tilt inflates the critic", "risk_premium", 21),
    ("update/evar_dual_x_mean", "x* = 1/beta*", "Solved EVaR dual variable", "evar_dual", 21),
]

RUN_DIR_RE = re.compile(r"^(?P<critic>c51|iqn)_alpha(?P<alpha>[\d.]+)_seed(?P<seed>\d+)_(?P<stamp>[\d-]+)$")


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average, edge-padded so the curve keeps its full span."""
    if window < 3 or y.size < window:
        return y
    pad = window // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(window) / window, mode="valid")[: y.size]


# -------------------------------------------------------------------- sources --
def collect_csv(results_root: str, x_column: str = "global_step") -> dict[str, list[dict]]:
    """Reads results/<env>/<critic>_alpha<a>_seed<s>_<stamp>/*.csv.

    Keeps only the **newest run per (group, seed)**: re-running a seed is normal,
    and the early smoke-test runs share seed 0 with the real sweep. Without this
    they would be averaged together as if they were independent seeds.
    """
    newest: dict[tuple[str, int], tuple[str, str]] = {}
    for env in sorted(os.listdir(results_root)):
        env_dir = os.path.join(results_root, env)
        if not os.path.isdir(env_dir):
            continue
        for name in os.listdir(env_dir):
            m = RUN_DIR_RE.match(name)
            if not m:
                continue
            group = f"{env}-{m['critic']}-alpha{m['alpha']}"
            seed = int(m["seed"])
            prev = newest.get((group, seed))
            if prev is None or m["stamp"] > prev[0]:
                newest[(group, seed)] = (m["stamp"], os.path.join(env_dir, name))

    by_group: dict[str, list[dict]] = defaultdict(list)
    for (group, seed), (_, run_dir) in sorted(newest.items()):
        tables: dict[str, dict[str, list[float]]] = {}
        for file in os.listdir(run_dir):
            kind = ("episode" if file.endswith("_episode.csv")
                    else "update" if file.endswith("_update.csv") else None)
            if kind is None:
                continue
            columns: dict[str, list[float]] = defaultdict(list)
            with open(os.path.join(run_dir, file)) as fh:
                for row in csv.DictReader(fh):
                    for col, value in row.items():
                        try:
                            columns[col].append(float(value))
                        except (TypeError, ValueError):
                            pass
            tables[kind] = columns

        curves = {}
        for key, *_ in PANELS:
            kind, column = key.split("/", 1)
            table = tables.get(kind, {})
            if column in table and x_column in table:
                x = np.asarray(table[x_column], float)
                y = np.asarray(table[column], float)
                if x.size > 2 and x.size == y.size:
                    curves[key] = (x, y)
        final = curves[FINAL_KEY][1][-1] if FINAL_KEY in curves else None
        by_group[group].append({"seed": seed, "curves": curves, "final": final})
    return dict(by_group)


def collect_wandb(entity: str, project: str) -> dict[str, list[dict]]:
    import wandb
    api = wandb.Api()
    by_group: dict[str, list[dict]] = defaultdict(list)
    for run in api.runs(f"{entity}/{project}"):
        if run.state == "crashed":
            continue
        keys = [key for key, *_ in PANELS]
        series: dict[str, tuple[list[float], list[float]]] = {k: ([], []) for k in keys}
        for row in run.scan_history(keys=["_step", *keys], page_size=10_000):
            for key in keys:
                if row.get(key) is not None:
                    series[key][0].append(row["_step"])
                    series[key][1].append(row[key])
        curves = {k: (np.asarray(x, float), np.asarray(y, float))
                  for k, (x, y) in series.items() if len(x) > 2}
        by_group[run.group or run.name].append(
            {"seed": run.config.get("seed"), "curves": curves,
             "final": run.summary.get(FINAL_KEY)})
    return dict(by_group)


# ------------------------------------------------------------------ aggregate --
def series_for(seeds: list[dict], key: str, points: int = 400):
    """Per-seed curves on a shared grid -> (grid, mean, lo, hi, n).

    Seeds are interpolated onto a common grid because each run logs at its own
    step boundaries; a raw mean would average over ragged x values.
    """
    curves = [s["curves"][key] for s in seeds if key in s["curves"]]
    if not curves:
        return None
    lo_x = max(c[0].min() for c in curves)
    hi_x = min(c[0].max() for c in curves)   # only the span every seed reached
    if hi_x <= lo_x:
        return None
    grid = np.linspace(lo_x, hi_x, points)
    stack = np.vstack([np.interp(grid, x, y) for x, y in curves])
    mean, n = stack.mean(axis=0), stack.shape[0]
    if n > 1:
        half = 1.96 * stack.std(axis=0, ddof=1) / np.sqrt(n)  # normal approx; n is small
        return grid, mean, mean - half, mean + half, n
    return grid, mean, mean, mean, n


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


def render(by_group: dict[str, list[dict]], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written, order = [], sorted(by_group)

    for key, ylabel, title, stem, window in PANELS:
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
        fig.patch.set_facecolor(SURFACE)
        plotted, x_lo, x_hi = 0, None, None
        for i, group in enumerate(order):
            data = series_for(by_group[group], key)
            if data is None:
                print(f"  note: '{group}' has no usable {key}; skipped in {stem}")
                continue
            grid, mean, lo, hi, n = data
            mean, lo, hi = smooth(mean, window), smooth(lo, window), smooth(hi, window)
            colour = SERIES[i % len(SERIES)]
            ax.fill_between(grid, lo, hi, color=colour, alpha=0.15, linewidth=0, zorder=2)
            ax.plot(grid, mean, color=colour, linewidth=2.0, zorder=3, label=f"{group}  (n={n})")
            ax.annotate(f" {group.replace('cartpole-', '')}", (grid[-1], mean[-1]), color=colour,
                        fontsize=9, va="center", ha="left", zorder=4)
            x_lo = grid[0] if x_lo is None else min(x_lo, grid[0])
            x_hi = grid[-1] if x_hi is None else max(x_hi, grid[-1])
            plotted += 1

        if not plotted:
            plt.close(fig)
            continue
        style_axes(ax, X_LABEL[0], ylabel, title)
        if plotted > 1:
            leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
            for text in leg.get_texts():
                text.set_color(INK_SOFT)   # text wears ink, not the series colour
        # Pad only the right, for the direct labels: a symmetric margin puts the
        # axis at negative environment steps, a range that cannot exist.
        ax.set_xlim(x_lo, x_hi + 0.20 * (x_hi - x_lo))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = os.path.join(out_dir, f"{stem}.{ext}")
            fig.savefig(path, facecolor=SURFACE)
            written.append(path)
        plt.close(fig)
    return written


def summary_table(by_group: dict[str, list[dict]], out_dir: str) -> str:
    path = os.path.join(out_dir, "summary.csv")
    rows = []
    for group in sorted(by_group):
        finals = [s["final"] for s in by_group[group] if s["final"] is not None]
        if not finals:
            continue
        arr = np.asarray(finals, float)
        rows.append({"group": group, "seeds": len(arr),
                     "final_return_mean": round(float(arr.mean()), 2),
                     "final_return_std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 2),
                     "final_return_min": round(float(arr.min()), 2),
                     "final_return_max": round(float(arr.max()), 2)})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["group"])
        writer.writeheader()
        writer.writerows(rows)
    if rows:
        print(f"\n{'GROUP':<30} {'SEEDS':>5} {'FINAL RETURN (mean +/- sd)':>28} {'MIN':>7} {'MAX':>7}")
        for r in rows:
            print(f"{r['group']:<30} {r['seeds']:>5} "
                  f"{r['final_return_mean']:>14.1f} +/- {r['final_return_std']:<10.1f} "
                  f"{r['final_return_min']:>7.1f} {r['final_return_max']:>7.1f}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["wandb", "csv"], default="wandb")
    parser.add_argument("--x", choices=["episode", "steps"], default="episode",
                        help="common x-axis. 'episode' is the default because runs are "
                             "episode-budgeted: a weaker policy takes fewer env steps for "
                             "the same 500 episodes, so a step axis truncates the "
                             "comparison at the worst seed's step count")
    parser.add_argument("--results", default="results", help="results root for --source csv")
    parser.add_argument("--entity", default=os.environ.get(
        "WANDB_ENTITY", "deepg98-technical-university-of-munich"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "evar-deeprl"))
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="*", default=["pipeline-smoke"])
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()

    x_column = "episode" if args.x == "episode" else "global_step"
    by_group = (collect_csv(args.results, x_column) if args.source == "csv"
                else collect_wandb(args.entity, args.project))
    by_group = {g: s for g, s in by_group.items()
                if (not args.groups or g in args.groups) and g not in (args.exclude or [])}
    if not by_group:
        print("no runs matched -- check --source/--results/--groups")
        return
    for group, seeds in sorted(by_group.items()):
        print(f"{group}: {len(seeds)} seed(s) {sorted(s['seed'] for s in seeds if s['seed'] is not None)}")

    X_LABEL[0] = "Episode" if args.x == "episode" else "Environment steps"
    written = render(by_group, args.out)
    written.append(summary_table(by_group, args.out))
    print("\nwrote:")
    for path in written:
        print(" ", path)


if __name__ == "__main__":
    main()
