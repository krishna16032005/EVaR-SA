"""Plot training vs. evaluation return curves saved by run_cartpole.py / run_invpend.py.

For each ``*_episode.csv`` this draws the noisy on-policy training return (thin,
translucent -- see the caveat in ``evar_deeprl/agents/base.py``'s module docstring)
and, if a matching ``*_eval.csv`` exists alongside it (same run directory), overlays
the deterministic evaluation return (bold, with a shaded +/-1 std band) -- the actual
"how good is the policy" signal.

Usage:
    python experiments/plot_results.py results/cartpole/*/*_episode.csv
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values: list[float], window: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def read_columns(path: str, columns: list[str]) -> dict[str, list[float]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {col: [float(row[col]) for row in rows] for col in columns}


def main() -> None:
    patterns = sys.argv[1:] or [os.path.join("results", "**", "*_episode.csv")]
    paths = sorted({p for pattern in patterns for p in glob.glob(pattern, recursive=True)})
    if not paths:
        print("No *_episode.csv files found (produced by run_cartpole.py / run_invpend.py).")
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    plt.figure(figsize=(9, 5.5))
    for i, path in enumerate(paths):
        label = os.path.basename(path).replace("_episode.csv", "")
        color = colors[i % len(colors)]

        train_return = read_columns(path, ["episode_return"])["episode_return"]
        smoothed = moving_average(train_return)
        plt.plot(smoothed, color=color, alpha=0.35, linewidth=1, label=f"{label} (train)")

        eval_path = path.replace("_episode.csv", "_eval.csv")
        if os.path.exists(eval_path):
            eval_cols = read_columns(eval_path, ["episode", "eval_return_mean", "eval_return_std"])
            episodes = eval_cols["episode"]
            mean = np.asarray(eval_cols["eval_return_mean"])
            std = np.asarray(eval_cols["eval_return_std"])
            plt.plot(episodes, mean, color=color, linewidth=2.2, marker="o", markersize=3, label=f"{label} (eval)")
            plt.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.15)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("EVaR actor-critic: training return (thin) vs. deterministic evaluation return (bold)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out_path = os.path.join("results", "episode_returns.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
