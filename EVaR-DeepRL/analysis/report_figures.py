"""Figures for the collaborator report that no other script produces.

Three things needed showing that the learning-curve panels cannot show, each one a
result about the *operator* rather than about a policy:

1. ``x* = 1/beta*`` scales linearly with the spread of the distribution it is
   handed. This is why a fixed search interval is a bug rather than a tuning
   choice, and it is the measurement behind the adaptive-interval change.
2. The solver's accuracy against brute-force minimisation, as a function of the
   bisection budget -- the evidence that 20 steps is not a corner cut.
3. What the dual variable is actually doing: G(x) for a few distributions, with
   the solved optimum marked, so a reader can see that the minimum is interior and
   what "pinned at a bound" would look like.

    python analysis/report_figures.py --out results/figures/report
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evar_deeprl.risk.evar import (EVaRConfig, evar_from_distribution,
                                   last_solve_diagnostics)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8)


def fig_scale(out):
    """x* against sd(Z), with and without the interval scaled to the distribution."""
    torch.manual_seed(0)
    stds = np.logspace(-3, 2, 18)
    fixed_x, auto_x, fixed_pin, auto_pin, fixed_err = [], [], [], [], []
    for s in stds:
        z = torch.randn(512, 64) * float(s)
        ref, _ = evar_from_distribution(z, None, EVaRConfig(alpha=0.1))
        a_x = float(_last_x(z, EVaRConfig(alpha=0.1)))
        auto_x.append(a_x)
        auto_pin.append(last_solve_diagnostics()["at_bound_frac"])
        cfg_f = EVaRConfig(alpha=0.1, x_min=1e-2, x_max=1e4, auto_scale_bounds=False)
        f_ev, f_xs = evar_from_distribution(z, None, cfg_f)
        fixed_x.append(float(f_xs.mean()))
        fixed_pin.append(last_solve_diagnostics()["at_bound_frac"])
        fixed_err.append(float(((f_ev - ref).abs() / ref.abs().clamp_min(1e-12)).mean()))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), dpi=170)
    fig.patch.set_facecolor(SURFACE)

    ax = axes[0]
    ax.loglog(stds, auto_x, color=SERIES[0], lw=2, marker="o", ms=3,
              label="interval scaled to sd(Z)")
    ax.loglog(stds, fixed_x, color=SERIES[1], lw=2, marker="s", ms=3,
              label="fixed interval [1e-2, 1e4]")
    ax.loglog(stds, 0.34 * stds, color=INK_SOFT, lw=1, ls="--",
              label=r"$0.34\,\mathrm{sd}(Z)$")
    ax.axhline(1e-2, color=SERIES[3], lw=1, ls=":", label=r"$x_{\min}=10^{-2}$")
    style(ax, "The dual optimum has units of return", "sd(Z)", "solved $x^*$")
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK_SOFT)

    ax = axes[1]
    ax.semilogx(stds, np.array(fixed_err) * 100, color=SERIES[1], lw=2, marker="s", ms=3,
                label="relative error, fixed interval")
    ax.semilogx(stds, np.array(fixed_pin) * 100, color=SERIES[3], lw=1.6, ls="--",
                label="% rows pinned, fixed interval")
    ax.semilogx(stds, np.array(auto_pin) * 100, color=SERIES[0], lw=2,
                label="% rows pinned, scaled interval")
    style(ax, "Consequence: a silent overestimate at small return scales",
          "sd(Z)", "percent")
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK_SOFT)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"solver_scale.{ext}"), facecolor=SURFACE)
    plt.close(fig)


def _last_x(z, cfg):
    _, xs = evar_from_distribution(z, None, cfg)
    return xs.mean()


def fig_precision(out):
    """Accuracy against a 60-step reference, as a function of bisection budget."""
    torch.manual_seed(0)
    budgets = [8, 10, 12, 14, 16, 18, 20, 24, 30]
    scales = [(0.01, "sd = 0.01"), (1.0, "sd = 1"), (100.0, "sd = 100")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    for i, (s, lbl) in enumerate(scales):
        z = torch.randn(512, 64) * s
        ref, _ = evar_from_distribution(z, None, EVaRConfig(alpha=0.1, solver_steps=60))
        errs = []
        for b in budgets:
            e, _ = evar_from_distribution(z, None, EVaRConfig(alpha=0.1, solver_steps=b))
            errs.append(float(((e - ref).abs() / ref.abs().clamp_min(1e-12)).max()))
        ax.semilogy(budgets, errs, color=SERIES[i], lw=2, marker="o", ms=3.5, label=lbl)
    ax.axvline(20, color=INK_SOFT, lw=1, ls="--")
    ax.text(20.4, ax.get_ylim()[1] * 0.3, "shipped budget", color=INK_SOFT, fontsize=8)
    ax.axhline(1.2e-7, color=SERIES[4], lw=1, ls=":")
    ax.text(8.2, 1.5e-7, "float32 epsilon", color=SERIES[4], fontsize=8)
    style(ax, "Bisection budget vs accuracy", "bisection steps",
          "max relative error vs 60-step reference")
    leg = ax.legend(frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK_SOFT)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"solver_precision.{ext}"), facecolor=SURFACE)
    plt.close(fig)


def fig_dual(out):
    """G(x) for distributions of different shape, with the solved optimum marked."""
    torch.manual_seed(0)
    alpha = 0.1
    cases = [
        ("spread (Gaussian, sd 12)", torch.randn(1, 4096) * 12.0),
        ("narrow (Gaussian, sd 0.5)", torch.randn(1, 4096) * 0.5),
        ("90% mass at the maximum",
         torch.where(torch.rand(1, 4096) < 0.9, torch.full((1, 4096), 26.0),
                     torch.randn(1, 4096) * 5 + 10)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    for i, ((name, z), ax) in enumerate(zip(cases, axes)):
        xs = torch.logspace(-3, 3, 600)
        n = z.shape[-1]
        G = []
        for x in xs:
            lm = torch.logsumexp(z / x - np.log(n), dim=-1)
            G.append(float(x * (lm - np.log(alpha))))
        G = np.array(G)
        ev, xstar = evar_from_distribution(z, None, EVaRConfig(alpha=alpha))
        ax.semilogx(xs.numpy(), G, color=SERIES[i], lw=2)
        ax.axvline(float(xstar), color=INK, lw=1.2, ls="--")
        ax.plot([float(xstar)], [float(ev)], marker="o", ms=6, color=INK)
        ax.annotate(f"$x^*$ = {float(xstar):.3g}\nEVaR = {float(ev):.3g}",
                    xy=(float(xstar), float(ev)), xytext=(6, 18),
                    textcoords="offset points", fontsize=8, color=INK)
        style(ax, name, "$x = 1/\\beta$", "$G(x)$" if i == 0 else "")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"dual_objective.{ext}"), facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/figures/report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_scale(args.out)
    print("  solver_scale")
    fig_precision(args.out)
    print("  solver_precision")
    fig_dual(args.out)
    print("  dual_objective")


if __name__ == "__main__":
    main()
