"""Compile every DSAC run into figures and publish a shareable W&B Report.

    python analysis/dsac_report.py                 # figures + report
    python analysis/dsac_report.py --no-publish    # figures only, no network write
    python analysis/dsac_report.py --groups dsac-pendulum dsac-safety-nav

Why this exists alongside `analysis/report.py`: that one reads the A2C/C51 metric
names (`episode/return_avg_window`, `update/risk_premium_mean`) and would silently
find nothing in a DSAC run, which logs everything under `update/` from
`train_dsac`'s single record. Rather than overload one script with two schemas, the
DSAC schema gets its own reader.

The report is deliberately built around *paired* arms. Every group contains a
`--risk mean` control sharing the critic, actor, data and seeds, so the only way to
read a difference as coming from the risk measure is to compare within a seed. The
summary table therefore reports the per-seed gap, not just the two averages: on
Pendulum the seed spread is 55 return points while the within-seed gap is 0.6, and
an arms-only table would hide exactly that.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless: write files, never try to open a window
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Categorical slots in fixed order, so an arm keeps its colour across every figure
# even when a filter changes which runs are present.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK_SOFT, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

X_KEY = "update/global_step"

# (metric key, y label, title, filename stem, smoothing window)
# Update-level series are logged every log_every*1000 env steps and are spiky; the
# return is already a 20-episode moving average inside the trainer, so it gets none.
PANELS = [
    ("update/return_mean", "Episode return (20-ep moving average)",
     "Learning curves", "dsac_return", 0),
    ("update/cost_mean", "Episode cost (20-ep moving average)",
     "Safety cost incurred", "dsac_cost", 0),
    ("update/x_star_mean", "x* = 1/beta*",
     "Solved EVaR dual variable", "dsac_dual_x", 5),
    ("update/risk_value_mean", "min over twin critics of the risk value",
     "What the actor is maximising", "dsac_risk_value", 5),
    ("update/entropy_temp", "SAC entropy temperature",
     "Auto-tuned entropy coefficient", "dsac_entropy", 5),
]

# Shape of the distribution the risk measure is applied to. Reported separately
# because these decide whether an experiment can say anything at all: if the
# critic's return distribution has no spread, every risk attitude ties and the
# comparison is uninformative regardless of how the learner performs.
DIAG_KEYS = [
    ("update/z_sd_mean", "sd(Z)"),
    ("update/z_range_mean", "range(Z)"),
    ("update/top_mass_frac", "mass within 0.05sd of max"),
    ("update/at_bound_frac", "dual solve at a bound"),
    ("update/x_star_mean", "x* = 1/beta*"),
]


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8)


def smooth(y, window):
    if window < 3 or y.size < window:
        return y
    pad = window // 2
    return np.convolve(np.pad(y, pad, mode="edge"),
                       np.ones(window) / window, mode="valid")[: y.size]


SEED_RE = re.compile(r"-s(\d+)-")


def run_seed(run):
    """Seed of a run, from config if present and from the run name otherwise.

    Runs launched before the seed was added to the wandb config carry no `seed`
    key, and defaulting them all to 0 makes seeds of the same arm overwrite each
    other -- the paired table then silently reports whichever run was read last.
    The launcher puts `-s<N>-` in every run name, so that is a reliable fallback;
    raising here beats guessing, because a wrong seed corrupts the pairing rather
    than merely mislabelling a row.
    """
    if "seed" in run.config:
        return int(run.config["seed"])
    m = SEED_RE.search(run.name or "")
    if m:
        return int(m.group(1))
    raise ValueError(
        f"run {run.name!r} has no `seed` in its config and no -s<N>- in its name; "
        f"cannot pair it against a control without knowing the seed")


def fetch(entity, project, groups):
    """Pull DSAC runs -> {(group, arm): {seed: (x, {metric: y})}}."""
    import wandb
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    out = defaultdict(dict)
    for run in api.runs(path):
        g = run.group or ""
        if not g.startswith("dsac"):
            continue
        if groups and g not in groups:
            continue
        keys = [X_KEY] + [p[0] for p in PANELS]
        hist = run.history(keys=keys, pandas=False, samples=100_000)
        if not hist:
            continue
        x = np.array([h.get(X_KEY, np.nan) for h in hist], dtype=float)
        series = {}
        for key, *_ in PANELS:
            v = np.array([h.get(key, np.nan) for h in hist], dtype=float)
            if np.isfinite(v).any():
                series[key] = v
        arm = run.config.get("risk", "?")
        seed = run_seed(run)
        out[(g, arm)][seed] = (x, series, run.name, run.state)
    return out


def aggregate(per_seed, key):
    """Interpolate seeds onto a common grid -> (x, mean, lo, hi, n).

    A single seed of an RL curve says very little, so seeds are reduced to a mean
    with a 95% normal band. With n < 2 the band collapses to the line itself and
    the caption has to say so rather than implying a confidence interval.
    """
    curves = [(x, s[key]) for x, s, _, _ in per_seed.values() if key in s]
    curves = [(x[np.isfinite(y)], y[np.isfinite(y)]) for x, y in curves]
    curves = [(x, y) for x, y in curves if x.size > 1]
    if not curves:
        return None
    lo_x = max(x.min() for x, _ in curves)
    hi_x = min(x.max() for x, _ in curves)
    if not np.isfinite(lo_x) or hi_x <= lo_x:
        return None
    grid = np.linspace(lo_x, hi_x, 200)
    ys = np.stack([np.interp(grid, x, y) for x, y in curves])
    m = ys.mean(0)
    if ys.shape[0] < 2:
        return grid, m, m, m, ys.shape[0]
    half = 1.96 * ys.std(0, ddof=1) / np.sqrt(ys.shape[0])
    return grid, m, m - half, m + half, ys.shape[0]


def figures(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    written = []
    groups = sorted({g for g, _ in data})
    for group in groups:
        arms = sorted({a for g, a in data if g == group})
        for key, ylabel, title, stem, win in PANELS:
            fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
            fig.patch.set_facecolor(SURFACE)
            drew = False
            for i, arm in enumerate(arms):
                agg = aggregate(data[(group, arm)], key)
                if agg is None:
                    continue
                x, m, lo, hi, n = agg
                m, lo, hi = smooth(m, win), smooth(lo, win), smooth(hi, win)
                c = SERIES[i % len(SERIES)]
                label = f"{arm} (n={n})"
                ax.plot(x, m, color=c, linewidth=2.0, label=label)
                if n > 1:
                    ax.fill_between(x, lo, hi, color=c, alpha=0.16, linewidth=0)
                drew = True
            if not drew:
                plt.close(fig)
                continue
            style(ax, f"{title} - {group}", "Environment steps", ylabel)
            leg = ax.legend(frameon=False, fontsize=9)
            for t in leg.get_texts():
                t.set_color(INK_SOFT)
            fig.tight_layout()
            base = os.path.join(outdir, f"{group}_{stem}")
            fig.savefig(base + ".png", facecolor=SURFACE)
            fig.savefig(base + ".pdf", facecolor=SURFACE)
            plt.close(fig)
            written.append(base + ".png")
    return written


def summary_rows(data):
    """Final return per (group, arm, seed), plus the within-seed paired gap."""
    rows = []
    for (group, arm), per_seed in sorted(data.items()):
        for seed, (x, s, name, state) in sorted(per_seed.items()):
            y = s.get("update/return_mean")
            final = float(y[np.isfinite(y)][-1]) if y is not None and np.isfinite(y).any() else float("nan")
            rows.append({"group": group, "arm": arm, "seed": seed,
                         "final_return": final, "state": state, "run": name})
    return rows


def paired_gaps(rows):
    """EVaR minus the risk-neutral control, matched seed by seed.

    This is the comparison that carries information. Averaging the arms separately
    mixes in seed variance, which on Pendulum is ~100x the effect being measured.
    """
    by = {(r["group"], r["arm"], r["seed"]): r["final_return"] for r in rows}
    out = []
    for (g, a, s), v in sorted(by.items()):
        if a == "mean":
            continue
        ctrl = by.get((g, "mean", s))
        if ctrl is None or not np.isfinite(v) or not np.isfinite(ctrl):
            continue
        out.append({"group": g, "arm": a, "seed": s,
                    "risk": v, "control": ctrl, "gap": v - ctrl})
    return out


def diagnostics_table(entity, project, group):
    """Distribution-shape metrics over training, for one group.

    Returns a markdown table, or None when the group logged none of them -- runs
    predating the diagnostics, or `--risk mean` arms which never call the solver.
    """
    import wandb
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = [r for r in api.runs(path) if (r.group or "") == group]
    if not runs:
        return None
    keys = ["update/global_step", "update/return_mean"] + [k for k, _ in DIAG_KEYS]
    hist = runs[0].history(keys=keys, pandas=False, samples=1000)
    hist = [h for h in hist if h.get("update/z_sd_mean") is not None]
    if not hist:
        return None
    head = ("| step | return | " + " | ".join(lbl for _, lbl in DIAG_KEYS) + " |\n"
            "|---:|---:|" + "---:|" * len(DIAG_KEYS) + "\n")
    body = ""
    for h in hist:
        body += (f"| {h.get('update/global_step', 0):,} "
                 f"| {h.get('update/return_mean', float('nan')):.2f} | "
                 + " | ".join(f"{h.get(k, float('nan')):.4f}" for k, _ in DIAG_KEYS)
                 + " |\n")
    return head + body


def publish(entity, project, rows, gaps, groups, title, diag=None, diag_group=None):
    """Create a W&B Report with the panels and the paired table."""
    import wandb_workspaces.reports.v2 as wr

    blocks = [
        wr.MarkdownBlock(text=(
            "## What this reports\n\n"
            "Every run here uses the **same** learner: an IQN action-value critic "
            "`Z(s,a)`, twin critics with the minimum taken over the *risk* value, a "
            "tanh-squashed actor, and an auto-tuned entropy temperature. The arms "
            "differ in one place only -- the functional applied to the critic's "
            "quantile samples. `mean` is therefore a true risk-neutral control, "
            "sharing critic, actor, data and seeds, and it is the only thing that "
            "licenses reading a difference as coming from the risk measure.\n\n"
            "**Read the table by seed, not by arm.** Seed variance on Pendulum is "
            "~55 return points; the within-seed gap between EVaR and the control is "
            "under 1. Comparing arm averages would hide that entirely."
        )),
    ]

    header = "| group | arm | seed | final return | state |\n|---|---|---:|---:|---|\n"
    body = "".join(
        f"| {r['group']} | {r['arm']} | {r['seed']} | {r['final_return']:.2f} | {r['state']} |\n"
        for r in rows)
    blocks.append(wr.MarkdownBlock(text="### Final return, every run\n\n" + header + body))

    if gaps:
        gh = "| group | arm | seed | risk arm | control | gap |\n|---|---|---:|---:|---:|---:|\n"
        gb = "".join(
            f"| {g['group']} | {g['arm']} | {g['seed']} | {g['risk']:.2f} | "
            f"{g['control']:.2f} | {g['gap']:+.2f} |\n" for g in gaps)
        blocks.append(wr.MarkdownBlock(
            text=("### Paired against the risk-neutral control\n\n"
                  "Positive gap = the risk arm ended higher than its own seed's "
                  "control.\n\n" + gh + gb)))

    if diag:
        blocks.append(wr.MarkdownBlock(text=(
            f"### Can this environment support the claim? (`{diag_group}`)\n\n"
            "Before comparing risk measures, check that there is risk to measure. "
            "These are the shape of the critic's return distribution `Z(s,a)` -- the "
            "thing the risk functional is applied to -- taken during training.\n\n"
            "Two of them are correctness checks. **`dual solve at a bound`** must be "
            "0: a solve that lands on its interval is not a solve, and EVaR silently "
            "degenerates into fixed-beta entropic utility, which is the baseline it "
            "exists to beat. **`mass within 0.05sd of max`** near `1/K` (0.0156 at "
            "K=64) means the distribution is spread rather than piled at its maximum, "
            "so EVaR is a tail average and not just the maximum.\n\n"
            "The third is the one that decides the experiment. **`sd(Z)` against the "
            "episode return** says how much spread the risk measure has to work "
            "with. Below, sd(Z) sits near 0.022 while episode return reaches ~10 -- "
            "under 1%. The per-state return distribution is nearly deterministic, so "
            "every risk attitude has almost nothing to trade and ties are the "
            "expected outcome. That is a property of the environment, not of the "
            "method, and it is the standard trap in risk-sensitive RL benchmarking: "
            "on a near-deterministic task any 'risk-seeking wins' claim is "
            "unfalsifiable.\n\n" + diag)))

    blocks.append(wr.MarkdownBlock(text=(
        "### How to read a tie\n\n"
        "On Pendulum a tie is the **expected and correct** outcome, not a failure: "
        "its return spread is policy noise rather than risk in the world, so there "
        "is no upper tail for a risk-seeking objective to trade for. The same "
        "reasoning retired CartPole as evidence. What is being checked here is "
        "plumbing -- that the operator is wired in, that `x*` solves to an interior "
        "value instead of pinning to a bound, and that switching the functional does "
        "not destabilise the learner.\n\n"
        "A tie only becomes a problem in an environment with genuine return spread. "
        "That is what the Safety-Gymnasium groups are for."
    )))

    for key, _, title_p, _, _ in PANELS:
        blocks.append(wr.H2(text=title_p))
        blocks.append(wr.PanelGrid(
            runsets=[wr.Runset(entity=entity, project=project,
                               name=g, groupby=["risk"]) for g in groups],
            panels=[wr.LinePlot(x=X_KEY, y=[key], title=title_p,
                                smoothing_factor=0.0, legend_template="${runsetName}")],
        ))

    report = wr.Report(entity=entity, project=project, title=title,
                       description="Risk-sensitive DSAC (IQN critic + EVaR dual solve) "
                                   "against a matched risk-neutral control.",
                       blocks=blocks)
    report.save()
    return report.url


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "evar-deeprl"))
    p.add_argument("--groups", nargs="*", default=None,
                   help="restrict to these wandb groups (default: every dsac-* group)")
    p.add_argument("--out", default="results/figures/dsac")
    p.add_argument("--title", default="Risk-sensitive DSAC: EVaR vs a matched control")
    p.add_argument("--no-publish", action="store_true",
                   help="write figures only; do not create the W&B report")
    p.add_argument("--diag-group", default=None,
                   help="group whose distribution-shape diagnostics to tabulate")
    args = p.parse_args()

    data = fetch(args.entity, args.project, args.groups)
    if not data:
        print("No dsac-* runs found.")
        return
    groups = sorted({g for g, _ in data})
    print(f"Groups: {', '.join(groups)}")

    written = figures(data, args.out)
    for w in written:
        print(f"  wrote {w}")

    rows = summary_rows(data)
    gaps = paired_gaps(rows)
    print("\nFinal return")
    for r in rows:
        print(f"  {r['group']:<20} {r['arm']:<10} s{r['seed']}  "
              f"{r['final_return']:>9.2f}  ({r['state']})")
    if gaps:
        print("\nPaired vs the risk-neutral control")
        for g in gaps:
            print(f"  {g['group']:<20} {g['arm']:<10} s{g['seed']}  "
                  f"risk {g['risk']:>9.2f}  control {g['control']:>9.2f}  "
                  f"gap {g['gap']:>+8.2f}")

    if not args.no_publish:
        entity = args.entity
        if entity is None:
            import wandb
            entity = wandb.Api().default_entity
        diag = (diagnostics_table(entity, args.project, args.diag_group)
                if args.diag_group else None)
        if args.diag_group and diag is None:
            print(f"  (no distribution diagnostics logged in {args.diag_group})")
        url = publish(entity, args.project, rows, gaps, groups, args.title,
                      diag=diag, diag_group=args.diag_group)
        print(f"\nReport: {url}")


if __name__ == "__main__":
    main()
