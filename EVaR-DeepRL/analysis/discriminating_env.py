"""Search for a gridworld where different risk measures want *different* policies.

Why this is the blocker
-----------------------
A benchmark only means something if the methods can be told apart. On the current
gridworld at alpha=0.1 every measure reaches ~100% of the achievable EVaR at every
training budget from 20k to 250k steps -- they all find the same optimum, so no
comparison between them is possible. On SafetyPointGoal1 the numbers do differ, but
the learner barely learns there and each run sees ~150 episodes, so those
differences are not trustworthy either. Tables from either one would look
authoritative and mean nothing.

What makes a testbed discriminating is not that the measures produce different
*values* -- they always do, they are different functionals -- but that they select
different *policies*. Then "EVaR found its own optimum and CVaR did not" is a
falsifiable claim about a method rather than an artefact of scoring.

This brute-forces that: for a candidate set of segments, compute each measure's
optimal deterministic policy exactly (the return distribution of any policy is
enumerable here), and keep configurations that induce as many distinct optima as
possible.
"""
from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from evar_deeprl.envs.lottery_gridworld import (
    LotteryGridWorld, Segment, deterministic_policies, lane_sequence,
    return_distribution)
from evar_deeprl.risk.evar import EVaRConfig
from evar_deeprl.risk.measures import RiskConfig, apply_risk


def measure_set(alpha=0.1):
    """The measures to separate, at their natural parameters."""
    return [
        ("mean", RiskConfig("mean")),
        (f"evar{alpha:g}", RiskConfig("evar", alpha=alpha,
                                      evar_cfg=EVaRConfig(alpha=alpha, x_min=1e-3,
                                                          x_max=4000.0))),
        (f"cvar{alpha:g}", RiskConfig("cvar", alpha=alpha)),
        ("wang0.75", RiskConfig("wang", eta=0.75)),
        ("entropic0.05", RiskConfig("entropic", beta=0.05)),
        ("meanvar1", RiskConfig("meanvar", kappa=1.0)),
    ]


def optimal_policy_for(env, risk_cfg):
    """Brute-force the optimal deterministic policy under any risk functional."""
    best = None
    for table in deterministic_policies(env):
        def pol(row, col, _t=table):
            return (1.0, 0.0) if _t[(row, col)] == 0 else (0.0, 1.0)
        v, p = return_distribution(env, pol)
        val = float(apply_risk(torch.as_tensor(v, dtype=torch.float64),
                               torch.as_tensor(p, dtype=torch.float64).unsqueeze(0),
                               risk_cfg).item())
        if best is None or val > best[0]:
            best = (val, table)
    return best


def profile(env, alpha=0.1):
    """(measure -> lane sequence, value) for one environment."""
    out = {}
    for name, cfg in measure_set(alpha):
        val, table = optimal_policy_for(env, cfg)
        out[name] = (lane_sequence(env, table), val)
    return out


def n_distinct(prof) -> int:
    return len({v[0] for v in prof.values()})


def search(max_configs=400, alpha=0.1, seed=0):
    """Look for segment triples that split the measures across distinct optima."""
    rng = np.random.default_rng(seed)
    safes = (10.0,)
    ps = (0.02, 0.05, 0.10, 0.20, 0.35, 0.55)
    his = (18.0, 25.0, 40.0, 70.0, 120.0, 220.0)
    los = (0.0, 3.0, 6.0, 8.0, 9.0)

    pool = []
    for p, hi, lo in itertools.product(ps, his, los):
        mean = p * hi + (1 - p) * lo
        if hi <= 10.0 or mean >= 10.0 or mean < 7.5:
            continue
        pool.append(Segment(safe=10.0, p=p, hi=hi, lo=lo))
    print(f"{len(pool)} candidate segments")

    best = []
    seen = set()
    for _ in range(max_configs):
        idx = tuple(sorted(rng.choice(len(pool), size=3, replace=False)))
        if idx in seen:
            continue
        seen.add(idx)
        segs = tuple(pool[i] for i in idx)
        try:
            env = LotteryGridWorld(segs)
        except ValueError:
            continue
        prof = profile(env, alpha)
        k = n_distinct(prof)
        best.append((k, idx, segs, prof))
    best.sort(key=lambda x: -x[0])
    return best


def main() -> None:
    alpha = float(os.environ.get("ALPHA", "0.1"))
    print(f"Searching for a configuration where the measures disagree (alpha={alpha})\n")
    results = search(max_configs=int(os.environ.get("N", "300")), alpha=alpha)

    print("\nBest configurations by number of distinct optimal policies:")
    for k, idx, segs, prof in results[:4]:
        print(f"\n  {k} distinct optima:")
        for s in segs:
            print(f"    Segment(safe={s.safe}, p={s.p}, hi={s.hi}, lo={s.lo})"
                  f"   mean {s.risky_mean:.2f}")
        for name, (seq, val) in prof.items():
            print(f"      {name:<14} {seq:<20} value {val:9.2f}")

    if results:
        print(f"\n  best achieved: {results[0][0]} distinct optima out of "
              f"{len(measure_set(alpha))} measures")
        if results[0][0] < 3:
            print("  NOTE: fewer than 3 distinct optima means the measures still mostly")
            print("  agree, and a benchmark on this environment cannot separate them.")


if __name__ == "__main__":
    main()
