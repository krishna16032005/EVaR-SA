"""Does per-state EVaR still recover the trajectory optimum when segments are dependent?

Why this is the deciding experiment
-----------------------------------
The action-value advantage EVaR( r(s,a) + Z(s') ) recovered the trajectory optimum
exactly on the existing gridworld -- 0.00% regret at every alpha. But that gridworld
has *independent* segments, which makes log E[exp(Z/x)] additive and the objective
decouple at a single global x*. Independence is precisely the condition under which
a per-state proxy is exact, so 0.00% there says nothing about the general case.

The literature says per-state risk is a proxy for trajectory-level risk and can be
arbitrarily suboptimal (Zhou et al. 2023; Wang et al. 2024). If that shows up here
once independence is broken, then trajectory consistency -- not the risk measure --
is the open problem, and state augmentation is the remedy worth building. If the
per-state form stays exact even with dependence, the premise is wrong and this
direction should be dropped before any work goes into it.

Dependence is introduced by making the lottery you face depend on the lane you
arrived in: a "committed" path where taking risky early unlocks a heavier tail
later, which is exactly the structure a trajectory-level measure should exploit and
a per-state proxy should miss.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evar_deeprl.envs.lottery_gridworld import (
    DEFAULT_SEGMENTS, LotteryGridWorld, Segment, evar_exact, lane_sequence,
    optimal_policy, return_distribution)
from analysis.c3_attribution import (fixed_point, greedy_improve,
                                     greedy_improve_action_value, table_policy)

ALPHAS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)

# Path-dependent payoffs. Column 0 has no history yet (both rows share a segment).
# From column 1 on, the lane you arrived in changes the lottery: the risky lane
# unlocks a much heavier tail, the safe lane does not. A trajectory-level measure
# can plan for that; a per-state proxy evaluates each state's own distribution and
# has no mechanism to.
SAFE_BASE = Segment(safe=10.0, p=0.05, hi=25.0, lo=8.0)
# Every lottery must stay worse in mean, or a risk-neutral agent would take it too
# and alpha=1.0 would stop being the all-safe control. Checked, not assumed: an
# earlier draft had (1,2) at mean 10.8 > 10, which silently moved the alpha=1
# optimum off all-safe.
PATH_SEGMENTS = {
    (0, 0): SAFE_BASE,
    (1, 0): SAFE_BASE,
    # arrived safe -> ordinary lotteries
    (0, 1): Segment(safe=10.0, p=0.05, hi=22.0, lo=9.0),
    (0, 2): Segment(safe=10.0, p=0.05, hi=22.0, lo=9.0),
    # arrived risky -> the committed path, with a far heavier tail
    (1, 1): Segment(safe=10.0, p=0.10, hi=60.0, lo=4.0),
    (1, 2): Segment(safe=10.0, p=0.10, hi=90.0, lo=0.5),
}

for _k, _s in PATH_SEGMENTS.items():
    assert _s.risky_mean < _s.safe, (
        f"path segment {_k} has risky mean {_s.risky_mean:.2f} >= safe {_s.safe}")


def run(env, label):
    print(f"\n{label}")
    print("  %5s  %-18s %-18s %-18s %8s %8s"
          % ("alpha", "trajectory optimum", "state-value fp", "action-value fp",
             "V regret", "Q regret"))
    print("  " + "-" * 88)
    worst_v = worst_q = 0.0
    for alpha in ALPHAS:
        ev_opt, _, table_opt, _, _ = optimal_policy(env, alpha)
        seq_opt = lane_sequence(env, table_opt)
        out = {}
        for tag, improver in (("V", greedy_improve), ("Q", greedy_improve_action_value)):
            best = None
            for init in (0, 1):
                start = {(r, c): init for c in range(env.n_segments) for r in (0, 1)}
                fp, _ = fixed_point(env, alpha, start, improver=improver)
                v, p = return_distribution(env, table_policy(fp))
                ev, _ = evar_exact(v, p, alpha)
                if best is None or ev > best[0]:
                    best = (ev, lane_sequence(env, fp))
            out[tag] = best
        rv = 100 * (ev_opt - out["V"][0]) / abs(ev_opt)
        rq = 100 * (ev_opt - out["Q"][0]) / abs(ev_opt)
        worst_v, worst_q = max(worst_v, rv), max(worst_q, rq)
        print("  %5.2f  %-18s %-18s %-18s %7.2f%% %7.2f%%"
              % (alpha, seq_opt, out["V"][1], out["Q"][1], rv, rq))
    print(f"  worst regret: state-value {worst_v:.2f}%   action-value {worst_q:.2f}%")
    return worst_q


def main() -> None:
    indep = LotteryGridWorld(DEFAULT_SEGMENTS)
    q_indep = run(indep, "INDEPENDENT segments (the existing gridworld)")

    dep = LotteryGridWorld(DEFAULT_SEGMENTS, path_segments=PATH_SEGMENTS)
    q_dep = run(dep, "PATH-DEPENDENT segments (independence broken)")

    print()
    print("=" * 90)
    if q_dep > 1.0:
        print(f"  The per-state action-value form is NOT exact once segments depend on")
        print(f"  the path: worst regret {q_dep:.2f}% against {q_indep:.2f}% with")
        print(f"  independence. Trajectory consistency is a real gap here, and state")
        print(f"  augmentation is the remedy worth building.")
    else:
        print(f"  The per-state action-value form stays exact ({q_dep:.2f}%) even with")
        print(f"  path dependence. The premise for a trajectory-consistency contribution")
        print(f"  does not hold on this construction -- do not build on it.")


if __name__ == "__main__":
    main()
