"""C3: does the per-state EVaR operator point at the trajectory-EVaR optimum?

The deep method tilts *per state* through the critic; the paper's objective tilts
*per trajectory*. The README concedes the difference and the plan calls it the
claim a reviewer will press. Until now it was unmeasured. Here it is computable,
because the lottery gridworld gives exact return distributions from every state.

The experiment removes learning entirely. Give the method a **perfect critic** --
the exact return-to-go distribution under the current policy -- and let the actor
act greedily on the advantage the code actually uses,

    A(s, a) = E[r(s, a)] + EVaR_alpha( Z^pi(s') )  -  EVaR_alpha( Z^pi(s) )

iterating policy improvement to a fixed point. Then compare that fixed point with
the brute-forced trajectory-EVaR optimum.

  * fixed point == optimum      -> the per-state approximation is exact here, and
                                   any regret a trained agent shows is optimiser
                                   error, not the method.
  * fixed point != optimum      -> the approximation itself is biased, the gap is
                                   a property of the method, and this is its size.

Why a gap is plausible: because segments are independent, the trajectory objective
decouples at a *single global* x*, so every segment should be judged at the same x.
The per-state solve gives each state its *own* x*(s), so segments get judged at
different tilts. That is exactly the structural difference between the two, and it
predicts positional errors -- picking the right number of lotteries but the wrong
ones, which is what the trained runs showed at alpha = 0.05 (4 of 5 seeds).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from evar_deeprl.envs.lottery_gridworld import (
    DEFAULT_SEGMENTS, LotteryGridWorld, evar_exact, lane_sequence, optimal_policy,
    return_distribution)

ALPHAS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def table_policy(table):
    def pol(row, col):
        return (1.0, 0.0) if table[(row, col)] == 0 else (0.0, 1.0)
    return pol


def per_state_values(env, table, alpha):
    """EVaR_alpha of the exact return-to-go at every state under this policy."""
    pol = table_policy(table)
    V, X = {}, {}
    for col in range(env.n_cols):
        for row in (0, 1):
            values, probs = return_distribution(env, pol, start=(row, col))
            ev, x = evar_exact(values, probs, alpha)
            V[(row, col)] = ev
            X[(row, col)] = x
    return V, X


def greedy_improve(env, table, alpha):
    """One sweep under the per-state EVaR advantage, as the code computes it.

    ``r + EVaR(Z(s')) - EVaR(Z(s))``. In expectation the immediate reward enters
    through its conditional mean, so this compares ``safe`` against the lottery's
    *mean* -- which is where the risk sensitivity is lost.
    """
    V, _ = per_state_values(env, table, alpha)
    new = {}
    for col in range(env.n_segments):
        for row in (0, 1):
            seg = env.segments[col]
            q_safe = seg.safe + V[(0, col + 1)]
            q_risky = seg.risky_mean + V[(1, col + 1)]
            new[(row, col)] = 0 if q_safe >= q_risky else 1
    return new


def action_value_dist(env, table, row, col, action):
    """Distribution of taking ``action`` at (row, col) then following ``table``.

    This is the object the fix needs: the immediate reward's *randomness* convolved
    with the return-to-go, rather than a sampled scalar added to it. EVaR is
    translation-equivariant, so adding a sampled reward to EVaR(Z(s')) gives back
    the current advantage exactly -- the tilt only reaches the reward when the
    critic represents its distribution.
    """
    pol = table_policy(table)
    tail_v, tail_p = return_distribution(env, pol, start=(action, col + 1))
    seg = env.segments[col]
    if action == 0:
        rv, rp = np.array([seg.safe]), np.array([1.0])
    else:
        rv, rp = np.array([seg.hi, seg.lo]), np.array([seg.p, 1.0 - seg.p])
    vals = (rv[:, None] + tail_v[None, :]).ravel()
    prs = (rp[:, None] * tail_p[None, :]).ravel()
    order = np.argsort(vals)
    return vals[order], prs[order]


def greedy_improve_action_value(env, table, alpha):
    """One sweep under the proposed action-value advantage.

    ``EVaR_alpha( r(s,a) + Z(s') )`` rather than ``r + EVaR_alpha( Z(s') )``.
    """
    new = {}
    for col in range(env.n_segments):
        for row in (0, 1):
            qs = []
            for a in (0, 1):
                v, p = action_value_dist(env, table, row, col, a)
                qs.append(evar_exact(v, p, alpha)[0])
            new[(row, col)] = 0 if qs[0] >= qs[1] else 1
    return new


def fixed_point(env, alpha, start_table, max_sweeps=50, improver=None):
    improver = improver or greedy_improve
    table = dict(start_table)
    seen = [tuple(sorted(table.items()))]
    for _ in range(max_sweeps):
        nxt = improver(env, table, alpha)
        key = tuple(sorted(nxt.items()))
        if key == seen[-1]:
            return nxt, True          # converged
        if key in seen:
            return nxt, False         # cycling between policies
        seen.append(key)
        table = nxt
    return table, False


IMPROVER = greedy_improve


def main() -> None:
    global IMPROVER
    env = LotteryGridWorld(DEFAULT_SEGMENTS)
    if "--action-value" in sys.argv:
        IMPROVER = greedy_improve_action_value
        print("PROPOSED operator:  EVaR( r(s,a) + Z(s') )  -- tilt reaches the reward")
    else:
        print("CURRENT operator:   r + EVaR( Z(s') )       -- tilt misses the reward")
    print("Per-state EVaR operator vs the trajectory-EVaR optimum")
    print("Perfect critic, greedy actor, no learning. Regret is exact.")
    print("=" * 94)
    print("  %5s  %-18s %-18s %-8s %9s %9s %8s"
          % ("alpha", "trajectory optimum", "per-state fixed pt", "match",
             "EVaR opt", "EVaR fp", "regret%"))
    print("-" * 94)
    rows = []
    for alpha in ALPHAS:
        ev_opt, x_opt, table_opt, _, _ = optimal_policy(env, alpha)
        seq_opt = lane_sequence(env, table_opt)

        # Start from both extremes so a fixed point that depends on initialisation
        # is visible rather than hidden by a lucky start.
        results = []
        for init in (0, 1):
            start = {(r, c): init for c in range(env.n_segments) for r in (0, 1)}
            fp, converged = fixed_point(env, alpha, start, improver=IMPROVER)
            v, p = return_distribution(env, table_policy(fp))
            ev_fp, _ = evar_exact(v, p, alpha)
            results.append((lane_sequence(env, fp), ev_fp, converged))
        # Report the better of the two starts: the method's best case.
        seq_fp, ev_fp, converged = max(results, key=lambda r: r[1])
        regret = 100 * (ev_opt - ev_fp) / abs(ev_opt)
        match = "YES" if seq_fp == seq_opt else "no"
        flag = "" if converged else "  (no fixed point; cycles)"
        both = "" if results[0][0] == results[1][0] else "  [init-dependent]"
        print("  %5.2f  %-18s %-18s %-8s %9.3f %9.3f %7.2f%s%s"
              % (alpha, seq_opt, seq_fp, match, ev_opt, ev_fp, regret, flag, both))
        rows.append((alpha, seq_opt, seq_fp, ev_opt, ev_fp, regret))

    print()
    mism = [r for r in rows if r[1] != r[2]]
    worst = max(r[5] for r in rows)
    print(f"  mismatches: {len(mism)}/{len(rows)} alphas   worst exact regret {worst:.2f}%")
    if mism:
        print("  VERDICT: the per-state approximation is biased. With a perfect critic and")
        print("  greedy improvement it does not reach the trajectory optimum, so the gap is")
        print("  a property of the method, not of the optimiser. Sizes above.")
    else:
        print("  VERDICT: the per-state approximation recovers the trajectory optimum at")
        print("  every alpha. Regret seen in trained runs is optimiser error, not C3.")

    print()
    print("Why: the trajectory objective decouples at ONE global x*, but the per-state")
    print("solve uses a different x*(s) at each state. Global x* vs per-state x*(s):")
    print("  %5s %10s   %s" % ("alpha", "global x*", "per-state x*(s) at the decision states"))
    for alpha in (0.05, 0.1, 0.3):
        ev_opt, x_opt, table_opt, _, _ = optimal_policy(env, alpha)
        _, X = per_state_values(env, table_opt, alpha)
        cells = "  ".join(f"col{c}:{X[(0, c)]:7.2f}" for c in range(env.n_segments))
        print("  %5.2f %10.2f   %s" % (alpha, x_opt, cells))
    print()
    print("Segment switching thresholds x_i (lottery beats safe iff x < x_i):")
    for i, seg in enumerate(env.segments):
        xs = np.geomspace(0.05, 500.0, 20000)
        ce = np.array([x * (max(seg.hi, seg.lo) / x + np.log(
            seg.p * np.exp(seg.hi / x - max(seg.hi, seg.lo) / x)
            + (1 - seg.p) * np.exp(seg.lo / x - max(seg.hi, seg.lo) / x))) for x in xs])
        above = ce > seg.safe
        xi = xs[np.argmax(~above)] if not above.all() else float("inf")
        print(f"    segment {i}: x_i = {xi:6.2f}")


if __name__ == "__main__":
    main()
