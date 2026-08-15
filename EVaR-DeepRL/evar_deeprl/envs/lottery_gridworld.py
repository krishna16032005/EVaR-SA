"""A gridworld built so that risk-seeking and risk-neutral optima are different.

Why a new environment rather than the existing one
--------------------------------------------------
``gridworld/gridworld_reinforce_evar.py`` is a deterministic maze (-1 per step,
-2 for an obstacle, 0 at the goal). With no aleatoric noise the return
distribution is induced entirely by policy stochasticity, so the shortest path
maximises the mean, the median and every upper-tail statistic at once -- the same
structural defect that makes CartPole unable to support C1. Reusing it would
repeat that mistake.

The design
----------
Two lanes running left to right across ``n_segments`` columns:

    row 0 (SAFE)    deterministic reward per segment
    row 1 (RISKY)   a lottery: ``hi`` with probability ``p``, else ``lo``

From ``(row, col)`` an action picks the lane of the *next* segment, pays that
segment's reward, and advances to ``(action, col + 1)``. Column ``n_segments`` is
terminal. So the grid is genuinely a grid -- beta*(s) can be drawn as a heatmap
over it -- while remaining small enough to enumerate exactly.

Each segment is sized so the lottery's **mean is below the safe reward** but its
**upper tail is far above it**. A risk-neutral agent must therefore take every
safe lane, and a sufficiently risk-seeking one must take the lotteries. That the
two optima differ is not a matter of tuning: it is verified by enumeration in
:func:`optimal_policy`.

Why this is the environment where the deep extension can be argued
------------------------------------------------------------------
SPSA optimises a single trajectory-level scalar and has one global ``beta*`` for
the whole return distribution; it cannot know *which* decision produced the tail.
A distributional critic solves for ``beta*(s)`` at every state. The measurable
signature of that difference is the spread of ``x* = 1/beta*`` across states --
``evar_dual_x_std / evar_dual_x_mean``, already logged. On CartPole it sits at
0.045-0.065, i.e. beta* is essentially constant and the per-state machinery buys
nothing. Here risk is *localised*: states in front of a big lottery face a wide
return distribution, states past the last fork face none at all, so beta* has to
move. Staggering the lotteries by depth is what forces that.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class Segment:
    """One column: a safe reward and a lottery, with the lottery mean kept below."""

    safe: float
    p: float      # probability of the high payout on the risky lane
    hi: float
    lo: float

    @property
    def risky_mean(self) -> float:
        return self.p * self.hi + (1.0 - self.p) * self.lo

    def validate(self, index: int) -> None:
        if not 0.0 < self.p < 1.0:
            raise ValueError(f"segment {index}: p must be in (0,1), got {self.p}")
        if self.risky_mean >= self.safe:
            raise ValueError(
                f"segment {index}: risky mean {self.risky_mean:.3f} >= safe {self.safe}. "
                "The lottery must be worse in expectation, or a risk-neutral agent would "
                "take it too and the environment could not separate the optima."
            )
        if self.hi <= self.safe:
            raise ValueError(
                f"segment {index}: hi {self.hi} <= safe {self.safe}; the lottery has no "
                "upper tail worth seeking."
            )


# The lotteries are staged so the optimal *number* of them falls step by step as
# alpha rises, rather than flipping all at once -- a graded response is much
# stronger evidence for C1 than a single switch.
#
# The lever follows from the segments being independent, which makes
# ``log E[exp(Z/x)]`` additive, so at a common ``x`` the objective decouples:
#
#     G(x) = sum_i  x * log E[exp(Z_i / x)]  -  x * log alpha
#
# Each segment is therefore chosen on its own, taking the lottery iff
#
#     x * log( p*exp(hi/x) + (1-p)*exp(lo/x) )  >  safe
#
# whose crossing point ``x_i`` depends only on that segment. Below ``x_i`` the
# lottery wins, above it the safe lane does; ``x*`` grows with alpha (12.7 at
# alpha = 0.01 up to 58.8 at 0.5 for the segments below), so segments with separated
# ``x_i`` drop out one at a time and the ground truth is a staircase:
#
#     alpha       0.01  0.05  0.1  0.2  0.3  0.5  1.0
#     lotteries      3     2    2    2    2    1    0
#
# Two earlier hand-picked designs came out optimal-all-risky for every alpha from
# 0.01 to 0.5, flipping only at alpha = 1.0: their lotteries were nearly free, so no
# risk-seeking alpha ever declined one. Matching ``alpha ~ p`` does not control this
# either, because EVaR of the *summed* return is not separable that way.
#
# The lotteries are also chosen so that going risky genuinely pays. They have to be
# worse in mean -- otherwise a risk-neutral agent takes them too and the optima never
# separate -- but only slightly. Taking all three gives up **6%** of the mean return
# (28.10 against 30) and puts the best case at **270, nine times** what the safe route
# can ever pay. An earlier set had a long shot costing 80% of its segment's expected
# reward: it separated the optima, but made risk-seeking look pathological rather than
# like a bet worth taking.
#
# Those two goals pull against each other -- gentler lotteries stay worth taking at
# every alpha, which flattens the staircase back into a single flip -- so the segments
# are not hand-tuned. They come from a search over the actual objective (>= 3 distinct
# risky-lane counts, every lottery keeping >= 85% of the safe mean, upside maximised),
# using the decoupling above to evaluate each candidate triple in one vectorised pass.
#
# The staggering also varies risk by depth, which forces beta*(s) to move across
# states -- the signature that distinguishes this method from SPSA's single global
# beta, and the thing CartPole could not exhibit (x*_std/x*_mean ~ 0.05 there).
DEFAULT_SEGMENTS = (
    Segment(safe=10.0, p=0.02, hi=45.0, lo=8.5),     # mean 9.23,  8% below safe, pays 4.5x
    Segment(safe=10.0, p=0.03, hi=45.0, lo=8.5),     # mean 9.59,  4% below safe, pays 4.5x
    Segment(safe=10.0, p=0.03, hi=180.0, lo=4.0),    # mean 9.28,  7% below safe, pays 18x
)


class LotteryGridWorld:
    """Gymnasium-style 5-tuple env over a 2 x (n_segments+1) grid."""

    def __init__(self, segments=DEFAULT_SEGMENTS, seed: int | None = None):
        self.segments = tuple(segments)
        for i, s in enumerate(self.segments):
            s.validate(i)
        self.n_segments = len(self.segments)
        self.n_cols = self.n_segments + 1
        self.n_states = 2 * self.n_cols
        self.n_actions = 2                       # 0 = safe lane, 1 = risky lane
        self._rng = np.random.default_rng(seed)
        self.row = 0
        self.col = 0

    # -- state helpers -----------------------------------------------------
    def index(self, row: int, col: int) -> int:
        return row * self.n_cols + col

    def decode(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.n_cols)

    def _obs(self) -> np.ndarray:
        v = np.zeros(self.n_states, dtype=np.float32)
        v[self.index(self.row, self.col)] = 1.0
        return v

    # -- gym API -----------------------------------------------------------
    def reset(self, seed: int | None = None, **kwargs):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.row, self.col = 0, 0
        return self._obs(), {}

    def step(self, action: int):
        if self.col >= self.n_segments:
            raise RuntimeError("step() called on a terminal state")
        seg = self.segments[self.col]
        action = int(action)
        if action == 0:
            reward = seg.safe
        else:
            reward = seg.hi if self._rng.random() < seg.p else seg.lo
        self.row, self.col = action, self.col + 1
        terminated = self.col >= self.n_segments
        return self._obs(), float(reward), terminated, False, {}

    def close(self) -> None:
        pass


# --------------------------------------------------------------- exact math --
def return_distribution(env: LotteryGridWorld, policy, start=(0, 0)
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Exact distribution of the return-to-go from ``start`` under ``policy``.

    ``policy(row, col) -> (p_safe, p_risky)``. Enumerates every action sequence and
    every lottery outcome, so this is the true ``J_EVaR`` integrand -- no sampling,
    no critic. That is the whole point of this environment: C1 and C3 can be checked
    against a number that is *known*, not estimated.

    With the default ``start`` this is the trajectory return. From an interior state
    it is what the critic is supposed to represent, which is what makes the per-state
    versus per-trajectory comparison (C3) computable here.
    """
    acc: dict[float, float] = {}

    def walk(row: int, col: int, total: float, prob: float) -> None:
        if prob == 0.0:
            return
        if col >= env.n_segments:
            acc[total] = acc.get(total, 0.0) + prob
            return
        seg = env.segments[col]
        p_safe, p_risky = policy(row, col)
        if p_safe > 0:
            walk(0, col + 1, total + seg.safe, prob * p_safe)
        if p_risky > 0:
            walk(1, col + 1, total + seg.hi, prob * p_risky * seg.p)
            walk(1, col + 1, total + seg.lo, prob * p_risky * (1.0 - seg.p))

    walk(start[0], start[1], 0.0, 1.0)
    values = np.array(sorted(acc), dtype=np.float64)
    probs = np.array([acc[v] for v in values], dtype=np.float64)
    return values, probs / probs.sum()


def render(env: LotteryGridWorld, policy=None, x_star=None) -> str:
    """ASCII view of the grid with the route the policy walks drawn through it.

    The in-terminal counterpart of the path figure in
    ``analysis/gridworld_report.py``, so the environment stays inspectable from a
    REPL or mid-training without opening a file. ``o`` starts the route, ``--->``
    reaches the goal, and a lane change shows as the route moving between rows.
    """
    W = 17
    lanes = None
    if policy is not None:
        lanes, row = [], 0
        for c in range(env.n_segments):
            ps, pr = policy(row, c)
            a = 0 if ps >= pr else 1
            lanes.append(a)
            row = a

    out = ["  col   " + "".join(f"{c:^{W}d}" for c in range(env.n_segments))]
    for row, name in ((0, "SAFE "), (1, "RISKY")):
        cells = []
        for c, seg in enumerate(env.segments):
            body = (f"[{seg.safe:.0f}]" if row == 0
                    else f"[{seg.hi:.0f} p{seg.p:g}]")
            on = lanes is not None and lanes[c] == row
            fill = "-" if (on and c > 0 and lanes[c - 1] == row) else " "
            cell = body.center(W, fill)
            if on and c == 0:
                cell = "o" + body.center(W - 1)
            cells.append(cell)
        tail = "---> GOAL" if (lanes is not None and lanes[-1] == row) else ""
        out.append(f"  {name} " + "".join(cells) + tail)
    if x_star is not None:
        cells = []
        for c in range(env.n_segments):
            r = lanes[c] if lanes else 0
            cells.append(f"x*={x_star.get((r, c), float('nan')):6.2f}".center(W))
        out.append("  x*(s) " + "".join(cells))
    out.append("  RISKY pays hi with probability p, else lo"
               + ("" if lanes is None else "; the drawn route is this policy's"))
    return "\n".join(out)


def evar_exact(values: np.ndarray, probs: np.ndarray, alpha: float,
               grid: int = 200_000) -> tuple[float, float]:
    """EVaR_alpha of an explicit discrete distribution, by direct minimisation.

    Deliberately independent of ``evar_deeprl.risk.evar``: this is the reference
    the solver is checked *against*, so sharing code would make the check circular.
    """
    if alpha >= 1.0:
        return float((values * probs).sum()), float("inf")
    p_max = float(probs[values == values.max()].sum())
    if p_max >= alpha:                      # dual optimum at x -> 0; EVaR is the max
        return float(values.max()), 0.0
    xs = np.geomspace(1e-4, 1e5, grid)[:, None]
    t = values[None, :] / xs
    m = t.max(axis=1, keepdims=True)
    log_mgf = (m + np.log((probs[None, :] * np.exp(t - m)).sum(axis=1, keepdims=True))).ravel()
    g = xs.ravel() * (log_mgf - np.log(alpha))
    i = int(np.argmin(g))
    return float(g[i]), float(xs.ravel()[i])


def deterministic_policies(env: LotteryGridWorld):
    """Every deterministic policy over the reachable decision states."""
    decisions = [(r, c) for c in range(env.n_segments) for r in (0, 1)]
    for combo in product((0, 1), repeat=len(decisions)):
        table = dict(zip(decisions, combo))
        yield table


def optimal_policy(env: LotteryGridWorld, alpha: float):
    """Brute-force the EVaR-optimal deterministic policy. Ground truth for C1."""
    best = None
    for table in deterministic_policies(env):
        def pol(row, col, _t=table):
            return (1.0, 0.0) if _t[(row, col)] == 0 else (0.0, 1.0)
        values, probs = return_distribution(env, pol)
        ev, x_star = evar_exact(values, probs, alpha)
        if best is None or ev > best[0]:
            best = (ev, x_star, table, values, probs)
    return best


def lane_sequence(env: LotteryGridWorld, table: dict) -> str:
    """The lanes a deterministic policy actually visits, e.g. 'RISKY-SAFE-RISKY'."""
    row, out = 0, []
    for col in range(env.n_segments):
        a = table[(row, col)]
        out.append("RISKY" if a else "SAFE")
        row = a
    return "-".join(out)
