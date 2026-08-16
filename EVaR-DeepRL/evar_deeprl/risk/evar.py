"""Differentiable Entropic Value-at-Risk (EVaR) of a learned return distribution.

The SPSA-based algorithm in ``EVaR-SA`` estimates ``J_EVaR(theta)`` for a whole policy
by rolling out many trajectories and running a two-timescale stochastic-approximation
recursion (see Eqs. 6-11 of Ganguly et al., 2025) purely to *estimate a scalar*, then
perturbs ``theta`` with SPSA because no gradient is available.

Here a distributional critic (:mod:`evar_deeprl.distributional.c51` or
:mod:`evar_deeprl.distributional.iqn`) already gives us an explicit, differentiable
return distribution Z(s) for every visited state (a set of atoms with either fixed
categorical weights or uniform quantile weights). That means EVaR can be evaluated
*per state* directly from Eq. (3)/(6) of the paper,

    EVaR_alpha[Z] = min_{beta>0} (1/beta) * (log E[exp(beta Z)] - log alpha),

via the convex reparameterisation ``x = 1/beta`` used in the paper's Proposition 1 /
Theorem 1 (``G(x)`` is strongly convex on a compact interval), instead of via an outer
SPSA loop. We solve the 1-D convex problem by bisecting its monotone derivative, and
recover gradients into the critic (and through the EVaR-advantage into the actor) via
the envelope theorem rather than by unrolling the solver.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class EVaRConfig:
    """Hyperparameters for the inner convex EVaR solve (mirrors Assumption 1 of the paper)."""

    alpha: float = 0.1
    x_min: float = 1e-2
    x_max: float = 50.0
    # x* = 1/beta* has *units of return*, so a fixed search interval is only valid
    # at one return scale. Measured: x* ~ 0.34 * sd(Z) across four orders of
    # magnitude, so with x_min = 1e-2 the solve starts pinning at sd ~ 0.03 (53% of
    # rows) and is fully pinned at sd = 0.01 (100%), where it returns 0.0278 for a
    # true EVaR of ~0.0198 -- a 40% overestimate. At the bound EVaR degenerates to
    # `mean + x_min * log(1/alpha)`, i.e. fixed-beta entropic utility, which is the
    # baseline the dual solve is supposed to beat.
    #
    # This is not hypothetical: Pendulum returns are O(100) and never exercised it,
    # but SafetyPointGoal1 returns are O(0.1) and sit inside the pinned regime.
    # With the interval derived per row from sd(Z) instead, nothing pins at any
    # scale. `x_min`/`x_max` are kept as absolute fallbacks for degenerate rows and
    # for callers that deliberately fix the interval (the x_smoothing trust region).
    auto_scale_bounds: bool = True
    rel_x_min: float = 1e-3      # multiples of sd(Z)
    rel_x_max: float = 1e3
    # 20 bisections of a 13.8-nat log-range leave ~1e-5 precision in log x, and the
    # resulting EVaR agrees with a 60-step reference to 2.5e-7 relative -- float32
    # machine epsilon -- across return scales 0.1 to 100 and shifts -50 to +500.
    # The previous 30 bought nothing measurable and each step costs ~12 kernel
    # launches, which is the dominant per-update cost once the nets run on a GPU.
    solver_steps: int = 20


# Diagnostics from the most recent solve. A module-level dict rather than a third
# return value so no caller signature changes; the trainer reads it right after the
# call it cares about.
_last_solve_diagnostics: dict[str, float] = {}


def last_solve_diagnostics() -> dict[str, float]:
    """Bound-hitting statistics from the most recent :func:`evar_from_distribution`.

    ``at_bound_frac`` is the one to watch: for ``alpha < 1`` a healthy solve holds it
    at 0. It sat at 1.0 for every update of every run while the Newton solve was
    broken, and again on any environment whose returns are small enough that a fixed
    ``x_min`` exceeds the optimum.
    """
    return dict(_last_solve_diagnostics)


def _log_mgf(atoms: torch.Tensor, log_weights: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return log E[exp(Z / x)] = logsumexp(log_weights + atoms / x) along the last dim.

    ``atoms``: (..., N) return samples/support values.
    ``log_weights``: (..., N) log-probabilities associated with ``atoms`` (broadcastable).
    ``x``: (..., 1) current reparameterised dual variable, x = 1 / beta.
    """
    return torch.logsumexp(log_weights + atoms / x, dim=-1)


def evar_from_distribution(
    atoms: torch.Tensor,
    weights: torch.Tensor | None,
    config: EVaRConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute EVaR_alpha of a batch of empirical return distributions.

    Implements Corollary 1 / Theorem 1: minimise the strongly convex
    ``G(x) = x * (log E[exp(Z/x)] - log alpha)`` over ``x in [x_min, x_max]`` by
    bisecting ``G'`` in log ``x``, then evaluate ``G`` at the solution.

    Args:
        atoms: (batch, N) tensor of return atoms/samples (e.g. C51 support or IQN
            quantile samples).
        weights: (batch, N) tensor of probabilities summing to 1 along dim -1, or
            ``None`` for uniform weights (the IQN / Monte-Carlo case).
        config: :class:`EVaRConfig` with ``alpha`` and the compact search interval.

    Returns:
        A ``(evar, x_star)`` pair of ``(batch,)`` tensors: the EVaR_alpha estimate and
        the solved dual variable ``x* = 1 / beta*`` (useful on its own as a training
        diagnostic -- see Fig. 3 of the paper for how ``x*``/``beta*`` move with alpha).
    """
    if atoms.dim() == 1:
        atoms = atoms.unsqueeze(0)
    batch = atoms.shape[0]
    n = atoms.shape[-1]

    if weights is None:
        log_weights = atoms.new_full(atoms.shape, -math.log(float(n)))
    else:
        # A floor of 1e-12 here is not a harmless guard against log(0): it *invents*
        # mass of 1e-12 on atoms whose true probability is zero, and the tilt
        # exp(Z/x) then multiplies that fake mass by e^(z_max/x). On the C51 support
        # (up to 500) with x ~ 11 that is a factor of e^45, so a floored atom at the
        # top of the support can dominate Q_x outright and pull EVaR up by ~13%.
        # Mask genuine zeros to -inf instead -- logsumexp and softmax both handle it
        # exactly -- and floor only to the dtype's smallest normal, which is ~7x
        # further down in log space than 1e-12.
        tiny = torch.finfo(weights.dtype).tiny
        log_weights = torch.log(weights.clamp_min(tiny))
        log_weights = torch.where(
            weights > 0, log_weights, torch.full_like(log_weights, -float("inf"))
        )

    # alpha = 1 is the risk-neutral control, and there EVaR_1[Z] = E[Z] exactly:
    # the dual optimum sits at x -> infinity, so *any* finite x_max leaves a
    # positive bias of order Var[Z] / (2 x_max). That bias is small (~2%), which is
    # precisely what makes it dangerous -- it is easy to read as sampling noise in
    # the one experiment whose whole job is to detect a broken operator.
    if config.alpha >= 1.0:
        mean = (log_weights.exp() * atoms).sum(dim=-1)
        x_star = atoms.new_full((batch,), config.x_max)
        # No solve happens on this path, so the tripwire must not carry a stale
        # reading from a previous batch into the risk-neutral control's logs.
        _last_solve_diagnostics.update(at_bound_frac=0.0, at_lower_frac=0.0,
                                       at_upper_frac=0.0, saturated_frac=0.0)
        return mean, x_star

    log_alpha = math.log(config.alpha)

    # The saturated regime. G'(0+) = log(p_max / alpha), where p_max is the mass on
    # the largest atom, so as soon as p_max >= alpha the objective is non-decreasing
    # on the whole interval and the infimum sits at x -> 0, where G -> z_max. EVaR
    # then *is* the maximum, exactly -- there is no interior optimum to find.
    #
    # This is not a corner case here. Returns are integer-valued and CartPole caps
    # episodes at 500, so ties at the maximum are routine: a policy that reaches the
    # ceiling on more than an alpha-fraction of evaluation episodes saturates the
    # measure. Bisecting anyway lands on x_min and returns z_max + x_min*log(1/alpha),
    # overshooting the true value and breaking the E[Z] <= EVaR <= max(Z) invariant
    # by ~0.005 -- which is how this regime first showed up, as bounds-check failures
    # on 5-10% of evals. Returning z_max exactly keeps the invariant; x* is reported
    # at x_min so `at_bound` still flags that the estimate has saturated and is no
    # longer a tail *average*.
    weights_lin = log_weights.exp()
    positive = weights_lin > 0
    masked = torch.where(positive, atoms, atoms.new_full((), -float("inf")).expand_as(atoms))
    z_max = masked.max(dim=-1).values
    p_max = (weights_lin * (atoms == z_max.unsqueeze(-1)) * positive).sum(dim=-1)
    saturated = p_max >= config.alpha

    def _g_prime(x: torch.Tensor) -> torch.Tensor:
        """G'(x) = (log E[exp(Z/x)] - log alpha) - E_Qx[Z] / x, with x of shape (batch, 1)."""
        tilt_logits = log_weights + atoms / x
        log_mgf = torch.logsumexp(tilt_logits, dim=-1)
        mean_qx = (torch.softmax(tilt_logits, dim=-1) * atoms).sum(dim=-1)
        return (log_mgf - log_alpha) - mean_qx / x.squeeze(-1)

    # Bisection on G', not Newton on G. G is strongly convex, so G' is monotonically
    # increasing and a sign change brackets the minimiser -- bisection therefore
    # converges unconditionally. Projected Newton does not: when the tilted measure
    # Q_x collapses onto one atom, Var_Qx -> 0 and the curvature G'' = Var_Qx / x^3
    # underflows, so the step G'/G'' explodes and is projected straight onto a bound,
    # from which it bounces to the other bound and cycles forever. The returned x*
    # was then just whichever bound the step-count parity landed on -- constant
    # across genuinely different return distributions, and biased upward by 25-590%.
    # Bisecting in log x also makes the iteration scale-free, so a run whose returns
    # grow by an order of magnitude during training keeps the same precision.
    with torch.no_grad():
        if config.auto_scale_bounds:
            # Per-row interval in the row's own return units. Degenerate rows (sd 0)
            # fall back to the absolute bounds, but they are also exactly the rows
            # the saturated branch above has already claimed, so the value they get
            # from the bisection is discarded either way.
            mean_z = (weights_lin * atoms.nan_to_num(0.0)).sum(dim=-1, keepdim=True)
            var_z = (weights_lin * (atoms - mean_z).nan_to_num(0.0) ** 2).sum(dim=-1)
            sd = var_z.clamp_min(0.0).sqrt()
            usable = sd > 0
            lo = torch.where(usable, (sd * config.rel_x_min).log(),
                             torch.full_like(sd, math.log(config.x_min)))
            hi = torch.where(usable, (sd * config.rel_x_max).log(),
                             torch.full_like(sd, math.log(config.x_max)))
        else:
            lo = atoms.new_full((batch,), math.log(config.x_min))
            hi = atoms.new_full((batch,), math.log(config.x_max))
        # Keep the interval the search *started* from. Bisection drives lo and hi
        # onto x_star, so comparing the solution against the post-loop bounds
        # compares it against itself and reports every solve as bound-hitting --
        # which is exactly what the first version of this diagnostic did.
        lo0, hi0 = lo.clone(), hi.clone()
        for _ in range(config.solver_steps):
            mid = 0.5 * (lo + hi)
            below = _g_prime(mid.exp().unsqueeze(-1)) < 0
            lo = torch.where(below, mid, lo)
            hi = torch.where(below, hi, mid)
        x_star = (0.5 * (lo + hi)).exp()

    # x* is detached deliberately. By the envelope theorem dG/dtheta = partial G /
    # partial theta at the optimum, because partial G / partial x vanishes there, so
    # gradients into the critic are exact without unrolling the solver through
    # autograd -- and unlike an unrolled loop they cannot be corrupted by the
    # iteration's own conditioning.
    x_star = x_star.detach()
    log_mgf_final = _log_mgf(atoms, log_weights, x_star.unsqueeze(-1))
    evar = x_star * (log_mgf_final - log_alpha)

    # Splice in the saturated elements. z_max carries a gradient to the maximising
    # atom, which is the correct derivative there: in this regime EVaR *is* that atom.
    evar = torch.where(saturated, z_max, evar)
    x_star = torch.where(saturated, x_star.new_full((), config.x_min), x_star)

    # Standing tripwire. A solve that lands on either end of its interval is not a
    # solve -- EVaR degenerates to entropic utility at a fixed beta, which is the
    # baseline this is meant to beat, and it does so without raising anything. That
    # has now happened twice for opposite reasons: x_max, when the Newton solve
    # cycled between bounds, and x_min, when a fixed interval met returns of order
    # 0.1. Reporting the fraction is what turns the next occurrence into a number on
    # a dashboard instead of a month of confused results. Saturated rows are excluded
    # because their x_min is a label, not a solution.
    with torch.no_grad():
        # One bisection's worth of slack: after n steps the solution can be no
        # closer to a bound than the final interval width without having converged
        # onto it.
        tol = (hi0 - lo0).abs() / (2 ** config.solver_steps) + 1e-6
        at_lo = (x_star.log() - lo0).abs() < tol
        at_hi = (hi0 - x_star.log()).abs() < tol
        live = ~saturated
        n_live = live.sum().clamp_min(1)
        # Shape of the distribution being tilted. When at_lower_frac is high these
        # say whether the solver or the critic is at fault: the dual optimum runs to
        # x -> 0 exactly when the tilted measure collapses onto the top of the
        # support, so a large `top_mass_frac` means the *critic* has produced a
        # near-degenerate Z and EVaR correctly reports something close to its
        # maximum. A small one with the same pinning would indict the solver.
        # `saturated` only catches exact ties at the maximum; mass merely *near* it
        # drives the same behaviour without tripping that branch.
        sd_z = ((weights_lin * (atoms - (weights_lin * atoms).sum(-1, keepdim=True)) ** 2)
                .sum(-1).clamp_min(0.0).sqrt())
        near_top = (atoms >= (z_max.unsqueeze(-1) - 0.05 * sd_z.unsqueeze(-1)))
        top_mass = (weights_lin * near_top).sum(dim=-1)
        _last_solve_diagnostics.update(
            at_bound_frac=float(((at_lo | at_hi) & live).sum() / n_live),
            at_lower_frac=float((at_lo & live).sum() / n_live),
            at_upper_frac=float((at_hi & live).sum() / n_live),
            saturated_frac=float(saturated.float().mean()),
            z_sd_mean=float(sd_z.mean()),
            z_range_mean=float((z_max - masked.min(dim=-1).values).mean()),
            top_mass_frac=float(top_mass.mean()),
        )
    return evar, x_star


def evar_from_c51(
    support: torch.Tensor, probs: torch.Tensor, config: EVaRConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper for a C51 categorical head.

    Args:
        support: (N,) fixed atom locations shared across the batch.
        probs: (batch, N) categorical probabilities (already softmax-ed).

    Returns: ``(evar, x_star)``, see :func:`evar_from_distribution`.
    """
    atoms = support.unsqueeze(0).expand(probs.shape[0], -1)
    return evar_from_distribution(atoms, probs, config)


def evar_from_iqn(
    quantile_samples: torch.Tensor, config: EVaRConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper for an IQN head.

    Args:
        quantile_samples: (batch, K) sampled quantile values theta_tau(s) for K
            samples tau ~ U(0, 1). Weighted uniformly, i.e. a plain Monte-Carlo
            estimate of the moment-generating function.

    Returns: ``(evar, x_star)``, see :func:`evar_from_distribution`.
    """
    return evar_from_distribution(quantile_samples, None, config)
