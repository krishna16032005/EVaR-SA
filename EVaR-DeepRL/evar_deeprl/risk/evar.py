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
    # 20 bisections of a 13.8-nat log-range leave ~1e-5 precision in log x, and the
    # resulting EVaR agrees with a 60-step reference to 2.5e-7 relative -- float32
    # machine epsilon -- across return scales 0.1 to 100 and shifts -50 to +500.
    # The previous 30 bought nothing measurable and each step costs ~12 kernel
    # launches, which is the dominant per-update cost once the nets run on a GPU.
    solver_steps: int = 20


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
        lo = atoms.new_full((batch,), math.log(config.x_min))
        hi = atoms.new_full((batch,), math.log(config.x_max))
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
