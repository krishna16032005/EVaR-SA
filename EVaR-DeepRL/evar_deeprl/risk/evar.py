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
SPSA loop. We solve the 1-D convex problem with a handful of projected Newton steps,
unrolled so that gradients can flow back into both the critic and (through the
resulting EVaR-advantage) the actor.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EVaRConfig:
    """Hyperparameters for the inner convex EVaR solve (mirrors Assumption 1 of the paper)."""

    alpha: float = 0.1
    x_min: float = 1e-2
    x_max: float = 50.0
    newton_steps: int = 15
    init_x: float = 1.0


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
    ``G(x) = x * (log E[exp(Z/x)] - log alpha)`` over ``x in [x_min, x_max]`` with
    projected Newton iterations, then evaluate ``G`` at the solution.

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
        log_weights = atoms.new_full(atoms.shape, -float(torch.log(torch.tensor(float(n)))))
    else:
        log_weights = torch.log(weights.clamp_min(1e-12))

    log_alpha = float(torch.log(torch.tensor(config.alpha)))

    x = atoms.new_full((batch, 1), config.init_x)
    x = x.clamp(config.x_min, config.x_max)

    for _ in range(config.newton_steps):
        log_mgf = _log_mgf(atoms, log_weights, x)  # (batch,)
        # tilted measure Q_x, used for both the first- and second-order terms.
        tilt_logits = log_weights + atoms / x
        tilt = torch.softmax(tilt_logits, dim=-1)  # probabilities under Q_x
        mean_qx = (tilt * atoms).sum(dim=-1)
        var_qx = (tilt * atoms.pow(2)).sum(dim=-1) - mean_qx.pow(2)
        var_qx = var_qx.clamp_min(1e-8)

        g_prime = (log_mgf - log_alpha) - mean_qx / x.squeeze(-1)
        g_double_prime = var_qx / x.squeeze(-1).pow(3)
        g_double_prime = g_double_prime.clamp_min(1e-8)

        step = g_prime / g_double_prime
        x = (x.squeeze(-1) - step).clamp(config.x_min, config.x_max).unsqueeze(-1)

    log_mgf_final = _log_mgf(atoms, log_weights, x)
    x_star = x.squeeze(-1)
    evar = x_star * (log_mgf_final - log_alpha)
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
