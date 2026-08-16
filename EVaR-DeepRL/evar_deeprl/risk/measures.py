"""Risk measures over a categorical return distribution, for like-for-like comparison.

The question this exists to answer is where EVaR stands against the risk measures
the distributional-RL literature actually uses. That comparison is only clean if
everything else is held fixed, so all of these consume the same object -- the
critic's ``Z(s,a)`` as atoms and probabilities -- and are dropped into the same PPO
learner at the same point. What varies between runs is the functional and nothing
else: same nets, same data, same optimiser, same seeds.

Implemented, with the literature they come from:

* ``mean``      -- risk-neutral floor.
* ``evar``      -- this project's measure. EVaR_alpha is the *beta-optimised*
                   entropic risk, which is exactly why ``entropic`` below is its
                   most informative rival.
* ``cvar``      -- upper-tail CVaR at the same alpha. The standard risk-sensitive
                   comparison, and the one reviewers reach for first.
* ``wang``      -- Wang's distortion, g(u) = Phi(Phi^-1(u) + eta). A distortion
                   risk measure from Dabney et al.'s IQN, risk-seeking for eta > 0.
* ``cpw``       -- cumulative probability weighting (Tversky & Kahneman), the other
                   distortion IQN reports. **Not an upper-tail measure**: it is
                   inverse-S shaped and overweights *both* tails, because it models
                   human probability perception rather than a risk attitude. It can
                   therefore fall *below* the mean (measured: 40.53 against a mean
                   of 42.78 on a bimodal return) and is not monotone in eta
                   (13.47, 16.19, 16.27, 5.25 as eta goes 1.0 -> 0.3). Kept because
                   IQN reports it, excluded from RISK_SEEKING_KINDS, and not to be
                   read as "more risk-seeking than X" in a comparison table.
* ``entropic``  -- entropic risk at *fixed* beta. EVaR optimises beta rather than
                   fixing it, so this isolates what the dual solve buys and is the
                   sharpest test of whether it earns its cost.
* ``meanvar``   -- mean + kappa * std, the classical Markowitz-style objective DSAC
                   also reports.

Upper-tail conventions throughout: this is risk-*seeking*, so the measures reward
the good tail. A lower-tail CVaR would score the method backwards.

All of these are evaluated under ``no_grad`` -- the actor consumes them as scalar
advantages and the critic is trained by regression onto observed returns -- so none
of them needs to be differentiable in the probabilities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from evar_deeprl.risk.evar import EVaRConfig, evar_from_c51


@dataclass
class RiskConfig:
    """Which measure, and its one parameter."""

    kind: str = "evar"          # mean | evar | cvar | wang | cpw | entropic | meanvar
    alpha: float = 0.1          # evar, cvar
    eta: float = 0.75           # wang, cpw
    beta: float = 0.1           # entropic (fixed)
    kappa: float = 1.0          # meanvar
    evar_cfg: EVaRConfig = None

    def label(self) -> str:
        if self.kind in ("evar", "cvar"):
            return f"{self.kind}{self.alpha:g}"
        if self.kind in ("wang", "cpw"):
            return f"{self.kind}{self.eta:g}"
        if self.kind == "entropic":
            return f"entropic{self.beta:g}"
        if self.kind == "meanvar":
            return f"meanvar{self.kappa:g}"
        return self.kind


def _mean(support, probs):
    return (probs * support).sum(-1)


def _cvar_upper(support, probs, alpha):
    """Mean of the best ``alpha`` fraction of the mass.

    Walks down from the largest atom accumulating mass until ``alpha`` is reached,
    splitting the final atom so the result is exact rather than quantised to atom
    boundaries.
    """
    order = torch.argsort(support, descending=True)
    z, p = support[order], probs[..., order]
    cum = p.cumsum(-1)
    take = torch.clamp(alpha - (cum - p), min=0.0)
    take = torch.minimum(take, p)
    return (take * z).sum(-1) / alpha


def _distorted(support, probs, g):
    """Distorted expectation with distortion ``g`` on the *survival* function.

    ``sum_i z_i * [ g(S_{i-1}) - g(S_i) ]`` where ``S_i`` is the probability of
    exceeding atom ``i``. A ``g`` that is concave near 0 overweights the good tail,
    which is the risk-seeking direction.
    """
    order = torch.argsort(support, descending=True)
    z, p = support[order], probs[..., order]
    surv = p.cumsum(-1)                       # P(Z >= z_i), atoms descending
    prev = torch.cat([torch.zeros_like(surv[..., :1]), surv[..., :-1]], dim=-1)
    w = g(surv) - g(prev)
    return (w * z).sum(-1)


def _wang(support, probs, eta):
    normal = torch.distributions.Normal(0.0, 1.0)

    def g(u):
        u = u.clamp(1e-6, 1 - 1e-6)
        return normal.cdf(normal.icdf(u) + eta)

    return _distorted(support, probs, g)


def _cpw(support, probs, eta):
    def g(u):
        u = u.clamp(1e-6, 1 - 1e-6)
        return u ** eta / ((u ** eta + (1 - u) ** eta) ** (1.0 / eta))

    return _distorted(support, probs, g)


def _entropic(support, probs, beta):
    """(1/beta) log E[exp(beta Z)] -- risk-seeking for beta > 0.

    EVaR is the infimum of this over beta (after the reparameterisation x = 1/beta),
    so holding beta fixed is precisely the ablation of the dual solve.
    """
    m = (beta * support).max()
    return (m + torch.log((probs * torch.exp(beta * support - m)).sum(-1))) / beta


def _meanvar(support, probs, kappa):
    mu = _mean(support, probs)
    var = (probs * support.pow(2)).sum(-1) - mu.pow(2)
    return mu + kappa * var.clamp_min(0).sqrt()


def apply_risk(support: torch.Tensor, probs: torch.Tensor, cfg: RiskConfig):
    """Scalar risk value per row of ``probs`` (shape ``(..., n_atoms)``)."""
    kind = cfg.kind
    if kind == "mean":
        return _mean(support, probs)
    if kind == "evar":
        ev, _ = evar_from_c51(support, probs, cfg.evar_cfg or EVaRConfig(alpha=cfg.alpha))
        return ev
    if kind == "cvar":
        return _cvar_upper(support, probs, cfg.alpha)
    if kind == "wang":
        return _wang(support, probs, cfg.eta)
    if kind == "cpw":
        return _cpw(support, probs, cfg.eta)
    if kind == "entropic":
        return _entropic(support, probs, cfg.beta)
    if kind == "meanvar":
        return _meanvar(support, probs, cfg.kappa)
    raise ValueError(f"unknown risk measure {kind!r}")


KINDS = ("mean", "evar", "cvar", "wang", "cpw", "entropic", "meanvar")

# The set that is actually comparable as risk-*seeking* objectives: each is >= the
# mean and increases as its parameter is pushed toward the upper tail. `cpw` is
# deliberately absent -- see its note above.
RISK_SEEKING_KINDS = ("mean", "evar", "cvar", "wang", "entropic", "meanvar")
