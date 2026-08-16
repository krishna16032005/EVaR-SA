"""IQN action-value critic ``Z(s,a)`` for continuous actions, with a robust EVaR solve.

This is the critic for the DSAC-style learner. It carries forward the constraints
this project established the hard way; each one is here because ignoring it produced
a measurable failure.

**Action-value, not state-value.** The advantage must be ``EVaR(Z(s,a))``, not
``r + EVaR(Z(s'))``. EVaR is translation-equivariant, so a *sampled scalar* reward
inside the tilt changes nothing -- the reward enters through its conditional mean and
at a terminal step alpha drops out entirely. Measured exactly on the lottery
gridworld, the state-value form costs up to 86.35% of the optimum while the
action-value form is exact.

**Quantiles, not a fixed categorical support.** C51 needs ``v_min``/``v_max`` chosen in
the units the critic actually represents -- discounted return, not undiscounted
episode return. Getting that wrong left ~9 of 51 atoms carrying any mass on CartPole
and ~5 of 51 on InvertedPendulum. Quantiles have no support to mis-size, and they
place resolution where the distribution is rather than uniformly.

**alpha > 1/K.** On a K-sample empirical measure the top sample carries mass 1/K, and
once that reaches alpha the dual optimum runs to ``x_min`` and EVaR collapses onto the
sample maximum. With quantile samples K is a knob, so this is checked rather than
hoped for.

**Bisection, not Newton.** The projected-Newton solve never converged: when the tilted
measure collapses onto one atom the curvature underflows, the step explodes, and it
cycles between the bounds forever, returning whichever bound the step-count parity
landed on. Bisection on the monotone ``G'`` converges unconditionally.

The *robust* part is new. The dual variable ``x*`` is re-solved from scratch every
update against a critic that is itself moving, so it can jump between updates even
when the underlying distribution barely changes. ``x_smoothing`` keeps an exponential
moving average of ``x*`` per batch and solves within a trust region around it, which
bounds how far the risk tilt can move in one step. Set it to 0 to recover the plain
solve -- it is a knob so its value can be measured, not assumed.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from evar_deeprl.risk.evar import EVaRConfig, evar_from_distribution
from evar_deeprl.risk.measures import RiskConfig, apply_risk


class IQNQCritic(nn.Module):
    """``Z(s,a)`` as implicit quantiles, for continuous actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embedding_dim: int = 128,
        n_cos: int = 64,
        hidden_sizes: tuple[int, ...] = (256, 256),
        huber_kappa: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_cos = n_cos
        self.huber_kappa = huber_kappa

        layers, in_dim = [], state_dim + action_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.sa_encoder = nn.Sequential(*layers)

        self.cos_embedding = nn.Linear(n_cos, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
            nn.Linear(embedding_dim, 1))
        self.register_buffer("i_pi", math.pi * torch.arange(n_cos, dtype=torch.float32))

    # -- quantiles ---------------------------------------------------------
    def sample_taus(self, batch: int, k: int, device) -> torch.Tensor:
        return torch.rand(batch, k, device=device)

    def quantiles(self, state, action, taus):
        """Z(s,a) at quantile levels ``taus`` -> (batch, K)."""
        sa = self.sa_encoder(torch.cat([state, action], dim=-1))      # (B, E)
        angles = taus.unsqueeze(-1) * self.i_pi.view(1, 1, -1)        # (B, K, n_cos)
        phi = torch.relu(self.cos_embedding(torch.cos(angles)))       # (B, K, E)
        return self.head(sa.unsqueeze(1) * phi).squeeze(-1)           # (B, K)

    def sample_z(self, state, action, k=32):
        taus = self.sample_taus(state.shape[0], k, state.device)
        return self.quantiles(state, action, taus)

    def mean_value(self, state, action, k=32):
        return self.sample_z(state, action, k).mean(dim=-1)

    # -- risk --------------------------------------------------------------
    def risk_value(self, state, action, risk_cfg: RiskConfig, k=32,
                   x_prev: torch.Tensor | None = None, x_smoothing: float = 0.0):
        """Risk functional of Z(s,a), plus the solved dual variable.

        Quantile samples are equally weighted, so ``apply_risk`` consumes them
        directly as a support with uniform probabilities.
        """
        z = self.sample_z(state, action, k)                           # (B, K)
        if risk_cfg.kind != "evar":
            probs = torch.full_like(z, 1.0 / z.shape[-1])
            # apply_risk sorts internally; pass per-row support via a loop-free path
            vals = torch.stack([
                apply_risk(z[i], probs[i].unsqueeze(0), risk_cfg).squeeze(0)
                for i in range(z.shape[0])])
            return vals, None

        cfg = risk_cfg.evar_cfg or EVaRConfig(alpha=risk_cfg.alpha)
        if x_smoothing > 0.0 and x_prev is not None:
            # Trust region around the previous solution: the critic moves every
            # update, so an unconstrained re-solve can swing x* even when the
            # distribution barely changed.
            lo = float(max(cfg.x_min, x_prev.mean().item() / (1.0 + x_smoothing)))
            hi = float(min(cfg.x_max, x_prev.mean().item() * (1.0 + x_smoothing)))
            if hi > lo:
                cfg = EVaRConfig(alpha=cfg.alpha, x_min=lo, x_max=hi,
                                 solver_steps=cfg.solver_steps)
        ev, x_star = evar_from_distribution(z, None, cfg)
        return ev, x_star

    # -- learning ----------------------------------------------------------
    def quantile_loss(self, state, action, target_z, k=32):
        """Pairwise quantile Huber regression against target samples.

        ``target_z`` is (B, K') of Bellman targets; the loss is the standard IQN
        asymmetric Huber over every (tau_i, target_j) pair.
        """
        b = state.shape[0]
        taus = self.sample_taus(b, k, state.device)
        z = self.quantiles(state, action, taus)                       # (B, K)
        diff = target_z.unsqueeze(1) - z.unsqueeze(-1)                # (B, K, K')
        kappa = self.huber_kappa
        huber = torch.where(diff.abs() <= kappa,
                            0.5 * diff.pow(2),
                            kappa * (diff.abs() - 0.5 * kappa))
        weight = (taus.unsqueeze(-1) - (diff.detach() < 0).float()).abs()
        return (weight * huber / kappa).sum(dim=-1).mean(dim=-1).mean()


def check_alpha_vs_k(alpha: float, k: int) -> None:
    """``alpha`` must exceed the mass on the top quantile sample, i.e. 1/K."""
    if alpha < 1.0 and alpha <= 1.0 / k:
        raise ValueError(
            f"alpha={alpha} <= 1/K={1.0/k:.4g}: the top of {k} quantile samples "
            f"carries mass 1/K, so the dual optimum runs to x_min and EVaR collapses "
            f"onto the sample maximum. Raise the number of quantile samples above "
            f"{int(math.ceil(1.0/alpha))}.")
