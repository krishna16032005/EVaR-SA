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


def risk_from_samples(z: torch.Tensor, cfg: RiskConfig) -> torch.Tensor:
    """Risk functional of equally-weighted quantile samples, batched over rows.

    ``measures.apply_risk`` takes a single shared support, which is right for a
    categorical critic but wrong here: every row of ``z`` is its own support. Looping
    it per row cost the learner two orders of magnitude -- the first DSAC smoke test
    ran at 11 steps/s, essentially all of it in that Python loop.
    """
    B, K = z.shape
    kind = cfg.kind
    if kind == "mean":
        return z.mean(-1)
    if kind == "meanvar":
        return z.mean(-1) + cfg.kappa * z.std(-1)
    if kind == "entropic":
        b = cfg.beta
        m = (b * z).max(dim=-1, keepdim=True).values
        return (m.squeeze(-1) + torch.log(torch.exp(b * z - m).mean(-1))) / b

    zs, _ = torch.sort(z, dim=-1, descending=True)                # best first
    if kind == "cvar":
        n = max(1, int(math.ceil(cfg.alpha * K)))
        return zs[:, :n].mean(-1)

    # distortion measures act on the survival function of the sorted samples
    surv = torch.arange(1, K + 1, device=z.device, dtype=z.dtype).expand(B, K) / K
    prev = surv - 1.0 / K
    if kind == "wang":
        nrm = torch.distributions.Normal(0.0, 1.0)
        g = lambda u: nrm.cdf(nrm.icdf(u.clamp(1e-6, 1 - 1e-6)) + cfg.eta)
    elif kind == "cpw":
        e = cfg.eta
        g = lambda u: (u.clamp(1e-6, 1 - 1e-6) ** e
                       / ((u.clamp(1e-6, 1 - 1e-6) ** e
                           + (1 - u.clamp(1e-6, 1 - 1e-6)) ** e) ** (1.0 / e)))
    else:
        raise ValueError(f"unknown risk measure {kind!r}")
    return ((g(surv) - g(prev)) * zs).sum(-1)


def risk_value_from_z(z: torch.Tensor, risk_cfg: RiskConfig,
                      x_prev: torch.Tensor | None = None,
                      x_smoothing: float = 0.0):
    """Risk functional of quantile samples ``z`` (B, K), plus the solved dual variable.

    Split out of :meth:`IQNQCritic.risk_value` so the twin critics can be solved in
    *one* call on a stacked ``(2B, K)`` batch. Rows are independent in the solver, so
    stacking is exactly equivalent -- and it halves the bisection's kernel launches,
    which is where the GPU time goes: on Pendulum the EVaR arm ran at 49 steps/s
    against the risk-neutral control's 105, while on CPU (where the nets dominate
    instead) the same solve costs only 12%.
    """
    if risk_cfg.kind != "evar":
        return risk_from_samples(z, risk_cfg), None

    cfg = risk_cfg.evar_cfg or EVaRConfig(alpha=risk_cfg.alpha)
    if x_smoothing > 0.0 and x_prev is not None:
        # Trust region around the previous solution: the critic moves every update,
        # so an unconstrained re-solve can swing x* even when the distribution
        # barely changed.
        lo = float(max(cfg.x_min, x_prev.mean().item() / (1.0 + x_smoothing)))
        hi = float(min(cfg.x_max, x_prev.mean().item() * (1.0 + x_smoothing)))
        if hi > lo:
            cfg = EVaRConfig(alpha=cfg.alpha, x_min=lo, x_max=hi,
                             solver_steps=cfg.solver_steps)
    return evar_from_distribution(z, None, cfg)


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
        return risk_value_from_z(z, risk_cfg, x_prev, x_smoothing)

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
