"""Tanh-squashed Gaussian policy, as SAC uses it.

Why not reuse :class:`~evar_deeprl.policies.gaussian.GaussianPolicy`: that one draws
from a Normal and *clamps* the sample to the action bound, returning the density of
the unclamped point. Any objective that compares densities of the executed action
then evaluates them at a boundary where the density swings violently once the mean
drifts outside the bound. In the PPO learner that produced approx KL of 64 and
1.4e16 and killed runs outright.

Squashing with tanh keeps actions in range without a boundary: the sample is
unbounded, the transform is smooth and invertible, and the log-probability carries
the exact Jacobian correction, so densities stay finite and comparable everywhere.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class SquashedGaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes: tuple[int, ...] = (256, 256),
                 action_bound: float = 1.0):
        super().__init__()
        layers, in_dim = [], state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, state, deterministic=False, with_logp=True):
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw = mean if deterministic else dist.rsample()
        action = torch.tanh(raw)

        logp = None
        if with_logp:
            # log pi(a) = log N(raw) - sum log(1 - tanh(raw)^2), computed in the
            # numerically stable form: log(1 - tanh(u)^2) = 2*(log 2 - u - softplus(-2u))
            logp = dist.log_prob(raw).sum(-1)
            logp = logp - (2 * (math.log(2.0) - raw
                                - torch.nn.functional.softplus(-2.0 * raw))).sum(-1)
        return action * self.action_bound, logp

    def act(self, state, deterministic=False):
        return self.forward(state, deterministic=deterministic)
