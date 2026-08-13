"""Diagonal Gaussian actor policy for continuous-action environments (e.g. InvertedPendulum)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        action_bound: float = 3.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_bound = action_bound

        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

    def distribution(self, state: torch.Tensor) -> Normal:
        features = self.trunk(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp())

    def act(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        raw_action = dist.rsample()
        action = torch.clamp(raw_action, -self.action_bound, self.action_bound)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        return action, log_prob

    def act_with_entropy(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One forward pass yielding action, log-prob and entropy together.

        The rollout needs all three every step; calling ``act`` then ``entropy``
        builds the distribution -- and so runs the network -- twice per env step.
        """
        dist = self.distribution(state)
        raw_action = dist.rsample()
        action = torch.clamp(raw_action, -self.action_bound, self.action_bound)
        return action, dist.log_prob(raw_action).sum(dim=-1), dist.entropy().sum(dim=-1)

    def act_deterministic(self, state: torch.Tensor) -> torch.Tensor:
        """Mean action, used for evaluation -- no exploration noise."""
        return torch.clamp(self.distribution(state).mean, -self.action_bound, self.action_bound)

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).log_prob(action).sum(dim=-1)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).entropy().sum(dim=-1)

    def diagnostics(self, state: torch.Tensor) -> dict[str, float]:
        """Extra scalars logged alongside the standard actor-critic metrics."""
        dist = self.distribution(state)
        saturated = (dist.mean.abs() >= self.action_bound).float().mean()
        return {
            "policy_std_mean": dist.stddev.mean().item(),
            "policy_mean_action_abs_mean": dist.mean.abs().mean().item(),
            "policy_mean_action_saturated_frac": saturated.item(),
        }
