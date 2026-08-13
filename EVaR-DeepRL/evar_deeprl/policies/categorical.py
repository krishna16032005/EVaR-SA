"""Categorical actor policy for discrete-action environments (e.g. CartPole)."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class CategoricalPolicy(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def distribution(self, state: torch.Tensor) -> Categorical:
        logits = self.net(state)
        return Categorical(logits=logits)

    def act(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        action = dist.sample()
        return action, dist.log_prob(action)

    def act_with_entropy(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One forward pass yielding action, log-prob and entropy together.

        The rollout needs all three every step; calling ``act`` then ``entropy``
        builds the distribution -- and so runs the network -- twice per env step.
        """
        dist = self.distribution(state)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def act_deterministic(self, state: torch.Tensor) -> torch.Tensor:
        """Greedy (argmax) action, used for evaluation -- no exploration noise."""
        return self.net(state).argmax(dim=-1)

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).log_prob(action)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).entropy()

    def diagnostics(self, state: torch.Tensor) -> dict[str, float]:
        """Extra scalars logged alongside the standard actor-critic metrics."""
        probs = self.distribution(state).probs
        return {"policy_max_action_prob_mean": probs.max(dim=-1).values.mean().item()}
