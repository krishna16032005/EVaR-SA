"""C51 categorical distributional state-value critic.

Standard C51 (Bellemare, Dabney & Munos, 2017) is normally paired with a discrete-action
Q-function and greedy action selection. Here the critic instead models the
state-*value* distribution Z(s) (no action argument) because the policy comes from a
separate actor network (categorical for CartPole, Gaussian for InvertedPendulum) -- so
the same critic works for both discrete and continuous action spaces. The Bellman
target is the usual one-step categorical projection, just applied to V instead of Q.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from evar_deeprl.risk.evar import EVaRConfig, evar_from_c51


class C51Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 500.0,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        support = torch.linspace(v_min, v_max, n_atoms)
        self.register_buffer("support", support)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_atoms))
        self.net = nn.Sequential(*layers)

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def probs(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(state), dim=-1)

    def mean_value(self, state: torch.Tensor) -> torch.Tensor:
        p = self.probs(state)
        return (p * self.support).sum(dim=-1)

    def evar(self, state: torch.Tensor, config: EVaRConfig) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(evar, x_star)`` -- see :func:`evar_deeprl.risk.evar.evar_from_c51`."""
        p = self.probs(state)
        return evar_from_c51(self.support, p, config)

    def project_target(
        self,
        rewards: torch.Tensor,
        next_probs: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Categorical projection Phi(T Z(s')) onto the fixed support (Algorithm 1 of
        Bellemare et al., 2017), applied to the one-step value target
        ``r + gamma * (1 - done) * support``.
        """
        batch = rewards.shape[0]
        device = rewards.device
        support = self.support

        tz = rewards.unsqueeze(-1) + gamma * (1.0 - dones.unsqueeze(-1)) * support.unsqueeze(0)
        tz = tz.clamp(self.v_min, self.v_max)
        b = (tz - self.v_min) / self.delta_z
        lower = b.floor().long()
        upper = b.ceil().long()

        # Fix the edge case where b is exactly an integer (lower == upper), which
        # would otherwise drop all mass for that atom.
        lower_eq_upper = lower == upper
        lower = torch.where((lower_eq_upper) & (lower > 0), lower - 1, lower)
        upper = torch.where((lower_eq_upper) & (upper < self.n_atoms - 1), upper + 1, upper)

        target_probs = torch.zeros(batch, self.n_atoms, device=device)
        offset = (
            torch.linspace(0, (batch - 1) * self.n_atoms, batch, device=device)
            .long()
            .unsqueeze(1)
            .expand(batch, self.n_atoms)
        )
        target_probs.view(-1).index_add_(
            0, (lower + offset).view(-1), (next_probs * (upper.float() - b)).view(-1)
        )
        target_probs.view(-1).index_add_(
            0, (upper + offset).view(-1), (next_probs * (b - lower.float())).view(-1)
        )
        return target_probs

    def loss(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        target_net: "C51Critic | None" = None,
    ) -> torch.Tensor:
        target_net = target_net or self
        with torch.no_grad():
            next_probs = target_net.probs(next_states)
            target_probs = self.project_target(rewards, next_probs, dones, gamma)

        log_probs = F.log_softmax(self.logits(states), dim=-1)
        loss = -(target_probs * log_probs).sum(dim=-1)
        return loss.mean()
