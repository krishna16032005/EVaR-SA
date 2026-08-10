"""Implicit Quantile Network (IQN) distributional state-value critic.

Dabney et al. (2018). As with :mod:`evar_deeprl.distributional.c51`, this models the
state-value distribution Z(s) (no action argument), trained by the standard pairwise
quantile-regression (Huber) loss, and exposes an ``evar`` method that plugs its sampled
quantiles into the same convex EVaR solver used by the C51 head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from evar_deeprl.risk.evar import EVaRConfig, evar_from_iqn


class IQNCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        embedding_dim: int = 64,
        n_cos: int = 32,
        hidden_sizes: tuple[int, ...] = (128, 128),
        huber_kappa: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_cos = n_cos
        self.huber_kappa = huber_kappa

        state_layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            state_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        state_layers.append(nn.Linear(in_dim, embedding_dim))
        self.state_encoder = nn.Sequential(*state_layers)

        self.cos_embedding = nn.Linear(n_cos, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        i_pi = torch.pi * torch.arange(n_cos, dtype=torch.float32)
        self.register_buffer("i_pi", i_pi)

    def _cos_features(self, taus: torch.Tensor) -> torch.Tensor:
        # taus: (batch, K) -> (batch, K, n_cos)
        angles = taus.unsqueeze(-1) * self.i_pi.view(1, 1, -1)
        return torch.cos(angles)

    def quantiles(self, state: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """Return theta_tau(s) for each (state, tau) pair.

        Args:
            state: (batch, state_dim)
            taus: (batch, K) samples in (0, 1)
        Returns:
            (batch, K) quantile values.
        """
        state_embed = self.state_encoder(state)  # (batch, d)
        cos_features = self._cos_features(taus)  # (batch, K, n_cos)
        tau_embed = torch.relu(self.cos_embedding(cos_features))  # (batch, K, d)
        merged = state_embed.unsqueeze(1) * tau_embed  # (batch, K, d)
        return self.head(merged).squeeze(-1)  # (batch, K)

    def sample_taus(self, batch_size: int, k: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, k, device=device)

    def mean_value(self, state: torch.Tensor, k: int = 32) -> torch.Tensor:
        taus = self.sample_taus(state.shape[0], k, state.device)
        return self.quantiles(state, taus).mean(dim=-1)

    def evar(self, state: torch.Tensor, config: EVaRConfig, k: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(evar, x_star)`` -- see :func:`evar_deeprl.risk.evar.evar_from_iqn`."""
        taus = self.sample_taus(state.shape[0], k, state.device)
        samples = self.quantiles(state, taus)
        return evar_from_iqn(samples, config)

    def loss(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        target_net: "IQNCritic | None" = None,
        n_tau: int = 8,
        n_tau_prime: int = 8,
    ) -> torch.Tensor:
        target_net = target_net or self
        batch = states.shape[0]
        device = states.device

        taus = self.sample_taus(batch, n_tau, device)
        online_quantiles = self.quantiles(states, taus)  # (batch, n_tau)

        with torch.no_grad():
            taus_prime = target_net.sample_taus(batch, n_tau_prime, device)
            next_quantiles = target_net.quantiles(next_states, taus_prime)  # (batch, n_tau')
            targets = rewards.unsqueeze(-1) + gamma * (1.0 - dones.unsqueeze(-1)) * next_quantiles

        # Pairwise TD errors: (batch, n_tau, n_tau')
        td_error = targets.unsqueeze(1) - online_quantiles.unsqueeze(2)
        huber = torch.where(
            td_error.abs() <= self.huber_kappa,
            0.5 * td_error.pow(2),
            self.huber_kappa * (td_error.abs() - 0.5 * self.huber_kappa),
        )
        quantile_weight = (taus.unsqueeze(2) - (td_error.detach() < 0).float()).abs()
        loss = (quantile_weight * huber / self.huber_kappa).sum(dim=1).mean(dim=1)
        return loss.mean()
