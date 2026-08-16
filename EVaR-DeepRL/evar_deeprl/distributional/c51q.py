"""C51 categorical distributional *action*-value critic, for discrete actions.

Why this exists
---------------
:class:`~evar_deeprl.distributional.c51.C51Critic` models the state-value
distribution ``Z(s)``, and the actor's advantage is then

    A(s,a) = r + gamma * EVaR_alpha( Z(s') ) - EVaR_alpha( Z(s) )

which applies the risk tilt only to the *future* return. The immediate reward is a
sampled scalar, and EVaR is translation-equivariant, so ``EVaR(r + Z(s'))`` with a
scalar ``r`` collapses straight back to ``r + EVaR(Z(s'))`` -- adding the reward
inside the tilt changes nothing. In expectation the reward therefore enters through
its conditional *mean*, and at a terminal step, where ``Z(s')`` is degenerate, alpha
cannot enter the comparison at all.

Measured on the lottery gridworld with a perfect critic and greedy improvement,
that costs up to **86.35%** of the optimum and lands on the risk-neutral policy at
every alpha. Modelling ``Z(s,a)`` instead -- so the reward's *randomness* is inside
the distribution being tilted -- recovers the trajectory-EVaR optimum **exactly, at
every alpha, 0.00% regret** on the same test. See ``analysis/c3_attribution.py``,
which runs both operators.

It also fixes the signal-to-noise. On the final segment of that environment the
state-value form asks the actor to detect a differential of 0.72 regardless of
alpha; the action-value form gives 157.28 at alpha=0.05 down to 61.22 at 0.5 --
85x to 218x more, and unlike 0.72 it moves with alpha.

Discrete actions only: it needs one distribution per action. The continuous-action
scripts keep the state-value critic, where the same issue applies and the remedy is
not this one.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from evar_deeprl.distributional.c51 import C51Critic
from evar_deeprl.risk.evar import EVaRConfig, evar_from_c51


class C51QCritic(C51Critic):
    """``Z(s, a)`` over a shared fixed support. Inherits the categorical projection."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 500.0,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ):
        super().__init__(state_dim, n_atoms=n_atoms, v_min=v_min, v_max=v_max,
                         hidden_sizes=hidden_sizes)
        self.n_actions = n_actions
        # Replace the state-value head with one that emits n_actions x n_atoms.
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions * n_atoms))
        self.net = nn.Sequential(*layers)

    # -- distributions -----------------------------------------------------
    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).view(*state.shape[:-1], self.n_actions, self.n_atoms)

    def probs(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.logits(state), dim=-1)          # (..., A, N)

    def mean_value(self, state: torch.Tensor, action_probs: torch.Tensor | None = None):
        """State value under ``action_probs`` (uniform-free: the actor supplies them)."""
        q = (self.probs(state) * self.support).sum(dim=-1)     # (..., A)
        if action_probs is None:
            return q.max(dim=-1).values
        return (q * action_probs).sum(dim=-1)

    def mean_value_taken(self, state: torch.Tensor, actions: torch.Tensor):
        """Mean of Z(s,a) for the action actually taken -- the discrete counterpart
        of the continuous critic's ``mean_value(state, action)``."""
        q = (self.probs(state) * self.support).sum(dim=-1)          # (B, A)
        return q.gather(1, actions.long().view(-1, 1)).squeeze(1)

    def evar_taken(self, state: torch.Tensor, config: EVaRConfig, actions: torch.Tensor):
        """EVaR of Z(s,a) for the action actually taken."""
        ev, x = self.evar_all_actions(state, config)
        idx = actions.long().view(-1, 1)
        return ev.gather(1, idx).squeeze(1), x.gather(1, idx).squeeze(1)

    def evar_all_actions(self, state: torch.Tensor, config: EVaRConfig):
        """``(evar, x_star)`` per action, each ``(batch, A)``."""
        p = self.probs(state)
        b, a, n = p.shape
        ev, x = evar_from_c51(self.support, p.reshape(b * a, n), config)
        return ev.view(b, a), x.view(b, a)

    def evar(self, state: torch.Tensor, config: EVaRConfig,
             action_probs: torch.Tensor | None = None):
        """State-level EVaR: the actor-weighted mixture over per-action EVaRs.

        Used only for logging and for the baseline in the advantage; the actor's
        decisions come from :meth:`evar_all_actions`.
        """
        ev, x = self.evar_all_actions(state, config)
        if action_probs is None:
            idx = ev.argmax(dim=-1, keepdim=True)
            return ev.gather(-1, idx).squeeze(-1), x.gather(-1, idx).squeeze(-1)
        return (ev * action_probs).sum(-1), (x * action_probs).sum(-1)

    # -- learning ----------------------------------------------------------
    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        next_action_probs: torch.Tensor,
        target_net: "C51QCritic | None" = None,
    ) -> torch.Tensor:
        """Expected-SARSA categorical loss.

        The bootstrap mixes the next state's per-action distributions under the
        actor's own probabilities rather than taking a max, because the critic must
        evaluate the *current* policy -- the actor is what improves it, and a max
        here would silently make the critic optimistic relative to the policy the
        advantage is computed for.
        """
        target_net = target_net or self
        with torch.no_grad():
            next_p = target_net.probs(next_states)                       # (B, A, N)
            mixed = (next_p * next_action_probs.unsqueeze(-1)).sum(dim=1)  # (B, N)
            target_probs = self.project_target(rewards, mixed, dones, gamma)

        log_p = F.log_softmax(self.logits(states), dim=-1)               # (B, A, N)
        idx = actions.long().view(-1, 1, 1).expand(-1, 1, self.n_atoms)
        taken = log_p.gather(1, idx).squeeze(1)                          # (B, N)
        return -(target_probs * taken).sum(dim=-1).mean()

    def regression_loss(self, states, actions, targets):
        """Fit Z(s,a) to observed scalar returns -- see the continuous version."""
        t = targets.clamp(self.v_min, self.v_max)
        b = (t - self.v_min) / self.delta_z
        lo = b.floor().long().clamp(0, self.n_atoms - 1)
        hi = b.ceil().long().clamp(0, self.n_atoms - 1)
        eq = lo == hi
        lo = torch.where(eq & (lo > 0), lo - 1, lo)
        hi = torch.where(eq & (hi < self.n_atoms - 1), hi + 1, hi)
        target = torch.zeros(t.shape[0], self.n_atoms, device=t.device)
        target.scatter_add_(1, lo.unsqueeze(1), (hi.float() - b).unsqueeze(1))
        target.scatter_add_(1, hi.unsqueeze(1), (b - lo.float()).unsqueeze(1))
        log_p = F.log_softmax(self.logits(states), dim=-1)
        idx = actions.long().view(-1, 1, 1).expand(-1, 1, self.n_atoms)
        taken = log_p.gather(1, idx).squeeze(1)
        return -(target * taken).sum(-1).mean()
