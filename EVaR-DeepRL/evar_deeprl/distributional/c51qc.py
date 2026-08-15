"""C51 categorical action-value critic for *continuous* actions: ``Z(s, a)``.

The discrete version (:mod:`evar_deeprl.distributional.c51q`) emits one
distribution per action, which is only possible when the actions are enumerable.
Here the action is an input alongside the state, so the same object works for a
Gaussian actor.

Why this is needed and not merely tidy: the state-value critic makes the advantage

    A(s,a) = r + gamma * EVaR_alpha( Z(s') ) - EVaR_alpha( Z(s) )

which tilts only the future return. EVaR is translation-equivariant, so putting a
*sampled scalar* reward inside the tilt changes nothing -- the reward enters through
its conditional mean, and at a terminal step alpha cannot enter at all. Measured
exactly on the lottery gridworld, that costs up to 86.35% of the optimum and lands
on the risk-neutral policy at every alpha, while the action-value form is exact
(0.00%, 7 of 7 alphas). See ``analysis/c3_attribution.py``.

``run_invpend.py`` and ``run_safety.py`` are continuous-action, so without this they
carry that defect -- which matters most for Safety-Gymnasium, where the whole point
is that risk lives in the immediate cost of touching a hazard.

The baseline for the advantage is estimated by sampling ``baseline_samples`` actions
from the current policy and averaging their EVaR, since the discrete sum over
actions is unavailable. Any function of ``s`` is a valid baseline, so this changes
variance rather than the expected gradient.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from evar_deeprl.distributional.c51 import C51Critic
from evar_deeprl.risk.evar import EVaRConfig, evar_from_c51


class C51QContinuousCritic(C51Critic):
    """``Z(s, a)`` with the action fed in as input."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 500.0,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ):
        super().__init__(state_dim, n_atoms=n_atoms, v_min=v_min, v_max=v_max,
                         hidden_sizes=hidden_sizes)
        self.action_dim = action_dim
        layers = []
        in_dim = state_dim + action_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_atoms))
        self.net = nn.Sequential(*layers)

    def logits(self, state: torch.Tensor, action: torch.Tensor | None = None):
        if action is None:
            raise ValueError("C51QContinuousCritic needs an action; it models Z(s,a)")
        return self.net(torch.cat([state, action], dim=-1))

    def probs(self, state: torch.Tensor, action: torch.Tensor | None = None):
        return F.softmax(self.logits(state, action), dim=-1)

    def mean_value(self, state: torch.Tensor, action: torch.Tensor | None = None):
        return (self.probs(state, action) * self.support).sum(dim=-1)

    def evar(self, state: torch.Tensor, config: EVaRConfig,
             action: torch.Tensor | None = None):
        """``(evar, x_star)`` for the given state-action pairs."""
        return evar_from_c51(self.support, self.probs(state, action), config)

    def baseline_evar(self, state: torch.Tensor, actor, config: EVaRConfig,
                      samples: int = 4):
        """EVaR averaged over actions drawn from the current policy.

        Stands in for the discrete ``sum_a pi(a|s) EVaR(Z(s,a))``. A baseline may be
        any function of the state, so this affects variance only.
        """
        with torch.no_grad():
            total = None
            for _ in range(samples):
                a, _ = actor.act(state)
                ev, _ = self.evar(state, config, a)
                total = ev if total is None else total + ev
            return total / samples

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        next_actions: torch.Tensor,
        target_net: "C51QContinuousCritic | None" = None,
    ) -> torch.Tensor:
        """Categorical loss bootstrapping from ``a' ~ pi(.|s')``.

        Sampling the next action from the current policy rather than maximising over
        actions keeps the critic an evaluation of the policy the advantage is
        computed for -- the continuous analogue of the discrete expected-SARSA
        bootstrap, and the reason this is not DDPG's max.
        """
        target_net = target_net or self
        with torch.no_grad():
            next_probs = target_net.probs(next_states, next_actions)
            target_probs = self.project_target(rewards, next_probs, dones, gamma)
        log_p = F.log_softmax(self.logits(states, actions), dim=-1)
        return -(target_probs * log_p).sum(dim=-1).mean()
