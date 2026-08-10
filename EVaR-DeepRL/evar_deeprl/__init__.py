"""EVaR-DeepRL: actor-critic risk-seeking RL with distributional critics (C51, IQN).

Replaces the SPSA-based multi-timescale EVaR optimizer in ``EVaR-SA`` with an
end-to-end differentiable actor-critic: a distributional critic learns the return
distribution Z(s), EVaR_alpha[Z(s)] is solved in closed form (a strongly-convex 1-D
program, per Theorem 1 of Ganguly et al. 2025) and used as the critic signal for a
policy-gradient actor.
"""
