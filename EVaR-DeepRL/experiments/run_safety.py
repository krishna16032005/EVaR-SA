"""EVaR actor-critic on Safety-Gymnasium: an environment with genuine risk.

Why this environment exists in the project
------------------------------------------
CartPole cannot test a risk-*seeking* objective, and not merely because it is
noisy. It is deterministic apart from a +/-0.05 initial state, so the return
spread EVaR measures there is the policy's own exploration noise rather than
anything about the world. Its return is capped at 500 and integer valued, which
censors the upper tail the operator is defined on -- once a policy is decent,
more than an alpha-fraction of episodes tie at the cap, p_max >= alpha, and EVaR
saturates onto the maximum by construction. Worst of all, its risk-neutral and
risk-seeking optima are the *same policy*: balancing forever maximises the mean,
the median and every upper-tail statistic at once, so alpha cannot matter and the
experiment is structurally incapable of supporting C1.

Safety-Gymnasium supplies what is missing: stochastic hazard layouts, genuinely
catastrophic events, and -- once cost is priced in below -- policies that trade
mean return against tail shape.

Pricing cost into the reward is what creates the risk
-----------------------------------------------------
Safety-Gymnasium is a *constrained* benchmark: it returns reward and cost
separately, and its usual reading is risk-averse (keep cumulative cost under a
budget). Left as-is there is no risk-return tension for a risk-seeking agent --
hazards cost nothing it can see, so the fastest route is simply best.

``--cost-penalty lambda`` folds them into one scalar,

    r_eff = r - lambda * c

which is what makes the return distribution genuinely bimodal: the short route
runs close to the hazards and pays off well *when it gets away with it*, the
detour is safe and mediocre. Those are two policies with different means and
different upper tails -- exactly the choice alpha is supposed to govern, and
exactly what CartPole cannot offer. lambda is therefore the knob that sets how
much risk there is to be sought; at lambda = 0 the environment degenerates back
into a plain reward-maximisation task with no tail structure worth measuring.

Both raw reward and raw cost are still logged per episode, so the constrained
reading of the benchmark remains available and a risk-seeking policy can be
reported honestly: "higher return, and here is the cost it incurred".

Environment
-----------
safety-gymnasium pins gymnasium==0.28.1 / mujoco==2.3.3, which collide with the
main environment (gymnasium 1.3.0, mujoco 3.11.0 -- run_invpend.py needs
InvertedPendulum-v5, an id 0.28 does not have). It therefore lives in its own
interpreter at ~/envs/safety, and the queue points jobs at it with
``python = "..."``. See DOCKER.md.

Usage:
    ~/envs/safety/bin/python experiments/run_safety.py --env SafetyPointGoal1-v0 \
        --critic c51 --alpha 0.1 --episodes 500
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from evar_deeprl.agents.base import TrainConfig, train
from evar_deeprl.distributional.c51 import C51Critic
from evar_deeprl.distributional.iqn import IQNCritic
from evar_deeprl.logging_utils import add_wandb_args, wandb_config_from_args
from evar_deeprl.policies.gaussian import GaussianPolicy
from evar_deeprl.risk.evar import EVaRConfig
from evar_deeprl.utils import new_run_tag, resolve_device, save_records, state_to_tensor

# Reward and cost live on very different scales here, so the C51 support has to be
# derived from lambda rather than fixed. Measured on SafetyPointGoal1-v0 under a
# random policy, 15 episodes of 1000 steps:
#
#     raw reward return   mean   0.19   sd   1.01   max    3.55
#     raw cost   return   mean 104.27   sd 206.10   max  669.00
#
# Cost is ~100x reward at lambda = 1, which would make the priced return simply
# "negative cost" and the task pure avoidance -- no risk/reward tension, and so
# nothing for alpha to trade off. `cost_rate_max` is the observed worst-case cost
# per step (669/1000 ~ 0.67); `reward_max` is a generous ceiling for a *trained*
# agent (a goal is worth ~1 and a good policy reaches a few tens per episode).
#
# Support that clips the lower tail would quietly bias every risk statistic taken
# from the critic, which is why this is computed rather than guessed -- the first
# smoke test returned -510 against a hand-picked v_min of -80.
ENV_PRESETS = {
    "SafetyPointGoal1-v0": dict(reward_max=40.0, cost_rate_max=0.70, max_steps=1000),
    "SafetyPointGoal2-v0": dict(reward_max=40.0, cost_rate_max=0.90, max_steps=1000),
    "SafetyCarGoal1-v0": dict(reward_max=40.0, cost_rate_max=0.70, max_steps=1000),
    "SafetyPointButton1-v0": dict(reward_max=40.0, cost_rate_max=0.80, max_steps=1000),
}


def support_bounds(preset: dict, cost_penalty: float) -> tuple[float, float]:
    """C51 support wide enough for the worst priced return lambda can produce."""
    v_max = preset["reward_max"]
    v_min = -(cost_penalty * preset["cost_rate_max"] * preset["max_steps"]) - 5.0
    return v_min, v_max


class CostPricedEnv:
    """Adapts safety-gymnasium's 6-tuple step to the 5-tuple the trainer expects.

    ``step`` returns ``(obs, reward, cost, terminated, truncated, info)`` there;
    everything downstream in this repo assumes the Gymnasium 5-tuple. Folding the
    two signals into ``r - lambda * c`` here (rather than inside the trainer)
    keeps the risk pricing visible at the experiment level, where it is a
    modelling choice rather than a detail of the algorithm.
    """

    def __init__(self, env, cost_penalty: float):
        self.env = env
        self.cost_penalty = cost_penalty
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.episode_cost = 0.0
        self.episode_reward_raw = 0.0

    def reset(self, **kwargs):
        self.episode_cost = 0.0
        self.episode_reward_raw = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        self.episode_cost += float(cost)
        self.episode_reward_raw += float(reward)
        info = dict(info)
        # Kept so a risk-seeking win can be reported with the cost it actually
        # incurred, rather than only through the priced-in scalar.
        info["cost"] = float(cost)
        info["episode_cost"] = self.episode_cost
        info["episode_reward_raw"] = self.episode_reward_raw
        return obs, float(reward) - self.cost_penalty * float(cost), terminated, truncated, info

    def close(self):
        self.env.close()


def build_critic(kind: str, state_dim: int, v_min: float, v_max: float, n_atoms: int):
    if kind == "c51":
        return C51Critic(state_dim, n_atoms=n_atoms, v_min=v_min, v_max=v_max)
    if kind == "iqn":
        return IQNCritic(state_dim)
    raise ValueError(f"Unknown critic kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=str, default="SafetyPointGoal1-v0", choices=list(ENV_PRESETS))
    parser.add_argument("--critic", choices=["c51", "iqn"], default="c51")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1, help="EVaR confidence level")
    parser.add_argument("--cost-penalty", type=float, default=0.25,
                        help="lambda in r_eff = r - lambda*c. 0 removes the risk/reward "
                             "tension entirely; 1.0 is also wrong here because raw cost "
                             "outweighs raw reward ~100x, collapsing the task to pure "
                             "avoidance. 0.25 puts the two terms within a factor of a few "
                             "for a trained agent -- but it is a modelling choice that "
                             "should be swept, not trusted")
    parser.add_argument("--n-atoms", type=int, default=101,
                        help="C51 atoms. The priced-return support is wide (it must cover "
                             "lambda * worst-case cost), so 51 atoms would leave the "
                             "resolution too coarse to represent the tail")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default=os.path.join("results", "safety"))
    parser.add_argument("--histogram-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-risk-episodes", type=int, default=500,
                        help="stochastic episodes for the EVaR of the return distribution. "
                             "Must satisfy alpha > 1/n: on an n-sample empirical measure the "
                             "top sample carries mass 1/n, and once that reaches alpha the "
                             "dual optimum runs to x_min and EVaR collapses onto the sample "
                             "maximum. At n=50 an alpha=0.01 run reports the max every time.")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    add_wandb_args(parser)
    args = parser.parse_args()

    if args.eval_risk_episodes and args.alpha < 1.0:
        if args.alpha <= 1.0 / args.eval_risk_episodes:
            parser.error(
                f"alpha={args.alpha} <= 1/eval_risk_episodes=1/{args.eval_risk_episodes}"
                f"={1/args.eval_risk_episodes:.4g}: the EVaR estimate would saturate onto "
                f"the sample maximum on every evaluation. Raise --eval-risk-episodes above "
                f"{int(np.ceil(1/args.alpha))}."
            )

    import safety_gymnasium  # imported here so --help works without the pinned env

    device = resolve_device(args.device)
    preset = ENV_PRESETS[args.env]
    base_env = safety_gymnasium.make(args.env, max_episode_steps=preset["max_steps"])
    env = CostPricedEnv(base_env, cost_penalty=args.cost_penalty)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])

    v_min, v_max = support_bounds(preset, args.cost_penalty)
    critic = build_critic(args.critic, state_dim, v_min=v_min, v_max=v_max, n_atoms=args.n_atoms)
    actor = GaussianPolicy(state_dim, action_dim, action_bound=action_bound)
    print(f"[setup] lambda={args.cost_penalty}  C51 support=[{v_min:.1f}, {v_max:.1f}] "
          f"over {args.n_atoms} atoms  (alpha must exceed 1/n_atoms = "
          f"{1/args.n_atoms:.4f}; alpha={args.alpha})")
    if args.alpha < 1.0 and args.alpha <= 1.0 / args.n_atoms:
        parser.error(
            f"alpha={args.alpha} <= 1/n_atoms={1/args.n_atoms:.4g}: the per-state EVaR "
            f"would saturate onto the top atom of the critic support. Raise --n-atoms "
            f"above {int(np.ceil(1/args.alpha))}."
        )

    x_max = max(abs(v_min), abs(v_max)) + 1.0

    run_tag = new_run_tag()
    run_dir = os.path.join(
        args.results_dir, f"{args.critic}_alpha{args.alpha}_seed{args.seed}_{run_tag}")
    run_name = (args.wandb_run_name
                or f"{args.env}-{args.critic}-alpha{args.alpha}-lam{args.cost_penalty}"
                   f"-seed{args.seed}-{run_tag}")

    cfg = TrainConfig(
        critic_kind=args.critic,
        gamma=args.gamma,
        n_steps=args.n_steps,
        max_episodes=args.episodes,
        max_steps_per_episode=preset["max_steps"],
        evar=EVaRConfig(alpha=args.alpha, x_min=1e-2, x_max=x_max),
        seed=args.seed,
        wandb=wandb_config_from_args(args),
        histogram_every=args.histogram_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_risk_episodes=args.eval_risk_episodes,
        torch_threads=args.torch_threads,
    )
    cfg.wandb.run_name = run_name
    cfg.wandb.tags = tuple(cfg.wandb.tags) + (args.env, args.critic)

    def action_to_env(action_t: torch.Tensor):
        return action_t.detach().cpu().numpy()

    logs = train(
        env, actor, critic, cfg, state_to_tensor, action_to_env,
        device=device,
        run_config_extra={
            "env_id": args.env,
            "device": str(device),
            "cost_penalty": args.cost_penalty,
        },
    )
    save_records(run_dir, f"{args.env}_{args.critic}_alpha{args.alpha}", logs)
    print(f"Results dir: {run_dir}")
    if logs["wandb_run_id"]:
        print(f"wandb run id: {logs['wandb_run_id']}")
    env.close()


if __name__ == "__main__":
    main()
