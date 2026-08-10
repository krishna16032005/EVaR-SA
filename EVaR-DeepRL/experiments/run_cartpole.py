"""Preliminary EVaR actor-critic experiment on CartPole-v1 (discrete actions).

Usage:
    python experiments/run_cartpole.py --critic c51 --episodes 500 --alpha 0.1
    python experiments/run_cartpole.py --critic iqn --episodes 500 --alpha 0.1

    # with Weights & Biases (offline, no account needed -- sync later with `wandb sync`):
    python experiments/run_cartpole.py --critic c51 --wandb-mode offline
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import torch

from evar_deeprl.agents.base import TrainConfig, train
from evar_deeprl.distributional.c51 import C51Critic
from evar_deeprl.distributional.iqn import IQNCritic
from evar_deeprl.logging_utils import add_wandb_args, wandb_config_from_args
from evar_deeprl.policies.categorical import CategoricalPolicy
from evar_deeprl.risk.evar import EVaRConfig
from evar_deeprl.utils import new_run_tag, save_records, state_to_tensor

ENV_ID = "CartPole-v1"


def build_critic(kind: str, state_dim: int, v_min: float, v_max: float):
    if kind == "c51":
        return C51Critic(state_dim, n_atoms=51, v_min=v_min, v_max=v_max)
    if kind == "iqn":
        return IQNCritic(state_dim)
    raise ValueError(f"Unknown critic kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic", choices=["c51", "iqn"], default="c51")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1, help="EVaR confidence level")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default=os.path.join("results", "cartpole"))
    parser.add_argument("--histogram-every", type=int, default=50, help="episodes between critic return-distribution snapshots; 0 disables")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="episodes between actor/critic checkpoints; 0 disables")
    parser.add_argument("--eval-every", type=int, default=20, help="episodes between deterministic evaluation passes; 0 disables")
    parser.add_argument("--eval-episodes", type=int, default=5, help="episodes per evaluation pass (fixed seed set)")
    add_wandb_args(parser)
    args = parser.parse_args()

    env = gym.make(ENV_ID)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # CartPole-v1 gives +1 reward per step for up to 500 steps, so returns lie in
    # [0, 500]. A small negative pad keeps the C51 support away from the boundary.
    critic = build_critic(args.critic, state_dim, v_min=-5.0, v_max=500.0)
    actor = CategoricalPolicy(state_dim, n_actions)

    # Each run gets its own timestamped subdirectory so successive runs never
    # overwrite each other's CSVs/checkpoints.
    run_tag = new_run_tag()
    run_dir = os.path.join(args.results_dir, f"{args.critic}_alpha{args.alpha}_seed{args.seed}_{run_tag}")
    run_name = args.wandb_run_name or f"cartpole-{args.critic}-alpha{args.alpha}-seed{args.seed}-{run_tag}"

    cfg = TrainConfig(
        critic_kind=args.critic,
        gamma=args.gamma,
        n_steps=args.n_steps,
        max_episodes=args.episodes,
        max_steps_per_episode=500,
        evar=EVaRConfig(alpha=args.alpha, x_min=1e-2, x_max=200.0),
        seed=args.seed,
        wandb=wandb_config_from_args(args),
        histogram_every=args.histogram_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )
    cfg.wandb.run_name = run_name
    cfg.wandb.tags = tuple(cfg.wandb.tags) + (ENV_ID, args.critic)

    def action_to_env(action_t: torch.Tensor):
        return int(action_t.item())

    logs = train(
        env, actor, critic, cfg, state_to_tensor, action_to_env,
        run_config_extra={"env_id": ENV_ID},
    )
    save_records(run_dir, f"cartpole_{args.critic}_alpha{args.alpha}", logs)
    print(f"Results dir: {run_dir}")
    if logs["wandb_run_id"]:
        print(f"wandb run id: {logs['wandb_run_id']}")
    env.close()


if __name__ == "__main__":
    main()
