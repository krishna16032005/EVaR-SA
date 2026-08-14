"""Preliminary EVaR actor-critic experiment on InvertedPendulum-v4 (continuous action).

Requires the MuJoCo Gymnasium extra (``pip install "gymnasium[mujoco]"``). If MuJoCo
is unavailable, pass ``--env Pendulum-v1`` to run the same actor-critic on Gymnasium's
classic-control (no MuJoCo dependency) continuous-control pendulum instead.

Usage:
    python experiments/run_invpend.py --critic c51 --episodes 500 --alpha 0.1
    python experiments/run_invpend.py --critic iqn --episodes 500 --alpha 0.1

    # with Weights & Biases (offline, no account needed -- sync later with `wandb sync`):
    python experiments/run_invpend.py --critic c51 --wandb-mode offline
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
from evar_deeprl.policies.gaussian import GaussianPolicy
from evar_deeprl.risk.evar import EVaRConfig
from evar_deeprl.utils import (discounted_support, new_run_tag, resolve_device,
                               save_records, state_to_tensor)

# Per-step reward bounds and horizon. The C51 support is derived from these with
# the discount applied, because the critic represents the *discounted* return: at
# gamma = 0.99 InvertedPendulum tops out near 100, not the 1000 of its undiscounted
# episode return. The old v_max of 1000 left ~5 of 51 atoms carrying any mass, and
# EVaR is a tail statistic read off exactly that histogram.
ENV_PRESETS = {
    "InvertedPendulum-v4": dict(r_min=0.0, r_max=1.0, max_steps=1000),
    "Pendulum-v1": dict(r_min=-16.27, r_max=0.0, max_steps=200),
}


def build_critic(kind: str, state_dim: int, v_min: float, v_max: float):
    if kind == "c51":
        return C51Critic(state_dim, n_atoms=51, v_min=v_min, v_max=v_max)
    if kind == "iqn":
        return IQNCritic(state_dim)
    raise ValueError(f"Unknown critic kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=str, default="InvertedPendulum-v4", choices=list(ENV_PRESETS))
    parser.add_argument("--critic", choices=["c51", "iqn"], default="c51")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.1, help="EVaR confidence level")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default=os.path.join("results", "invpend"))
    parser.add_argument("--histogram-every", type=int, default=50, help="episodes between critic return-distribution snapshots; 0 disables")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="episodes between actor/critic checkpoints; 0 disables")
    parser.add_argument("--eval-every", type=int, default=20, help="episodes between deterministic evaluation passes; 0 disables")
    parser.add_argument("--eval-episodes", type=int, default=5, help="episodes per evaluation pass (fixed seed set)")
    parser.add_argument("--eval-risk-episodes", type=int, default=30, help="stochastic episodes used to measure EVaR of the return distribution; 0 disables")
    parser.add_argument("--torch-threads", type=int, default=1, help="intra-op threads per run; 1 is fastest when several runs share cores")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="'auto' uses CUDA when available, else CPU")
    add_wandb_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    preset = ENV_PRESETS[args.env]
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])

    v_min, v_max = discounted_support(
        preset["r_min"], preset["r_max"], preset["max_steps"], args.gamma)
    critic = build_critic(args.critic, state_dim, v_min=v_min, v_max=v_max)
    actor = GaussianPolicy(state_dim, action_dim, action_bound=action_bound)

    x_max = max(abs(v_min), abs(v_max)) + 1.0

    # Each run gets its own timestamped subdirectory so successive runs never
    # overwrite each other's CSVs/checkpoints.
    run_tag = new_run_tag()
    run_dir = os.path.join(args.results_dir, f"{args.critic}_alpha{args.alpha}_seed{args.seed}_{run_tag}")
    run_name = args.wandb_run_name or f"{args.env}-{args.critic}-alpha{args.alpha}-seed{args.seed}-{run_tag}"

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
        run_config_extra={"env_id": args.env, "device": str(device)},
    )
    save_records(run_dir, f"{args.env}_{args.critic}_alpha{args.alpha}", logs)
    print(f"Results dir: {run_dir}")
    if logs["wandb_run_id"]:
        print(f"wandb run id: {logs['wandb_run_id']}")
    env.close()


if __name__ == "__main__":
    main()
