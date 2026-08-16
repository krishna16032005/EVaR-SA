"""Run the vectorized PPO learner on CartPole, MuJoCo, or Safety-Gymnasium.

    # sanity: the learner must clear this before anything harder is meaningful
    python experiments/run_ppo.py --env CartPole-v1 --total-steps 300000 --risk mean

    # the paper's env
    python experiments/run_ppo.py --env InvertedPendulum-v5 --total-steps 1000000

    # safety-gymnasium, in its own interpreter (see DOCKER.md)
    MUJOCO_GL=egl ~/envs/safety/bin/python experiments/run_ppo.py \
        --env SafetyPointGoal1-v0 --total-steps 2000000 --cost-penalty 0.25

`--risk` selects the functional applied to Z(s,a): evar, cvar, wang, entropic,
meanvar, or mean. They all read the same critic at the same point, so sweeping it
varies the risk measure and nothing else -- same nets, same data, same optimiser,
same seeds. `mean` is the risk-neutral floor and `entropic` at fixed beta is the
sharpest rival, since EVaR is entropic risk with beta optimised.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from evar_deeprl.agents.ppo_evar import PPOConfig, train_ppo
from evar_deeprl.distributional.c51q import C51QCritic
from evar_deeprl.distributional.c51qc import C51QContinuousCritic
from evar_deeprl.logging_utils import add_wandb_args, wandb_config_from_args
from evar_deeprl.policies.categorical import CategoricalPolicy
from evar_deeprl.policies.gaussian import GaussianPolicy
from evar_deeprl.risk.evar import EVaRConfig
from evar_deeprl.risk.measures import KINDS, RiskConfig
from evar_deeprl.utils import new_run_tag, resolve_device, save_records


def make_vector_env(env_id: str, n_envs: int, max_steps: int | None):
    """Vector env for whichever package owns this id."""
    if env_id in ("LotteryGridWorld", "LotteryGridWorldDisc"):
        import gymnasium as gym
        from evar_deeprl.envs.lottery_gridworld import (
            DEFAULT_SEGMENTS, DISCRIMINATING_SEGMENTS, LotteryGridWorld)
        segs = (DISCRIMINATING_SEGMENTS if env_id.endswith("Disc")
                else DEFAULT_SEGMENTS)
        return gym.vector.SyncVectorEnv(
            [(lambda i=i: LotteryGridWorld(segs, seed=1000 + i))
             for i in range(n_envs)]), False
    if env_id.startswith("Safety"):
        import safety_gymnasium
        kwargs = {} if max_steps is None else {"max_episode_steps": max_steps}
        return safety_gymnasium.vector.make(env_id, num_envs=n_envs, **kwargs), True
    import gymnasium as gym
    fns = [(lambda i=i: gym.make(env_id)) for i in range(n_envs)]
    return gym.vector.SyncVectorEnv(fns), False


def support_for(env_id: str, gamma: float, cost_penalty: float) -> tuple[float, float]:
    """C51 support, in the units the critic represents: discounted return.

    Sized from the reward scale and the effective horizon 1/(1-gamma), not from the
    undiscounted episode return -- the same units error that once left 9 of 51 atoms
    carrying any mass on CartPole.
    """
    # gamma = 1 is legitimate for short episodic tasks (the gridworld is 3 steps and
    # its objective is the undiscounted trajectory return), so guard the horizon.
    horizon = 1.0 / max(1.0 - gamma, 1e-3)
    if env_id == "LotteryGridWorldDisc":
        return -5.0, 80.0             # max 25+25+18 = 68
    if env_id == "LotteryGridWorld":
        return -5.0, 285.0            # 3 segments, max 45+45+180 = 270
    if env_id.startswith("CartPole"):
        return -0.05 * horizon, 1.05 * horizon
    if env_id.startswith("Safety"):
        # reward ~1 per goal, cost <=1 per step and priced by lambda
        return -(cost_penalty * 1.05 * horizon) - 5.0, 40.0
    if env_id.startswith("Pendulum"):
        return -16.3 * horizon * 1.05, 1.0
    return -0.05 * horizon, 1.05 * horizon        # reward-1-per-step MuJoCo tasks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--risk", choices=list(KINDS), default="evar",
                   help="risk functional applied to Z(s,a). All read the same critic "
                        "at the same point, so a sweep over this varies the measure "
                        "and nothing else")
    p.add_argument("--eta", type=float, default=0.75, help="wang / cpw parameter")
    p.add_argument("--beta", type=float, default=0.05, help="fixed-beta entropic risk")
    p.add_argument("--kappa", type=float, default=1.0, help="mean-variance weight")
    p.add_argument("--cost-penalty", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--n-atoms", type=int, default=101)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--log-every", type=int, default=10,
                   help="updates between console lines")
    p.add_argument("--results-dir", default=os.path.join("results", "ppo"))
    add_wandb_args(p)
    args = p.parse_args()

    device = resolve_device(args.device)
    env, is_safety = make_vector_env(args.env, args.n_envs, args.max_episode_steps)

    obs_dim = int(np.prod(env.single_observation_space.shape))
    discrete = hasattr(env.single_action_space, "n")
    v_min, v_max = support_for(args.env, args.gamma, args.cost_penalty)

    if discrete:
        n_actions = int(env.single_action_space.n)
        actor = CategoricalPolicy(obs_dim, n_actions)
        critic = C51QCritic(obs_dim, n_actions, n_atoms=args.n_atoms,
                            v_min=v_min, v_max=v_max)
    else:
        act_dim = int(np.prod(env.single_action_space.shape))
        bound = float(env.single_action_space.high[0])
        actor = GaussianPolicy(obs_dim, act_dim, action_bound=bound)
        critic = C51QContinuousCritic(obs_dim, act_dim, n_atoms=args.n_atoms,
                                      v_min=v_min, v_max=v_max)

    if args.alpha < 1.0 and args.alpha <= 1.0 / args.n_atoms:
        p.error(f"alpha={args.alpha} <= 1/n_atoms={1/args.n_atoms:.4g}: the per-state "
                f"EVaR would saturate onto the top atom of the support.")

    run_tag = new_run_tag()
    run_dir = os.path.join(args.results_dir,
                           f"{args.env}_{args.risk}_a{args.alpha}_s{args.seed}_{run_tag}")
    wandb_cfg = wandb_config_from_args(args)
    wandb_cfg.run_name = (args.wandb_run_name
                          or f"ppo-{args.env}-{args.risk}-a{args.alpha}-s{args.seed}-{run_tag}")
    wandb_cfg.tags = tuple(wandb_cfg.tags) + (args.env, "ppo", args.risk)

    risk_cfg = RiskConfig(kind=args.risk, alpha=args.alpha, eta=args.eta,
                          beta=args.beta, kappa=args.kappa,
                          evar_cfg=EVaRConfig(alpha=args.alpha, x_min=1e-2,
                                              x_max=4.0 * max(abs(v_min), abs(v_max))))
    cfg = PPOConfig(
        n_envs=args.n_envs, n_steps=args.n_steps, total_steps=args.total_steps,
        gamma=args.gamma, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        epochs=args.epochs, minibatches=args.minibatches, clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        risk=risk_cfg, cost_penalty=args.cost_penalty,
        seed=args.seed, wandb=wandb_cfg, torch_threads=args.torch_threads,
        log_every=args.log_every)

    print(f"[setup] {args.env}  {'discrete' if discrete else 'continuous'}  "
          f"obs {obs_dim}  support [{v_min:.1f}, {v_max:.1f}] x {args.n_atoms} atoms  "
          f"risk={risk_cfg.label()}")

    logs = train_ppo(env, actor, critic, cfg, device,
                     run_config_extra={"env_id": args.env, "device": str(device)})
    save_records(run_dir, f"ppo_{args.env}_{args.risk}", logs)
    print(f"Results dir: {run_dir}")
    if logs["wandb_run_id"]:
        print(f"wandb run id: {logs['wandb_run_id']}")
    env.close()


if __name__ == "__main__":
    main()
