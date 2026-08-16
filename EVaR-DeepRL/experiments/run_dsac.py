"""Run the risk-sensitive DSAC learner.

    # sanity: continuous control the previous learners struggled with
    python experiments/run_dsac.py --env Pendulum-v1 --total-steps 60000 --risk mean

    # safety-gymnasium, in its own interpreter (see DOCKER.md)
    MUJOCO_GL=egl ~/envs/safety/bin/python experiments/run_dsac.py \
        --env SafetyPointGoal1-v0 --risk evar --alpha 0.1 --cost-penalty 0.25

`--risk mean` is the risk-neutral control: same critic, same actor, same data, same
seeds, with the distribution's mean in place of EVaR. It is the only way to
attribute a difference to the risk measure rather than to the algorithm.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from evar_deeprl.agents.dsac_evar import DSACConfig, train_dsac
from evar_deeprl.logging_utils import add_wandb_args, wandb_config_from_args
from evar_deeprl.risk.evar import EVaRConfig
from evar_deeprl.risk.measures import KINDS, RiskConfig
from evar_deeprl.utils import new_run_tag, resolve_device, save_records


def make_env(env_id: str, max_steps: int | None, seed: int):
    if env_id.startswith("Safety"):
        import safety_gymnasium
        kw = {} if max_steps is None else {"max_episode_steps": max_steps}
        return safety_gymnasium.make(env_id, **kw)
    import gymnasium as gym
    return gym.make(env_id)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--start-steps", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--risk", choices=list(KINDS), default="evar")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--eta", type=float, default=0.75)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--x-smoothing", type=float, default=0.0,
                   help="trust region on the dual variable between updates; 0 = off")
    p.add_argument("--n-quantiles-risk", type=int, default=64,
                   help="quantile samples the risk value is computed from; "
                        "alpha must exceed 1/K")
    p.add_argument("--cost-penalty", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--log-every", type=int, default=20, help="thousands of env steps")
    p.add_argument("--results-dir", default=os.path.join("results", "dsac"))
    add_wandb_args(p)
    args = p.parse_args()

    device = resolve_device(args.device)
    env = make_env(args.env, args.max_episode_steps, args.seed)

    risk = RiskConfig(kind=args.risk, alpha=args.alpha, eta=args.eta,
                      beta=args.beta, kappa=args.kappa,
                      evar_cfg=EVaRConfig(alpha=args.alpha, x_min=1e-2, x_max=1e4))

    run_tag = new_run_tag()
    run_dir = os.path.join(args.results_dir,
                           f"{args.env}_{risk.label()}_s{args.seed}_{run_tag}")
    wcfg = wandb_config_from_args(args)
    wcfg.run_name = (args.wandb_run_name
                     or f"dsac-{args.env}-{risk.label()}-s{args.seed}-{run_tag}")
    wcfg.tags = tuple(wcfg.tags) + (args.env, "dsac", args.risk)

    cfg = DSACConfig(total_steps=args.total_steps, start_steps=args.start_steps,
                     batch_size=args.batch_size, gamma=args.gamma, risk=risk,
                     x_smoothing=args.x_smoothing,
                     n_quantiles_risk=args.n_quantiles_risk,
                     cost_penalty=args.cost_penalty, seed=args.seed,
                     log_every=args.log_every, wandb=wcfg,
                     torch_threads=args.torch_threads)

    print(f"[setup] {args.env}  obs {env.observation_space.shape}  "
          f"act {env.action_space.shape}  risk={risk.label()}  "
          f"K_risk={args.n_quantiles_risk}  x_smoothing={args.x_smoothing}")

    logs = train_dsac(env, cfg, device,
                      run_config_extra={"env_id": args.env, "device": str(device)})
    save_records(run_dir, f"dsac_{args.env}_{risk.label()}", logs)
    print(f"Results dir: {run_dir}")
    if logs["wandb_run_id"]:
        print(f"wandb run id: {logs['wandb_run_id']}")
    env.close()


if __name__ == "__main__":
    main()
