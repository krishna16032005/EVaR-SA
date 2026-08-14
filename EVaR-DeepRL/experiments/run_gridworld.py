"""EVaR actor-critic on the lottery gridworld, against analytic ground truth.

This is the only environment in the project where ``J_EVaR(theta)`` is *known*
rather than estimated: the return distribution of any policy is obtained by
enumerating trajectories, so the learned policy can be scored against the exact
EVaR-optimal one. It therefore carries C1 and C3 together, and -- because episodes
are three steps long -- it is also the only place a matched-sample-budget
comparison against the paper's SPSA code is actually runnable.

What each run reports
---------------------
* **C1** -- the learned lane sequence against the brute-forced optimum for that
  alpha, plus exact EVaR of the learned policy against the optimum's. The ground
  truth is a staircase: RISKY-RISKY-RISKY at alpha = 0.01, SAFE-RISKY-RISKY at
  0.05-0.1, SAFE-SAFE-RISKY at 0.2-0.5, SAFE-SAFE-SAFE at 1.0.
* **The rationale for the deep extension** -- ``x*(s) = 1/beta*(s)`` at every
  state. SPSA has a single global beta for the whole trajectory distribution and
  cannot know which decision produced the tail; a distributional critic solves per
  state. The measurable signature is the spread of ``x*`` across states, and on
  CartPole it was 0.045-0.065 (i.e. flat, buying nothing). Here risk is localised
  by construction, so a working method should show it moving.

Note gamma defaults to 1.0. The objective is EVaR of the *undiscounted* trajectory
return, which is what the analytic reference computes; discounting would compare
the learner against a different quantity. Episodes are three steps, so there is
nothing to stabilise with a discount.
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
from evar_deeprl.envs.lottery_gridworld import (
    DEFAULT_SEGMENTS, LotteryGridWorld, evar_exact, lane_sequence, optimal_policy,
    return_distribution)
from evar_deeprl.logging_utils import add_wandb_args, wandb_config_from_args
from evar_deeprl.policies.categorical import CategoricalPolicy
from evar_deeprl.risk.evar import EVaRConfig
from evar_deeprl.utils import new_run_tag, resolve_device, save_records, state_to_tensor


def return_bounds(env: LotteryGridWorld) -> tuple[float, float]:
    """Range of the *return-to-go*, which is what the critic represents.

    Not the range of the full trajectory return: the critic is queried at every
    state, and the remaining return shrinks with depth -- it is exactly 0 at a
    terminal state. Sizing from the full-trajectory range put v_min at 3.3 and so
    could not represent the terminal value at all. The bound has to be the min and
    max over every *suffix*, including the empty one.
    """
    lo = hi = 0.0
    lo_best, hi_best = 0.0, 0.0            # empty suffix: terminal, return-to-go 0
    for seg in reversed(env.segments):
        lo += min(seg.safe, seg.lo)
        hi += max(seg.safe, seg.hi)
        lo_best, hi_best = min(lo_best, lo), max(hi_best, hi)
    return lo_best, hi_best


def learned_policy_fn(actor, env, device):
    """Wrap the actor as ``policy(row, col) -> (p_safe, p_risky)`` for exact scoring."""
    def policy(row, col):
        obs = np.zeros(env.n_states, dtype=np.float32)
        obs[env.index(row, col)] = 1.0
        with torch.no_grad():
            probs = actor.distribution(state_to_tensor(obs).to(device)).probs
        p = probs.squeeze(0).cpu().numpy()
        return float(p[0]), float(p[1])
    return policy


def dual_variable_map(critic, env, cfg, device):
    """``x*(s)`` at every state -- the per-state risk attribution SPSA cannot produce."""
    out = {}
    for col in range(env.n_cols):
        for row in (0, 1):
            if col == 0 and row == 1:
                continue                      # unreachable: the walk starts at (0,0)
            obs = np.zeros(env.n_states, dtype=np.float32)
            obs[env.index(row, col)] = 1.0
            s = state_to_tensor(obs).to(device).unsqueeze(0)
            with torch.no_grad():
                if cfg.critic_kind == "c51":
                    _, x_star = critic.evar(s, cfg.evar)
                else:
                    _, x_star = critic.evar(s, cfg.evar, k=cfg.iqn_k_eval)
            out[(row, col)] = float(x_star.item())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic", choices=["c51", "iqn"], default="c51")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--n-atoms", type=int, default=151,
                        help="returns here are integer valued in [10, 144], so delta_z=1 "
                             "represents the distribution exactly")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=str, default=os.path.join("results", "gridworld"))
    parser.add_argument("--histogram-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-risk-episodes", type=int, default=2000,
                        help="alpha must exceed 1/n; at alpha=0.01 that needs n > 100")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    add_wandb_args(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    env = LotteryGridWorld(DEFAULT_SEGMENTS, seed=args.seed)

    v_lo, v_hi = return_bounds(env)
    pad = 0.05 * (v_hi - v_lo)
    v_min, v_max = v_lo - pad, v_hi + pad
    if args.critic == "c51":
        critic = C51Critic(env.n_states, n_atoms=args.n_atoms, v_min=v_min, v_max=v_max)
    else:
        critic = IQNCritic(env.n_states)
    actor = CategoricalPolicy(env.n_states, env.n_actions)

    if args.alpha < 1.0 and args.alpha <= 1.0 / args.n_atoms:
        parser.error(f"alpha={args.alpha} <= 1/n_atoms={1/args.n_atoms:.4g}: the per-state "
                     f"EVaR would saturate onto the top atom.")

    ev_opt, x_opt, table_opt, _, _ = optimal_policy(env, args.alpha)
    seq_opt = lane_sequence(env, table_opt)
    print(f"[ground truth] alpha={args.alpha}: optimal lanes {seq_opt}, EVaR {ev_opt:.3f}")
    print(f"[setup] C51 support [{v_min:.1f}, {v_max:.1f}] over {args.n_atoms} atoms")

    run_tag = new_run_tag()
    run_dir = os.path.join(args.results_dir,
                           f"{args.critic}_alpha{args.alpha}_seed{args.seed}_{run_tag}")
    run_name = (args.wandb_run_name
                or f"gridworld-{args.critic}-alpha{args.alpha}-seed{args.seed}-{run_tag}")

    cfg = TrainConfig(
        critic_kind=args.critic,
        gamma=args.gamma,
        n_steps=args.n_steps,
        entropy_coef=args.entropy_coef,
        max_episodes=args.episodes,
        max_steps_per_episode=env.n_segments,
        evar=EVaRConfig(alpha=args.alpha, x_min=1e-2, x_max=4.0 * (v_hi - v_lo)),
        seed=args.seed,
        wandb=wandb_config_from_args(args),
        histogram_every=args.histogram_every,
        checkpoint_every=0,
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_risk_episodes=args.eval_risk_episodes,
        torch_threads=args.torch_threads,
    )
    cfg.wandb.run_name = run_name
    cfg.wandb.tags = tuple(cfg.wandb.tags) + ("lottery-gridworld", args.critic)

    logs = train(env, actor, critic, cfg, state_to_tensor, lambda a: int(a.item()),
                 device=device,
                 run_config_extra={"env_id": "LotteryGridWorld", "device": str(device),
                                   "optimal_lanes": seq_opt, "optimal_evar": ev_opt})

    # ---- C1: exact EVaR of the learned policy, no sampling ----
    pol = learned_policy_fn(actor, env, device)
    values, probs = return_distribution(env, pol)
    ev_learned, x_learned = evar_exact(values, probs, args.alpha)
    greedy = {}
    for col in range(env.n_segments):
        for row in (0, 1):
            ps, pr = pol(row, col)
            greedy[(row, col)] = 0 if ps >= pr else 1
    seq_learned = lane_sequence(env, greedy)

    print()
    print("=" * 72)
    print(f"C1  alpha={args.alpha}  seed={args.seed}")
    print(f"  optimal lanes : {seq_opt:<22} EVaR {ev_opt:9.3f}")
    print(f"  learned lanes : {seq_learned:<22} EVaR {ev_learned:9.3f}  "
          f"(exact, under the learned stochastic policy)")
    print(f"  match         : {'YES' if seq_learned == seq_opt else 'NO'}   "
          f"regret {ev_opt - ev_learned:8.3f}  ({100*(ev_opt-ev_learned)/abs(ev_opt):.2f}%)")

    # ---- the deep-extension rationale: does beta* vary across states? ----
    xmap = dual_variable_map(critic, env, cfg, device)
    xs = np.array(list(xmap.values()))
    print()
    print("  x*(s) across states (SPSA has one global beta for all of these):")
    for (row, col), x in sorted(xmap.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lane = "safe " if row == 0 else "risky"
        tag = "terminal" if col == env.n_segments else f"before segment {col}"
        print(f"    ({lane}, col {col})  x* = {x:8.3f}   {tag}")
    spread = float(xs.std() / max(abs(xs.mean()), 1e-9))
    print(f"  x* spread across states = {spread:.3f}   "
          f"(CartPole: 0.045-0.065, i.e. flat and uninformative)")
    print("=" * 72)

    logs["episode_records"] = logs["episode_records"]
    save_records(run_dir, f"gridworld_{args.critic}_alpha{args.alpha}", logs)
    with open(os.path.join(run_dir, "c1_summary.txt"), "w") as fh:
        fh.write(f"alpha\t{args.alpha}\nseed\t{args.seed}\n"
                 f"optimal_lanes\t{seq_opt}\nlearned_lanes\t{seq_learned}\n"
                 f"optimal_evar\t{ev_opt}\nlearned_evar\t{ev_learned}\n"
                 f"match\t{int(seq_learned == seq_opt)}\n"
                 f"x_star_spread\t{spread}\n")
    print(f"Results dir: {run_dir}")
    if logs["wandb_run_id"]:
        print(f"wandb run id: {logs['wandb_run_id']}")


if __name__ == "__main__":
    main()
