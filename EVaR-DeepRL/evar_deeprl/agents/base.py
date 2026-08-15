"""EVaR actor-critic training loop.

Replaces the outer SPSA loop of ``EVaR-SA`` (finite-difference perturbation of the
whole policy, Eq. 12-14 of Ganguly et al. 2025) with a standard on-policy n-step
actor-critic where the critic is *distributional* (C51 or IQN) and the advantage used
by the actor is built from EVaR_alpha of the critic's learned return distribution
instead of its mean:

    A_t = r_t + gamma * EVaR_alpha[Z(s_{t+1})] - EVaR_alpha[Z(s_t)]

which is the direct entropic-risk analogue of the classical TD advantage
``r_t + gamma * V(s_{t+1}) - V(s_t)``. Because both the critic (C51 categorical head /
IQN quantile head) and the EVaR solve (:mod:`evar_deeprl.risk.evar`) are ordinary
differentiable torch modules, the actor is optimised by plain backprop -- no
finite-difference perturbations or multi-timescale stochastic-approximation
bookkeeping is needed.

Everything the loop computes (losses, gradient norms, advantage/EVaR/value
statistics, policy diagnostics, timing, and periodic critic-distribution snapshots)
is routed through :class:`evar_deeprl.logging_utils.RunLogger` so it lands in both the
local CSV records and (optionally) Weights & Biases -- see ``README.md`` for the full
list of tracked quantities.

Note the ``episode_return`` logged every episode is a *training-time* number: it comes
from the still-exploring, still-updating policy (``actor.act`` samples from the
distribution, and ``flush_update`` fires mid-episode every ``n_steps`` env steps), so
it is a noisy on-policy signal, not a clean measurement of "how good is the policy
right now." :func:`evaluate_policy` (run every ``cfg.eval_every`` episodes, logged
separately as ``eval_records``/``eval/*``) is the actual answer to that question:
frozen weights, deterministic actions, no gradient updates, fixed eval seeds.
"""
from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn

from evar_deeprl.logging_utils import RunLogger, WandbConfig
from evar_deeprl.risk.evar import EVaRConfig, evar_from_distribution


def make_target(net: nn.Module) -> nn.Module:
    target = copy.deepcopy(net)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1.0 - tau).add_(tau * sp)


@dataclass
class TrainConfig:
    critic_kind: Literal["c51", "iqn", "c51q"] = "c51"
    gamma: float = 0.99
    n_steps: int = 16
    max_episodes: int = 500
    max_steps_per_episode: int = 500
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 1e-3
    target_tau: float = 0.01
    iqn_k_eval: int = 32
    grad_clip: float = 5.0
    # Per-batch advantage normalisation. On by default (standard A2C), but
    # switchable: see the note at its use site -- it is a candidate source of
    # risk-seeking behaviour independent of the EVaR operator.
    normalize_advantage: bool = True
    evar: EVaRConfig = field(default_factory=EVaRConfig)
    log_every: int = 10
    seed: int = 0
    # Diagnostics logging.
    wandb: WandbConfig = field(default_factory=WandbConfig)
    histogram_every: int = 50  # episodes between critic-return-distribution snapshots; 0 disables.
    histogram_samples: int = 256  # IQN Monte-Carlo samples used for the histogram snapshot.
    checkpoint_every: int = 0  # episodes between actor/critic checkpoints; 0 disables.
    checkpoint_dir: str = "checkpoints"
    eval_every: int = 20  # episodes between deterministic evaluation passes; 0 disables.
    eval_episodes: int = 5  # episodes per evaluation pass, run on a fixed seed set.
    # These nets are tiny, so intra-op parallelism costs more in thread sync than
    # it saves -- and when a sweep runs N of these at once on the same cores, each
    # process spawning one thread per core makes them fight. One thread per run,
    # many runs in parallel, is strictly faster here. 0 leaves torch's default.
    torch_threads: int = 1
    # Episodes for the stochastic pass that measures the objective itself. Needs
    # to be large enough for a tail estimate: EVaR at alpha=0.1 over 5 episodes is
    # decided by a single sample.
    eval_risk_episodes: int = 30


@dataclass
class RolloutBuffer:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)   # needed by the action-value critic
    log_probs: list = field(default_factory=list)
    entropies: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    next_states: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.entropies.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)


def _critic_evar(critic: nn.Module, states: torch.Tensor, cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(evar, x_star)`` -- see the critics' ``evar()`` methods."""
    if cfg.critic_kind == "c51":
        return critic.evar(states, cfg.evar)
    return critic.evar(states, cfg.evar, k=cfg.iqn_k_eval)


def _critic_mean_value(critic: nn.Module, states: torch.Tensor, cfg: TrainConfig) -> torch.Tensor:
    if cfg.critic_kind == "c51":
        return critic.mean_value(states)
    return critic.mean_value(states, k=cfg.iqn_k_eval)


def _critic_loss(
    critic: nn.Module,
    target_critic: nn.Module,
    states: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    cfg: TrainConfig,
    actions: torch.Tensor | None = None,
    next_action_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    if cfg.critic_kind == "c51q":
        return critic.loss(states, actions, rewards, next_states, dones, cfg.gamma,
                           next_action_probs, target_net=target_critic)
    return critic.loss(states, rewards, next_states, dones, cfg.gamma, target_net=target_critic)


def _log_return_distribution(logger: RunLogger, critic: nn.Module, probe_state: torch.Tensor, cfg: TrainConfig, step: int) -> None:
    """Snapshot the critic's predicted return distribution at a fixed probe state.

    Tracking this over training is the per-run analogue of Fig. 9(b) in the paper
    (return-probability heat map): it shows whether the critic is learning to spread
    mass towards the risk-seeking tail the EVaR objective rewards.
    """
    with torch.no_grad():
        if cfg.critic_kind == "c51":
            probs = critic.probs(probe_state).squeeze(0).cpu().numpy()
            support = critic.support.cpu().numpy()
            delta = critic.delta_z
            edges = np.concatenate([support - delta / 2, [support[-1] + delta / 2]])
            logger.log_np_histogram("return_distribution/c51_probe_state", probs, edges, step=step)
        else:
            taus = critic.sample_taus(1, cfg.histogram_samples, probe_state.device)
            samples = critic.quantiles(probe_state, taus).squeeze(0).cpu().numpy()
            logger.log_histogram("return_distribution/iqn_probe_state", samples, step=step)


def objective_metrics(returns: np.ndarray, cfg: TrainConfig) -> dict:
    """The paper's objective, measured on returns rather than estimated by the critic.

    ``J_EVaR(theta) = EVaR_alpha[R(tau)]`` with ``tau ~ pi_theta``, so this is the
    quantity the method claims to maximise -- computed here from an empirical
    sample of trajectory returns with the same solver the critic uses, which is
    what makes it comparable to the SPSA results in the paper.

    The dual search interval is re-derived from the observed spread: the training
    ``EVaRConfig`` sizes ``[x_min, x_max]`` to the *critic's* support, and using
    that here would clip ``x*`` whenever the empirical returns are wider, silently
    biasing the number that goes in the table.
    """
    if returns.size == 0:
        return {}
    spread = float(returns.max() - returns.min())
    eval_cfg = replace(cfg.evar, x_min=1e-3, x_max=max(2.0 * spread, 1.0))
    atoms = torch.as_tensor(returns, dtype=torch.float32).unsqueeze(0)
    evar, x_star = evar_from_distribution(atoms, None, eval_cfg)

    # Risk-*seeking* means the upper tail is the target, so the tail statistics
    # reported here are upper-tail: the mean of the best alpha-fraction, not the
    # worst. Reporting the conventional lower-tail CVaR would score the method on
    # the opposite of what it optimises.
    ordered = np.sort(returns)[::-1]
    k = max(1, int(np.ceil(cfg.evar.alpha * ordered.size)))
    evar_v = float(evar.item())
    # EVaR is a coherent upper-tail measure, so E[Z] <= EVaR_alpha[Z] <= max(Z) holds
    # for every alpha. Logged rather than asserted: a violation means the solver is
    # wrong, and that is a fact about the run which belongs on the dashboard next to
    # the number it invalidates -- not a crash that loses the rest of the sweep.
    lo, hi = float(returns.mean()), float(returns.max())
    tol = 1e-6 * max(1.0, abs(hi))
    return {
        "eval_evar": evar_v,
        "eval_evar_dual_x": float(x_star.item()),
        "eval_evar_within_bounds": float(lo - tol <= evar_v <= hi + tol),
        "eval_evar_at_bound": float(
            x_star.item() <= eval_cfg.x_min * 1.001 or x_star.item() >= eval_cfg.x_max * 0.999
        ),
        "eval_cvar_upper": float(ordered[:k].mean()),
        "eval_top_decile_mean": float(ordered[: max(1, ordered.size // 10)].mean()),
        "eval_return_p90": float(np.percentile(returns, 90)),
        "eval_return_p10": float(np.percentile(returns, 10)),
    }


def evaluate_policy(
    env,
    actor: nn.Module,
    cfg: TrainConfig,
    state_to_tensor: Callable[[np.ndarray], torch.Tensor],
    action_to_env: Callable[[torch.Tensor], object],
    device: torch.device,
) -> dict:
    """Deterministic policy evaluation, decoupled from the noisy training curve.

    Unlike ``episode_return`` in the main loop (collected while the policy is still
    exploring *and* being updated mid-episode -- see the module docstring), this
    freezes the actor, disables exploration (``act_deterministic``: argmax for
    :class:`CategoricalPolicy`, distribution mean for :class:`GaussianPolicy`), takes
    no gradient steps, and always resets on the same fixed seed set
    (``cfg.seed + 1_000_000 + i``) so successive evaluation passes measure genuine
    policy improvement rather than a different random set of initial states.
    """
    was_training = actor.training
    actor.eval()
    returns: list[float] = []
    lengths: list[int] = []
    with torch.no_grad():
        for i in range(cfg.eval_episodes):
            obs, _ = env.reset(seed=cfg.seed + 1_000_000 + i)
            ep_return = 0.0
            ep_len = 0
            for _ in range(cfg.max_steps_per_episode):
                state_t = state_to_tensor(obs).to(device).unsqueeze(0)
                action_t = actor.act_deterministic(state_t).squeeze(0)
                # (deterministic pass -- kept for a stable, comparable curve)
                env_action = action_to_env(action_t)
                obs, reward, terminated, truncated, _ = env.step(env_action)
                ep_return += reward
                ep_len += 1
                if terminated or truncated:
                    break
            returns.append(ep_return)
            lengths.append(ep_len)

        # Stochastic pass: the objective is EVaR over trajectories drawn from the
        # policy, so it can only be measured by sampling actions. The
        # deterministic pass above has (near) zero return spread, which would make
        # the measured EVaR collapse onto the mean and hide the whole effect.
        risk_returns: list[float] = []
        for i in range(cfg.eval_risk_episodes):
            obs, _ = env.reset(seed=cfg.seed + 2_000_000 + i)
            ep_return = 0.0
            for _ in range(cfg.max_steps_per_episode):
                state_t = state_to_tensor(obs).to(device).unsqueeze(0)
                action_t, _, _ = actor.act_with_entropy(state_t)
                obs, reward, terminated, truncated, _ = env.step(
                    action_to_env(action_t.squeeze(0)))
                ep_return += reward
                if terminated or truncated:
                    break
            risk_returns.append(ep_return)
    actor.train(was_training)

    returns_arr = np.asarray(returns, dtype=float)
    risk_arr = np.asarray(risk_returns, dtype=float)
    return {
        "eval_episodes": cfg.eval_episodes,
        "eval_return_mean": float(returns_arr.mean()),
        "eval_return_std": float(returns_arr.std()) if len(returns_arr) > 1 else 0.0,
        "eval_return_min": float(returns_arr.min()),
        "eval_return_max": float(returns_arr.max()),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_risk_episodes": int(risk_arr.size),
        "eval_risk_return_mean": float(risk_arr.mean()) if risk_arr.size else 0.0,
        "eval_risk_return_std": float(risk_arr.std()) if risk_arr.size > 1 else 0.0,
        **objective_metrics(risk_arr, cfg),
    }


def train(
    env,
    actor: nn.Module,
    critic: nn.Module,
    cfg: TrainConfig,
    state_to_tensor: Callable[[np.ndarray], torch.Tensor],
    action_to_env: Callable[[torch.Tensor], object],
    device: torch.device | None = None,
    run_config_extra: dict | None = None,
) -> dict:
    """Run the EVaR actor-critic training loop.

    Args:
        env: a Gymnasium environment.
        actor: a :class:`CategoricalPolicy` or :class:`GaussianPolicy`.
        critic: a :class:`C51Critic` or :class:`IQNCritic`.
        cfg: :class:`TrainConfig`.
        state_to_tensor: converts a raw env observation (numpy) to a (state_dim,) tensor.
        action_to_env: converts a sampled action tensor (shape depends on policy) to
            whatever ``env.step`` expects.
        run_config_extra: extra key/value pairs (e.g. env id) merged into the wandb config.

    Returns:
        ``{"episode_records": [...], "update_records": [...], "eval_records": [...],
        "wandb_run_id": str | None}`` -- the first three are one flat dict per episode
        / gradient update / evaluation pass, suitable for
        :func:`evar_deeprl.utils.save_records` or direct ``pandas.DataFrame``
        construction; ``wandb_run_id`` is set whenever wandb logging was active.
    """
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    device = device or torch.device("cpu")
    actor.to(device)
    critic.to(device)
    target_critic = make_target(critic).to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_config = {
        "critic_kind": cfg.critic_kind,
        "gamma": cfg.gamma,
        "n_steps": cfg.n_steps,
        "max_episodes": cfg.max_episodes,
        "max_steps_per_episode": cfg.max_steps_per_episode,
        "actor_lr": cfg.actor_lr,
        "critic_lr": cfg.critic_lr,
        "entropy_coef": cfg.entropy_coef,
        "target_tau": cfg.target_tau,
        "iqn_k_eval": cfg.iqn_k_eval,
        "grad_clip": cfg.grad_clip,
        "evar_alpha": cfg.evar.alpha,
        "evar_x_min": cfg.evar.x_min,
        "evar_x_max": cfg.evar.x_max,
        "evar_solver_steps": cfg.evar.solver_steps,
        "seed": cfg.seed,
        **(run_config_extra or {}),
    }
    logger = RunLogger(cfg.wandb, run_config)
    logger.watch(actor, critic)

    buffer = RolloutBuffer()
    global_step = 0
    update_count = 0
    best_return = -float("inf")
    return_window: list[float] = []
    probe_state: torch.Tensor | None = None

    def flush_update(episode: int) -> tuple[float, float]:
        nonlocal global_step, update_count
        if len(buffer) == 0:
            return 0.0, 0.0
        states = torch.stack(buffer.states).to(device)
        next_states = torch.stack(buffer.next_states).to(device)
        rewards = torch.tensor(buffer.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(buffer.dones, dtype=torch.float32, device=device)
        log_probs = torch.stack(buffer.log_probs).to(device)
        entropies = torch.stack(buffer.entropies).to(device)

        actions = (torch.stack(buffer.actions).to(device)
                   if cfg.critic_kind == "c51q" else None)
        next_action_probs = None
        if cfg.critic_kind == "c51q":
            with torch.no_grad():
                next_action_probs = actor.distribution(next_states).probs
        critic_loss = _critic_loss(critic, target_critic, states, rewards, next_states,
                                   dones, cfg, actions, next_action_probs)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
        critic_opt.step()
        polyak_update(target_critic, critic, cfg.target_tau)

        with torch.no_grad():
            # One critic forward and one Newton solve over the concatenated batch
            # instead of two of each: the EVaR solve is the costliest part of the
            # update, and s / s' need identical treatment anyway.
            batch = states.shape[0]
            if cfg.critic_kind == "c51q":
                # Action-value form. The tilt is applied to Z(s,a), which carries the
                # immediate reward's randomness, so alpha reaches the reward itself --
                # the state-value form cannot, because EVaR is translation-equivariant
                # and the sampled reward is a scalar. The baseline is the actor-weighted
                # mixture, i.e. the state's value under its own policy, which keeps the
                # advantage centred without changing the argmax.
                pi = actor.distribution(states).probs
                evar_a, x_a = critic.evar_all_actions(states, cfg.evar)     # (B, A)
                evar_s = (evar_a * pi).sum(-1)
                taken = evar_a.gather(1, actions.long().view(-1, 1)).squeeze(1)
                raw_advantage = taken - evar_s
                evar_s_next = evar_s                     # logged only; no bootstrap here
                x_star_s = (x_a * pi).sum(-1)
                value_s = critic.mean_value(states, action_probs=pi)
            else:
                both = torch.cat([states, next_states], dim=0)
                evar_both, x_star_both = _critic_evar(critic, both, cfg)
                evar_s, evar_s_next = evar_both[:batch], evar_both[batch:]
                x_star_s = x_star_both[:batch]
                value_s = _critic_mean_value(critic, states, cfg)
                raw_advantage = rewards + cfg.gamma * (1.0 - dones) * evar_s_next - evar_s
            # Switchable because it is a suspect, not a detail. In expectation this
            # advantage is risk-neutral in the immediate reward -- the tilt only
            # reaches EVaR(Z(s')) -- yet trained agents still take lotteries, so the
            # risk-taking has to be entering somewhere else. Per-batch normalisation
            # is nonlinear in the batch, so a rare huge reward does not contribute in
            # proportion to its probability; turning it off tests whether that, and
            # not the EVaR operator, is what makes the policy risk-seeking.
            if cfg.normalize_advantage and raw_advantage.numel() > 1:
                norm_advantage = (raw_advantage - raw_advantage.mean()) / (raw_advantage.std() + 1e-2)
            else:
                norm_advantage = raw_advantage

        actor_loss = -(log_probs * norm_advantage).mean() - cfg.entropy_coef * entropies.mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
        actor_opt.step()

        update_count += 1
        record = {
            "update": update_count,
            "episode": episode,
            "global_step": global_step,
            "batch_size": len(buffer),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_grad_norm": float(critic_grad_norm),
            "actor_grad_norm": float(actor_grad_norm),
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std().item()) if rewards.numel() > 1 else 0.0,
            "done_frac": float(dones.mean().item()),
            "entropy_mean": float(entropies.mean().item()),
            "entropy_std": float(entropies.std().item()) if entropies.numel() > 1 else 0.0,
            "value_mean": float(value_s.mean().item()),
            "evar_mean": float(evar_s.mean().item()),
            "evar_next_mean": float(evar_s_next.mean().item()),
            "risk_premium_mean": float((evar_s - value_s).mean().item()),
            "evar_dual_x_mean": float(x_star_s.mean().item()),
            "evar_dual_x_std": float(x_star_s.std().item()) if x_star_s.numel() > 1 else 0.0,
            # Tripwire. For alpha < 1 the dual optimum is interior, so a healthy solve
            # leaves this at 0. It sat at 1.0 for every update of every earlier run --
            # x* pinned to a bound, which silently degrades EVaR into fixed-beta
            # entropic utility at beta = 1/x_max. Watch it in wandb: if this leaves 0,
            # the risk operator is not the one the paper describes.
            "evar_dual_x_at_bound_frac": float(
                (
                    (x_star_s <= cfg.evar.x_min * 1.001)
                    | (x_star_s >= cfg.evar.x_max * 0.999)
                ).float().mean().item()
            ),
            "advantage_raw_mean": float(raw_advantage.mean().item()),
            "advantage_raw_std": float(raw_advantage.std().item()) if raw_advantage.numel() > 1 else 0.0,
            "advantage_raw_min": float(raw_advantage.min().item()),
            "advantage_raw_max": float(raw_advantage.max().item()),
        }
        if hasattr(actor, "diagnostics"):
            with torch.no_grad():
                record.update(actor.diagnostics(states))
        logger.log_update(record)

        buffer.clear()
        return record["critic_loss"], record["actor_loss"]

    train_start = time.time()
    try:
        for episode in range(cfg.max_episodes):
            obs, _ = env.reset(seed=cfg.seed + episode)
            if probe_state is None:
                probe_state = state_to_tensor(obs).to(device).unsqueeze(0)

            episode_return = 0.0
            episode_steps = 0
            last_critic_loss = 0.0
            last_actor_loss = 0.0
            episode_start = time.time()

            for step in range(cfg.max_steps_per_episode):
                state_t = state_to_tensor(obs).to(device)
                action_t, log_prob, entropy = actor.act_with_entropy(state_t.unsqueeze(0))

                env_action = action_to_env(action_t.squeeze(0))
                next_obs, reward, terminated, truncated, _ = env.step(env_action)

                buffer.states.append(state_t)
                buffer.actions.append(action_t.squeeze(0).detach())
                buffer.log_probs.append(log_prob.squeeze(0))
                buffer.entropies.append(entropy.squeeze(0))
                buffer.rewards.append(float(reward))
                buffer.next_states.append(state_to_tensor(next_obs))
                buffer.dones.append(1.0 if terminated else 0.0)

                episode_return += reward
                episode_steps += 1
                global_step += 1
                obs = next_obs

                if len(buffer) >= cfg.n_steps or terminated or truncated:
                    last_critic_loss, last_actor_loss = flush_update(episode + 1)

                if terminated or truncated:
                    break

            best_return = max(best_return, episode_return)
            return_window.append(episode_return)
            if len(return_window) > cfg.log_every:
                return_window.pop(0)

            elapsed = time.time() - train_start
            episode_record = {
                "episode": episode + 1,
                "global_step": global_step,
                "episode_return": episode_return,
                "episode_length": episode_steps,
                "return_avg_window": float(np.mean(return_window)),
                "return_best": best_return,
                "last_critic_loss": last_critic_loss,
                "last_actor_loss": last_actor_loss,
                "steps_per_sec": episode_steps / max(time.time() - episode_start, 1e-8),
                "elapsed_s": elapsed,
            }
            logger.log_episode(episode_record)

            if cfg.eval_every and (episode + 1) % cfg.eval_every == 0:
                eval_stats = evaluate_policy(env, actor, cfg, state_to_tensor, action_to_env, device)
                eval_record = {"episode": episode + 1, "global_step": global_step, **eval_stats}
                logger.log_eval(eval_record)
                print(
                    f"  [eval] episode {episode + 1:4d}  "
                    f"mean_return={eval_stats['eval_return_mean']:8.2f} +/- {eval_stats['eval_return_std']:6.2f}  "
                    f"(min={eval_stats['eval_return_min']:.2f}, max={eval_stats['eval_return_max']:.2f}, "
                    f"over {cfg.eval_episodes} episodes)"
                )

            if cfg.histogram_every and (episode + 1) % cfg.histogram_every == 0:
                _log_return_distribution(logger, critic, probe_state, cfg, step=global_step)

            if cfg.checkpoint_every and (episode + 1) % cfg.checkpoint_every == 0:
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"{cfg.critic_kind}_ep{episode + 1}.pt")
                torch.save(
                    {"actor": actor.state_dict(), "critic": critic.state_dict(), "episode": episode + 1},
                    ckpt_path,
                )

            if (episode + 1) % cfg.log_every == 0:
                print(
                    f"[{cfg.critic_kind}] episode {episode + 1:4d}  "
                    f"return={episode_return:8.2f}  avg{cfg.log_every}={np.mean(return_window):8.2f}  "
                    f"critic_loss={last_critic_loss:.4f}  actor_loss={last_actor_loss:.4f}"
                )
    finally:
        run_id = logger.run_id
        logger.finish()

    return {
        "episode_records": logger.episode_records,
        "update_records": logger.update_records,
        "eval_records": logger.eval_records,
        "wandb_run_id": run_id,
    }
