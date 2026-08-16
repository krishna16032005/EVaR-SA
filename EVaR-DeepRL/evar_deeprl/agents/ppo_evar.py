"""Vectorized PPO learner with a distributional critic and an EVaR advantage.

Why this replaces `agents/base.py` for anything at scale
-------------------------------------------------------
The n-step A2C in `base.py` cannot do continuous control. Measured on
`SafetyPointGoal1-v0` with cost priced at zero -- pure navigation, no risk
machinery involved -- reward converges to random-policy level and stays there over
1500 episodes. Three things are wrong with it, and all three are standard:

* **One environment.** Updates come from consecutive steps of a single env, so
  every gradient is drawn from a highly correlated batch.
* **One epoch, no trust region.** A single gradient step per batch, nothing
  constraining how far the policy moves.
* **No observation normalization.** MuJoCo-class observations differ in scale by
  orders of magnitude per coordinate.

They are fixed together because they are not separable: vectorization is what makes
multiple epochs worth doing, and clipping is what makes multiple epochs safe.

What is kept from the EVaR work
-------------------------------
The advantage is the action-value form:

    A(s,a) = EVaR_alpha( Z(s,a) )  -  E_{a'~pi} EVaR_alpha( Z(s,a') )

`Z(s,a)` carries the immediate reward's randomness, which is what lets alpha reach
it. The state-value form cannot: EVaR is translation-equivariant, so a *sampled
scalar* reward inside the tilt changes nothing, and at a terminal step alpha drops
out entirely. Measured exactly on the lottery gridworld that costs up to 86.35% of
the optimum, against 0.00% for this form -- `analysis/c3_attribution.py`.

`risk_objective="mean"` swaps EVaR for the distribution's mean using the same nets
and the same data, which is the risk-neutral control (baseline 3 in the plan) and
the only way to attribute a difference to the risk operator rather than to the
distributional critic.

Critic targets
--------------
The critic is fit to the empirical discounted return-to-go, bootstrapped at the
rollout boundary, projected onto the categorical support -- rather than a one-step
distributional Bellman backup. One-step targets propagate too slowly over a
1000-step horizon, and the alternative (an n-step distributional backup) is fiddly
enough that a silent indexing bug is the likely outcome. Fitting the observed
return is simple enough to be obviously correct, and across a batch the fitted
categorical head still recovers a distribution rather than a point.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evar_deeprl.logging_utils import RunLogger, WandbConfig
from evar_deeprl.risk.evar import EVaRConfig


@dataclass
class PPOConfig:
    n_envs: int = 16
    n_steps: int = 128                 # per env; batch is n_envs * n_steps
    total_steps: int = 2_000_000
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    epochs: int = 10
    minibatches: int = 4
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    grad_clip: float = 0.5
    target_kl: float | None = 0.03
    anneal_lr: bool = True             # standard PPO; the run collapses without it
    normalize_advantage: bool = True
    normalize_obs: bool = True
    evar: EVaRConfig = field(default_factory=EVaRConfig)
    risk_objective: str = "evar"       # "evar" | "mean"
    baseline_samples: int = 4
    cost_penalty: float = 0.0          # lambda for safety-gymnasium's cost channel
    seed: int = 0
    log_every: int = 10
    wandb: WandbConfig = field(default_factory=WandbConfig)
    torch_threads: int = 1


class RunningNorm:
    """Welford running mean/std over observations."""

    def __init__(self, shape, eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        bm, bv, bc = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc
                    + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(z, -10.0, 10.0).astype(np.float32)


def policy_logp_entropy(actor, obs, actions):
    """log pi(a|s) and entropy, for both the categorical and Gaussian actors."""
    dist = actor.distribution(obs)
    logp = dist.log_prob(actions)
    ent = dist.entropy()
    if logp.dim() > 1:                 # Gaussian: sum over action dimensions
        logp = logp.sum(-1)
        ent = ent.sum(-1)
    return logp, ent


def _discrete(critic) -> bool:
    return hasattr(critic, "evar_all_actions")


def mean_of(critic, obs, actions):
    """Mean of Z(s,a) for the taken action, for either critic."""
    if _discrete(critic):
        return critic.mean_value_taken(obs, actions)
    return critic.mean_value(obs, actions)


def risk_value(critic, obs, actions, cfg):
    """EVaR (or mean) of Z(s,a) -- the quantity the actor is pushed toward."""
    if cfg.risk_objective == "mean":
        return mean_of(critic, obs, actions)
    if _discrete(critic):
        return critic.evar_taken(obs, cfg.evar, actions)[0]
    ev, _ = critic.evar(obs, cfg.evar, actions)
    return ev


def baseline_value(critic, obs, actor, cfg):
    """E_{a~pi}[ risk_value(s,a) ], the state-dependent baseline.

    Any function of the state is a valid baseline, so this changes variance and not
    the expected gradient -- and variance is the whole point here. With discrete
    actions the expectation is a closed-form sum over the action set, so it is taken
    exactly; sampling it (which is unavoidable for continuous actions) injects noise
    into every advantage for no reason.
    """
    with torch.no_grad():
        if _discrete(critic):
            pi = actor.distribution(obs).probs                    # (B, A)
            if cfg.risk_objective == "mean":
                q = (critic.probs(obs) * critic.support).sum(-1)  # (B, A)
            else:
                q, _ = critic.evar_all_actions(obs, cfg.evar)     # (B, A)
            return (pi * q).sum(-1)
        total = None
        for _ in range(cfg.baseline_samples):
            a, _ = actor.act(obs)
            v = risk_value(critic, obs, a, cfg)
            total = v if total is None else total + v
        return total / cfg.baseline_samples


def train_ppo(env, actor, critic, cfg: PPOConfig, device, run_config_extra=None):
    """Vectorized PPO with a distributional action-value critic."""
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger = RunLogger(cfg.wandb, run_config={**(run_config_extra or {}),
                                          "algo": "ppo-evar",
                                          "risk_objective": cfg.risk_objective,
                                          "alpha": cfg.evar.alpha,
                                          "n_envs": cfg.n_envs,
                                          "n_steps": cfg.n_steps,
                                          "cost_penalty": cfg.cost_penalty})
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

    obs_shape = env.single_observation_space.shape
    obs_norm = RunningNorm(obs_shape)
    raw_obs, _ = env.reset(seed=cfg.seed)
    N, T = cfg.n_envs, cfg.n_steps

    ep_ret = np.zeros(N)
    ep_cost = np.zeros(N)
    ep_len = np.zeros(N, dtype=int)
    finished_returns: list[float] = []
    finished_costs: list[float] = []
    finished_lengths: list[int] = []

    global_step = 0
    update = 0
    start = time.time()

    while global_step < cfg.total_steps:
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf = np.zeros((T, N), dtype=np.float32)
        done_buf = np.zeros((T, N), dtype=np.float32)

        if cfg.normalize_obs:
            obs_norm.update(raw_obs)
        for t in range(T):
            norm = obs_norm(raw_obs) if cfg.normalize_obs else np.asarray(raw_obs, np.float32)
            obs_t = torch.as_tensor(norm, device=device)
            with torch.no_grad():
                action, _ = actor.act(obs_t)
                logp, _ = policy_logp_entropy(actor, obs_t, action)
            step_out = env.step(action.cpu().numpy())
            if len(step_out) == 6:                     # safety-gymnasium
                nxt, reward, cost, term, trunc, _ = step_out
                cost = np.asarray(cost, dtype=np.float32)
                reward = np.asarray(reward, dtype=np.float32) - cfg.cost_penalty * cost
                ep_cost += cost
            else:
                nxt, reward, term, trunc, _ = step_out
                reward = np.asarray(reward, dtype=np.float32)
            term = np.asarray(term, dtype=bool)
            trunc = np.asarray(trunc, dtype=bool)

            obs_buf.append(obs_t)
            act_buf.append(action)
            logp_buf.append(logp)
            rew_buf[t] = reward
            # `terminated` only: a time-limit truncation must still bootstrap, or the
            # agent is punished for surviving to the horizon.
            done_buf[t] = term.astype(np.float32)

            ep_ret += reward
            ep_len += 1
            for i in np.nonzero(np.logical_or(term, trunc))[0]:
                finished_returns.append(float(ep_ret[i]))
                finished_costs.append(float(ep_cost[i]))
                finished_lengths.append(int(ep_len[i]))
                ep_ret[i] = ep_cost[i] = 0.0
                ep_len[i] = 0
            raw_obs = nxt
            global_step += N

        obs_b = torch.stack(obs_buf)                      # (T, N, obs)
        act_b = torch.stack(act_buf)
        logp_b = torch.stack(logp_buf)

        # ---- targets: empirical discounted return-to-go, bootstrapped at T ----
        with torch.no_grad():
            last = obs_norm(raw_obs) if cfg.normalize_obs else np.asarray(raw_obs, np.float32)
            last_t = torch.as_tensor(last, device=device)
            boot_a, _ = actor.act(last_t)
            boot = mean_of(critic, last_t, boot_a).cpu().numpy()
        returns = np.zeros((T, N), dtype=np.float32)
        running = boot
        for t in reversed(range(T)):
            running = rew_buf[t] + cfg.gamma * (1.0 - done_buf[t]) * running
            returns[t] = running
        ret_t = torch.as_tensor(returns.reshape(-1), device=device)

        # ---- advantage from the critic, computed once under the old policy ----
        flat_obs = obs_b.reshape(T * N, -1)
        flat_act = act_b.reshape(T * N, -1) if act_b.dim() == 3 else act_b.reshape(-1)
        with torch.no_grad():
            q = risk_value(critic, flat_obs, flat_act, cfg)
            b = baseline_value(critic, flat_obs, actor, cfg)
            adv = q - b
            if cfg.normalize_advantage:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat_logp = logp_b.reshape(-1)

        # ---- epochs ----
        # Linear LR decay over training, and a KL early-stop checked *per minibatch*
        # rather than per epoch. Checking only between epochs let a single epoch run
        # the KL to 0.165 against a 0.03 target, after which return fell 169 -> 10
        # and explained variance went negative.
        if cfg.anneal_lr:
            frac = max(0.0, 1.0 - global_step / cfg.total_steps)
            for g in actor_opt.param_groups:
                g["lr"] = cfg.actor_lr * frac
            for g in critic_opt.param_groups:
                g["lr"] = cfg.critic_lr * frac

        batch = T * N
        mb = batch // cfg.minibatches
        idx = np.arange(batch)
        approx_kl = 0.0
        stop = False
        for _ in range(cfg.epochs):
            if stop:
                break
            np.random.shuffle(idx)
            for s in range(0, batch, mb):
                j = torch.as_tensor(idx[s:s + mb], device=device)
                new_logp, ent = policy_logp_entropy(actor, flat_obs[j], flat_act[j])
                ratio = (new_logp - flat_logp[j]).exp()
                a_j = adv[j]
                loss1 = -a_j * ratio
                loss2 = -a_j * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                actor_loss = torch.max(loss1, loss2).mean() - cfg.entropy_coef * ent.mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                a_gn = nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
                actor_opt.step()

                critic_loss = critic.regression_loss(flat_obs[j], flat_act[j], ret_t[j])
                critic_opt.zero_grad()
                critic_loss.backward()
                c_gn = nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
                critic_opt.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1) - (new_logp - flat_logp[j])).mean())
                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    stop = True
                    break

        update += 1
        recent = finished_returns[-50:]
        rec_cost = finished_costs[-50:]
        with torch.no_grad():
            pred = mean_of(critic, flat_obs, flat_act).cpu().numpy()
        record = {
            "update": update,
            "global_step": global_step,
            "episodes_finished": len(finished_returns),
            "return_mean": float(np.mean(recent)) if recent else float("nan"),
            "return_max": float(np.max(recent)) if recent else float("nan"),
            "cost_mean": float(np.mean(rec_cost)) if rec_cost else float("nan"),
            "episode_len_mean": float(np.mean(finished_lengths[-50:])) if finished_lengths else float("nan"),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "approx_kl": approx_kl,
            "entropy": float(ent.mean().item()),
            "advantage_std": float(adv.std().item()),
            "risk_value_mean": float(q.mean().item()),
            "baseline_mean": float(b.mean().item()),
            "explained_variance": float(1 - np.var(returns.reshape(-1) - pred) / (np.var(returns) + 1e-8)),
            "actor_grad_norm": float(a_gn),
            "critic_grad_norm": float(c_gn),
            "steps_per_sec": global_step / max(time.time() - start, 1e-8),
        }
        logger.log_update(record)
        if update % cfg.log_every == 0:
            print(f"[ppo] step {global_step:>9,}  ep {len(finished_returns):>6}  "
                  f"return {record['return_mean']:8.2f}  cost {record['cost_mean']:7.2f}  "
                  f"ev {record['explained_variance']:6.3f}  kl {approx_kl:.4f}  "
                  f"{record['steps_per_sec']:.0f} steps/s")

    logs = {"update_records": logger.update_records,
            "episode_records": [{"episode": i + 1, "episode_return": r, "episode_cost": c,
                                 "episode_length": l}
                                for i, (r, c, l) in enumerate(
                                    zip(finished_returns, finished_costs, finished_lengths))],
            "eval_records": logger.eval_records,
            "wandb_run_id": logger.run_id}
    logger.finish()
    return logs
