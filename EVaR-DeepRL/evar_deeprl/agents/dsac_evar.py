"""Risk-sensitive DSAC: off-policy SAC with an IQN action-value critic and EVaR.

This is the learner the project actually needs. The earlier ones each failed for a
recorded reason:

* the n-step A2C could not do continuous control at all -- on SafetyPointGoal1 with
  cost priced at zero it sat at random-policy reward for 1500 episodes;
* the on-policy PPO learner works but is sample-hungry, and continuous-control
  benchmarks in this literature are run off-policy.

Design, with the reason for each choice:

**Off-policy with a replay buffer.** DSAC and SAC are the comparisons that matter on
continuous control, and they are off-policy. It also means the tail of the return
distribution is estimated from the whole buffer rather than from one rollout, which
is the regime the risk measure needs.

**IQN action-value critic.** ``EVaR(Z(s,a))``, never ``r + EVaR(Z(s'))`` -- the latter
is translation-equivariant in the sampled reward and therefore risk-neutral in it
(measured: up to 86.35% regret). Quantiles avoid the fixed-support sizing that left
~9 of 51 atoms in use on CartPole.

**Twin critics, minimum over the pair.** Standard SAC overestimation control, applied
to the *risk value* rather than the mean, since that is what the actor maximises.

**Tanh-squashed actor.** A clamped Gaussian evaluates densities at a boundary and
blows up the objective; that produced approx KL of 1.4e16 in the PPO learner.

**Automatic entropy temperature.** Hand-tuning the entropy coefficient wasted real
time earlier: at 1e-3 and 1e-2 policy entropy collapsed identically because the
coefficient is scaled against normalised advantages, and the fix was not a better
constant but removing the constant.

The risk functional is pluggable, so ``--risk mean`` is the risk-neutral control
sharing every other component -- the only way to attribute a difference to the risk
measure rather than to the algorithm.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from evar_deeprl.distributional.iqn_q import (
    IQNQCritic, check_alpha_vs_k, risk_value_from_z)
from evar_deeprl.logging_utils import RunLogger, WandbConfig
from evar_deeprl.policies.squashed_gaussian import SquashedGaussianPolicy
from evar_deeprl.risk.evar import EVaRConfig
from evar_deeprl.risk.measures import RiskConfig


@dataclass
class DSACConfig:
    total_steps: int = 1_000_000
    start_steps: int = 10_000          # uniform-random actions before learning
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005                 # polyak
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4             # entropy temperature
    updates_per_step: int = 1
    n_quantiles: int = 32              # K for the critic's own samples
    n_quantiles_target: int = 32
    n_quantiles_risk: int = 64         # K for the risk value; alpha > 1/K must hold
    risk: RiskConfig = field(default_factory=RiskConfig)
    x_smoothing: float = 0.0           # trust region on the dual variable; 0 = off
    cost_penalty: float = 0.0
    autotune_entropy: bool = True
    target_entropy_scale: float = 1.0
    seed: int = 0
    log_every: int = 20                # in thousands of env steps
    eval_every: int = 0
    wandb: WandbConfig = field(default_factory=WandbConfig)
    torch_threads: int = 1


class Replay:
    def __init__(self, size, obs_dim, act_dim, device):
        self.o = np.zeros((size, obs_dim), dtype=np.float32)
        self.a = np.zeros((size, act_dim), dtype=np.float32)
        self.r = np.zeros(size, dtype=np.float32)
        self.o2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.d = np.zeros(size, dtype=np.float32)
        self.size, self.ptr, self.full = size, 0, False
        self.device = device

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n):
        idx = np.random.randint(0, len(self), size=n)
        t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


def polyak(target, source, tau):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(tau * sp)


def train_dsac(env, cfg: DSACConfig, device, run_config_extra=None):
    """Single-env off-policy loop. Returns the same log dict shape as the others."""
    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    bound = float(env.action_space.high[0])

    if cfg.risk.kind == "evar":
        check_alpha_vs_k(cfg.risk.alpha, cfg.n_quantiles_risk)

    actor = SquashedGaussianPolicy(obs_dim, act_dim, action_bound=bound).to(device)
    q1 = IQNQCritic(obs_dim, act_dim).to(device)
    q2 = IQNQCritic(obs_dim, act_dim).to(device)
    q1t, q2t = IQNQCritic(obs_dim, act_dim).to(device), IQNQCritic(obs_dim, act_dim).to(device)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    for p in list(q1t.parameters()) + list(q2t.parameters()):
        p.requires_grad_(False)

    a_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    c_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=cfg.critic_lr)

    # Entropy temperature, tuned rather than guessed.
    target_entropy = -cfg.target_entropy_scale * act_dim
    log_ent = torch.zeros(1, requires_grad=True, device=device)
    e_opt = torch.optim.Adam([log_ent], lr=cfg.alpha_lr)

    buf = Replay(cfg.buffer_size, obs_dim, act_dim, device)
    # `seed` in particular is not optional. Without it every run of a group reports
    # the same default, seeds collide when results are keyed by (arm, seed), and the
    # paired comparison against the risk-neutral control silently reduces to
    # whichever run was read last. The rest is what a reader of the report needs to
    # tell two runs apart without opening the launch script.
    logger = RunLogger(cfg.wandb, run_config={**(run_config_extra or {}),
                                              "algo": "dsac-evar",
                                              "risk": cfg.risk.label(),
                                              "risk_kind": cfg.risk.kind,
                                              "alpha": cfg.risk.alpha,
                                              "seed": cfg.seed,
                                              "total_steps": cfg.total_steps,
                                              "batch_size": cfg.batch_size,
                                              "gamma": cfg.gamma,
                                              "n_quantiles_risk": cfg.n_quantiles_risk,
                                              "x_smoothing": cfg.x_smoothing,
                                              "cost_penalty": cfg.cost_penalty})

    obs, _ = env.reset(seed=cfg.seed)
    ep_ret = ep_cost = 0.0
    ep_len = 0
    returns, costs = [], []
    x_prev = None
    start = time.time()
    last_log = 0
    stats = {}

    for step in range(cfg.total_steps):
        if step < cfg.start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _ = actor.act(torch.as_tensor(obs, dtype=torch.float32,
                                                 device=device).unsqueeze(0))
            action = a.squeeze(0).cpu().numpy()

        out = env.step(action)
        if len(out) == 6:                                   # safety-gymnasium
            obs2, reward, cost, term, trunc, _ = out
            ep_cost += float(cost)
            reward = float(reward) - cfg.cost_penalty * float(cost)
        else:
            obs2, reward, term, trunc, _ = out
            reward = float(reward)
        ep_ret += reward
        ep_len += 1
        # `terminated` only: a time-limit truncation must still bootstrap.
        buf.add(obs, action, reward, obs2, float(term))
        obs = obs2

        if term or trunc:
            returns.append(ep_ret)
            costs.append(ep_cost)
            obs, _ = env.reset()
            ep_ret = ep_cost = 0.0
            ep_len = 0

        if step >= cfg.start_steps:
            for _ in range(cfg.updates_per_step):
                o, a, r, o2, d = buf.sample(cfg.batch_size)

                # ---- critic: quantile regression on the distributional target ----
                with torch.no_grad():
                    a2, logp2 = actor.act(o2)
                    z1 = q1t.sample_z(o2, a2, k=cfg.n_quantiles_target)
                    z2 = q2t.sample_z(o2, a2, k=cfg.n_quantiles_target)
                    # min over the twin pair, per quantile
                    zt = torch.min(z1, z2)
                    ent = log_ent.exp().detach()
                    target = (r.unsqueeze(-1)
                              + cfg.gamma * (1 - d).unsqueeze(-1)
                              * (zt - ent * logp2.unsqueeze(-1)))
                closs = (q1.quantile_loss(o, a, target, k=cfg.n_quantiles)
                         + q2.quantile_loss(o, a, target, k=cfg.n_quantiles))
                c_opt.zero_grad()
                closs.backward()
                c_opt.step()

                # ---- actor: maximise the RISK value, not the mean ----
                anew, logp = actor.act(o)
                # One solve on the stacked twin batch rather than two. Rows are
                # independent inside the solver, so this is exactly equivalent and
                # halves the bisection's kernel launches -- the dominant per-update
                # cost on GPU, where the nets themselves are nearly free.
                zz = torch.cat([q1.sample_z(o, anew, k=cfg.n_quantiles_risk),
                                q2.sample_z(o, anew, k=cfg.n_quantiles_risk)], dim=0)
                rr, x1 = risk_value_from_z(zz, cfg.risk, x_prev, cfg.x_smoothing)
                r1, r2 = rr.split(o.shape[0], dim=0)
                rq = torch.min(r1, r2)
                aloss = (log_ent.exp().detach() * logp - rq).mean()
                a_opt.zero_grad()
                aloss.backward()
                a_opt.step()
                if x1 is not None:
                    x_prev = x1.detach()

                if cfg.autotune_entropy:
                    eloss = -(log_ent.exp() * (logp.detach() + target_entropy)).mean()
                    e_opt.zero_grad()
                    eloss.backward()
                    e_opt.step()

                polyak(q1t, q1, cfg.tau)
                polyak(q2t, q2, cfg.tau)

            stats = {"critic_loss": float(closs), "actor_loss": float(aloss),
                     "entropy_temp": float(log_ent.exp()),
                     "logp_mean": float(logp.mean()),
                     "risk_value_mean": float(rq.mean()),
                     "x_star_mean": float(x1.mean()) if x1 is not None else float("nan")}

        if step - last_log >= 1000 * cfg.log_every and returns:
            last_log = step
            rec = {"global_step": step, "episodes": len(returns),
                   "return_mean": float(np.mean(returns[-20:])),
                   "cost_mean": float(np.mean(costs[-20:])),
                   "steps_per_sec": step / max(time.time() - start, 1e-8), **stats}
            logger.log_update(rec)
            print(f"[dsac] step {step:>8,}  ep {len(returns):>5}  "
                  f"return {rec['return_mean']:8.2f}  cost {rec['cost_mean']:7.2f}  "
                  f"ent {stats.get('entropy_temp', float('nan')):.4f}  "
                  f"x* {stats.get('x_star_mean', float('nan')):.2f}  "
                  f"{rec['steps_per_sec']:.0f} steps/s")

    logs = {"update_records": logger.update_records,
            "episode_records": [{"episode": i + 1, "episode_return": r,
                                 "episode_cost": c}
                                for i, (r, c) in enumerate(zip(returns, costs))],
            "eval_records": [], "wandb_run_id": logger.run_id}
    logger.finish()
    return logs
