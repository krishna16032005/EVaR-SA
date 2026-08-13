# EVaR-DeepRL

Actor-critic risk-seeking RL with **distributional critics** (C51, IQN), replacing the
SPSA-based multi-timescale optimizer in `../EVaR-SA` (the `Risk-Seeking Reinforcement
Learning via Multi-Timescale EVaR Optimization` paper, `4774_Risk_Seeking_Reinforcemen.pdf`
in this folder).

## Why move off SPSA

`EVaR-SA` estimates `J_EVaR(theta) = EVaR_alpha[R(tau)]` for a *whole policy* by rolling
out many trajectories per candidate `theta +/- c_t * delta_t`, running a two-timescale
stochastic-approximation recursion to get a scalar EVaR estimate (Eqs. 6-11 of the
paper), then taking a simultaneous-perturbation finite-difference step (Eqs. 12-14).
Every outer policy update therefore costs `2 * N_t` on-policy rollouts and only ever
gets a noisy zeroth-order gradient.

Here, instead:

1. A **distributional critic** (C51 categorical head, or IQN implicit quantile head)
   learns the return distribution `Z(s)` for every visited state via ordinary one-step
   distributional TD learning (categorical projection for C51, quantile regression for
   IQN) - the same class of update DQN-family algorithms use, just applied to a
   state-value function `V(s)` instead of `Q(s, a)`, so it plugs into any actor.
2. `EVaR_alpha[Z(s)]` is solved **in closed form** from that learned distribution:
   `EVaR_alpha[Z] = min_{beta>0} (1/beta)(log E[exp(beta Z)] - log alpha)`. Using the
   paper's `x = 1/beta` reparameterization (Proposition 1 / Theorem 1: `G(x)` is
   strongly convex on a compact interval), the 1-D minimization is solved with a
   handful of projected Newton steps (`evar_deeprl/risk/evar.py`) - no rollouts needed,
   and the whole thing is differentiable.
3. The **actor** (categorical policy for discrete actions, diagonal Gaussian for
   continuous actions) is trained with a plain policy-gradient using an EVaR-TD
   advantage, the direct entropic-risk analogue of the classical actor-critic
   advantage:

   ```
   A_t = r_t + gamma * EVaR_alpha[Z(s_{t+1})] - EVaR_alpha[Z(s_t)]
   ```

   instead of `r_t + gamma * E[Z(s_{t+1})] - E[Z(s_t)]`. Because everything is a
   differentiable torch module, the actor is updated by backprop, on-policy, every
   `n_steps` environment steps - no finite-difference perturbations of the policy and
   no separate multi-timescale bookkeeping for `(vartheta_t, omega_t, x_t)`.

This is a simplification relative to the paper's full trajectory-level exponential
tilting (Eq. 5): the tilting here happens locally, per-state, through the critic's
learned distribution rather than through explicit importance-reweighting of whole
trajectories. It is the natural actor-critic instantiation of the same objective, and a
much cheaper starting point for preliminary experiments.

## Layout

```
evar_deeprl/
  risk/evar.py                  differentiable EVaR-from-distribution solver (shared)
  distributional/c51.py         C51 categorical state-value critic
  distributional/iqn.py         IQN implicit-quantile state-value critic
  policies/categorical.py       discrete-action actor (CartPole)
  policies/gaussian.py          continuous-action actor (InvertedPendulum)
  agents/base.py                EVaR actor-critic training loop (TrainConfig, train())
  logging_utils.py              RunLogger + WandbConfig: unified CSV / wandb logging
  utils.py                      tidy-CSV writer (save_records)
experiments/
  run_cartpole.py               CartPole-v1, discrete actions
  run_invpend.py                InvertedPendulum-v4 (MuJoCo) or Pendulum-v1, continuous
  plot_results.py               moving-average episode-return plots from saved CSVs
results/                        per-run CSV logs, checkpoints and plots land here
```

## Running the preliminary experiments

```bash
pip install -r requirements.txt   # torch, gymnasium[classic-control,mujoco], numpy, matplotlib, wandb

# CartPole-v1 (discrete actor), both distributional critics
python experiments/run_cartpole.py --critic c51 --episodes 500 --alpha 0.1
python experiments/run_cartpole.py --critic iqn --episodes 500 --alpha 0.1

# InvertedPendulum-v4 (continuous actor), both distributional critics
python experiments/run_invpend.py --critic c51 --episodes 500 --alpha 0.1
python experiments/run_invpend.py --critic iqn --episodes 500 --alpha 0.1

# If MuJoCo isn't installed, swap to the classic-control continuous env instead:
python experiments/run_invpend.py --env Pendulum-v1 --critic c51

python experiments/plot_results.py
```

`--alpha` is the EVaR confidence level (paper's `alpha`; smaller -> more risk-seeking,
heavier emphasis on the upper tail).

`--device {auto,cpu,cuda}` selects the compute device (`auto` = CUDA when visible).
Note that with these small MLPs and single-env rollouts the loop is env/Python-bound,
so CPU is usually as fast or faster; the GPU matters for wider networks or many IQN
quantile samples.

For the push-to-run experiment lifecycle (edit a queue file, push, read wandb), see [AUTOMATION.md](AUTOMATION.md).

To run on a GPU server in a container, see [DOCKER.md](DOCKER.md): part A covers the
long-lived SSH dev box (`Dockerfile.ssh`, the `ssh-server/<name>` workflow used on
TUM's gpud.model), part B the one-shot batch container (`Dockerfile` +
`docker/run.sh`) for plain Docker hosts and Slurm/Enroot clusters. Both cover W&B
credentials, multi-seed sweeps, and offline sync.

## Logging: what's tracked, and where it goes

Every run gets its own timestamped subdirectory,
`results/<env>/<critic>_alpha<alpha>_seed<seed>_<YYYYMMDD-HHMMSS>/` (built by
`evar_deeprl.utils.new_run_tag`), so **successive runs never overwrite each other's
CSVs or checkpoints** -- rerun the same command as many times as you like. The same
tag is folded into the wandb run name so a local directory and its wandb run are easy
to match up by eye; `logs["wandb_run_id"]` (also printed at the end of each script) is
the exact link.

Every run always writes **tidy local CSVs** (one row per episode / per gradient
update, joinable on `global_step`) via `evar_deeprl.utils.save_records`:

- `<run_dir>/<prefix>_episode.csv` - one row per episode:
  `episode`, `global_step`, `episode_return`, `episode_length`, `return_avg_window`,
  `return_best`, `last_critic_loss`, `last_actor_loss`, `steps_per_sec`, `elapsed_s`,
  `wall_time_s`.
- `<run_dir>/<prefix>_update.csv` - one row per gradient update (there are several
  of these per episode, every `n_steps` env steps):
  `update`, `episode`, `global_step`, `batch_size`,
  `critic_loss`, `actor_loss`, `critic_grad_norm`, `actor_grad_norm` (pre-clip norms,
  from `clip_grad_norm_`'s return value),
  `reward_mean`, `reward_std`, `done_frac`,
  `entropy_mean`, `entropy_std`,
  `value_mean` (plain expected return from the critic),
  `evar_mean`, `evar_next_mean` (EVaR_alpha of Z(s) / Z(s')),
  `risk_premium_mean` = `evar_mean - value_mean` (how much the entropic tilt inflates
  the critic's estimate over the risk-neutral one -- should stay positive and is a
  direct read on how "risk-seeking" the critic's signal actually is),
  `evar_dual_x_mean` (the solved dual variable `x* = 1/beta*` from the EVaR Newton
  solve -- the per-batch analogue of Fig. 3 in the paper),
  `advantage_raw_{mean,std,min,max}` (the EVaR-TD advantage *before* batch
  normalization -- the normalized version is what the actor loss actually uses, but
  the raw values are what reveal skew/outliers),
  plus policy-specific diagnostics from `actor.diagnostics()`:
  `policy_max_action_prob_mean` (categorical: mean confidence of the argmax action) or
  `policy_std_mean` / `policy_mean_action_abs_mean` / `policy_mean_action_saturated_frac`
  (Gaussian: action std, mean magnitude, fraction of states whose mean action is
  clipped at the action bound).
- `<run_dir>/<prefix>_eval.csv` - one row every `--eval-every` episodes (default 20),
  from a genuine **held-out evaluation pass**, not the training curve above -- see
  next section.

### Training return vs. evaluation return -- these are not the same thing

`episode_return` in `*_episode.csv` is collected **while the policy is still
exploring and still being updated**: `actor.act()` samples from the (stochastic)
policy, and a gradient step fires every `n_steps` environment steps -- possibly
several times within a single episode. It's a real signal for "is training
progressing," but it is *not* a clean measurement of policy quality at any given
point, since the policy that produced the back half of an episode's reward isn't the
same policy that produced the front half.

`*_eval.csv` (from `evar_deeprl.agents.base.evaluate_policy`, called every
`--eval-every` episodes, default 20) is the actual answer to "how good is the policy
right now": the actor is frozen (`.eval()`, `torch.no_grad()`), actions are
**deterministic** (argmax for `CategoricalPolicy`, distribution mean for
`GaussianPolicy` -- no exploration noise), and every evaluation pass resets on the
*same fixed seed set* (`cfg.seed + 1_000_000 + i` for `i` in `range(eval_episodes)`),
so improvement across evaluation checkpoints reflects the policy, not a different
random set of initial conditions. Columns: `episode`, `global_step`, `eval_episodes`,
`eval_return_mean`, `eval_return_std`, `eval_return_min`, `eval_return_max`,
`eval_length_mean`. `experiments/plot_results.py` plots both on the same axes (thin
translucent training curve, bold eval curve with a shaded +/-1 std band) so the
distinction is visible at a glance.

Optional extras (enabled via CLI flags):

- `--eval-every N` (default 20 episodes, `--eval-episodes` default 5): see above. Set
  to `0` to disable.
- `--histogram-every N` (default 50 episodes): snapshots the critic's predicted
  return distribution at a fixed probe state (the very first state seen in training)
  -- the per-run analogue of Fig. 9(b) in the paper (return-probability heat map).
  Requires wandb (`--wandb-mode offline` or `online`) since histograms aren't written
  to CSV. Set to `0` to disable.
- `--checkpoint-every N` (default 0/off): saves `{actor, critic}.state_dict()` to
  `<run_dir>/checkpoints/` every N episodes.

### Weights & Biases

`--wandb-mode online` is the **default** and mirrors every record above under
`update/*` and `episode/*`, plus `wandb.watch(actor, critic)` (gradient/parameter
histograms) and the return-distribution snapshots described above. It streams live to
whichever account you're logged into (checked via `wandb login` / `WANDB_API_KEY` /
the netrc credential file wandb itself manages) -- the run URL is printed as soon as
training starts:

```bash
python experiments/run_cartpole.py --critic c51   # streams to wandb by default now

# force a local-only run (no account needed), sync later if you want:
python experiments/run_cartpole.py --critic c51 --wandb-mode offline
wandb sync wandb/offline-run-...

# skip wandb entirely, CSV logging only:
python experiments/run_cartpole.py --critic c51 --wandb-mode disabled
```

If `--wandb-mode online` is used without a resolvable login, the run automatically
falls back to offline mode with a printed note rather than blocking on an interactive
login prompt (so it's always safe to leave as the default even in a fresh
environment). If the `wandb` package isn't installed at all, training proceeds with
CSV-only logging and a one-line notice -- wandb is never a hard dependency for
actually running an experiment.

Full config (`gamma`, learning rates, `evar.alpha`/`x_min`/`x_max`, env id, etc.) is
logged as `wandb.config` / visible via `run_config_extra` in `agents/base.train`.

## Notes / next steps

- `TrainConfig.evar` (`EVaRConfig` in `risk/evar.py`) exposes `x_min`/`x_max`, the
  compact search interval `I` from Assumption 1 of the paper - set from each env's
  reward scale in the experiment scripts (`ENV_PRESETS` in `run_invpend.py`).
- The critic uses a Polyak-averaged target network (`target_tau` in `TrainConfig`) for
  TD-target stability, standard for both C51 and IQN.
- These are preliminary, single-seed runs meant to validate the approach end-to-end
  (`python experiments/run_cartpole.py --episodes 20` finishes in seconds and is a good
  smoke test). Longer runs, multi-seed averaging, and hyperparameter sweeps (analogous
  to the alpha-sensitivity and learning-rate studies in the paper, Section 3.6) are the
  natural next step before comparing against the SPSA baseline in `../EVaR-SA` -- the
  per-update CSVs / wandb logs above are sized for exactly that kind of analysis.
