# From SPSA to distributional deep RL: the plan

Goal: restate the TMLR result (`Risk-Seeking RL via Multi-Timescale EVaR
Optimization`, SPSA-based) as a distributional deep RL method, with evidence that
survives review and a codebase that scales to harder environments.

## What changes, and what must be proved

The paper optimizes `J_EVaR(theta) = EVaR_alpha[R(tau)]` with a two-timescale SPSA
recursion: `2 * N_t` rollouts per policy update, zeroth-order gradient. The deep
version replaces that with a distributional critic plus a differentiable EVaR
solve, so one backprop replaces `2 * N_t` rollouts.

Three claims have to be defended:

| Claim | Evidence needed |
|---|---|
| C1. It optimizes the same objective | measured `EVaR_alpha` of *trajectory returns* rises; `alpha` monotonically moves the tail; `alpha = 1` recovers risk-neutral |
| C2. It is cheaper | EVaR vs env-steps against SPSA at a matched sample budget |
| C3. The approximation is sound | per-state tilting vs the paper's trajectory-level tilting (Eq. 5) -- quantified where the exact optimum is computable |

C3 is the one a reviewer will press. The README already concedes the deep method
tilts per-state through the critic rather than per-trajectory. Answer it on
**gridworld**, where trajectory EVaR is exactly computable: compare EVaR-AC, SPSA,
and the analytic optimum. A small measured gap is a defensible approximation; an
unmeasured one is a hole.

## Measuring the objective, not a proxy

Until now, evaluation reported mean return -- which scores a risk-seeking method
on the thing it is entitled to sacrifice. `objective_metrics` in
`agents/base.py` now reports, from a *stochastic* eval pass (the objective is
defined over trajectories drawn from `pi_theta`, so a deterministic pass has no
return spread and its EVaR collapses onto the mean):

- `eval_evar` -- EVaR_alpha of empirical returns, same solver as the critic
- `eval_evar_dual_x` -- solved `x* = 1/beta*`, the paper's Fig. 3 analogue
- `eval_cvar_upper`, `eval_top_decile_mean`, `eval_return_p90/p10`

Upper-tail statistics, deliberately: risk-*seeking* targets the good tail, and
reporting conventional lower-tail CVaR would score the method backwards.

## Baselines, in the order they earn their place

1. **alpha = 1.0 control** (in the queue now) -- must match risk-neutral. Free, and
   invalidates everything if it fails.
2. **SPSA (`../_spsa.py`, the paper's own code)** -- same env, same sample budget.
   This is C2, and the reason the paper exists.
3. **Risk-neutral A2C with the same nets** -- isolates the EVaR operator from the
   distributional critic. Needs a `--risk-objective {evar,mean}` switch (not yet
   implemented).
4. **Fixed-beta entropic / exponential utility** -- EVaR *is* beta-optimized
   entropic risk, so this shows the dual solve earns its cost.
5. **IQN with distortion measures** (Dabney et al. 2018: CVaR, Wang, CPW) -- the
   standard risk-sensitive distributional RL comparison, and what "SOTA baseline"
   means to this audience.
6. **CVaR-AC** -- you already have `../swimmer_cvar.py` and
   `../gridworld_reinforce_cvar.py`; port the objective, reuse the harness.
7. **DSAC / risk-sensitive SAC** -- only once continuous-control envs are the
   focus; off-policy, so it is a different sample-efficiency regime and must be
   compared on env steps, not updates.

## The environment ladder

1. **Gridworld** -- exact EVaR; C3 verification against SPSA and the optimum
2. **CartPole / Pendulum** -- plumbing and alpha-sweep sanity only
3. **InvertedPendulum, InvertedDoublePendulum** -- the paper's envs (phase 3 in the queue)
4. **Swimmer, HalfCheetah** -- head-to-head with the existing SPSA results
5. **Deliberately stochastic variants** -- see below

**The stochasticity trap.** MuJoCo is near-deterministic: with only initial-state
randomness the return distribution is nearly a point mass, so EVaR ≈ CVaR ≈ mean
and every risk attitude ties. Any "risk-seeking wins" claim on stock MuJoCo is
unfalsifiable. Add and *report* real return spread: action noise, per-episode
randomized mass/friction, or heavy-tailed reward perturbation. This is the first
thing a risk-sensitive-RL reviewer checks.

## Performance work

Done:

- **one torch thread per run** (`torch_threads`) -- tiny nets gain nothing from
  intra-op parallelism, and N concurrent runs each spawning one thread per core
  is the worst case. One thread per run, many runs in parallel; `max_parallel`
  raised 3 -> 8.
- **one actor forward per env step** -- `act_with_entropy` replaces `act()` then
  `entropy()`, which built the distribution (and ran the net) twice per step.
- **one critic forward and one Newton solve per update** -- `s` and `s'`
  concatenated; the EVaR solve is the costliest part of the update.

Next, in expected-payoff order:

1. **Vectorized envs** (`gymnasium.vector`): N envs stepped together turns N
   Python-level policy calls into one batched forward. This is the big one --
   expect several times the throughput -- and it also decorrelates the on-policy
   batch. Requires reworking episode accounting in the train loop.
2. **Preallocated rollout tensors** instead of per-step list appends + `stack`.
3. **`torch.compile`** on critic and actor -- worth measuring only after (1),
   since compile overhead dominates at the current batch sizes.
4. GPU stays pointless until nets are wide or IQN quantile counts are large; the
   A40 is for the scaled-up phase, not this one.

## Statistics for the paper

10 seeds, and report IQM with stratified bootstrap CIs (rliable) rather than
mean ± CI -- the current normal-approximation band over 5 seeds is indicative
only. Cheap here: these runs are CPU-bound and the box has 10 pinned cores.

## Sequence

1. Verify the new eval metrics on one short run (**the queue should not be
   approved before this**)
2. Phase 1 alpha sweep + the `alpha = 1` control -> C1
3. Gridworld exactness study -> C3
4. `--risk-objective` switch, then baselines 3 and 4
5. Phase 3 MuJoCo, with a stochastic variant, vs SPSA -> C2
6. Vectorized envs, then Swimmer / HalfCheetah at scale
