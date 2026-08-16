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

**C3 is answered, and it was not an approximation gap.** It was a formulation
error, which is a far better position to write from.

Measured on the lottery gridworld (`evar_deeprl/envs/lottery_gridworld.py`), where
trajectory EVaR is exactly computable, with a *perfect* critic and greedy policy
improvement -- no learning, so the number is exact:

| advantage | alphas correct | worst regret |
|---|---|---|
| `r + EVaR(Z(s'))` -- what the code did | 1 of 7 | **86.35%** |
| `EVaR(r(s,a) + Z(s'))` -- action-value | **7 of 7** | **0.00%** |

The state-value form applies the risk tilt only to the *future* return, so the
immediate reward enters through its conditional mean. At a terminal step `Z(s')` is
degenerate, alpha cannot enter at all, and the comparison is safe-reward against
lottery-*mean*; that makes the previous step deterministic and the collapse
propagates back to the start. The fixed point is the risk-neutral policy at every
alpha.

Moving the reward inside the tilt is not sufficient on its own: EVaR is
translation-equivariant, so `EVaR(r + Z(s'))` with a *sampled scalar* `r` is
identically `r + EVaR(Z(s'))`. The tilt reaches the reward only when the critic
represents the reward's distribution -- hence `Z(s,a)`, implemented as
`C51QCritic` (discrete actions; expected-SARSA bootstrap so the critic evaluates
the current policy rather than an optimistic one).

Caveat worth keeping attached: 0.00% is exact *on this environment*, where segments
are independent and the objective decouples. It establishes that the state-value
form is the source of the bias and that the action-value form removes it here, not
that per-state tilting is exact in general.

SPSA head-to-head at a matched sample budget is still to run, and this is the
environment where it is actually runnable -- episodes are three steps.

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

### The operator itself, and what it took to trust it

The first sweep to run these metrics invalidated the solver rather than the
policy. The projected-Newton inner solve never converged: when the tilted measure
collapses onto one atom, `Var_Qx -> 0`, the curvature `G'' = Var_Qx / x^3`
underflows its clamp, and the step is projected onto a bound and cycles between
the bounds forever. `x*` was whichever bound the step-count parity landed on --
constant at 0.301 across genuinely different return distributions -- and EVaR was
inflated 25-590%, with the error *growing in the scale of returns*, i.e. in the
same direction as C1 itself. It also pinned `x*` to `x_max`, silently degrading
the method into fixed-beta entropic utility at `beta = 1/x_max`: baseline 4 below,
the very thing the dual solve is supposed to beat.

`G` is strongly convex, so `G'` is monotone and a sign change brackets the
minimiser; bisection converges unconditionally and now matches brute-force
minimisation to 0.00%. Gradients come from the envelope theorem rather than an
unrolled solver (checked against finite differences at 5e-9).

Three invariants are worth stating because each one caught a real fault:

- **`E[Z] <= EVaR_alpha[Z] <= max(Z)`.** Logged per eval as
  `eval_evar_within_bounds`. It is what exposed the saturated regime below.
- **`alpha` must exceed the mass on the largest atom**, which for an `n`-point
  empirical measure means **`alpha > 1/n`**. This binds twice and independently:
  on the critic support (`n_atoms = 51` -> `alpha > 0.0196`) and on the evaluation
  sample (`eval_risk_episodes = 50` -> `alpha > 0.02`). The `alpha = 0.01` arm of
  the first clean sweep was invalid in all 500 of its eval rows because of the
  second. `eval_risk_episodes` is now 500, and the run scripts refuse an `alpha`
  below either floor rather than reporting a saturated number.
- **When `p_max >= alpha`, EVaR *is* `max(Z)`** and is returned exactly. This is
  not exotic: integer returns plus an episode cap make ties at the maximum
  routine, and a policy that reaches the ceiling on more than an `alpha`-fraction
  of episodes saturates the measure by construction. `x*` is reported at `x_min`
  so `at_bound` still flags that the estimate is a maximum, not a tail average.

`update/evar_dual_x_at_bound_frac` is the standing tripwire. For `alpha < 1` a
healthy solve holds it at 0; it sat at 1.0 for every update of every run before
the fix.

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
7. **DSAC / risk-sensitive SAC** -- ~~only once continuous-control envs are the
   focus~~. Built (`agents/dsac_evar.py`), and it is now *our own learner* rather
   than an external baseline: IQN action-value critic, twin critics with the min
   taken over the **risk** value, tanh-squashed actor, auto entropy temperature.
   Still off-policy, so it must be compared on env steps, not updates.

## The environment ladder

1. **Gridworld** -- exact EVaR; C3 verification against SPSA and the optimum
2. **CartPole / Pendulum** -- plumbing only. **Settled: CartPole cannot support C1.**
3. **Safety-Gymnasium** -- stochastic hazards, real catastrophic events
4. **InvertedPendulum, InvertedDoublePendulum** -- the paper's envs (phase 3 in the queue)
5. **Swimmer, HalfCheetah** -- head-to-head with the existing SPSA results
6. **Deliberately stochastic variants** -- see below

**Why CartPole is finished as evidence.** Three compounding reasons, and the third
is fatal on its own:

- It is deterministic bar a +/-0.05 initial state, so the return spread EVaR
  measures there is the policy's own exploration noise, not risk in the world.
- Its return is capped at 500 and integer valued, which *censors the upper tail
  the operator is defined on*. Once a policy is decent, more than an
  alpha-fraction of episodes tie at the cap and EVaR saturates onto it.
- **Its risk-neutral and risk-seeking optima are the same policy.** Balancing
  forever maximises the mean, the median and every upper-tail statistic
  simultaneously, so no policy trades mean against tail and `alpha` cannot matter.

The first sweep run with a *correct* solver and correct eval sizing confirms it:
top-decile mean across `alpha` = 0.01, 0.05, 0.1, 0.3, 1.0 came out 121.8, 104.2,
110.9, 48.4, **197.8** -- non-monotone, with the *risk-neutral control highest*,
which is the opposite of C1's prediction. Seed sd was +/-50 to +/-180. That is
what a structurally null experiment looks like, and it is only interpretable
because the operator underneath it had already been verified.

**Safety-Gymnasium, and why cost has to be priced in.** It supplies the
stochastic hazards and catastrophic events CartPole lacks, but out of the box it
reports reward and cost separately and is read risk-aversely, so a risk-seeking
agent sees no reason to avoid a hazard and there is still no tradeoff.
`run_safety.py` folds them into `r_eff = r - lambda*c`, which makes the return
distribution bimodal: the short route past the hazards pays well *when it gets
away with it*, the detour is safe and mediocre. Those are two policies with
different means and different upper tails -- the choice `alpha` is supposed to
govern. `lambda` must be calibrated before any alpha sweep: measured on
`SafetyPointGoal1-v0` over 1000 random steps, raw reward return is ~0.19 (max
3.55) against raw cost ~104 (max 669), so at `lambda = 1` the priced return is
essentially negative cost and the task collapses to pure avoidance.

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

Measured, on 100-episode runs: **4x** per-step throughput single-run (336 -> 1358
steps/s), and **175x** in the regime that actually matters, 8 runs in parallel
(7.5 -> 1314 steps/s per run). The old code had each of 8 processes spawning 21
threads onto 10 cores; the full 30-job sweep went from hours to minutes.

Two further defects, found by auditing the learner rather than the throughput:

- **The C51 support was in the wrong units.** It was written as the undiscounted
  episode return (500 for CartPole, 1000 for InvertedPendulum) but the critic
  represents the *discounted* return, which at `gamma = 0.99` cannot exceed
  `r_max/(1-gamma) ~ 100`. Measured over 10k episodes, discounted returns spanned
  7.7 to 99.3 against a support of [-10, 500]: **~9 of 51 atoms carried any mass**
  (~5 of 51 on InvertedPendulum). EVaR is a tail statistic read off exactly that
  histogram. `discounted_support()` now derives the bounds from `gamma`, giving
  ~42 usable atoms, and it eliminated a real pathology -- terminal TD errors had
  been 200-sigma outliers (`|adv|` p99 ~100 against `adv_std` ~0.5) because a
  9-atom critic could not represent "this state is about to end"; p99 is now 3-6.
- **The entropy bonus is scaled against normalised advantages.** Because
  `raw_advantage` is normalised to unit variance, the policy-gradient term is
  O(1) while a conventional `entropy_coef = 0.01` contributes `0.01 * 0.693 ~
  0.007`, under 1% of the loss. Entropy fell from 0.69 (the maximum for two
  actions) to ~0.09 by a quarter of training at both 1e-3 and 1e-2. At 0.05 it
  holds near 0.25 and prevents *total* collapse -- at 0.01 one seed reached
  entropy 0.00 and return 9 -- but it does not fix the instability.

**The learner still does not solve CartPole**, and this is now the binding
constraint on every experiment downstream. At 2000 episodes, alpha = 1.0, mean
final return is ~150-180 across all three configurations above, with peaks
touching 500 and finals oscillating between 100 and 280. A method whose evidence
rests on comparing tail statistics across `alpha` cannot be evaluated on a
learner with this much seed variance -- the error bars swallow any real effect,
which is exactly what the first clean sweep showed.

Next, in expected-payoff order:

1. **Vectorized envs** (`gymnasium.vector`) -- and this is a *stability* fix, not
   only a speedup, which is why it is now first by a wide margin. The trainer
   currently updates on 16 consecutive steps from a single environment: those
   states are highly correlated, so every gradient is noisy and biased toward
   whatever region the policy happens to be in. Standard A2C uses N envs stepped
   together (16 x 5 rather than 1 x 16) precisely to decorrelate the batch, and
   its absence is the leading explanation for the oscillate-and-collapse pattern
   that survived both fixes above. Requires reworking episode accounting.
2. **Preallocated rollout tensors** instead of per-step list appends + `stack`.
3. **`torch.compile`** on critic and actor -- worth measuring only after (1),
   since compile overhead dominates at the current batch sizes.
4. GPU stays pointless until nets are wide or IQN quantile counts are large; the
   A40 is for the scaled-up phase, not this one.

## The learner that finally does continuous control

`agents/dsac_evar.py`, validated on Pendulum-v1 at 60k steps, 2 seeds, with the
`--risk mean` control sharing every other component:

| arm | seed 0 | seed 1 |
|---|---|---|
| `mean` | -96.99 | -152.44 |
| `evar` alpha=0.1 | -96.36 | -151.92 |

Random policy is ~-1200, so both solve the task; the n-step A2C never left random
on anything continuous. **EVaR tying the risk-neutral control here is the correct
result, not a null one**: Pendulum's return spread is almost entirely policy noise,
so there is no tail to seek and `alpha` should not matter.

Read the table by seed rather than by arm, because that is where the signal is.
Seed variance is 55 return points; the mean-vs-EVaR gap *within* a seed is 0.6 and
0.5. The two arms are not merely close on average, they track each other run for
run -- which is what a correctly wired risk operator does on a distribution with no
tail to exploit, and is much harder to get by accident than a matching average. It is the same reasoning
that retired CartPole, and it is why this run is plumbing validation rather than
evidence. `x*` settles at 0.37-0.41 instead of pinning to a bound, which is the
diagnostic that the dual solve is live.

### Where the wall-clock goes, measured rather than guessed

The EVaR arm ran at 49 steps/s against `mean` at 105, which looks like the risk
solve being expensive. Profiling `SafetyPointGoal1-v0` says otherwise:

| | |
|---|---|
| MuJoCo sim alone | 617 steps/s |
| learner, `mean`, CPU | 40 updates/s |
| learner, `evar`, CPU | 35 updates/s |

The simulator is 30x faster than the learner, so the environment is never the
bottleneck -- and on CPU the EVaR solve costs only **12%**, not 2x. The cost is
therefore not arithmetic but *kernel launches*: on GPU the 256x256 nets are nearly
free and the bisection's ~12 launches per step are what remains. Fixed by cutting
`solver_steps` 30 -> 20 (measured equivalent to 2.5e-7 relative, float32 epsilon)
and solving both twin critics in one stacked call (bit-identical, verified).

**Infrastructure note:** safety-gymnasium 1.0.0 pins `gymnasium 0.28.1` against the
main environment's `1.3.0`, so it lives in its own venv -- which shipped CPU-only
torch. The two environments cannot be merged; the venv needs its own CUDA build.

## The gap that actually blocks the paper

Everything above establishes that the port is *correct*. Nothing yet establishes
that it is *better*. On the discriminating gridworld EVaR reaches 97.9% of its own
optimum -- the operator works -- but it is **statistically indistinguishable from
CVaR and Wang**. A method that merely ties the standard distortion measures has no
claim to make.

So the remaining work is not solver quality. It is: (a) identify the environment
class where optimizing `beta` buys something a *fixed* distortion cannot, and (b)
the adaptive-`beta` mechanism, which is the contribution as such. EVaR's dual is
the only one of these measures that adapts its tilt to the distribution it is
handed; the experiment has to be one where that adaptivity is load-bearing --
which means a return distribution whose shape *changes during training*, since a
fixed distortion is tuned once and cannot follow it.

## Statistics for the paper

10 seeds, and report IQM with stratified bootstrap CIs (rliable) rather than
mean ± CI -- the current normal-approximation band over 5 seeds is indicative
only. Cheap here: these runs are CPU-bound and the box has 10 pinned cores.

## Sequence

Done:

1. ~~Verify the new eval metrics on one short run~~ -- did that, and it caught a
   solver that never converged. See "The operator itself" above.
2. ~~Phase 1 alpha sweep + the `alpha = 1` control -> C1~~ -- ran clean, and
   returned a **structural null**: CartPole cannot support C1. The `alpha = 1`
   control is now exact, and the sweep is worth keeping only as a measurement
   regression test.

3. ~~Gridworld exactness study -> C1 and C3 together.~~ Built, with lotteries that
   pay (all-risky gives up 6% of mean for 9x the best case) and a graded ground
   truth of 3, 2, 2, 2, 2, 1, 0 lotteries across alpha. C3 answered above. C1: with
   the action-value critic, exact regret falls from 4.4-6.4% to **0.1-0.3%** at
   alpha 0.05-0.3, and -- the bigger result -- policy decisiveness goes from
   0.10-0.29 to **0.49-0.50** of a maximum 0.5. The old agent was never really
   choosing.

Now, in order:

4. **The alpha = 0.5 failure mode.** 4 of 5 seeds reach 0.55% regret; one collapses
   to all-safe at 50.93%, which is exactly the state-value operator's fixed point.
   Cause is critic coverage: the unchosen action's value is never trained (measured
   error +46.36 on a never-visited state-action), and once the policy commits at
   p ~ 0.998 the advantage `Q(s,a) - sum_a pi Q` collapses toward zero, so nothing
   corrects a wrong commitment. Raising the entropy bonus does not fix it
   (0.4% -> 0.5% -> 1.4% at 0.05 / 0.2 / 0.5). Needs a coverage mechanism.
5. **An action-value form for continuous actions.** `C51QCritic` is discrete-action
   only, so `run_invpend.py` and `run_safety.py` still carry the state-value critic
   and therefore the C3 defect. Everything queued for Safety-Gymnasium is currently
   built on the broken form. This blocks the whole continuous-control ladder.
6. ~~**A learner that can do continuous control.**~~ **Done** -- see "The learner
   that finally does continuous control" above. What follows was the diagnosis.
   Measured on `SafetyPointGoal1-v0` at `lambda = 0` -- cost priced at zero, so
   pure navigation with no risk machinery involved -- reward converges to
   random-policy level over 1500 episodes and stays: `c51qc` at -0.11 and -0.15,
   `c51` at -0.19 with one seed diverging to -11.68, against a random-policy probe
   of +0.05 to -0.10. The environment and the operator are fine; there is no
   policy. Needed: a proper on-policy update (GAE, clipped objective, several
   epochs per batch) and vectorized envs, which also fixes the correlated-batch
   problem. These benchmarks are normally run at 1e6-1e7 steps with PPO/SAC-class
   algorithms; this is n-step A2C at 800k-1.5M.
7. ~~**Safety-Gymnasium**: calibrate `lambda`.~~ Attempted, and it is blocked on
   (6) rather than on anything about the environment. Worth recording that
   Safety-Gymnasium *does* supply what CartPole could not: cost is non-zero
   (249-704 episodes per run) and the priced return has genuine spread (sd 8.6 to
   45.0). But `lambda` cannot be calibrated against a random walk -- cost rose with
   `lambda` (Goal2: 66.9 -> 92.2 -> 102.1 at 0.1/0.25/0.5) where a policy
   responding to a price would do the opposite. Redo once (6) lands.
8. **SPSA head-to-head** on gridworld at a matched sample budget -> C2.
9. `--risk-objective` switch, then baselines 3 and 4.
10. Phase 3 MuJoCo with a stochastic variant; Swimmer / HalfCheetah at scale.
