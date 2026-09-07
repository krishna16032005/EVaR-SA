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
| C2. It is cheaper | **Settled analytically, by decision.** SPSA spends `2*N_t` rollouts per policy update purely to estimate a scalar; the distributional form spends one backprop. The paper's own reported numbers stand for the SPSA side, so no head-to-head is run |
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

SPSA head-to-head at a matched sample budget will **not** be run. C2 is argued from
the structure instead: `2*N_t` rollouts per update against one backprop, with the
paper's published numbers standing for the SPSA side. The gridworld remains the
place it *would* have been cheap (three-step episodes) if that is ever revisited.

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
2. ~~**SPSA (the paper's own code)** -- same env, same sample budget.~~ **Dropped.**
   C2 is argued analytically (see the claims table). The code now lives in
   `../../legacy/` if this is revisited.
3. **Risk-neutral A2C with the same nets** -- isolates the EVaR operator from the
   distributional critic. Needs a `--risk-objective {evar,mean}` switch (not yet
   implemented).
4. **Fixed-beta entropic / exponential utility** -- EVaR *is* beta-optimized
   entropic risk, so this shows the dual solve earns its cost.
5. **IQN with distortion measures** (Dabney et al. 2018: CVaR, Wang, CPW) -- the
   standard risk-sensitive distributional RL comparison, and what "SOTA baseline"
   means to this audience.
6. **CVaR-AC** -- harness available at `../../legacy/swimmer_cvar.py` and
   `../../legacy/gridworld/`; port the objective, reuse the harness.
7. **DSAC / risk-sensitive SAC** -- ~~only once continuous-control envs are the
   focus~~. Built (`agents/dsac_evar.py`), and it is now *our own learner* rather
   than an external baseline: IQN action-value critic, twin critics with the min
   taken over the **risk** value, tanh-squashed actor, auto entropy temperature.
   Still off-policy, so it must be compared on env steps, not updates.

## The environment ladder

This section records *what each environment was tried for and what it settled*. For
what to run next and in what order, see "Where to go from here" below -- that
supersedes the ordering here.

| # | Environment | Purpose | Verdict |
|---|---|---|---|
| 1 | Gridworld (lottery) | exact EVaR; C1 and C3 against the true optimum | **works**, the reference testbed |
| 2 | CartPole | C1 | **structurally null** -- see below |
| 3 | Pendulum | plumbing for continuous control | **works as plumbing**; no return spread, so ties are expected |
| 4 | Safety-Gymnasium (`lambda = 0`) | stochastic hazards, catastrophic events | learner **works**; environment **cannot discriminate** at `lambda = 0` (sd(Z) under 1% of return scale) |
| 5 | InvertedPendulum / Double | the source paper's environments | not re-run; superseded by the Safe Velocity suite for comparability |
| 6 | Swimmer, HalfCheetah | comparable to the paper's published SPSA numbers | pending, and only worth running with the stochastic variants |
| 7 | Deliberately stochastic variants | escape the stochasticity trap | **not yet built** -- now a priority, see below |

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

## SafetyPointGoal1: the learner works, the environment does not discriminate

300k steps, cost priced at zero, with the matched risk-neutral control:

| arm | final return | cost |
|---|---|---|
| `mean` | 26.99 | 56.30 |
| `evar` alpha=0.1 | 27.39 | 49.30 |

**The learner solves it.** That is the result to keep: the n-step A2C sat at
random-policy reward here for 1500 episodes, and this reaches ~27 and holds. The
continuous-control ladder is unblocked.

**The environment cannot discriminate risk attitudes**, and this is now measured
rather than suspected. A probe run logging the shape of `Z(s,a)` during training:

| step | return | x* | sd(Z) | top-mass | at-bound |
|---|---|---|---|---|---|
| 5,000 | -0.00 | 0.0451 | 0.0754 | 0.017 | 0.00 |
| 20,000 | 0.49 | 0.0084 | 0.0218 | 0.019 | 0.00 |
| 35,000 | 10.80 | 0.0122 | 0.0228 | 0.017 | 0.00 |

Read it in three parts. `at-bound` holds at 0, so the dual solve is finding
interior optima and EVaR is not degenerating into fixed-beta entropic utility.
`top-mass` sits at 0.017 against a floor of 1/K = 0.0156, so the distribution is
spread rather than piled at its maximum and EVaR is a genuine tail average. Both
operator checks pass.

Then `sd(Z) ~ 0.022` against an episode return reaching ~10 -- **under 1% spread**.
The per-state return distribution is nearly deterministic, so there is almost
nothing for any risk attitude to trade, and the 1.5% gap between the arms is what
that looks like. This is the stochasticity trap this document already warned about
for MuJoCo, now confirmed on the Safety-Gymnasium task that was supposed to escape
it. Pricing cost at zero is part of the cause -- it removes the risk/reward tension
by construction -- but the deeper point stands: navigation with dense shaped reward
is not a risky task.

The consequence for the ladder: `cost_penalty > 0` is not an optional refinement to
try later, it is the only thing that makes these environments risk experiments at
all. `lambda` calibration moves back to the critical path, and now it can actually
be done, because there is finally a policy to calibrate against rather than a
random walk.

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

## Where to go from here

Written after the DSAC learner landed and the two reports were compiled. The short
version: **the method is finished as an engineering problem and unstarted as an
empirical one.** Everything below is about producing a comparison that means
something.

### Step 0: the screening test, before any full run

The single most useful thing learned from SafetyPointGoal1 is that an environment
can look ideal and be structurally incapable of supporting the claim. Cost is
non-zero, hazards are stochastic, episode returns reach ~27 -- and the critic's
per-state return distribution still has `sd(Z) ~ 0.022`, **under 1% of the return
scale**. With no spread there is no tail, and every risk attitude ties by
construction.

That is now a cheap, mechanical pre-check rather than a discovery:

```
python experiments/run_dsac.py --env <ENV> --risk evar --alpha 0.1 \
    --total-steps 40000 --log-every 5     # then read update/z_sd_mean
```

| Reading | Meaning | Action |
|---|---|---|
| `at_bound_frac > 0` | dual solve pinned; EVaR is degenerate | fix before interpreting anything |
| `top_mass_frac >> 1/K` | critic distribution piled at its maximum | EVaR is reporting `max(Z)`, not a tail |
| `z_sd_mean / |return| < ~5%` | no spread to trade | **do not run the alpha sweep** |
| `z_sd_mean / |return| > ~10%` | usable tail | proceed |

**Run this on every candidate environment before spending GPU time on it.** It costs
about 10 minutes and it is the difference between a null result and a wasted week.

### The asymmetry that decides which benchmarks are worth using

This deserves stating plainly because it cuts against the obvious choices, and it is
the main reason the safe-RL suites have not delivered.

**Almost all risk-sensitive and safe-RL benchmarks are built for risk *aversion*.**
Safety-Gymnasium, GUARD, WCSAC, the CMDP literature generally: they add *downside*
hazards and ask the agent to avoid them. Constraint satisfaction is the goal, and
the interesting tail is the bad one.

This method is risk-*seeking*. It needs an **upside** worth chasing -- a return
distribution with a good tail that a risk-neutral agent leaves on the table. A
hazard field supplies variance in the wrong direction: avoiding it is simply
correct, and no `alpha` makes recklessness pay.

So the environment must have one of these properties:

1. **A genuine bimodal payoff.** A risky branch that beats the safe one *when it
   works*. Pricing cost (`r_eff = r - lambda*c`) manufactures exactly this in
   Safety-Gymnasium: the short route past the hazards pays well when it gets away
   with it, the detour is safe and mediocre. This is why `lambda > 0` moved from
   optional refinement to the critical path.
2. **Exogenous stochasticity the agent cannot remove.** Opponents, randomized
   dynamics, heavy-tailed rewards. If the only randomness is the agent's own
   exploration noise, `sd(Z)` collapses as the policy converges -- which is exactly
   what the probe table shows happening.
3. **A distribution whose shape changes during training.** This is where an
   *adaptive* tilt should beat a fixed distortion, and therefore where the
   contribution lives. A fixed CVaR or Wang parameter is chosen once; EVaR re-solves
   its dual against whatever distribution it is handed.

Property 3 is the one to design for. Properties 1 and 2 are necessary; 3 is what
makes the result *ours* rather than a tie with CVaR.

### Environment shortlist, in priority order

| # | Environment | Why | Cost | Gate |
|---|---|---|---|---|
| 1 | **SafetyPointGoal1/2 with `lambda` swept** | Already running; pricing cost is the cheapest way to manufacture bimodality. Calibrate `lambda` against a *trained* policy -- the earlier attempt failed only because it calibrated against a random walk | ~30 min per `lambda` | Step 0 |
| 2 | **Stochastic MuJoCo variants** (per-episode randomized mass/friction, heavy-tailed reward perturbation) | Directly attacks the stochasticity trap. The spread is ours to set and, critically, to *report* -- a reviewer checks this first | ~1 h per config | Step 0 |
| 3 | **GUARD interactive tasks** (Chase, Defense) | The one suite whose tasks are *genuinely* stochastic rather than hazard-static: an opponent supplies exogenous variance that survives policy convergence. 8 agents x 4 tasks x 8 constraints gives a systematic difficulty sweep | integration effort | Step 0 on one instance first |
| 4 | **Safe Velocity** (Hopper, HalfCheetah, Swimmer, Walker2d, Ant) | What recent risk-sensitive papers report, so it buys direct comparability. Expect a tie unless combined with (2) -- stock MuJoCo is near-deterministic | ~4 h per env, 10 seeds | run *with* (2) |
| 5 | **Lottery gridworld, discriminating configs** | Already shows 5 of 6 measures selecting distinct optima. The place to demonstrate adaptive-`beta` against fixed distortions where ground truth is exact | minutes | none |

Deliberately **not** prioritised: Atari with distortion measures (risk-averse
framing, and the return distributions are dominated by score scale), and offline
suites (D4RL/ORAAC/CODAC) -- a different problem setting that would need its own
algorithm.

### SOTA baselines, and what each one isolates

Every baseline must share the critic, actor, data and seeds, and be run against the
`--risk mean` control. A comparison that does not hold those fixed cannot attribute
a difference to the risk measure.

| Baseline | Isolates | Status |
|---|---|---|
| `--risk mean` | the risk measure vs the algorithm | **done**, built in |
| Fixed-`beta` entropic utility | **whether the dual solve earns its cost.** EVaR *is* `beta`-optimised entropic risk, so this is the most important single baseline and the one a reviewer will demand | implemented, not yet run head-to-head |
| IQN + distortion (CVaR, Wang, CPW) | the standard risk-sensitive distributional comparison | implemented, needs the sweep |
| CVaR-AC | an actor-critic built around a different coherent measure; harness at `../../legacy/` | to port |
| DSAC proper | our learner against its published form, to show the risk machinery costs nothing | to run |
| WCSAC | the safety-constrained framing, for the Safety-Gym audience | to implement |

Note on GUARD: its own baselines (TRPO, CPO, PCPO, TRPO-Lagrangian/FAC/IPO/SL/USL)
are **constraint-based and on-policy**, not risk-measure-based. GUARD is valuable to
us as an *environment suite*, not as a source of directly comparable numbers. Do not
promise a GUARD baseline table; promise GUARD environments.

### The contribution, stated as an experiment

Adaptive `beta` is the claim. It needs an experiment that a fixed distortion cannot
pass, and the scale-adaptive interval already found this session is the first
concrete instance: `x* = 1/beta*` has units of return, so the tilt strength *must*
track the distribution or EVaR silently degenerates into fixed-`beta` entropic
utility (measured error up to 1100%).

The experiment that follows from that:

> Take an environment whose return distribution **changes scale or shape during
> training** -- which is the normal case, since returns grow as the policy improves.
> A fixed-`beta` entropic utility is tuned once and is therefore mis-scaled for most
> of training. EVaR re-solves `beta` every update. Measure both, report `x*` over
> training alongside return, and show the fixed-`beta` arm degrading exactly where
> the return scale moves away from its tuning point.

This is falsifiable, it is cheap, and it uses machinery that already exists and is
already instrumented. **If it works, it is the paper.** If EVaR ties fixed-`beta`
here too, that is a decisive negative result and worth knowing early.

### Ordered sequence, with gates

1. **`lambda` sweep on SafetyPointGoal1** (0, 0.05, 0.1, 0.25), screening each with
   Step 0. *Gate:* does any `lambda` lift `z_sd_mean` above ~10% of the return
   scale? If none does, stop using this environment for risk claims and record it.
2. **Fixed-`beta` vs EVaR, on whichever setting passes (1).** This is the
   contribution experiment above. Report `x*` trajectories, not just returns.
3. **Stochastic MuJoCo variant** with reported spread; re-run (2) there. Two
   independent settings is the minimum for a claim.
4. **Full risk-measure panel** (mean, EVaR, CVaR, Wang, CPW, fixed-entropic,
   mean-variance) on the settings that passed, 10 seeds.
5. **GUARD Chase/Defense integration**, screened on one instance before committing
   to the matrix.
6. **Safe Velocity suite** for comparability, run *with* the stochastic variants
   from (3) rather than stock.

Steps 1-2 are days, not weeks, and they decide whether there is a paper. Everything
from 3 on is scale-up and should not start before 2 returns.

## Statistics for the paper

10 seeds, and report IQM with stratified bootstrap CIs (rliable) rather than
mean ± CI. The current continuous-control numbers are **1-2 seeds and should be read
as smoke tests, not evidence** -- both reports say so explicitly, and so should any
draft.

Costing this correctly matters for planning, and the old note here was wrong. The
A2C runs were CPU-bound and packed 8 to a box; DSAC is GPU-resident and runs at
49-75 steps/s with EVaR. A 300k-step run is ~1.5 h, so a 10-seed x 2-arm comparison
on one environment is ~30 GPU-hours, or overnight with a few in parallel. Budget per
environment, not per run, and screen with Step 0 first so that budget is never spent
on an environment that cannot discriminate.
