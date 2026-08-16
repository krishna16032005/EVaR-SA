# What the contribution is, and what has to be reproduced to defend it

This is the plan for turning the port into a result. It is written against measured
numbers, not intentions -- every claim below points at a run or a test in this repo.

## Where we actually stand

Six risk measures, one PPO learner, one distributional critic, one common scoring
panel. Only the functional applied to `Z(s,a)` changes between runs.

| risk | gridworld: % of optimal EVaR | Safety: EVaR(0.1) | Safety: cost |
|---|---|---|---|
| mean | 18.2% | 5.34 | 19.9 |
| evar | 98.7% | **1.58** | **158.3** |
| cvar | 99.8% | 6.95 | 111.1 |
| wang | 66.3% | **7.17** | 35.2 |
| entropic | **99.9%** | 2.33 | 28.1 |
| meanvar | 101.4% | 1.00 | 68.2 |

Two facts follow, and they are the whole basis for the contribution:

1. **EVaR is competitive where the return distribution is well estimated and
   collapses where it is not.** The gridworld gives ~30,000 episodes over a 3-step
   horizon; SafetyPointGoal1 gives ~150 over 1000 steps. EVaR degrades hardest
   because the dual solve tilts onto the extreme upper tail -- the least-sampled
   region of the critic -- so tail estimation error feeds straight into the
   advantage. CVaR *averages* the tail and is more robust; Wang spreads weight wider
   still and survives best.
2. **The dual solve is not currently earning its cost.** Fixed-beta entropic risk
   ties EVaR on the gridworld (99.9% vs 98.7%). Since EVaR *is* entropic risk with
   beta optimised, a tie means the optimisation is buying nothing there.

## What the contribution therefore has to be

Not "adaptive beta". EVaR's beta is already per-state by construction -- that is what
the dual solve does -- and online risk adaptation is prior art (DRL-ORA, below). The
defensible gap is narrower and is exactly what the measurements point at:

> **A dual solve that is robust to tail estimation error**, so EVaR keeps its
> gridworld behaviour in the sampling regime where risk-sensitive RL actually
> matters -- plus the trajectory-consistency fix that per-state risk needs in
> general.

Three components, in dependency order:

1. **Robust dual solve.** The current solve minimises `G(x)` against the critic's
   raw upper tail. Candidates, cheapest first: shrink the tail toward the bulk before
   solving; solve against a lower-confidence bound on the tail rather than the point
   estimate; put a trust region on `x*` between updates so it cannot chase critic
   noise. The gridworld-vs-Safety gap is the measurement that motivates this, and
   the same pair of environments tests whether it worked.
2. **Trajectory consistency.** Per-state risk is a proxy for trajectory-level risk
   and can be arbitrarily suboptimal in general (Zhou et al. 2023; Wang et al. 2024).
   Measured here with a perfect critic, the state-value advantage costs up to 86.35%
   of the optimum; the action-value form recovers it exactly -- but on an environment
   with independent segments, which is not a general claim. State augmentation with
   accumulated reward (Pires et al. 2025) is the literature's fix and is testable on
   the gridworld against ground truth.
3. **Tail-aware critic.** A fixed-support C51 spends atoms uniformly; the tail is
   where the decision is made. An IQN-style quantile critic, or non-uniform atom
   placement, targets the same failure from the representation side.

The honest framing of the result: *per-state EVaR is competitive with the best
distortion baselines when the return distribution is well estimated, and the
contribution is what makes that hold when it is not.*

## Papers to reproduce, and why each one

Ordered by how much each is worth relative to the work it costs.

**Tier 1 -- same critic, functional swap. Already done.**
Cheap because they share this harness completely.

| paper | what it gives us | status |
|---|---|---|
| Dabney et al. 2018, *Implicit Quantile Networks* -- CVaR, Wang, CPW distortions | the standard distortion baselines; Wang is currently the best method on Safety | done as functionals |
| entropic risk at fixed beta | the ablation of the dual solve; EVaR must beat this or the contribution is empty | done |
| Markowitz / mean-variance (also reported by DSAC) | classical risk objective | done |

**Tier 2 -- different learners. Each is a real implementation.**

| paper | why it is needed |
|---|---|
| Ma et al. 2020, *DSAC: Distributional Soft Actor Critic* (arXiv 2004.14547) | the SOTA risk-sensitive continuous-control comparison; off-policy, so it is also a different sample-efficiency regime and must be compared on env steps |
| Dabney et al. 2018, IQN as an *algorithm* (not just distortions) | quantile critic rather than fixed-support categorical -- directly targets the tail-estimation failure diagnosed above |
| Luo et al., *DRL-ORA* (arXiv 2310.05179) | online risk-level adaptation via total-variation minimisation with Follow-The-Leader. **The closest prior art to "adaptive risk"** -- it must be cited and compared, and it is why "adaptive beta" alone is not the contribution |
| Schubert & Eimer, *Automatic Risk Adaptation in Distributional RL* | the other adaptive-risk-level line |
| Zhou et al. 2023 / Wang et al. 2024 / Han et al. 2025 -- trajectory-level vs per-state risk | establishes that per-state risk is a proxy; our 86.35% measurement is an instance of their claim |
| Pires et al. 2025 -- stock-augmented distributional RL | the state-augmentation fix for trajectory-level risk; component 2 above |
| Greenberg et al., CVaR mixture policy parameterisation (arXiv 2403.11062) | sample-efficiency technique for CVaR optimisation, transferable to EVaR |

**Explicitly not a baseline:** SPSA. It is the method being replaced, not a rival.

## Environments

Chosen so a claim is falsifiable rather than flattering.

**Where we have ground truth (keep as the diagnostic bed)**
* `LotteryGridWorld` -- optimum computable for every alpha; ~30k episodes per run.
  The only place a measure can be scored against truth rather than another curve.

**Standard risk-sensitive benchmarks (what reviewers expect)**
* **Safety-Gymnasium navigation** -- Goal / Button / Push / Circle at levels 1-2,
  agents Point / Car / Racecar / Doggo / Ant. Level 0 has zero cost and is unusable
  for a risk objective. Measured hazard activity per 500 random steps: Goal2 mean
  cost 23 (max 129), Circle1 50 (max 403), CarGoal1 17.9, Button1 2.5, Goal1 and
  Push1 ~0.
* **Safe Velocity suite** -- Hopper, HalfCheetah, Swimmer, Walker2d, Ant with
  velocity-threshold costs. This is what the recent risk-sensitive papers report, so
  it is the comparison reviewers will look for.
* **Risk-sensitive D4RL** -- stochastic catastrophic penalties (e.g. HalfCheetah
  with a failure probability above a velocity threshold). Offline, so it is a
  different regime, but it is where several 2025 papers benchmark.

**Where risk-seeking specifically should shine**
The upper tail has to be *exploitable*, which rules out most of the above -- they
are built for risk-*aversion*. Candidates: hazard-adjacent shortcuts in
Safety-Gymnasium (already what `--cost-penalty` sets up), heavy-tailed reward
perturbations on MuJoCo, and portfolio/allocation tasks, which are natively
heavy-tailed and are where the EVaR literature lives.

**Isaac Gym / Isaac Lab: not now.** It buys throughput, not risk structure, and its
tasks are near-deterministic in the same way stock MuJoCo is. Throughput is not the
bottleneck -- the learner and the tail estimation are.

## Order of work

1. Robust dual solve, tested on the gridworld/Safety pair that exposed the problem.
2. IQN-style quantile critic (attacks the same failure from the representation side).
3. DSAC, as the SOTA continuous-control comparison.
4. Safe Velocity suite, since that is what the recent papers report.
5. State augmentation for trajectory consistency, verified on the gridworld.

## Sources

- DSAC: https://arxiv.org/abs/2004.14547
- DRL-ORA: https://arxiv.org/abs/2310.05179
- CVaR mixture policy parameterisation: https://arxiv.org/pdf/2403.11062
- Beyond CVaR / static spectral risk measures: https://arxiv.org/pdf/2501.02087
- Static spectral risk, online and offline: https://arxiv.org/html/2507.03900
- Risk-sensitive policy with distributional RL: https://arxiv.org/pdf/2212.14743
