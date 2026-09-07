# EVaR-SA

Official code repository for the TMLR paper
**Risk-Seeking Reinforcement Learning via Multi-Timescale EVaR Optimization**
by **Deep Kumar Ganguly**, **Ajin George Joseph**, **Sarthak Girotra**, and **Sirish Sekhar**
(*Transactions on Machine Learning Research*, 2025) —
[OpenReview](https://openreview.net/forum?id=4nbEgNDsii)

— and of the follow-up work porting that objective into distributional deep RL.

---

## Repository layout

| Path | What it is |
|---|---|
| **[`EVaR-DeepRL/`](EVaR-DeepRL/)** | **Active work.** EVaR as a differentiable risk objective inside an off-policy distributional actor–critic (IQN critic + SAC). This is where current development happens. |
| [`legacy/`](legacy/) | The published paper's original SPSA implementation and its experiment scripts, kept for reference and reproduction. See [`legacy/README.md`](legacy/README.md). |

**If you came here from the paper**, the code accompanying the publication is in
[`legacy/`](legacy/) — it has been moved there unchanged, not modified.

---

## The follow-up work: EVaR in distributional deep RL

The paper estimates `J_EVaR(θ) = EVaR_α[R(τ)]` with a two-timescale stochastic
approximation recursion, spending `2·N_t` rollouts per policy update purely to
estimate a scalar, then perturbing `θ` with SPSA because no gradient is available.

A distributional critic supplies an explicit, differentiable return distribution,
so the EVaR dual can be solved directly per state and differentiated — one
backpropagation in place of `2·N_t` rollouts.

Start here:

- **[Practitioner's guide](EVaR-DeepRL/report/practitioner_guide.pdf)** (3 pages) —
  the algorithm, the full hyperparameter block, the headline numbers, and the
  failure modes worth knowing before you run it.
- **[Full status report](EVaR-DeepRL/report/evar_deeprl_report.pdf)** (13 pages) —
  method, measurements, and an explicit closed/refuted/open breakdown.
- **[`EVaR-DeepRL/RESEARCH_PLAN.md`](EVaR-DeepRL/RESEARCH_PLAN.md)** — the working
  plan, including results that were tested and refuted.

### One result worth surfacing here

The natural per-state advantage `r + EVaR(Z(s'))` is **risk-neutral in the reward**.
EVaR is translation-equivariant, so a *sampled scalar* reward passes through the
tilt untouched; at a terminal step `α` cannot enter at all, and the collapse
propagates backwards to give the risk-neutral policy at every `α`. Measured exactly
on a testbed where the trajectory optimum is computable: **86.35% regret** for that
form, **0.00%** once the critic is an action-value `Z(s,a)`.

---

## Abstract (TMLR 2025)

Tail-aware objectives shape an agent's behavior when navigating uncertainty and can
significantly differ from risk-neutral formulations. Risk measures such as **Value
at Risk (VaR)** and **Conditional Value at Risk (CVaR)** have been widely used in
reinforcement learning (RL).

In this work, we study the use of a relatively new **coherent risk measure**, the
**Entropic Value at Risk (EVaR)**, as a **high-return, risk-seeking** objective. We
propose a **multi-timescale stochastic approximation algorithm** (EVaR-SA) to learn
the optimal parameterized EVaR policy. Our approach enables efficient exploration of
high-return tails and provides a robust gradient approximation for optimizing the
EVaR objective.

Theoretical analysis establishes asymptotic convergence and finite-time behavior,
while experiments on discrete and continuous control benchmarks demonstrate that
EVaR policies achieve higher cumulative returns — confirming that EVaR is a
competitive risk-seeking objective in RL.

---

## Citation

```bibtex
@article{
  ganguly2025riskseeking,
  title={Risk-Seeking Reinforcement Learning via Multi-Timescale {EV}aR Optimization},
  author={Deep Kumar Ganguly and Ajin George Joseph and Sarthak Girotra and Sirish Sekhar},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=4nbEgNDsii}
}
```
