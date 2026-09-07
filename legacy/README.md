# Legacy: the TMLR paper's SPSA implementation

This is the code accompanying **Risk-Seeking Reinforcement Learning via
Multi-Timescale EVaR Optimization** (TMLR 2025). It was moved here unchanged when
the repository root was reorganised around the distributional deep RL follow-up in
[`../EVaR-DeepRL/`](../EVaR-DeepRL/). Nothing was edited, and nothing in the active
package imports from this directory.

## Contents

| Path | Contents |
|---|---|
| `_spsa.py` | SPSA driver. **Not standalone** — imports `spsa._defaults`, `spsa._utils`, `spsa.random.iterator` from the `spsa/` package inside `evar_optimization 2/` and `evar_optimization 3/`. |
| `evar_optimization 2/`, `evar_optimization 3/` | Two experiment sets, **not duplicates**: 84 files differ. `hopper_evar.py` / `hopper_cvar.py` exist only in `2`; `invpenddouble_evar.py` and `evar_sensitivity.py` only in `3`. Each carries its own copy of the `spsa/` package. |
| `swimmer_evar.py`, `swimmer_cvar.py`, `halfcheetah_evar.py`, `invpendulum_evar.py`, `invpenddouble_evar.py` | Per-environment MuJoCo experiment scripts. |
| `gridworld/`, `swimmer/` | Gridworld and Swimmer experiment directories, including REINFORCE-based EVaR/CVaR variants. |
| `iterator.py`, `test_swimmer.py`, `steps_to_execute.rtf` | Helpers and the original run notes. |
| `EVaR_TwoTimeScale_SA.ipynb` | The two-timescale SA notebook. |
| `data/` | Recorded objective and return traces (`evar_obj_*.csv`, `evar_ret_*.csv`) from the paper's runs. |

## Why this is kept rather than deleted

Two reasons, and the first is the load-bearing one.

**It is the published artifact.** This repository is the official code repository
for a TMLR paper. Whatever the follow-up work becomes, the code a reader arrives
looking for has to still be here.

**It is the reference for baselines the follow-up may still want.** The plan lists
`swimmer_cvar.py` and the gridworld REINFORCE-CVaR variant as the harness to port
for a CVaR actor–critic comparison.

The sample-efficiency claim (**C2** in the follow-up's terms) is now argued
analytically rather than by a head-to-head run: SPSA spends `2·N_t` rollouts per
policy update purely to estimate a scalar, where the distributional formulation
spends one backpropagation, and the original paper's own reported numbers stand for
the SPSA side. So this code is no longer on the critical path — but it is one `cd`
away if that decision is revisited, which is cheaper than reconstructing it from
git history along with its package dependencies.

## Running any of it

These scripts predate the follow-up's environment and pin their own dependencies.
They expect the `spsa/` package on the path, so run them from inside the
`evar_optimization 2/` or `evar_optimization 3/` directory that contains the copy
you want, in a **separate virtualenv** — the active package requires
`gymnasium >= 1.0`, which these do not target.
