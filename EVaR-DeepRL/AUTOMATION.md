# The experiment lifecycle: push → runs on the GPU box → results on wandb

The point of this file: you should never have to SSH anywhere to run an
experiment. Edit a queue file, push, and read the dashboard.

```
   laptop                     github                gpud.model                wandb
 edit queue.toml  ──push──▶   main   ◀──pull(60s)── runner daemon ──launch──▶ live charts
                                                    (inside evar-gade)         + email alerts
```

## The loop

1. Edit [`experiments/queue.toml`](experiments/queue.toml) — add a job, change a
   hyperparameter, flip `enabled`.
2. `git commit && git push`.
3. Within 60 seconds the runner inside the container pulls, sees jobs it has
   never run, and starts them (up to `max_parallel` at a time).
4. Runs stream to **https://wandb.ai/deepg98-technical-university-of-munich/evar-deeprl**,
   grouped so seeds of one config sit together.

No inbound connection to the server is needed — the container polls outward,
which is what makes this work behind the TUM firewall without any port opening
or CI runner registration.

## Adding an experiment

```toml
[[job]]
name = "cartpole-c51-alpha005"     # appears in the job id and log filename
script = "experiments/run_cartpole.py"
seeds = [0, 1, 2, 3, 4]            # one run per seed
group = "cartpole-c51-alpha0.05"   # wandb group: seeds compared on one plot
tags = ["c51", "ablation"]
enabled = true
[job.args]                         # becomes CLI flags; n_steps -> --n-steps
critic = "c51"
alpha = 0.05
episodes = 500
```

That's five runs. Push it and walk away.

## What re-runs, and what doesn't

A job's identity is a **hash of its own definition** (name, script, args, seed).
That gives the behaviour you want by default:

| You do this | What happens |
|---|---|
| change `alpha`, `episodes`, any arg | new identity → runs |
| add a seed | that seed runs, existing seeds don't |
| push unrelated code changes | nothing re-runs |
| container restarts / server reboots | finished work is not repeated |
| want to repeat an unchanged job | bump `rerun = 1` on it |

So pushing a bugfix to the training code does *not* silently redo your whole
sweep — and it also means a code fix alone will not re-run affected jobs, which
is the tradeoff. Bump `rerun` when a fix invalidates earlier results.

## Controlling it

```toml
[settings]
paused = false        # true = finish what is running, start nothing new
max_parallel = 3      # concurrent experiments (10 cores are pinned to this box)
```

Both take effect on the next poll after you push — `paused = true` is the
emergency brake and needs no SSH.

## Checking in

From your laptop, without entering the box:

```bash
ssh evar-box 'cd ~/EVaR-SA/EVaR-DeepRL && python automation/runner.py --status'
ssh evar-box 'tail -20 ~/EVaR-SA/EVaR-DeepRL/automation/logs/runner.log'
ssh evar-box 'tail -30 ~/EVaR-SA/EVaR-DeepRL/automation/logs/<job-id>.log'
```

`--status` prints every queued job as `pending` / `running` / `done` / `failed`
with its log path. But the dashboard is the better view — the runner exists so
that wandb is where you look.

## Getting results without watching

In **wandb → Settings → Alerts**, enable "Run finished" and "Run crashed". You
then get email (and Slack, if connected) per run, which is the "write the paper,
receive results" mode. The iOS app pushes the same events.

Worth knowing: a crashed *container* does not produce a crashed *run* — wandb
just goes quiet. If the dashboard stops updating, check
`ssh gpud 'sudo docker ps | grep evar'`.

## Turning it on

The daemon runs only when the container is started with `EVAR_AUTORUN=1`:

```bash
EVAR_AUTORUN=1 WANDB_API_KEY=<key> ./docker/bootstrap_gpud.sh    # via bootstrap
```

or on a raw `docker run`, add `-e EVAR_AUTORUN=1`. Without it you get a plain
dev box and nothing starts by itself. Run it by hand any time with
`python automation/runner.py --once`.

## Reusing this on the next project

The runner assumes only that your training script accepts `--seed`, `--device`,
and the `--wandb-*` flags this repo's scripts already take. To reuse it:

1. Copy `automation/runner.py` and `experiments/queue.toml` into the new repo.
2. Point `script = ` at the new entry points.
3. Build that project's box with `EVAR_AUTORUN=1` (its own image tag, container
   name and port — see [DOCKER.md](DOCKER.md)).

Everything else — job identity, state, logs, parallelism, the pause switch — is
project-agnostic.

## Why polling and not GitHub Actions

A self-hosted Actions runner would give the same push-to-run loop with nicer UI,
but it needs a registration token, an always-running agent registered to the
repo, and it executes arbitrary pushed code on a shared university machine. The
poller is ~200 lines, needs no secrets on GitHub, and fails safe: if the network
drops, it keeps running the checkout it already has. If you later want Actions,
the runner script stays useful as the thing the workflow calls.
