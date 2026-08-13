"""Queue-driven experiment runner: git pull -> launch new jobs -> log to wandb.

The loop this closes: you edit `experiments/queue.toml` on your laptop, push, and
within one poll interval the container on the GPU server pulls the change and
starts every job it has not run before. Results appear on the wandb dashboard;
nothing needs to be started by hand and no inbound connection to the server is
required (which matters on a firewalled university box).

Job identity is the sha1 of the job's own definition -- name, script, args, seed
-- so editing a job's hyperparameters makes it a *new* job that runs, while
leaving it alone means it is never re-run. To deliberately repeat an unchanged
job, bump its ``rerun`` counter.

State lives under ``automation/state/`` on the bind-mounted repo, so it survives
container restarts: a marker file per job, plus the exit code once it finishes.

Usage:
    python automation/runner.py --loop          # daemon (what the container runs)
    python automation/runner.py --once          # single pass, for testing
    python automation/runner.py --status        # what has run / is running
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(os.environ.get("EVAR_REPO", Path(__file__).resolve().parent.parent))
QUEUE_FILE = REPO / "experiments" / "queue.toml"
STATE_DIR = REPO / "automation" / "state"
LOG_DIR = REPO / "automation" / "logs"


def log(msg: str) -> None:
    print(f"[runner {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------- queue --
def load_queue() -> tuple[dict, list[dict]]:
    """Parses queue.toml into (settings, jobs), one job per (spec, seed) pair."""
    if not QUEUE_FILE.exists():
        return {}, []
    with open(QUEUE_FILE, "rb") as f:
        raw = tomllib.load(f)

    settings = raw.get("settings", {})
    defaults = raw.get("defaults", {})
    jobs: list[dict] = []

    for spec in raw.get("job", []):
        if not spec.get("enabled", True):
            continue
        merged = {**defaults, **spec}
        args = {**defaults.get("args", {}), **spec.get("args", {})}
        for seed in merged.get("seeds", [0]):
            jobs.append(
                {
                    "name": merged["name"],
                    "script": merged["script"],
                    "device": merged.get("device", "auto"),
                    "group": merged.get("group", merged["name"]),
                    "project": merged.get("wandb_project", "evar-deeprl"),
                    "entity": merged.get("wandb_entity"),
                    "tags": merged.get("tags", []),
                    "args": args,
                    "seed": seed,
                    "rerun": merged.get("rerun", 0),
                }
            )
    return settings, jobs


def job_id(job: dict) -> str:
    """Stable short hash of everything that defines the experiment."""
    payload = json.dumps(job, sort_keys=True, default=str)
    return f"{job['name']}-s{job['seed']}-{hashlib.sha1(payload.encode()).hexdigest()[:8]}"


def build_command(job: dict) -> list[str]:
    cmd = [sys.executable, job["script"],
           "--seed", str(job["seed"]),
           "--device", job["device"],
           "--wandb-mode", "online",
           "--wandb-project", job["project"],
           "--wandb-group", job["group"]]
    if job.get("entity"):
        cmd += ["--wandb-entity", job["entity"]]
    if job.get("tags"):
        cmd += ["--wandb-tags", *[str(t) for t in job["tags"]]]
    for key, value in job["args"].items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd += [flag, str(value)]
    return cmd


# --------------------------------------------------------------------- state --
def state_path(jid: str, suffix: str) -> Path:
    return STATE_DIR / f"{jid}.{suffix}"


def status_of(jid: str) -> str:
    for suffix in ("done", "failed", "running"):
        if state_path(jid, suffix).exists():
            return suffix
    return "pending"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def reconcile() -> None:
    """Turns finished/dead `.running` markers into `.done` or `.failed`.

    Runs on every pass so a container restart mid-experiment cannot leave a job
    wedged in `running` forever, silently occupying a concurrency slot.
    """
    for marker in STATE_DIR.glob("*.running"):
        jid = marker.stem
        try:
            pid = int(marker.read_text().strip())
        except (ValueError, OSError):
            pid = -1
        exit_file = state_path(jid, "exit")
        if exit_file.exists():
            code = exit_file.read_text().strip()
            marker.unlink(missing_ok=True)
            target = "done" if code == "0" else "failed"
            state_path(jid, target).write_text(f"exit={code}\n")
            log(f"{jid}: {target} (exit={code})")
        elif not pid_alive(pid):
            marker.unlink(missing_ok=True)
            state_path(jid, "failed").write_text("process vanished (container restart?)\n")
            log(f"{jid}: failed (process vanished)")


def running_count() -> int:
    return len(list(STATE_DIR.glob("*.running")))


# -------------------------------------------------------------------- launch --
def launch(job: dict, jid: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{jid}.log"
    exit_file = state_path(jid, "exit")
    exit_file.unlink(missing_ok=True)

    inner = " ".join(shlex.quote(c) for c in build_command(job))
    # The exit code is written by the wrapper rather than tracked in memory, so
    # status survives a runner (or container) restart.
    wrapped = f"cd {shlex.quote(str(REPO))} && {inner} >> {shlex.quote(str(log_file))} 2>&1; " \
              f"echo $? > {shlex.quote(str(exit_file))}"

    with open(log_file, "a") as fh:
        fh.write(f"\n=== {jid} :: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{inner}\n\n")

    proc = subprocess.Popen(["sh", "-c", wrapped], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    state_path(jid, "running").write_text(str(proc.pid))
    log(f"{jid}: launched (pid {proc.pid})")


def git_pull() -> None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "pull", "--ff-only"],
                             capture_output=True, text=True, timeout=120)
        msg = (out.stdout + out.stderr).strip().splitlines()
        if msg and "Already up to date" not in msg[0]:
            log(f"git: {msg[0]}")
    except Exception as exc:  # network blips must never kill the daemon
        log(f"git pull failed ({exc!r}); continuing with the current checkout")


# ---------------------------------------------------------------------- main --
def one_pass(max_parallel: int) -> None:
    git_pull()
    reconcile()
    settings, jobs = load_queue()
    if settings.get("paused"):
        log("queue is paused (settings.paused = true)")
        return
    max_parallel = settings.get("max_parallel", max_parallel)

    for job in jobs:
        jid = job_id(job)
        if status_of(jid) != "pending":
            continue
        if running_count() >= max_parallel:
            log(f"at capacity ({max_parallel}); {jid} waits for a free slot")
            return
        launch(job, jid)


def print_status() -> None:
    _, jobs = load_queue()
    if not jobs:
        print("queue is empty or missing:", QUEUE_FILE)
        return
    width = max(len(job_id(j)) for j in jobs)
    print(f"{'JOB':<{width}}  STATUS   LOG")
    for job in jobs:
        jid = job_id(job)
        print(f"{jid:<{width}}  {status_of(jid):<8} {LOG_DIR / (jid + '.log')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true", help="run continuously (daemon mode)")
    parser.add_argument("--once", action="store_true", help="single pass, then exit")
    parser.add_argument("--status", action="store_true", help="print job states and exit")
    parser.add_argument("--interval", type=int, default=int(os.environ.get("EVAR_INTERVAL", 60)),
                        help="seconds between passes in --loop mode")
    parser.add_argument("--max-parallel", type=int, default=int(os.environ.get("EVAR_MAX_PARALLEL", 3)),
                        help="most experiments running at once (queue.toml can override)")
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.status:
        print_status()
        return
    if args.once:
        one_pass(args.max_parallel)
        return
    if not args.loop:
        parser.error("pick one of --loop / --once / --status")

    log(f"watching {QUEUE_FILE} every {args.interval}s (max {args.max_parallel} parallel)")
    while True:
        try:
            one_pass(args.max_parallel)
        except Exception as exc:  # a bad queue edit must not kill the daemon
            log(f"pass failed: {exc!r}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
