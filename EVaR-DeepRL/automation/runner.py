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
import re
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
            entry = {
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
            # Only recorded when explicitly set. job_id() hashes this whole dict,
            # so adding the key unconditionally would change the identity of every
            # existing job and silently re-run finished work.
            if merged.get("python"):
                entry["python"] = merged["python"]
            jobs.append(entry)
    return settings, jobs


def job_id(job: dict) -> str:
    """Stable short hash of everything that defines the experiment."""
    payload = json.dumps(job, sort_keys=True, default=str)
    return f"{job['name']}-s{job['seed']}-{hashlib.sha1(payload.encode()).hexdigest()[:8]}"


def build_command(job: dict) -> list[str]:
    # `python` lets a job run in a different interpreter than the runner's own.
    # safety-gymnasium pins gymnasium==0.28.1 / mujoco==2.3.3, which cannot coexist
    # with the main environment, so its jobs point at ~/envs/safety instead.
    cmd = [os.path.expanduser(job.get("python") or sys.executable), job["script"],
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


def status_of(jid: str, require_approval: bool = False) -> str:
    """Terminal states first, then the approval gate, then 'pending'.

    The gate lives in the state directory on the server -- not in git -- which is
    the whole point: collaborators can push experiment proposals, but only
    someone with access to this machine can turn one into a running job.
    """
    for suffix in ("done", "failed", "running", "denied"):
        if state_path(jid, suffix).exists():
            return suffix
    if require_approval and not state_path(jid, "approved").exists():
        return "awaiting-approval"
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


def git_pull(branch: str | None = None) -> None:
    try:
        if branch:
            # Mirror the branch exactly rather than merging it. A fast-forward
            # merge fails the moment the checked-out branch and the release
            # branch diverge -- which is precisely the situation this setting
            # exists for (main collects proposals, the release branch is what
            # the owner has vetted). Nothing tracked is written on the server,
            # and results/logs are gitignored, so a hard reset is safe here.
            subprocess.run(["git", "-C", str(REPO), "fetch", "origin", branch],
                           capture_output=True, text=True, timeout=120)
            cmd = ["git", "-C", str(REPO), "reset", "--hard", f"origin/{branch}"]
        else:
            cmd = ["git", "-C", str(REPO), "pull", "--ff-only"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        msg = (out.stdout + out.stderr).strip().splitlines()
        if msg and "Already up to date" not in msg[0]:
            log(f"git: {msg[0]}")
    except Exception as exc:  # network blips must never kill the daemon
        log(f"git pull failed ({exc!r}); continuing with the current checkout")


def who_touched_queue() -> str:
    """Last author of experiments/queue.toml -- shown when listing proposals.

    Advisory only: git authorship is self-reported and trivially forged. It
    answers "who asked for this?", never "is this allowed to run?" -- that is
    what the approval marker on this machine is for.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "log", "-1", "--format=%an <%ae>, %ar", "--", str(QUEUE_FILE)],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------- main --
def one_pass(max_parallel: int) -> None:
    settings, _ = load_queue()
    git_pull(settings.get("branch"))
    reconcile()
    settings, jobs = load_queue()
    if settings.get("paused"):
        log("queue is paused (settings.paused = true)")
        return
    max_parallel = settings.get("max_parallel", max_parallel)
    require_approval = settings.get("require_approval", True)

    new_proposals = []
    for job in jobs:
        jid = job_id(job)
        state = status_of(jid, require_approval)
        if state == "awaiting-approval":
            # Announce each proposal once, not every poll.
            marker = state_path(jid, "proposed")
            if not marker.exists():
                marker.write_text(f"{job['name']} seed={job['seed']}\nqueued by: {who_touched_queue()}\n")
                new_proposals.append(jid)
            continue
        if state != "pending":
            continue
        if running_count() >= max_parallel:
            log(f"at capacity ({max_parallel}); {jid} waits for a free slot")
            return
        launch(job, jid)

    if new_proposals:
        log(f"{len(new_proposals)} job(s) awaiting your approval "
            f"(last queue edit by {who_touched_queue()}):")
        for jid in new_proposals:
            log(f"    {jid}")
        log("approve with: python automation/runner.py --approve <id|all>")


EPISODE_RE = re.compile(r"episode\s+(\d+)")


def progress_of(jid: str, job: dict) -> str:
    """How far a run has got, read from its own log tail.

    Uses the training script's own progress lines rather than wandb, so this
    still answers the question when the network is down or the run is offline.
    """
    log_file = LOG_DIR / f"{jid}.log"
    if not log_file.exists():
        return ""
    try:
        with open(log_file, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 8192))
            tail = fh.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""
    seen = EPISODE_RE.findall(tail)
    if not seen:
        return ""
    current = int(seen[-1])
    total = job["args"].get("episodes")
    if total:
        return f"ep {current}/{total} ({100 * current // int(total)}%)"
    return f"ep {current}"


def print_status() -> None:
    settings, jobs = load_queue()
    require_approval = settings.get("require_approval", True)
    if not jobs:
        print("queue is empty or missing:", QUEUE_FILE)
        return

    rows = [(job_id(j), status_of(job_id(j), require_approval), j) for j in jobs]
    counts: dict[str, int] = {}
    for _, state, _ in rows:
        counts[state] = counts.get(state, 0) + 1
    finished = counts.get("done", 0) + counts.get("failed", 0)
    total = len(rows)

    print(f"{finished}/{total} finished ({100 * finished // total}%)   "
          + "  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"approval gate: {'ON' if require_approval else 'OFF'}   "
          f"max_parallel: {settings.get('max_parallel', 3)}   "
          f"last queue edit: {who_touched_queue()}\n")

    width = max(len(r[0]) for r in rows)
    print(f"{'JOB':<{width}}  {'STATUS':<18} PROGRESS")
    for jid, state, job in rows:
        detail = progress_of(jid, job) if state in ("running", "done", "failed") else ""
        print(f"{jid:<{width}}  {state:<18} {detail}")

    if counts.get("awaiting-approval"):
        print(f"\n{counts['awaiting-approval']} job(s) need approval: "
              "python automation/runner.py --approve all")
    if not counts.get("running") and not counts.get("pending") and not counts.get("awaiting-approval"):
        print("\nQueue is drained -- nothing is running and nothing is waiting. "
              "Safe to stop the container.")


def approve(targets: list[str], deny: bool = False) -> None:
    """Marks jobs approved (or denied) on this machine, so the daemon may run them."""
    settings, jobs = load_queue()
    require_approval = settings.get("require_approval", True)
    verb = "denied" if deny else "approved"
    matched = 0
    for job in jobs:
        jid = job_id(job)
        state = status_of(jid, require_approval)
        wanted = "all" in targets or any(t == jid or t in job["name"] for t in targets)
        if not wanted or state not in ("awaiting-approval", "denied", "pending"):
            continue
        state_path(jid, "denied" if deny else "approved").write_text(
            f"{verb} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if deny:
            state_path(jid, "approved").unlink(missing_ok=True)
        else:
            state_path(jid, "denied").unlink(missing_ok=True)
        print(f"{verb}: {jid}")
        matched += 1
    if not matched:
        print("nothing matched; run --status to see job ids")
    elif not deny:
        print(f"\n{matched} job(s) will start within one poll interval.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true", help="run continuously (daemon mode)")
    parser.add_argument("--once", action="store_true", help="single pass, then exit")
    parser.add_argument("--status", action="store_true", help="print job states and exit")
    parser.add_argument("--approve", nargs="+", metavar="ID",
                        help="approve job ids, name substrings, or 'all' -- the daemon "
                             "only runs approved jobs while the gate is on")
    parser.add_argument("--deny", nargs="+", metavar="ID",
                        help="reject proposals so they stop appearing as pending")
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
    if args.approve:
        approve(args.approve)
        return
    if args.deny:
        approve(args.deny, deny=True)
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
    try:
        main()
    except BrokenPipeError:
        # `--status | head` closes the pipe early; that is not an error worth a
        # traceback. Redirect stdout to devnull so the interpreter's own flush
        # at exit cannot raise a second time.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    except KeyboardInterrupt:
        pass
