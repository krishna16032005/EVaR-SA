#!/usr/bin/env bash
# Run one training command inside the EVaR-DeepRL container on a GPU host.
#
#   ./docker/run.sh python experiments/run_cartpole.py --critic c51 --episodes 500
#   GPU=1 ./docker/run.sh python experiments/run_invpend.py --critic iqn
#   DETACH=1 NAME=evar-seed0 ./docker/run.sh python experiments/run_cartpole.py --seed 0
#
# Environment knobs:
#   IMAGE        image tag to run              (default evar-deeprl:latest)
#   GPU          "all", "none", or indices     (default all, e.g. GPU=0 or GPU=0,1)
#   RESULTS_DIR  host dir bind-mounted to /workspace/results (default ./results)
#   ENV_FILE     file holding WANDB_API_KEY=…  (default ./.env, optional)
#   DETACH=1     run in the background and print the container name
#   NAME         container name                (default evar-<timestamp>)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-evar-deeprl:latest}"
GPU="${GPU:-all}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
NAME="${NAME:-evar-$(date +%Y%m%d-%H%M%S)-$$}"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <command…>   e.g. $0 python experiments/run_cartpole.py --critic c51" >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR"

args=(--rm --name "$NAME" --shm-size=8g)

# GPU selection. `--gpus all` and `--gpus device=0,1` are different syntaxes, and
# GPU=none is the escape hatch for a CPU-only sanity run.
case "$GPU" in
    none|"") ;;
    all)     args+=(--gpus all) ;;
    *)       args+=(--gpus "device=$GPU") ;;
esac

# Run as the invoking user so results/ on the host is not owned by root.
args+=(--user "$(id -u):$(id -g)")
args+=(-v "$RESULTS_DIR:/workspace/results")

# WANDB_API_KEY from the environment wins; otherwise fall back to the env file.
if [ -n "${WANDB_API_KEY:-}" ]; then
    args+=(-e "WANDB_API_KEY=$WANDB_API_KEY")
elif [ -f "$ENV_FILE" ]; then
    args+=(--env-file "$ENV_FILE")
else
    echo "[run.sh] no WANDB_API_KEY and no $ENV_FILE -- wandb will fall back to offline mode." >&2
fi

# Pass through any wandb settings already exported in the shell (project/entity/mode).
for var in WANDB_PROJECT WANDB_ENTITY WANDB_MODE WANDB_TAGS; do
    if [ -n "${!var:-}" ]; then args+=(-e "$var=${!var}"); fi
done

if [ "${DETACH:-0}" = "1" ]; then
    args+=(-d)
elif [ -t 1 ]; then
    args+=(-it)   # only when attached to a real terminal; `-t` fails under nohup/cron
fi

docker run "${args[@]}" "$IMAGE" "$@"

if [ "${DETACH:-0}" = "1" ]; then
    echo "[run.sh] started '$NAME' -- follow with: docker logs -f $NAME"
fi
