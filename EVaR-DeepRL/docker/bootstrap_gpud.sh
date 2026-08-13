#!/usr/bin/env bash
# One-shot server-side setup: build the EVaR-DeepRL SSH dev box and start it.
# Run this ON gpud.model, from inside the cloned repo.
#
#   SSH_KEY=~/laptop_key.pub WANDB_API_KEY=xxxx ./docker/bootstrap_gpud.sh
#
# Knobs (all optional except SSH_KEY):
#   DEV_USER    login/sudo user inside the container   (default: $USER)
#               (named DEV_USER, not USERNAME: zsh treats USERNAME as a special
#                parameter, so `USERNAME=x ./bootstrap…` is silently ignored there)
#   SSH_KEY     path to, or contents of, your LAPTOP's public key   (required)
#   PORT        host port published to the container's 22   (default: 6730)
#   CPUS        --cpus value        (default: 10)
#   CPUSET      --cpuset-cpus value (default: unset; e.g. "38-47")
#               On the i7 group's shared box the convention is to pin each
#               container to its own 10-core block so users don't fight over the
#               same cores -- see /common/docker-server/current-containers.
#   MEMORY      --memory value      (default: 64g)
#   GPUS        --gpus value        (default: all; e.g. '"device=0"')
#   WANDB_API_KEY   bridged into SSH sessions by the container entrypoint
#   FORCE=1     replace an existing container with the same name
set -euo pipefail

DEV_USER="${DEV_USER:-$USER}"
PORT="${PORT:-6730}"
CPUS="${CPUS:-10}"
MEMORY="${MEMORY:-64g}"
GPUS="${GPUS:-all}"
IMAGE="${IMAGE:-ssh-server/$DEV_USER}"
CONTAINER="${CONTAINER:-evar-$DEV_USER}"
SSH_KEY="${SSH_KEY:-}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # …/EVaR-SA/EVaR-DeepRL
CLONE_ROOT="$(cd "$REPO_DIR/.." && pwd)"                      # …/EVaR-SA  (holds .git)

# Use plain docker if this user is in the docker group; fall back to sudo.
if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
else
    DOCKER=(sudo docker)
fi

die() { echo "[bootstrap] error: $*" >&2; exit 1; }
step() { echo; echo "[bootstrap] === $* ==="; }

# ---------------------------------------------------------------- preflight --
step "preflight"

command -v docker >/dev/null || die "docker is not installed on this host."

[ -n "$SSH_KEY" ] || die "SSH_KEY is required -- the PUBLIC key of the machine you
  will SSH in from (on your Mac: cat ~/.ssh/id_ed25519.pub). Pass a path or the
  key text: SSH_KEY=~/laptop.pub $0"

# Accept either a path to a key file or the key string itself.
if [ -f "$SSH_KEY" ]; then
    SSH_KEY_TEXT="$(cat "$SSH_KEY")"
else
    SSH_KEY_TEXT="$SSH_KEY"
fi
case "$SSH_KEY_TEXT" in
    ssh-*|ecdsa-*) ;;
    *) die "SSH_KEY does not look like a public key (expected it to start with
  'ssh-ed25519', 'ssh-rsa', …). A PRIVATE key here would be a mistake." ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo "[bootstrap] warning: nvidia-smi not found on the host -- GPU runs will fail."
fi

# Resolve our own container FIRST: when replacing it (FORCE=1) it is still
# holding $PORT, and the port check below would otherwise reject the rebuild
# because of the very container we are about to remove.
if "${DOCKER[@]}" ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    if [ "${FORCE:-0}" = "1" ]; then
        echo "[bootstrap] removing existing container '$CONTAINER' (FORCE=1)"
        "${DOCKER[@]}" rm -f "$CONTAINER" >/dev/null
    else
        die "container '$CONTAINER' already exists.
  Restart it:  ${DOCKER[*]} start $CONTAINER
  Replace it:  FORCE=1 $0   (destroys anything not on a bind mount -- see DOCKER.md A6)"
    fi
fi

# Now any remaining holder of $PORT genuinely belongs to someone else. gpud is a
# shared machine -- stealing a colleague's port would break their box.
if "${DOCKER[@]}" ps --format '{{.Names}} {{.Ports}}' | grep ":$PORT->"; then
    die "host port $PORT is already published by the container listed above.
  Pick another: PORT=$((PORT + 1)) $0"
fi

# -------------------------------------------------------------------- build --
step "building $IMAGE (first build pulls ~3 GB of CUDA base image)"

"${DOCKER[@]}" build -f "$REPO_DIR/Dockerfile.ssh" \
    --build-arg USERNAME="$DEV_USER" \
    --build-arg SSH_KEY="$SSH_KEY_TEXT" \
    --build-arg UID="$(id -u)" \
    -t "$IMAGE" "$REPO_DIR"

# ------------------------------------------------------------- gpu sanity ----
step "checking CUDA inside the image"

# The entrypoint execs whatever command follows, so this bypasses sshd entirely.
"${DOCKER[@]}" run --rm --gpus "$GPUS" "$IMAGE" \
    python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')" \
    || die "CUDA check failed. If this said 'could not select device driver', the
  NVIDIA Container Toolkit is missing -- see DOCKER.md section 0."

# ---------------------------------------------------------------- start it ---
step "starting $CONTAINER on port $PORT"

run_args=(
    -d --name "$CONTAINER"
    --cpus="$CPUS" --memory="$MEMORY" --shm-size=8g
    --gpus "$GPUS"
)
[ -n "${CPUSET:-}" ] && run_args+=(--cpuset-cpus="$CPUSET")
run_args+=(
    -p "$PORT:22"
    # The host clone, not the image's baked-in copy: keeps results across
    # `docker rm`, and carries the .git dir that lives at the repo root.
    -v "$CLONE_ROOT:/home/$DEV_USER/$(basename "$CLONE_ROOT")"
    --restart unless-stopped
)
[ -n "${WANDB_API_KEY:-}" ] && run_args+=(-e "WANDB_API_KEY=$WANDB_API_KEY")

"${DOCKER[@]}" run "${run_args[@]}" "$IMAGE" >/dev/null

sleep 2
"${DOCKER[@]}" ps --filter "name=$CONTAINER" --format '  {{.Names}}  {{.Status}}  {{.Ports}}'

# ------------------------------------------------------------------ done -----
HOST="$(hostname -f 2>/dev/null || hostname)"
cat <<EOF

[bootstrap] done.

  SSH in from your laptop:
      ssh -J <you>@shell.model.in.tum.de -p $PORT $DEV_USER@$HOST

  Then:
      cd ~/$(basename "$CLONE_ROOT")/$(basename "$REPO_DIR")
      tmux new -s train
      python experiments/run_cartpole.py --critic c51 --episodes 500 \\
          --device auto --wandb-mode online --wandb-project evar-deeprl

  sudo password inside the container: $DEV_USER
EOF
