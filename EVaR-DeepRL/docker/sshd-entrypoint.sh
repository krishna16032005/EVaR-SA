#!/bin/sh
# Bridges `docker run -e VAR=...` into SSH login shells.
#
# sshd is started by the container init, not by a shell, so variables passed with
# `docker run -e` exist only in PID 1's environment and are invisible to anything
# you type over SSH. Writing them into /etc/profile.d/20-runtime.sh (sourced from
# the user's .bashrc) is what makes `--wandb-mode online` pick the key up.
set -e

RUNTIME_ENV=/etc/profile.d/20-runtime.sh
: > "$RUNTIME_ENV"

for var in WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_TAGS HF_TOKEN; do
    eval "value=\${$var:-}"
    if [ -n "$value" ]; then
        printf "export %s='%s'\n" "$var" "$value" >> "$RUNTIME_ENV"
    fi
done

# Credentials, so keep it root-readable only... except the login shell that needs
# it runs as the dev user, so 644 is the working compromise inside a
# single-tenant container. See the `docker commit` warning in DOCKER.md.
chmod 644 "$RUNTIME_ENV"

# Persist SSH host keys across rebuilds.
#
# `ssh-keygen -A` runs at build time, so every rebuilt image ships fresh host
# keys and every client sees REMOTE HOST IDENTIFICATION HAS CHANGED -- the same
# warning a real man-in-the-middle would produce, which is exactly the warning
# you do not want people trained to click past. Keeping the keys on the bind
# mount makes the box's identity stable across rebuilds.
HOSTKEY_DIR="${EVAR_HOSTKEYS:-${EVAR_REPO:-/home/$EVAR_USER/EVaR-SA/EVaR-DeepRL}/automation/.hostkeys}"
if [ -d "$(dirname "$HOSTKEY_DIR")" ]; then
    if ls "$HOSTKEY_DIR"/ssh_host_*_key >/dev/null 2>&1; then
        cp -a "$HOSTKEY_DIR"/ssh_host_* /etc/ssh/
        chmod 600 /etc/ssh/ssh_host_*_key
        echo "[entrypoint] restored persisted SSH host keys (fingerprint unchanged)"
    else
        mkdir -p "$HOSTKEY_DIR"
        cp -a /etc/ssh/ssh_host_* "$HOSTKEY_DIR"/
        chmod 700 "$HOSTKEY_DIR"
        echo "[entrypoint] saved SSH host keys to $HOSTKEY_DIR for future rebuilds"
    fi
fi

# Optional experiment daemon. Started here rather than by cron so it inherits the
# container lifecycle: `docker start` brings it back, `--restart unless-stopped`
# survives a reboot, and there is no second thing to remember to launch.
if [ "${EVAR_AUTORUN:-0}" = "1" ]; then
    REPO="${EVAR_REPO:-/home/$EVAR_USER/EVaR-SA/EVaR-DeepRL}"
    if [ -f "$REPO/automation/runner.py" ]; then
        mkdir -p "$REPO/automation/logs"
        chown "$EVAR_USER" "$REPO/automation/logs" 2>/dev/null || true
        su - "$EVAR_USER" -c \
            "cd '$REPO' && EVAR_REPO='$REPO' nohup python automation/runner.py --loop \
             >> '$REPO/automation/logs/runner.log' 2>&1 &"
        echo "[entrypoint] experiment runner started (EVAR_AUTORUN=1), watching $REPO"
    else
        echo "[entrypoint] EVAR_AUTORUN=1 but $REPO/automation/runner.py is missing;" \
             "is the repo bind-mounted? Skipping the runner."
    fi
fi

exec "$@"
