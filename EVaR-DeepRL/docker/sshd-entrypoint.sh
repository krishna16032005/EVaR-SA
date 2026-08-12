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

exec "$@"
