#!/bin/bash
# Build an isolated env for safety-gymnasium.
#
# It pins gymnasium==0.28.1, mujoco==2.3.3, pygame==2.1.0 -- all of which
# conflict with the main env (gymnasium 1.3.0, mujoco 3.11.0) and would break
# run_invpend.py's InvertedPendulum-v5. Python 3.10 because pygame 2.1.0 has no
# cp311 wheel and building it needs SDL2 headers the container lacks.
set -e
PREFIX=$HOME/envs/safety
echo "started: $(date -Is)"

source /opt/conda/etc/profile.d/conda.sh
conda create -y -p "$PREFIX" python=3.10 pip
conda activate "$PREFIX"

python -V
pip install -q --upgrade pip
# CPU-only torch: every run in this project uses --device cpu.
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q safety-gymnasium wandb

echo "=== versions ==="
python - <<'PY'
import gymnasium, mujoco, torch, wandb
print("gymnasium", gymnasium.__version__)
print("mujoco   ", mujoco.__version__)
print("torch    ", torch.__version__)
print("wandb    ", wandb.__version__)
import safety_gymnasium
print("safety_gymnasium OK")
PY

echo "=== can we actually make and step a hazard env, headless? ==="
MUJOCO_GL=egl python - <<'PY'
import numpy as np, safety_gymnasium
env = safety_gymnasium.make("SafetyPointGoal1-v0")
obs, info = env.reset(seed=0)
print("obs", np.asarray(obs).shape, "action", env.action_space.shape)
tot_r = tot_c = 0.0
for _ in range(200):
    o, r, c, term, trunc, info = env.step(env.action_space.sample())
    tot_r += r; tot_c += c
    if term or trunc:
        o, info = env.reset()
print(f"200 random steps: return={tot_r:.3f} cost={tot_c:.3f}")
print("STEP OK")
PY
echo "finished: $(date -Is)"
