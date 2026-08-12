# EVaR-DeepRL training image (CUDA).
#
# The base already ships torch 2.5.1 built against CUDA 12.4, so the `torch>=2.0`
# line in requirements.txt resolves as already-satisfied and pip does not pull a
# second (CPU-only) wheel over it. CUDA 12.x has minor-version compatibility, so
# any host NVIDIA driver >= 525 can run this image.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # MuJoCo picks its GL backend from this; egl is the headless-server choice.
    MUJOCO_GL=egl \
    MPLBACKEND=Agg \
    # Everything below is written to at run time by a non-root, possibly
    # unknown-uid user (`docker run --user "$(id -u):$(id -g)"`), so it must not
    # live under a root-owned $HOME.
    HOME=/workspace/.container-home \
    MPLCONFIGDIR=/workspace/.container-home/matplotlib \
    WANDB_DIR=/workspace/results

# libgl1/libegl1/libosmesa6: MuJoCo links its renderer against these even when
# training headless. git: wandb records the code version of the repo it runs in.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libegl1 \
        libosmesa6 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Requirements first: dependency layers stay cached across code-only rebuilds.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# World-writable so an arbitrary --user uid can write results/wandb/checkpoints
# whether or not the host bind-mount shadows these paths.
RUN mkdir -p /workspace/results /workspace/.container-home \
    && chmod -R 777 /workspace/results /workspace/.container-home

CMD ["python", "experiments/run_cartpole.py", "--critic", "c51", "--wandb-mode", "offline"]
