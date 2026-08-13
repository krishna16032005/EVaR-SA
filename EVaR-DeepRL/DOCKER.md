# Running EVaR-DeepRL on gpud.model (TUM) with Weights & Biases

Two containers are defined in this repo, for two different habits:

| File | Shape | Use it when |
|---|---|---|
| [`Dockerfile.ssh`](Dockerfile.ssh) | long-lived **SSH dev box** (sshd on 22, `docker commit` to persist) | **gpud.model** — matches the group's `ssh-server/<name>` workflow |
| [`Dockerfile`](Dockerfile) | one-shot **batch job** (runs a training command, exits) | any plain Docker host, CI, or a Slurm/Enroot cluster |

Start with the SSH dev box — it's the one that fits your server.

---

# Part A — the gpud.model SSH dev box

## A0. The whole thing in three commands

```bash
# on your Mac
cat ~/.ssh/id_ed25519.pub                       # copy this

# on gpud.model
git clone https://github.com/krishna16032005/EVaR-SA.git && cd EVaR-SA/EVaR-DeepRL
SSH_KEY="ssh-ed25519 AAAA… you@laptop" WANDB_API_KEY=<key> ./docker/bootstrap_gpud.sh
```

> **Live deployment on gpud.model:** container `evar-gade`, image
> `ssh-server/gade`, host port **6731** — port 6730 from the group's standard
> recipe is already held by another user's container (`ssh-server/innushka`), and
> 6710-6713/6715/6727-6730 are taken too. gpud is shared; check
> `sudo docker ps` before claiming a port. Reach it with `ssh evar-box` once
> `~/.ssh/config` has the block from A3.

[`docker/bootstrap_gpud.sh`](docker/bootstrap_gpud.sh) runs the preflight checks,
builds the image with the right build args, verifies CUDA works inside it,
starts the container with the flags below, and prints your SSH command. Knobs:
`DEV_USER`, `PORT` (default 6730), `CPUS` (10), `MEMORY` (64g), `GPUS` (all),
`FORCE=1` to replace an existing container.

The rest of Part A is what that script does, step by step — read it when
something needs changing or goes wrong.

## A1. Build (run on gpud.model)

Get the code onto the server, then build with your group's build-arg contract:

```bash
ssh -J gade@shell.model.in.tum.de gade@gpud.model

git clone https://github.com/krishna16032005/EVaR-SA.git
cd EVaR-SA/EVaR-DeepRL

sudo docker build -f Dockerfile.ssh \
     --build-arg USERNAME=gade \
     --build-arg SSH_KEY="ssh-ed25519 AAAA... you@laptop" \
     --build-arg UID=$(id -u) \
     -t ssh-server/gade .
```

`UID` isn't part of your group's standard contract, but it matters here: it makes
the container user share your host uid, so files written into the bind-mounted
repo (A2) stay owned by you instead of by root.

`SSH_KEY` is your **laptop's public key** (`cat ~/.ssh/id_ed25519.pub` on your
Mac) — that's the key you'll use to SSH into the container. Not the server's key.
It's a build arg, so it must be quoted as one string.

The image keeps your group's conventions: login is key-only, and the account
password is the username (`gade`) — used for `sudo` *inside* the container, not
for logging in. On top of that it adds CUDA-enabled torch 2.5.1, this repo's
dependencies, MuJoCo's system libraries, and `tmux`.

First build pulls ~3 GB of CUDA base image. Rebuilds after a `git pull` reuse the
cached pip layer unless `requirements.txt` changed.

> **If your group's `ssh-server` base image is mandatory** (some setups require
> it for accounting/quota reasons), don't fight it — layer on top instead. Change
> `Dockerfile.ssh`'s first line to `FROM ssh-server/gade` and delete the
> `openssh-server`/`useradd`/sshd blocks; the pip and MuJoCo layers are the parts
> that matter, and the base already handles users and keys.

## A2. Run it

```bash
sudo docker run -d \
    --name evar-gade \
    --cpus="10" --memory="64g" --shm-size=8g \
    --gpus all \
    -p 6730:22 \
    -v /home/gade/EVaR-SA:/home/gade/EVaR-SA \
    -e WANDB_API_KEY="<your-40-char-key>" \
    ssh-server/gade
```

Beyond your standard flags:

- **`--shm-size=8g`** — Docker's 64 MB default `/dev/shm` is what makes PyTorch
  DataLoader workers die with "bus error"; cheap insurance.
- **`-v /home/gade/EVaR-SA:/home/gade/EVaR-SA`** — mounts your *host clone* into
  the container, and this is the path you should actually work in. Three reasons:
  results survive `docker rm` (a bind mount is deliberately excluded from
  `docker commit`, so commit alone would not save them); `git pull` works, because
  the `.git` directory lives at the repo root `EVaR-SA/` — one level above the
  build context, so the image's baked-in copy at `~/EVaR-DeepRL` has no git
  metadata; and wandb records the exact commit each run came from, which it can
  only do inside a real git checkout.

  The image's own `~/EVaR-DeepRL` copy stays as a self-contained fallback if you
  ever run without the mount.
- **`-e WANDB_API_KEY`** — see A4 for why this needs the entrypoint's help, and
  for the safer interactive alternative.
- **`--gpus all`** on a shared box takes visibility of every GPU. If others use
  gpud, prefer `--gpus '"device=0"'` (the quoting is required) and coordinate.

Pick a host port nobody else on gpud has claimed — `sudo docker ps` shows what's
already published. Reusing 6730 while another container holds it fails the run.

## A3. SSH into the container from your Mac

Two hops: jump host → gpud's published container port.

```bash
ssh -J gade@shell.model.in.tum.de -p 6730 gade@gpud.model
```

Worth putting in `~/.ssh/config` on your laptop so VS Code Remote-SSH, `rsync`,
and `scp` all just work:

```
Host tum-shell
    HostName shell.model.in.tum.de
    User gade

Host gpud
    HostName gpud.model
    User gade
    ProxyJump tum-shell

Host evar-box
    HostName gpud.model
    User gade
    Port 6730
    ProxyJump tum-shell
    ServerAliveInterval 60
```

Then `ssh evar-box`, `rsync -avz evar-box:~/EVaR-SA/EVaR-DeepRL/results/ ./results/`, or
open the container directly in VS Code (Remote-SSH → `evar-box`).

Rebuilding the image regenerates nothing key-wise, but replacing a container on
the same port can trip `REMOTE HOST IDENTIFICATION HAS CHANGED` on your laptop;
`ssh-keygen -R '[gpud.model]:6730'` clears it.

## A4. Weights & Biases inside the box

**The gotcha:** `sshd` is started by the container init, not by a shell, so
variables you pass with `docker run -e` live only in PID 1's environment and are
**invisible in your SSH session**. `echo $WANDB_API_KEY` would come back empty
and every run would silently fall back to offline mode.

[`docker/sshd-entrypoint.sh`](docker/sshd-entrypoint.sh) fixes this by writing
the wandb variables into `/etc/profile.d/20-runtime.sh`, which the user's
`.bashrc` sources. So after `docker run -e WANDB_API_KEY=...`, this works:

```bash
ssh evar-box
echo "${WANDB_API_KEY:0:6}…"    # non-empty means the bridge worked
```

The same mechanism carries `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_MODE`, and
`WANDB_TAGS` if you pass them.

**Alternative, if you'd rather not put the key in a `docker run` command** (it
lands in shell history and in `docker inspect` output): omit `-e` and log in
once from inside the container:

```bash
ssh evar-box
wandb login          # paste the key; writes ~/.netrc
```

> **Both paths matter for `docker commit`.** A commit bakes `~/.netrc` *and*
> `/etc/profile.d/20-runtime.sh` into the image, so anyone who later runs that
> image has your W&B account. Before committing an image you intend to share or
> push:
> ```bash
> sudo docker exec evar-gade rm -f /home/gade/.netrc /etc/profile.d/20-runtime.sh
> ```
> For a private, never-pushed snapshot it's your call — just know the key is in
> there.

## A5. Train

```bash
ssh evar-box
cd ~/EVaR-SA/EVaR-DeepRL     # the bind-mounted host clone from A2

# is the GPU actually visible in here?
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 5-episode smoke test, no wandb: catches import/env problems in seconds
python experiments/run_cartpole.py --episodes 5 --wandb-mode disabled

# the real thing, streaming live to wandb
python experiments/run_cartpole.py \
    --critic c51 --episodes 500 --alpha 0.1 --seed 0 \
    --device auto --wandb-mode online \
    --wandb-project evar-deeprl --wandb-group cartpole-c51
```

The run URL prints in the first few seconds (`[logging] wandb run '…'`).

**Always run long jobs under `tmux`** — an SSH drop otherwise kills training,
and this connection goes through two hops:

```bash
tmux new -s c51            # start
#   … launch training, then detach with Ctrl-b d
tmux attach -t c51         # come back later, even from a different laptop
```

Multiple seeds, each in its own tmux window, grouped for comparison in W&B:

```bash
for seed in 0 1 2 3 4; do
  tmux new-session -d -s "s$seed" \
    "cd ~/EVaR-SA/EVaR-DeepRL && python experiments/run_cartpole.py \
       --critic c51 --alpha 0.1 --seed $seed --episodes 500 \
       --device cpu --wandb-mode online --wandb-group cartpole-c51-alpha0.1"
done
tmux ls
```

> **`--device cpu` in that sweep is deliberate.** CartPole and InvertedPendulum
> here are tiny MLPs stepping one env at a time, so the loop is bound by Python
> and env overhead, not matrix math — CUDA is often *slower* than CPU for these
> configs. With `--cpus="10"` you can run ~10 seeds concurrently, which is the
> actual win from this server. Switch to `--device cuda` when you widen the
> critic, batch several envs, or push IQN's quantile-sample count up.

## A6. Persisting and stopping

```bash
sudo docker stop evar-gade         # frees the GPU; results/ on the host survive
sudo docker start evar-gade        # same container, same port, tmux sessions gone

# snapshot extra packages you pip-installed inside the box, per your workflow:
sudo docker commit evar-gade ssh-server/gade:v2
```

`docker commit` captures the container filesystem — so a package you installed
interactively is preserved, but anything under the bind-mounted `results/` is
not (it isn't part of the container filesystem). Read the A4 warning before
committing a container you've logged into W&B from.

Cleaner long-term habit: when you add a dependency, add it to
`requirements.txt`, `git push`, and rebuild. Then the image is reproducible from
the repo instead of from an undocumented commit chain.

## A7. If gpud has no outbound internet

Compute nodes are often firewalled. Check from inside the container:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' https://api.wandb.ai
```

No answer means train offline and sync from a host that can reach wandb.ai:

```bash
python experiments/run_cartpole.py --critic c51 --wandb-mode offline
wandb sync results/wandb/offline-run-*
```

The code degrades this way on its own too: `--wandb-mode online` with no
resolvable key falls back to offline with a printed note rather than hanging on a
login prompt. If even the *build* can't reach the internet, build where you can
and ship the image:

```bash
docker save ssh-server/gade | gzip > evar-box.tar.gz
scp evar-box.tar.gz gpud:~            # uses the ~/.ssh/config above
ssh gpud 'gunzip -c evar-box.tar.gz | sudo docker load'
```

## A8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Permission denied (publickey)` on port 6730 | `SSH_KEY` build arg didn't hold your laptop's *public* key — check with `sudo docker exec evar-gade cat /home/gade/.ssh/authorized_keys` |
| SSH connects but `python` is missing / wrong | shell startup files not sourced — confirm `/etc/profile.d/10-evar.sh` exists and line 1 of `~/.bashrc` sources it |
| `echo $WANDB_API_KEY` empty over SSH | the sshd-doesn't-inherit-`-e` issue in A4; the entrypoint must be running (`sudo docker inspect -f '{{.Path}}' evar-gade`) |
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing on gpud, or Docker wasn't restarted after `nvidia-ctk runtime configure` |
| `--device cuda requested but torch.cuda.is_available() is False` | container started without `--gpus all` |
| `CUDA driver version is insufficient` | host driver older than 525; rebuild `FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime` |
| `libGL.so.1: cannot open shared object file` | rebuild without cache: `sudo docker build --no-cache -f Dockerfile.ssh …` |
| MuJoCo EGL error on `render()` | run with `-e MUJOCO_GL=osmesa`, or verify `NVIDIA_DRIVER_CAPABILITIES` includes `graphics` |
| `port is already allocated` | another container holds 6730 — pick a free port (`sudo docker ps`) |
| Bus error / DataLoader worker crash | `/dev/shm` too small; keep `--shm-size=8g` |
| `results/` owned by root on the host | the container uid doesn't match yours — rebuild with `--build-arg UID=$(id -u)` |

---

# Part B — the one-shot batch container

For any plain Docker host (or to script sweeps without SSH-ing anywhere), the
[`Dockerfile`](Dockerfile) + [`docker/run.sh`](docker/run.sh) pair runs a single
training command and exits.

```bash
docker build -t evar-deeprl:latest .
echo "WANDB_API_KEY=<key>" > .env && chmod 600 .env      # gitignored

./docker/run.sh python experiments/run_cartpole.py --critic c51 --episodes 500

GPU=0 DETACH=1 NAME=evar-s0 ./docker/run.sh \
    python experiments/run_invpend.py --critic iqn --seed 0 --wandb-mode online
docker logs -f evar-s0
```

`docker/run.sh` handles the tedious flags: `--gpus all` vs `--gpus device=0`,
`--user "$(id -u):$(id -g)"` so results aren't root-owned, the results bind
mount, and reading `WANDB_API_KEY` from the environment or `.env`. Knobs:
`IMAGE`, `GPU` (`all`/`none`/indices), `RESULTS_DIR`, `ENV_FILE`, `DETACH`, `NAME`.

## On a Slurm cluster (LRZ AI Systems and similar)

Those don't allow `docker` on compute nodes; they consume the same image through
Enroot/Pyxis:

```bash
srun --gres=gpu:1 \
     --container-image=ghcr.io#<github-user>/evar-deeprl:latest \
     --container-mounts=$HOME/evar-results:/workspace/results \
     --export=WANDB_API_KEY \
     python experiments/run_cartpole.py --critic c51 --wandb-mode online
```

To publish the image for that (or to skip rebuilding on each machine):

```bash
echo "$GITHUB_PAT" | docker login ghcr.io -u <github-user> --password-stdin
docker build -t ghcr.io/<github-user>/evar-deeprl:latest .
docker push ghcr.io/<github-user>/evar-deeprl:latest
```

Or convert for Apptainer: `apptainer build evar.sif docker://ghcr.io/<github-user>/evar-deeprl:latest`,
then `apptainer exec --nv evar.sif python experiments/run_cartpole.py …`.

---

## Where outputs land (both parts)

```
results/<env>/<critic>_alpha<alpha>_seed<seed>_<timestamp>/
    *_episode.csv      # one row per episode
    *_update.csv       # one row per gradient update
    *_eval.csv         # one row per evaluation pass
    checkpoints/
results/wandb/         # wandb's own run directories
```

Every run gets its own timestamped directory, so reruns never overwrite each
other, and the same tag appears in the W&B run name for matching them up.

## The Safety-Gymnasium environment (a second interpreter)

`experiments/run_safety.py` cannot run in the main environment.
`safety-gymnasium` pins `gymnasium==0.28.1`, `mujoco==2.3.3` and `pygame==2.1.0`
exactly, while the box runs gymnasium 1.3.0 and mujoco 3.11.0 — and
`run_invpend.py` needs `InvertedPendulum-v5`, an id that does not exist in
gymnasium 0.28. Installing it into the shared env would downgrade both libraries
under every other experiment. Concretely it also fails outright: `safety_gymnasium`
imports `gymnasium.wrappers.compatibility.EnvCompatibility`, removed in
gymnasium 1.0.

So it gets its own interpreter:

```bash
./docker/setup_safety_env.sh      # builds ~/envs/safety, ~5 minutes
```

Python 3.10, because `pygame==2.1.0` has no cp311 wheel and building it needs
SDL2 headers the image does not carry. The script finishes by making and
stepping `SafetyPointGoal1-v0` headless, so a silent GL failure surfaces at build
time rather than inside a queued job.

Queue jobs reach it with the `python` field:

```toml
[[job]]
script = "experiments/run_safety.py"
python = "~/envs/safety/bin/python"
```

`python` is optional and is recorded on a job only when set, so adding it does
not change the identity hash of any job that does not use it. Rendering needs
`MUJOCO_GL=egl`, which the container already exports (the runner inherits it).

Run it by hand the same way:

```bash
MUJOCO_GL=egl ~/envs/safety/bin/python experiments/run_safety.py \
    --env SafetyPointGoal1-v0 --alpha 0.1 --cost-penalty 0.25 --episodes 1000
```
