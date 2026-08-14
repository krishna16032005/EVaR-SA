"""Small helpers shared by the experiment scripts."""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import torch


def state_to_tensor(obs: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(obs, dtype=torch.float32)


def resolve_device(choice: str = "auto") -> torch.device:
    """Maps a ``--device`` flag to a concrete device.

    ``"auto"`` picks CUDA when a GPU is visible (the usual case inside the
    container on a GPU host) and falls back to CPU otherwise, so the same command
    works on a laptop and on the server. An explicit ``"cuda"`` on a machine with
    no GPU is a mistake worth surfacing, so it raises rather than silently
    degrading to a 10x-slower CPU run.
    """
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but torch.cuda.is_available() is False. "
            "Inside Docker this usually means the container was started without "
            "`--gpus` (or without the NVIDIA container toolkit installed)."
        )
    return torch.device(choice)


def new_run_tag() -> str:
    """A timestamp suffix (e.g. ``20260810-153000``) used to keep each run's output
    directory and wandb run name unique so successive runs never overwrite each
    other's CSVs/checkpoints."""
    return time.strftime("%Y%m%d-%H%M%S")


def _write_records(path: str, records: list[dict]) -> None:
    if not records:
        return
    fieldnames = sorted({key for record in records for key in record})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_records(directory: str, prefix: str, logs: dict) -> None:
    """Write tidy (one row per episode / per update) CSVs from :func:`evar_deeprl.agents.base.train`'s output.

    ``logs`` is expected to have ``"episode_records"`` and ``"update_records"`` keys,
    each a list of flat dicts sharing a ``global_step`` column so the two files can be
    merged/aligned during offline analysis (e.g. with pandas). Any other (non-list)
    keys, such as ``"wandb_run_id"``, are ignored here.
    """
    os.makedirs(directory, exist_ok=True)
    for key, records in logs.items():
        if not key.endswith("_records"):
            continue
        suffix = key.replace("_records", "")
        path = os.path.join(directory, f"{prefix}_{suffix}.csv")
        _write_records(path, records)
        print(f"Saved {len(records)} rows to {path}")


def discounted_support(reward_min: float, reward_max: float, horizon: int,
                       gamma: float, margin: float = 0.05) -> tuple[float, float]:
    """C51 support bounds for the quantity the critic actually represents.

    The critic learns the *discounted* return ``E[sum_t gamma^t r_t]``, which for a
    bounded per-step reward can never exceed ``r_max * (1 - gamma^T) / (1 - gamma)``
    -- 99.3 for CartPole at gamma = 0.99, not the 500 of its undiscounted episode
    return. Sizing the support in undiscounted units is not merely untidy: it was
    leaving ~9 of 51 atoms carrying any mass (~5 of 51 on InvertedPendulum),
    measured over 10k episodes, and EVaR is a tail statistic read off exactly that
    histogram. A tail estimate from a 9-atom grid is dominated by discretisation.

    ``margin`` widens the range slightly so the extreme atoms are not pinned by the
    projection.
    """
    if gamma >= 1.0:
        scale = float(horizon)
    else:
        scale = (1.0 - gamma ** horizon) / (1.0 - gamma)
    lo, hi = reward_min * scale, reward_max * scale
    pad = margin * max(abs(lo), abs(hi), 1.0)
    return lo - pad, hi + pad
