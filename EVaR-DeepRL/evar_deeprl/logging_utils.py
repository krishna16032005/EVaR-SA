"""Unified experiment logging: local tidy-CSV records plus optional Weights & Biases.

Every scalar the training loop computes (losses, gradient norms, advantage/EVaR
statistics, policy diagnostics, timing) is captured as one flat dict per event and
handed to :class:`RunLogger`, which appends it to an in-memory record list (later
written to CSV by :func:`evar_deeprl.utils.save_records`) and, if enabled, mirrors it
to a wandb run. wandb is an optional dependency: if it isn't installed, or
``wandb_cfg.mode == "disabled"``, training proceeds with local CSV logging only.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None


def _has_wandb_credentials() -> bool:
    """True if wandb can resolve an API key (env var, netrc/_netrc, or local settings).

    Local-only check (no network call) so it's safe to run before every ``online``
    attempt without risking a hang or a slow request.
    """
    if wandb is None:
        return False
    try:
        return bool(wandb.Api().api_key)
    except Exception:
        return False


@dataclass
class WandbConfig:
    """Weights & Biases settings. ``mode="disabled"`` (the default) runs no wandb code."""

    mode: str = "disabled"  # "online", "offline", or "disabled"
    project: str = "evar-deeprl"
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()
    watch_models: bool = True
    watch_log_freq: int = 50


class RunLogger:
    """Accumulates per-update / per-episode / per-training-run records.

    All records share a single monotonic ``global_step`` (cumulative environment
    steps) so update-level and episode-level metrics line up on the same x-axis in
    wandb, and so the local CSVs can be joined/aligned during offline analysis.
    """

    def __init__(self, wandb_cfg: WandbConfig, run_config: dict[str, Any]):
        self.wandb_cfg = wandb_cfg
        self.update_records: list[dict[str, Any]] = []
        self.episode_records: list[dict[str, Any]] = []
        self.eval_records: list[dict[str, Any]] = []
        self._start_time = time.time()
        self._run = None
        self._run_id: str | None = None

        if wandb_cfg.mode == "disabled":
            return
        if wandb is None:
            print("[logging] wandb is not installed (`pip install wandb`); logging to CSV only.")
            return
        mode = wandb_cfg.mode
        if mode == "online" and not _has_wandb_credentials():
            print(
                "[logging] No wandb login found (checked WANDB_API_KEY and the netrc file; "
                "run `wandb login` to enable online sync); falling back to offline mode for this run."
            )
            mode = "offline"
        try:
            self._run = wandb.init(
                project=wandb_cfg.project,
                entity=wandb_cfg.entity,
                name=wandb_cfg.run_name,
                group=wandb_cfg.group,
                tags=list(wandb_cfg.tags),
                mode=mode,
                config=run_config,
            )
            self._run_id = self._run.id
            if mode == "online":
                print(f"[logging] wandb run '{self._run.id}': {self._run.url}")
            else:
                print(f"[logging] wandb offline run '{self._run.id}' (sync later with `wandb sync {self._run.dir}`)")
        except Exception as exc:  # wandb misconfiguration should never kill a training run.
            print(f"[logging] wandb.init failed ({exc!r}); continuing with CSV logging only.")
            self._run = None

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def active(self) -> bool:
        return self._run is not None

    def watch(self, actor, critic) -> None:
        if self.active and self.wandb_cfg.watch_models:
            wandb.watch([actor, critic], log="all", log_freq=self.wandb_cfg.watch_log_freq)

    def log_update(self, record: dict[str, Any]) -> None:
        record = {**record, "wall_time_s": time.time() - self._start_time}
        self.update_records.append(record)
        if self.active:
            wandb.log({f"update/{k}": v for k, v in record.items()}, step=record.get("global_step"))

    def log_episode(self, record: dict[str, Any]) -> None:
        record = {**record, "wall_time_s": time.time() - self._start_time}
        self.episode_records.append(record)
        if self.active:
            wandb.log({f"episode/{k}": v for k, v in record.items()}, step=record.get("global_step"))

    def log_eval(self, record: dict[str, Any]) -> None:
        record = {**record, "wall_time_s": time.time() - self._start_time}
        self.eval_records.append(record)
        if self.active:
            wandb.log({f"eval/{k}": v for k, v in record.items()}, step=record.get("global_step"))

    def log_histogram(self, name: str, values: Sequence[float], step: int | None = None) -> None:
        if self.active:
            wandb.log({name: wandb.Histogram(list(values))}, step=step)

    def log_np_histogram(self, name: str, counts: Sequence[float], edges: Sequence[float], step: int | None = None) -> None:
        if self.active:
            wandb.log({name: wandb.Histogram(np_histogram=(list(counts), list(edges)))}, step=step)

    def finish(self) -> None:
        if self.active:
            wandb.finish()
            self._run = None


def add_wandb_args(parser) -> None:
    """Adds the standard ``--wandb-*`` flags shared by every experiment script."""
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="'online' (default) streams live to your logged-in wandb account, "
        "falling back to offline automatically if no login/WANDB_API_KEY is found; "
        "'offline' always writes a local run only (sync later with `wandb sync`); "
        "'disabled' skips wandb entirely (CSV logging only).",
    )
    parser.add_argument("--wandb-project", type=str, default="evar-deeprl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=())


def wandb_config_from_args(args) -> WandbConfig:
    return WandbConfig(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        tags=tuple(args.wandb_tags),
    )
