"""Local joblib scheduler for one primary parallelism axis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from joblib import Parallel, delayed, parallel_config


def _evaluate_trial(evaluator: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pickle-friendly trial worker entry point."""
    return evaluator.evaluate(**kwargs)


@dataclass(frozen=True)
class SchedulerSpec:
    parallelism: str
    n_jobs: int
    inner_max_num_threads: int | None
    trial_timeout: float | None
    memmap_max_nbytes: int | str | None
    memmap_temp_folder: str | None
    joblib_batch_size: int | str


class TrialScheduler:
    """Bounded local trial scheduler; no distributed/shared-SQLite promise."""

    def __init__(self, spec: SchedulerSpec) -> None:
        self.spec = spec
        self.validate()

    def worker_count(self) -> int:
        if self.spec.n_jobs == 0:
            raise ValueError("n_jobs must not be 0")
        if self.spec.n_jobs > 0:
            return self.spec.n_jobs
        return max(1, (os.cpu_count() or 1) + 1 + self.spec.n_jobs)

    def validate(self) -> None:
        if self.spec.parallelism not in {"trials", "folds"}:
            raise ValueError("parallelism must be 'trials' or 'folds'")
        workers = self.worker_count()
        if self.spec.inner_max_num_threads is not None and self.spec.inner_max_num_threads < 1:
            raise ValueError("inner_max_num_threads must be a positive integer or None")
        if self.spec.trial_timeout is not None:
            if self.spec.trial_timeout <= 0:
                raise ValueError("trial_timeout must be a positive number or None")
            if self.spec.parallelism != "trials" or workers < 2:
                raise ValueError("trial_timeout requires parallelism='trials' and n_jobs >= 2")
        if self.spec.joblib_batch_size != "auto" and (
            not isinstance(self.spec.joblib_batch_size, int) or self.spec.joblib_batch_size < 1
        ):
            raise ValueError("joblib_batch_size must be 'auto' or a positive integer")

    def evaluate(self, evaluator: Any, kwargs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(kwargs) == 1 and self.spec.trial_timeout is None:
            return [_evaluate_trial(evaluator, kwargs[0])]
        config = {
            "backend": "loky",
            "n_jobs": self.worker_count() if self.spec.trial_timeout is not None else len(kwargs),
            "inner_max_num_threads": self.spec.inner_max_num_threads,
            "max_nbytes": self.spec.memmap_max_nbytes,
            "mmap_mode": "r",
            "temp_folder": self.spec.memmap_temp_folder,
        }
        with parallel_config(**config):
            try:
                return Parallel(timeout=self.spec.trial_timeout, batch_size=self.spec.joblib_batch_size)(
                    delayed(_evaluate_trial)(evaluator, item) for item in kwargs
                )
            except TimeoutError:
                if len(kwargs) != 1:
                    raise RuntimeError("scheduler timeout must evaluate exactly one trial")
                return [{
                    "params": kwargs[0]["params"],
                    "error": f"trial exceeded timeout of {self.spec.trial_timeout} seconds",
                    "error_type": "TrialTimeout",
                    "cancellation_reason": "trial_timeout",
                }]
