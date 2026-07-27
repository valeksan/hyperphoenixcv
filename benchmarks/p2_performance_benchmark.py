"""Reproducible P2.7 performance and scale gate.

Run a release record (five samples protects against a single noisy run)::

    python benchmarks/p2_performance_benchmark.py --output benchmarks/p2_baseline.json

Compare a candidate with a recorded release::

    python benchmarks/p2_performance_benchmark.py --baseline benchmarks/p2_baseline.json

The default resume sizes include 100,000 rows and can take several minutes on
durable SQLite.  Use ``--resume-sizes 1000`` for a fast local smoke run.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import resource
import statistics
import tempfile
import time
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification

from hyperphoenixcv import HyperPhoenixCV
from hyperphoenixcv.result_manager import ResultManager
from hyperphoenixcv.search_grid import ExhaustiveSearchStrategy
from hyperphoenixcv.storage.sqlite_store import SQLiteStudyStore
from hyperphoenixcv.study_identity import StudyIdentity, param_key


# Delay is per fold.  With cv=2, these yield genuine approximately 5/30/150ms
# trial durations, independent of host CPU speed or estimator convergence.
WORKLOADS = {
    "cheap": {"delay": 0.0025, "trials": 8},
    "medium": {"delay": 0.015, "trials": 8},
    "expensive": {"delay": 0.075, "trials": 8},
}
LARGE_SPACE = {"left": range(1_000), "right": range(1_000), "third": range(1_000)}
REGRESSION_METRICS = (
    "wall_seconds", "proposal_p95_seconds", "store_commit_p95_seconds",
    "peak_rss_kib", "serialized_bytes",
)


class DelayedMajorityClassifier(ClassifierMixin, BaseEstimator):
    """Deterministic estimator used only to make trial-cost taxonomy explicit."""

    def __init__(self, delay: float = 0.0, label: int = 0) -> None:
        self.delay = delay
        self.label = label

    def fit(self, X: Any, y: Any) -> "DelayedMajorityClassifier":
        time.sleep(self.delay)
        values, counts = np.unique(y, return_counts=True)
        self.classes_ = values
        self.majority_ = values[np.argmax(counts)]
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), self.majority_)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def _summary(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {"median": statistics.median([sample[key] for sample in samples]),
               "p95": _percentile([sample[key] for sample in samples], 95)}
        for key in samples[0]
    }


def _identity(name: str) -> StudyIdentity:
    return StudyIdentity.create(
        estimator=DelayedMajorityClassifier(), search_space={"label": [0]},
        scoring="accuracy", cv=2, random_state=7, dataset_id=f"p2-{name}",
        scorer_id=None, cv_id=None,
    )


def _proposal_latencies(strategy: ExhaustiveSearchStrategy, count: int) -> list[float]:
    strategy.restore([])
    latencies = []
    for _ in range(count):
        started = time.perf_counter()
        strategy.ask(1)
        latencies.append(time.perf_counter() - started)
    return latencies


def _store_metrics(root: Path, name: str, count: int = 100) -> tuple[list[float], int]:
    path = root / f"{name}-store.sqlite3"
    with SQLiteStudyStore(path) as store:
        study_id = store.open_study(_identity(f"store-{name}"))
        latencies = []
        for value in range(count):
            params = {"label": value}
            result = {"params": params, "mean_test_accuracy": 1.0}
            started = time.perf_counter()
            store.commit_trial(study_id, params, result)
            latencies.append(time.perf_counter() - started)
        assert store.connection is not None
        serialized = store.connection.execute(
            "SELECT COALESCE(SUM(length(params_json) + length(result_json)), 0) "
            "FROM trials WHERE study_id = ?", (study_id,),
        ).fetchone()[0]
    return latencies, int(serialized)


def _fit_workload(
    root: Path, name: str, *, parallelism: str = "trials", n_jobs: int = 1,
    inner_max_num_threads: int | None = None,
) -> tuple[float, float]:
    spec = WORKLOADS[name]
    X, y = make_classification(
        n_samples=80, n_features=6, n_informative=3, random_state=7,
    )
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    HyperPhoenixCV(
        estimator=DelayedMajorityClassifier(),
        search_space={"delay": [spec["delay"]], "label": list(range(spec["trials"]))},
        strategy="grid", scoring="accuracy", cv=2, n_jobs=n_jobs, refit=False,
        verbose=False, dataset_id=f"p2-fit-{name}", storage_path=str(root / f"{name}.sqlite3"),
        results_csv=str(root / f"{name}.csv"), parallelism=parallelism,
        inner_max_num_threads=inner_max_num_threads,
    ).fit(X, y)
    return time.perf_counter() - started_wall, time.process_time() - started_cpu


def measure_workload(name: str) -> dict[str, float]:
    """Measure one taxonomy workload; values are suitable for JSON baselines."""
    spec = WORKLOADS[name]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        proposal = _proposal_latencies(
            ExhaustiveSearchStrategy({"delay": [spec["delay"]], "label": range(200)}), 100,
        )
        store, serialized = _store_metrics(root, name)
        wall, cpu = _fit_workload(root, name)
    return {
        "wall_seconds": wall,
        "trials_per_second": spec["trials"] / wall,
        "proposal_p50_seconds": _percentile(proposal, 50),
        "proposal_p95_seconds": _percentile(proposal, 95),
        "store_commit_p50_seconds": _percentile(store, 50),
        "store_commit_p95_seconds": _percentile(store, 95),
        "peak_rss_kib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "serialized_bytes": float(serialized),
        "cpu_utilization_percent": 100 * cpu / wall if wall else 0.0,
        "target_trial_seconds": 2 * spec["delay"],
    }


def measure_large_space() -> dict[str, float]:
    """Prove 10^9 combinations can be counted/proposed without materializing them."""
    strategy = ExhaustiveSearchStrategy(LARGE_SPACE)
    started = time.perf_counter()
    candidates = strategy.total_candidates()
    first = list(islice(strategy.iter_parameters(), 10))
    elapsed = time.perf_counter() - started
    if candidates != 1_000_000_000 or len(first) != 10:
        raise AssertionError("large grid must stay lazy and contain exactly 10^9 candidates")
    return {"candidate_count": float(candidates), "first_ten_seconds": elapsed,
            "peak_rss_kib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}


def measure_parallelism() -> list[dict[str, float | int | str]]:
    """Compare supported axes under same workload; never run both axes at once."""
    report: list[dict[str, float | int | str]] = []
    for parallelism in ("trials", "folds"):
        with tempfile.TemporaryDirectory() as directory:
            wall, cpu = _fit_workload(
                Path(directory), "medium", parallelism=parallelism, n_jobs=2,
                inner_max_num_threads=1,
            )
        report.append({
            "parallelism": parallelism, "n_jobs": 2, "inner_max_num_threads": 1,
            "wall_seconds": wall, "cpu_utilization_percent": 100 * cpu / wall if wall else 0.0,
        })
    return report


def measure_resume_latency(size: int) -> dict[str, float]:
    """Time durable history read + hash restoration + next proposal."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "resume.sqlite3"
        with SQLiteStudyStore(path) as store:
            study_id = store.open_study(_identity(f"resume-{size}"))
            for value in range(size):
                params = {"label": value}
                store.commit_trial(study_id, params, {"params": params, "mean_test_accuracy": 1.0})
            strategy = ExhaustiveSearchStrategy({"label": range(size + 1)})
            started = time.perf_counter()
            strategy.restore(store.iter_results(study_id))
            proposal = strategy.ask(1)
            elapsed = time.perf_counter() - started
    if proposal != [{"label": size}]:
        raise AssertionError("resume did not skip durable terminal trials")
    return {"trials": float(size), "resume_seconds": elapsed,
            "peak_rss_kib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}


def measure_projection_bound() -> dict[str, float]:
    """Check P2.3's no-retention path at the documented 100k scale."""
    manager = ResultManager(["accuracy"], retain_results=False)
    total = 100_000
    for value in range(total):
        manager.add_result({"params": {"label": value}, "mean_test_accuracy": 1.0})
    if manager.results:
        raise AssertionError("projection bound violated")
    return {"retained_rows": float(len(manager.results)), "offered_rows": float(total)}


def _transport_size(payload: bytes) -> int:
    """Pickle-friendly joblib transport target; no application work."""
    return len(payload)


def measure_profiled_components() -> dict[str, float]:
    """Direct timings for hotspots hidden below cProfile's end-to-end top 30."""
    count = 100_000
    started = time.perf_counter()
    for value in range(count):
        param_key({"label": value})
    hash_seconds = time.perf_counter() - started

    started = time.perf_counter()
    projection = measure_projection_bound()
    projection_seconds = time.perf_counter() - started

    payload = b"x" * 1_000_000
    started = time.perf_counter()
    sizes = Parallel(n_jobs=2, backend="loky")(
        delayed(_transport_size)(payload) for _ in range(8)
    )
    transport_seconds = time.perf_counter() - started
    if sizes != [len(payload)] * 8:
        raise AssertionError("joblib transport altered payload")
    return {
        "param_key_100k_seconds": hash_seconds,
        "projection_100k_seconds": projection_seconds,
        "joblib_transport_8mib_seconds": transport_seconds,
        "projection_retained_rows": projection["retained_rows"],
    }


def profile_engine(output: Path) -> list[dict[str, Any]]:
    """Emit top cumulative functions; inspect before optimizing any hotspot."""
    profiler = cProfile.Profile()
    with tempfile.TemporaryDirectory() as directory:
        profiler.enable()
        _fit_workload(Path(directory), "medium")
        profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(30)
    output.write_text(stream.getvalue())
    return [{"profile": str(output), "top": stream.getvalue().splitlines()[-30:]}]


def compare(baseline: dict[str, Any], report: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Flag only median regressions beyond threshold; never fail on one sample."""
    regressions: dict[str, dict[str, float]] = {}
    for name in WORKLOADS:
        old, new = baseline["workloads"][name], report["workloads"][name]
        for metric in REGRESSION_METRICS:
            old_value, new_value = old[metric]["median"], new[metric]["median"]
            if old_value and (new_value - old_value) / old_value > threshold:
                regressions.setdefault(name, {})[metric] = (new_value - old_value) / old_value
    return {"median_regression_threshold": threshold, "regressions": regressions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--resume-sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--regression-threshold", type=float, default=0.15)
    parser.add_argument("--profile", type=Path, default=Path("benchmarks/p2_profile.txt"))
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be >= 3; a single noisy run is not a regression signal")
    report: dict[str, Any] = {
        "schema": 1, "platform": {"python": os.sys.version, "cpu_count": os.cpu_count()},
        "workloads": {name: _summary([measure_workload(name) for _ in range(args.runs)]) for name in WORKLOADS},
        "large_space": measure_large_space(),
        "parallelism": measure_parallelism(),
        "resume_latency": [measure_resume_latency(size) for size in args.resume_sizes],
        "projection_bound": measure_projection_bound(),
        "profiled_components": measure_profiled_components(),
        "profile": profile_engine(args.profile),
    }
    if args.baseline:
        report["comparison"] = compare(json.loads(args.baseline.read_text()), report, args.regression_threshold)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
