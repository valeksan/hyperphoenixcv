"""Record P1 scheduler baselines: proposal latency, RSS, throughput, store time.

Run: ``python benchmarks/p1_scheduler_benchmark.py --output benchmarks/p1_baseline.json``.
Use ``--baseline`` to print per-workload deltas against a previous JSON record.
"""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import HyperPhoenixCV
from hyperphoenixcv.search_strategies import RandomSearchStrategy
from hyperphoenixcv.storage.sqlite_store import SQLiteStudyStore
from hyperphoenixcv.study_identity import StudyIdentity


WORKLOADS = {"cheap": (80, 4, 2), "medium": (400, 12, 3), "expensive": (1200, 30, 3)}


def measure(name: str) -> dict[str, float]:
    samples, features, cv = WORKLOADS[name]
    X, y = make_classification(n_samples=samples, n_features=features, random_state=7)
    strategy = RandomSearchStrategy({"C": range(1, 1_001)}, n_trials=100, random_state=7)
    strategy.restore([])
    started = time.perf_counter()
    strategy.ask(100)
    proposal_latency = time.perf_counter() - started
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        started = time.perf_counter()
        search = HyperPhoenixCV(
            estimator=LogisticRegression(max_iter=200), search_space={"C": [0.1, 1.0, 10.0]},
            strategy="grid", scoring="accuracy",
            cv=cv, n_jobs=1, refit=False, verbose=False, dataset_id=f"benchmark-{name}",
            storage_path=str(root / "study.sqlite3"), results_csv=str(root / "results.csv"),
        ).fit(X, y)
        elapsed = time.perf_counter() - started
        identity = StudyIdentity.create(
            estimator=LogisticRegression(), search_space={"x": [1]}, scoring="accuracy", cv=2,
            random_state=7, dataset_id=f"store-{name}", scorer_id=None, cv_id=None,
        )
        with SQLiteStudyStore(str(root / "store.sqlite3")) as store:
            study_id = store.open_study(identity)
            started = time.perf_counter()
            for value in range(100):
                store.commit_trial(study_id, {"x": value}, {"params": {"x": value}, "mean_test_accuracy": 1.0})
            store_seconds = time.perf_counter() - started
    return {
        "proposal_latency_seconds": proposal_latency,
        "peak_rss_kib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "throughput_trials_per_second": 3 / elapsed,
        "store_commit_seconds_per_trial": store_seconds / 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    report = {name: measure(name) for name in WORKLOADS}
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        report["delta"] = {name: {key: values[key] - baseline[name][key] for key in values} for name, values in report.items()}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
