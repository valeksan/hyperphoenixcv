from __future__ import annotations

import joblib

from hyperphoenixcv.scheduler import SchedulerSpec, TrialScheduler
from hyperphoenixcv.search_strategies import create_search_strategy
from hyperphoenixcv.study_engine import StudySpec


def test_runtime_configuration_and_strategies_round_trip_through_joblib(tmp_path):
    objects = {
        "study_spec": StudySpec(
            scoring=("accuracy",), strategy="random", random_search=True,
            adaptive_search=True, early_stopping_patience=3, batch_size=1,
            total_candidates=10, optuna_directions=None,
        ),
        "scheduler": TrialScheduler(SchedulerSpec(
            parallelism="trials", n_jobs=1, inner_max_num_threads=None,
            trial_timeout=None, memmap_max_nbytes="1M", memmap_temp_folder=None,
            joblib_batch_size="auto",
        )),
        "grid": create_search_strategy({"C": [0.1, 1.0]}, "grid", None),
        "random": create_search_strategy(
            {"C": [0.1, 1.0]}, "random", 2, random_state=7,
        ),
    }
    path = tmp_path / "runtime.joblib"

    joblib.dump(objects, path)
    restored = joblib.load(path)

    assert restored["study_spec"] == objects["study_spec"]
    assert restored["scheduler"].worker_count() == 1
    assert restored["grid"].ask(1) == [{"C": 0.1}]
    assert len(restored["random"].ask(2)) == 2
