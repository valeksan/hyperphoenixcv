from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pytest
from contextlib import contextmanager

from hyperphoenixcv import HyperPhoenixCV
from hyperphoenixcv.storage.sqlite_store import SQLiteStudyStore


def make_search(tmp_path, **kwargs):
    values = {
        "estimator": LogisticRegression(max_iter=100),
        "search_space": {"C": [0.1, 1.0, 10.0]},
        "scoring": "accuracy",
        "cv": 2,
        "storage_path": str(tmp_path / "study.sqlite3"),
        "results_csv": str(tmp_path / "results.csv"),
        "dataset_id": "scheduler-test-v1",
        "verbose": False,
        "refit": False,
    }
    values.update(kwargs)
    return HyperPhoenixCV(**values)


def test_trial_parallelism_bounds_batch_by_workers_and_remaining_budget(tmp_path):
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    search = make_search(tmp_path, n_jobs=2, parallelism="trials")
    search.fit(X, y)

    assert search.cv_executor.n_jobs == 1
    assert len(search.cv_results_["params"]) == 3
    with SQLiteStudyStore(search._storage_path()) as store:
        state = store.study_state(search.study_id)["scheduler"]
    assert state == {
        "parallelism": "trials", "n_jobs": 2, "worker_count": 2,
        "inner_max_num_threads": None,
        "trial_timeout": None, "memmap_max_nbytes": "1M",
        "memmap_temp_folder": None, "joblib_batch_size": "auto",
        "attempts": 3, "cancellation_reason": None,
    }


def test_fold_parallelism_uses_cv_workers_and_sequential_trials(tmp_path):
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    search = make_search(tmp_path, n_jobs=2, parallelism="folds")
    search.fit(X, y)

    assert search.cv_executor.n_jobs == 2
    assert len(search.cv_results_["params"]) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"parallelism": "nested"}, "parallelism"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"inner_max_num_threads": 0}, "inner_max_num_threads"),
        ({"trial_timeout": 0}, "trial_timeout"),
        ({"trial_timeout": 1, "n_jobs": 1}, "trial_timeout"),
        ({"joblib_batch_size": 0}, "joblib_batch_size"),
    ],
)
def test_scheduler_rejects_invalid_resource_settings(tmp_path, kwargs, message):
    X, y = make_classification(n_samples=20, n_features=4, random_state=0)
    search = make_search(tmp_path, **kwargs)

    with pytest.raises(ValueError, match=message):
        search.fit(X, y)


def test_cancel_callback_stops_before_next_trial_and_persists_reason(tmp_path):
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    search = make_search(tmp_path, cancel_callback=lambda: "user_cancelled")
    search.fit(X, y)

    assert search.cv_results_ == {}
    with SQLiteStudyStore(search._storage_path()) as store:
        scheduler = store.study_state(search.study_id)["scheduler"]
    assert scheduler["cancellation_reason"] == "user_cancelled"
    assert scheduler["attempts"] == 0


def test_timeout_commits_resumable_failed_trial_and_reason(tmp_path, monkeypatch):
    import hyperphoenixcv.scheduler as scheduler_module

    class TimedOutParallel:
        def __init__(self, **kwargs):
            assert kwargs["timeout"] == 0.01

        def __call__(self, tasks):
            list(tasks)
            raise TimeoutError()

    monkeypatch.setattr(scheduler_module, "Parallel", TimedOutParallel)
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    search = make_search(
        tmp_path, search_space={"C": [0.1]}, n_jobs=2, trial_timeout=0.01,
    )
    search.fit(X, y)

    with SQLiteStudyStore(search._storage_path()) as store:
        results = store.results(search.study_id)
        scheduler = store.study_state(search.study_id)["scheduler"]
    assert results[0]["error_type"] == "TrialTimeout"
    assert scheduler["cancellation_reason"] == "trial_timeout"
    assert scheduler["attempts"] == 1


def test_trial_parallelism_passes_thread_and_memmap_limits_without_oversubscription(tmp_path, monkeypatch):
    import hyperphoenixcv.scheduler as scheduler_module

    seen = {}

    @contextmanager
    def fake_parallel_config(**kwargs):
        seen["config"] = kwargs
        yield

    class ImmediateParallel:
        def __init__(self, **kwargs):
            seen["parallel"] = kwargs

        def __call__(self, tasks):
            return [func(*args, **kwargs) for func, args, kwargs in tasks]

    monkeypatch.setattr(scheduler_module, "parallel_config", fake_parallel_config)
    monkeypatch.setattr(scheduler_module, "Parallel", ImmediateParallel)
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    search = make_search(
        tmp_path, search_space={"C": [0.1, 1.0]}, n_jobs=2,
        inner_max_num_threads=1, memmap_max_nbytes="2K", joblib_batch_size=1,
    )
    search.fit(X, y)

    assert seen["config"] == {
        "backend": "loky", "n_jobs": 2, "inner_max_num_threads": 1,
        "max_nbytes": "2K", "mmap_mode": "r", "temp_folder": None,
    }
    assert seen["parallel"] == {"timeout": None, "batch_size": 1}
