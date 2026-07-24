from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import pytest

from hyperphoenixcv import HyperPhoenixCV
from hyperphoenixcv.storage.sqlite_store import SQLiteStudyStore


def make_search(tmp_path, **kwargs):
    values = {
        "estimator": LogisticRegression(max_iter=100),
        "param_grid": {"C": [0.1, 1.0, 10.0]},
        "scoring": "accuracy",
        "cv": 2,
        "checkpoint_path": str(tmp_path / "study.sqlite3"),
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
    ],
)
def test_scheduler_rejects_invalid_resource_settings(tmp_path, kwargs, message):
    X, y = make_classification(n_samples=20, n_features=4, random_state=0)
    search = make_search(tmp_path, **kwargs)

    with pytest.raises(ValueError, match=message):
        search.fit(X, y)
