import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold

from hyperphoenixcv import HyperPhoenixCV


def make_search(tmp_path, **kwargs):
    values = {
        "estimator": LogisticRegression(max_iter=200),
        "param_grid": {"C": [0.1, 1.0]},
        "scoring": "accuracy",
        "cv": 2,
        "checkpoint_path": str(tmp_path / "study.sqlite3"),
        "results_csv": str(tmp_path / "results.csv"),
        "dataset_id": "sklearn-parity-v1",
        "verbose": False,
    }
    values.update(kwargs)
    return HyperPhoenixCV(**values)


def test_grid_search_parity_for_split_scores_and_refit(tmp_path):
    X, y = make_classification(n_samples=60, n_features=5, random_state=4)
    grid = GridSearchCV(
        LogisticRegression(max_iter=200), {"C": [0.1, 1.0]}, scoring="accuracy", cv=2
    ).fit(X, y)
    search = make_search(tmp_path)
    search.fit(X, y)

    assert search.cv_results_["params"] == grid.cv_results_["params"]
    assert search.cv_results_["mean_test_accuracy"] == pytest.approx(grid.cv_results_["mean_test_score"])
    assert search.cv_results_["split0_test_accuracy"] == pytest.approx(grid.cv_results_["split0_test_score"])
    assert search.best_params_ == grid.best_params_
    assert search.best_score_ == pytest.approx(grid.best_score_)


def test_multi_metric_refit_name_matches_grid_search(tmp_path):
    X, y = make_classification(n_samples=60, n_features=5, random_state=4)
    scoring = {"acc": "accuracy", "f1_score": "f1"}
    grid = GridSearchCV(
        LogisticRegression(max_iter=200), {"C": [0.1, 1.0]}, scoring=scoring, cv=2, refit="f1_score"
    ).fit(X, y)
    search = make_search(tmp_path, scoring=scoring, refit="f1_score")
    search.fit(X, y)

    assert search.best_params_ == grid.best_params_
    assert search.best_score_ == pytest.approx(grid.best_score_)
    assert search.cv_results_["mean_test_f1_score"] == pytest.approx(grid.cv_results_["mean_test_f1_score"])


def test_numeric_error_score_records_failed_candidate_like_grid_search(tmp_path):
    X, y = make_classification(n_samples=60, n_features=5, random_state=4)
    params = {"C": ["invalid", 1.0]}
    with pytest.warns():
        grid = GridSearchCV(
            LogisticRegression(max_iter=200), params, scoring="accuracy", cv=2, error_score=np.nan
        ).fit(X, y)
    search = make_search(tmp_path, param_grid=params, error_score=np.nan)
    search.fit(X, y)

    assert np.isnan(search.cv_results_["mean_test_accuracy"][0])
    assert np.isnan(grid.cv_results_["mean_test_score"][0])
    assert "error" in search._load_checkpoint()[0]


def test_groups_reach_group_kfold_like_grid_search(tmp_path):
    X, y = make_classification(n_samples=60, n_features=5, random_state=4)
    groups = np.repeat(np.arange(12), 5)
    cv = GroupKFold(n_splits=3)
    grid = GridSearchCV(
        LogisticRegression(max_iter=200), {"C": [0.1, 1.0]}, scoring="accuracy", cv=cv
    ).fit(X, y, groups=groups)
    search = make_search(tmp_path, cv=cv, cv_id="group-kfold-3-v1")
    search.fit(X, y, groups=groups)

    assert search.cv_results_["mean_test_accuracy"] == pytest.approx(grid.cv_results_["mean_test_score"])
