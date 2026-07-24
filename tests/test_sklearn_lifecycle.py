from pathlib import Path

import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import HyperPhoenixCV


def make_search(**overrides):
    kwargs = {
        "estimator": LogisticRegression(),
        "param_grid": {"C": [1.0]},
        "scoring": "accuracy",
        "cv": 2,
        "checkpoint_path": "checkpoint.pkl",
        "results_csv": "results.csv",
        "verbose": False,
    }
    kwargs.update(overrides)
    return HyperPhoenixCV(**kwargs)


def test_constructor_keeps_params_and_has_no_runtime_state(tmp_path):
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"existing checkpoint")
    scoring = "accuracy"

    search = make_search(checkpoint_path=str(checkpoint), scoring=scoring)

    assert search.scoring is scoring
    assert search.clear_checkpoint is False
    assert checkpoint.read_bytes() == b"existing checkpoint"
    assert not hasattr(search, "search_strategy")
    assert not hasattr(search, "checkpoint_manager")
    assert not hasattr(search, "result_manager")
    assert not hasattr(search, "cv_executor")
    assert not hasattr(search, "best_params_")
    assert not hasattr(search, "best_score_")
    assert not hasattr(search, "best_estimator_")
    assert not hasattr(search, "cv_results_")
    assert not hasattr(search, "best_index_")


def test_constructor_does_not_clear_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"existing checkpoint")

    make_search(checkpoint_path=str(checkpoint), clear_checkpoint=True)

    assert checkpoint.read_bytes() == b"existing checkpoint"


def test_clear_checkpoint_is_deprecated_at_fit_time(tmp_path):
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=30, n_features=4, random_state=0)
    search = make_search(
        checkpoint_path=str(tmp_path / "study.sqlite3"),
        results_csv=str(tmp_path / "results.csv"),
        clear_checkpoint=True,
    )

    with pytest.warns(FutureWarning, match="clear_checkpoint=True is deprecated"):
        search.fit(X, y)


def test_sklearn_clone_and_get_params_preserve_constructor_values():
    search = make_search(scoring="accuracy", clear_checkpoint=True)

    params = search.get_params(deep=False)
    assert params["scoring"] == "accuracy"
    assert params["clear_checkpoint"] is True
    cloned = clone(search)
    cloned_params = cloned.get_params(deep=False)
    assert cloned_params.pop("estimator") is not params["estimator"]
    assert cloned_params == {key: value for key, value in params.items() if key != "estimator"}
    assert not hasattr(cloned, "result_manager")


def test_predict_before_fit_raises_not_fitted_error():
    search = make_search()

    with pytest.raises(NotFittedError):
        search.predict([[0.0]])


def test_refit_false_does_not_publish_usable_estimator(tmp_path):
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=30, n_features=4, random_state=0)
    search = make_search(
        checkpoint_path=str(tmp_path / "checkpoint.pkl"),
        results_csv=str(tmp_path / "results.csv"),
        refit=False,
    )
    search.fit(X, y)

    assert not hasattr(search, "best_estimator_")
    with pytest.raises(NotFittedError):
        search.predict(X)


def test_multi_metric_refit_by_name(tmp_path):
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    search = make_search(
        checkpoint_path=str(tmp_path / "checkpoint.sqlite3"),
        results_csv=str(tmp_path / "results.csv"),
        param_grid={"C": [0.1, 1.0]},
        scoring={"acc": "accuracy", "f1_score": "f1"},
        refit="f1_score",
    )
    search.fit(X, y)

    assert search.best_score_ == max(search.cv_results_["mean_test_f1_score"])
    assert hasattr(search, "best_estimator_")


def test_multi_metric_refit_true_is_rejected_before_persistence(tmp_path):
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    checkpoint = tmp_path / "checkpoint.sqlite3"
    search = make_search(
        checkpoint_path=str(checkpoint),
        results_csv=str(tmp_path / "results.csv"),
        scoring={"acc": "accuracy", "f1_score": "f1"},
        refit=True,
    )

    with pytest.raises(ValueError, match="multi-metric"):
        search.fit(X, y)
    assert not checkpoint.exists()


def test_multi_metric_refit_callable_selects_cv_result_index(tmp_path):
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    search = make_search(
        checkpoint_path=str(tmp_path / "checkpoint.sqlite3"),
        results_csv=str(tmp_path / "results.csv"),
        param_grid={"C": [0.1, 1.0]},
        scoring={"acc": "accuracy", "f1_score": "f1"},
        refit=lambda cv_results: 1,
        scorer_id="refit-callable-test-v1",
    )
    search.fit(X, y)

    assert search.best_index_ == 1
    assert search.best_params_ == search.cv_results_["params"][1]
    assert hasattr(search, "best_estimator_")
