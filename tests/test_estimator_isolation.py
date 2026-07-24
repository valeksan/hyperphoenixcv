import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import HyperPhoenixCV
from hyperphoenixcv.cv_executor import CVExecutor
from hyperphoenixcv.result_manager import ResultManager


def test_cv_evaluation_does_not_mutate_source_estimator():
    estimator = LogisticRegression(C=1.0, max_iter=100)
    executor = CVExecutor(cv=2, scoring="accuracy", verbose=False)
    X, y = make_classification(n_samples=30, n_features=4, random_state=0)

    executor.evaluate(estimator, X, y, {"C": 0.25})

    assert estimator.get_params()["C"] == 1.0
    assert not hasattr(estimator, "coef_")


def test_second_fit_does_not_duplicate_checkpoint_results(tmp_path):
    X, y = make_classification(n_samples=30, n_features=4, random_state=0)
    estimator = LogisticRegression(max_iter=100)
    search = HyperPhoenixCV(
        estimator=estimator,
        param_grid={"C": [0.5, 1.0]},
        scoring="accuracy",
        cv=2,
        checkpoint_path=str(tmp_path / "checkpoint.pkl"),
        results_csv=str(tmp_path / "results.csv"),
        verbose=False,
    )

    search.fit(X, y)
    first_params = list(search.cv_results_["params"])
    search.fit(X, y)

    assert search.cv_results_["params"] == first_params
    assert len(search.cv_results_["params"]) == 2
    assert search.best_estimator_ is not estimator


def test_result_manager_duplicate_result_is_idempotent():
    manager = ResultManager(scoring=["accuracy"])
    result = {"params": {"C": np.float64(1.0)}, "mean_test_accuracy": 0.8}

    manager.add_result(result)
    manager.add_result(result.copy())

    assert manager.results == [result]
