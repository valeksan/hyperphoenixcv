import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from hyperphoenixcv import HyperPhoenixCV


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return np.array([" ".join(map(str, row)) for row in X]), y


@pytest.fixture
def sample_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


@pytest.fixture
def sample_param_grid():
    return {"tfidf__max_features": [10, 20], "clf__C": [0.1, 1.0]}


def make_search(tmp_path, pipeline, grid, **kwargs):
    scoring = kwargs.pop("scoring", "accuracy")
    return HyperPhoenixCV(
        estimator=pipeline,
        search_space=grid,
        scoring=scoring,
        cv=2,
        n_jobs=1,
        storage_path=str(tmp_path / "study.sqlite3"),
        results_csv=str(tmp_path / "results.csv"),
        dataset_id="hyperphoenixcv-tests-v1",
        verbose=False,
        **kwargs,
    )


def test_initialization_has_no_persistence_side_effect(sample_pipeline, sample_param_grid, tmp_path):
    search = make_search(tmp_path, sample_pipeline, sample_param_grid)

    assert search.storage_path.endswith("study.sqlite3")
    assert not (tmp_path / "study.sqlite3").exists()


def test_full_grid_search_persists_sqlite_and_csv(sample_data, sample_pipeline, sample_param_grid, tmp_path):
    X, y = sample_data
    search = make_search(tmp_path, sample_pipeline, sample_param_grid)
    search.fit(X, y)

    assert len(search.cv_results_["params"]) == len(list(ParameterGrid(sample_param_grid)))
    assert search.cv_results_["params"][search.best_index_] == search.best_params_
    assert (tmp_path / "study.sqlite3").exists()
    assert (tmp_path / "results.csv").exists()


def test_sqlite_resume_does_not_duplicate_trials(sample_data, sample_pipeline, sample_param_grid, tmp_path):
    X, y = sample_data
    first = make_search(tmp_path, sample_pipeline, sample_param_grid)
    first.fit(X, y)
    resumed = make_search(tmp_path, sample_pipeline, sample_param_grid)
    resumed.fit(X, y)

    assert resumed.cv_results_["params"] == first.cv_results_["params"]
    assert len(resumed.cv_results_["params"]) == 4


def test_random_search_and_multiple_metrics_use_distinct_studies(sample_data, sample_pipeline, sample_param_grid, tmp_path):
    X, y = sample_data
    random = make_search(tmp_path / "random", sample_pipeline, sample_param_grid,
                         strategy="random", n_trials=2, random_state=7)
    random.fit(X, y)
    assert len(random.cv_results_["params"]) == 2

    multi = make_search(tmp_path / "multi", sample_pipeline, sample_param_grid,
                        scoring=["accuracy", "f1"], refit=False)
    multi.fit(X, y)
    assert {"mean_test_accuracy", "mean_test_f1"} <= set(multi.cv_results_)


def test_load_results_from_sqlite(sample_data, sample_pipeline, sample_param_grid, tmp_path):
    X, y = sample_data
    search = make_search(tmp_path, sample_pipeline, sample_param_grid)
    search.fit(X, y)

    loaded = make_search(tmp_path, sample_pipeline, sample_param_grid).load_results_from_checkpoint(2)
    assert len(loaded) == 2
    assert loaded.iloc[0]["mean_test_accuracy"] >= loaded.iloc[1]["mean_test_accuracy"]


def test_failed_trial_is_stored_without_publishing_cv_result(sample_data, sample_pipeline, tmp_path):
    X, y = sample_data
    grid = {"tfidf__max_features": [10, 20], "clf__C": ["invalid", 1.0]}
    search = make_search(tmp_path, sample_pipeline, grid, error_score=np.nan)
    search.fit(X, y)

    assert len(search.cv_results_["params"]) == len(list(ParameterGrid(grid)))
    assert "split0_test_accuracy" in search.cv_results_
    assert "rank_test_accuracy" in search.cv_results_
    assert any("error" in result for result in search._load_checkpoint())


def test_final_refit_predicts(sample_data, sample_pipeline, sample_param_grid, tmp_path):
    X, y = sample_data
    search = make_search(tmp_path, sample_pipeline, sample_param_grid, refit=True)
    search.fit(X, y)

    assert len(search.predict(X)) == len(y)


def test_fit_routes_sample_weight_to_cv_and_refit(tmp_path):
    X, y = make_classification(n_samples=40, n_features=4, random_state=42)
    search = HyperPhoenixCV(
        estimator=LogisticRegression(max_iter=1000),
        search_space={"C": [1.0]}, scoring="accuracy", cv=2,
        storage_path=str(tmp_path / "study.sqlite3"),
        results_csv=str(tmp_path / "results.csv"), dataset_id="fit-params-v1",
        verbose=False,
    )
    search.fit(X, y, sample_weight=np.linspace(1.0, 2.0, len(y)))

    assert hasattr(search, "best_estimator_")
