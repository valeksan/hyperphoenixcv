import importlib.util
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.hyperphoenixcv.core import HyperPhoenixCV
from src.hyperphoenixcv.result_manager import ResultManager
from src.hyperphoenixcv.storage.sqlite_store import SQLiteStudyStore
from src.hyperphoenixcv.study_identity import StudyIdentity


def _identity():
    return StudyIdentity.create(
        estimator=LogisticRegression(), param_grid={"C": [0.1]}, scoring=["accuracy", "f1"],
        cv=2, random_state=1, dataset_id="multi-test", scorer_id=None, cv_id=None,
    )


def _search(**kwargs):
    refit = kwargs.pop("refit", False)
    return HyperPhoenixCV(
        LogisticRegression(), strategy="optuna", search_space={}, n_trials=2,
        scoring=["accuracy", "f1"], refit=refit, verbose=False, **kwargs,
    )


def test_vector_objective_and_diagnostics_round_trip(tmp_path):
    with SQLiteStudyStore(str(tmp_path / "study.sqlite3")) as store:
        study_id = store.open_study(_identity())
        result = {
            "params": {"C": 0.1}, "objective_values": {"accuracy": 0.8, "f1": 0.7},
            "trial_diagnostics": {"intermediate_reports": [{"step": 1, "value": 0.5}]},
        }
        assert store.commit_trial(study_id, result["params"], result)
        assert store.results(study_id) == [result]


def test_pareto_front_honors_mixed_directions():
    search = _search(optuna_directions={"accuracy": "maximize", "f1": "minimize"})
    search.result_manager = ResultManager(["accuracy", "f1"])
    search.result_manager.add_results([
        {"params": {"C": 1}, "objective_values": {"accuracy": 0.9, "f1": 0.3}},
        {"params": {"C": 2}, "objective_values": {"accuracy": 0.8, "f1": 0.4}},
        {"params": {"C": 3}, "objective_values": {"accuracy": 0.95, "f1": 0.5}},
    ])
    assert search._pareto_front() == [
        {"trial_index": 0, "params": {"C": 1}, "objective_values": {"accuracy": 0.9, "f1": 0.3}},
        {"trial_index": 2, "params": {"C": 3}, "objective_values": {"accuracy": 0.95, "f1": 0.5}},
    ]


def test_multiobjective_rejects_ambiguous_refit():
    search = _search(
        refit=True, optuna_directions={"accuracy": "maximize", "f1": "maximize"},
    )
    with pytest.raises(ValueError, match="Multi-objective"):
        search._validate_refit()


def test_directions_must_match_scoring_names():
    search = _search(optuna_directions={"accuracy": "maximize"})
    with pytest.raises(ValueError, match="exactly match"):
        search._validate_strategy()


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="optional Optuna dependency")
def test_multiobjective_fit_and_sqlite_resume_keep_pareto_front(tmp_path):
    X, y = make_classification(n_samples=80, n_features=5, random_state=7)
    import optuna

    settings = dict(
        estimator=LogisticRegression(max_iter=200), param_grid=None, strategy="optuna",
        search_space={"C": optuna.distributions.CategoricalDistribution([0.1, 1.0])},
        n_trials=2, scoring=["accuracy", "f1"],
        optuna_directions={"accuracy": "maximize", "f1": "maximize"},
        refit=False, random_state=7, cv=2, dataset_id="pareto-resume-test",
        checkpoint_path=str(tmp_path / "pareto.sqlite3"), results_csv=str(tmp_path / "pareto.csv"), verbose=False,
    )
    first = HyperPhoenixCV(**settings).fit(X, y)
    assert first.pareto_front_
    assert not hasattr(first, "best_params_")
    resumed = HyperPhoenixCV(**settings).fit(X, y)
    assert resumed.pareto_front_ == first.pareto_front_
    assert len(resumed.cv_results_["params"]) == 2


@pytest.mark.skipif(importlib.util.find_spec("optuna") is None, reason="optional Optuna dependency")
def test_intermediate_evaluator_failure_commits_diagnostics_and_plain_cv_is_not_pruned(tmp_path):
    X, y = make_classification(n_samples=60, n_features=5, random_state=3)
    import optuna

    common = dict(
        estimator=LogisticRegression(max_iter=200), param_grid=None, strategy="optuna",
        search_space={"C": optuna.distributions.CategoricalDistribution([0.1])}, n_trials=1,
        scoring="accuracy", random_state=3, cv=2, dataset_id="prune-test", verbose=False,
    )
    plain_path = tmp_path / "plain.sqlite3"
    plain = HyperPhoenixCV(**common, checkpoint_path=str(plain_path), results_csv=str(tmp_path / "plain.csv"), refit=False)
    plain.fit(X, y)
    with SQLiteStudyStore(str(plain_path)) as store:
        assert store.results(plain.study_id)[0].get("trial_state") != "pruned"

    def broken_evaluator(estimator, X, y, params, report, groups, fit_params):
        report(1, 0.5)
        raise RuntimeError("reporter exploded")

    broken_path = tmp_path / "broken.sqlite3"
    broken = HyperPhoenixCV(
        **common, checkpoint_path=str(broken_path), results_csv=str(tmp_path / "broken.csv"),
        intermediate_evaluator=broken_evaluator, refit=False, resume="never",
    ).fit(X, y)
    with SQLiteStudyStore(str(broken_path)) as store:
        result = store.results(broken.study_id)[0]
    assert "reporter exploded" in result["error"]
    assert result["trial_diagnostics"]["intermediate_reports"][0]["step"] == 1
