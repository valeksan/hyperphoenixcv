import pytest
from sklearn.linear_model import LogisticRegression

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
