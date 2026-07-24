from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import HyperPhoenixCV
from hyperphoenixcv.cv_executor import CVExecutor
from hyperphoenixcv.result_manager import ResultManager
from hyperphoenixcv.storage.sqlite_store import SQLiteStudyStore
from hyperphoenixcv.search_strategies import SearchStrategy


class SimulatedCrash(BaseException):
    """Fault-injection stand-in for abrupt process termination."""


def make_search(tmp_path: Path, *, refit: bool = True) -> HyperPhoenixCV:
    return HyperPhoenixCV(
        estimator=LogisticRegression(max_iter=100),
        param_grid={"C": [0.1, 1.0, 10.0]},
        scoring="accuracy",
        cv=2,
        random_state=7,
        dataset_id="crash-recovery-v1",
        checkpoint_path=str(tmp_path / "checkpoint.pkl"),
        results_csv=str(tmp_path / "results.csv"),
        verbose=False,
        refit=refit,
    )


def deterministic_evaluate(self, estimator, X, y, params, groups=None):
    score = float(params["C"])
    return {
        "params": params,
        "mean_test_accuracy": score,
        "std_test_accuracy": 0.0,
    }


def trial_results(search: HyperPhoenixCV) -> list[dict]:
    with SQLiteStudyStore(search._storage_path()) as store:
        study_id = store.open_study(search._identity_for_loading(), resume="must")
        return store.results(study_id)


@pytest.mark.parametrize("crash_point", ["during_evaluation", "before_commit"])
def test_uncommitted_trial_is_retried_after_crash(tmp_path, monkeypatch, crash_point):
    X, y = make_classification(n_samples=30, n_features=4, random_state=7)
    search = make_search(tmp_path)

    if crash_point == "during_evaluation":
        monkeypatch.setattr(
            CVExecutor,
            "evaluate",
            lambda *args, **kwargs: (_ for _ in ()).throw(SimulatedCrash()),
        )
    else:
        monkeypatch.setattr(CVExecutor, "evaluate", deterministic_evaluate)
        monkeypatch.setattr(
            SQLiteStudyStore,
            "commit_trial",
            lambda *args, **kwargs: (_ for _ in ()).throw(SimulatedCrash()),
        )

    with pytest.raises(SimulatedCrash):
        search.fit(X, y)
    assert trial_results(search) == []

    monkeypatch.undo()
    monkeypatch.setattr(CVExecutor, "evaluate", deterministic_evaluate)
    resumed = make_search(tmp_path)
    resumed.fit(X, y)

    assert len(trial_results(resumed)) == 3
    assert resumed.best_params_ == {"C": 10.0}


def test_committed_trial_survives_crash_before_next_proposal(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=30, n_features=4, random_state=7)
    search = make_search(tmp_path)
    monkeypatch.setattr(CVExecutor, "evaluate", deterministic_evaluate)
    original_commit = SQLiteStudyStore.commit_trial
    calls = 0

    def commit_then_crash(self, *args, **kwargs):
        nonlocal calls
        committed = original_commit(self, *args, **kwargs)
        calls += 1
        if calls == 1:
            raise SimulatedCrash()
        return committed

    monkeypatch.setattr(SQLiteStudyStore, "commit_trial", commit_then_crash)
    with pytest.raises(SimulatedCrash):
        search.fit(X, y)

    committed = trial_results(search)
    assert len(committed) == 1
    assert committed[0]["params"] == {"C": 0.1}

    monkeypatch.undo()
    evaluated = []

    def count_evaluation(*args, **kwargs):
        evaluated.append(kwargs["params"])
        return deterministic_evaluate(*args, **kwargs)

    monkeypatch.setattr(CVExecutor, "evaluate", count_evaluation)
    resumed = make_search(tmp_path)
    resumed.fit(X, y)

    assert evaluated == [{"C": 1.0}, {"C": 10.0}]
    assert len(trial_results(resumed)) == 3
    assert resumed.best_params_ == {"C": 10.0}


def test_random_sampler_replays_after_commit_before_tell(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=30, n_features=4, random_state=7)
    kwargs = dict(
        estimator=LogisticRegression(max_iter=100),
        param_grid={"C": [0.1, 1.0, 10.0, 100.0]},
        scoring="accuracy", cv=2, random_search=True, n_iter=3,
        random_state=None, dataset_id="random-post-commit-v1",
        checkpoint_path=str(tmp_path / "random.sqlite3"),
        results_csv=str(tmp_path / "random.csv"), verbose=False, refit=False,
    )
    monkeypatch.setattr(CVExecutor, "evaluate", deterministic_evaluate)
    original_tell = SearchStrategy.tell
    calls = 0

    def tell_then_crash(self, results):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SimulatedCrash()
        return original_tell(self, results)

    monkeypatch.setattr(SearchStrategy, "tell", tell_then_crash)
    with pytest.raises(SimulatedCrash):
        HyperPhoenixCV(**kwargs).fit(X, y)

    monkeypatch.undo()
    resumed = HyperPhoenixCV(**kwargs)
    resumed.fit(X, y)
    results = trial_results(resumed)

    assert len(results) == 3
    assert len({str(result["params"]) for result in results}) == 3
    with SQLiteStudyStore(resumed._storage_path()) as store:
        state = store.study_state(resumed.study_id)
    assert isinstance(state["sampler_random_state"], int)


def test_random_early_stopping_recovers_counter_after_post_commit_crash(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=30, n_features=4, random_state=7)
    search = HyperPhoenixCV(
        estimator=LogisticRegression(max_iter=100),
        param_grid={"C": [0.1, 1.0, 10.0, 100.0]},
        scoring="accuracy",
        cv=2,
        random_search=True,
        n_iter=4,
        random_state=7,
        early_stopping_patience=2,
        dataset_id="early-stop-v1",
        checkpoint_path=str(tmp_path / "checkpoint.sqlite3"),
        results_csv=str(tmp_path / "results.csv"),
        verbose=False,
        refit=False,
    )
    monkeypatch.setattr(
        CVExecutor,
        "evaluate",
        lambda self, estimator, X, y, params, groups=None: {
            "params": params,
            "mean_test_accuracy": 1.0,
            "std_test_accuracy": 0.0,
        },
    )
    original_commit = SQLiteStudyStore.commit_trial
    commits = 0

    def commit_then_crash(self, *args, **kwargs):
        nonlocal commits
        committed = original_commit(self, *args, **kwargs)
        commits += 1
        if commits == 2:
            raise SimulatedCrash()
        return committed

    monkeypatch.setattr(SQLiteStudyStore, "commit_trial", commit_then_crash)
    with pytest.raises(SimulatedCrash):
        search.fit(X, y)

    monkeypatch.undo()
    resumed = HyperPhoenixCV(**search.get_params(deep=False))
    resumed.fit(X, y)

    assert len(trial_results(resumed)) == 3
    with SQLiteStudyStore(resumed._storage_path()) as store:
        state = store.study_state(resumed.study_id)["early_stopping"]
    assert state["no_improvement_count"] == 2
    assert state["stop_reason"] == "patience_exhausted"


def test_csv_export_failure_preserves_trials_and_resume_does_not_re_evaluate(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=30, n_features=4, random_state=7)
    search = make_search(tmp_path)
    monkeypatch.setattr(CVExecutor, "evaluate", deterministic_evaluate)
    monkeypatch.setattr(
        ResultManager,
        "save_to_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("csv failed")),
    )

    with pytest.raises(RuntimeError, match="csv failed"):
        search.fit(X, y)
    assert len(trial_results(search)) == 3

    monkeypatch.undo()
    monkeypatch.setattr(
        CVExecutor,
        "evaluate",
        lambda *args, **kwargs: pytest.fail("committed trials must not be evaluated again"),
    )
    resumed = make_search(tmp_path)
    resumed.fit(X, y)

    assert resumed.best_params_ == {"C": 10.0}
    assert len(trial_results(resumed)) == 3


def test_final_refit_failure_preserves_trials_and_resume_refits(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=30, n_features=4, random_state=7)
    search = make_search(tmp_path)
    monkeypatch.setattr(CVExecutor, "evaluate", deterministic_evaluate)
    import hyperphoenixcv.core as core_module

    original_clone = core_module.clone

    def clone_with_failing_fit(estimator):
        cloned = original_clone(estimator)
        cloned.fit = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("refit failed"))
        return cloned

    monkeypatch.setattr(core_module, "clone", clone_with_failing_fit)
    with pytest.raises(RuntimeError, match="refit failed"):
        search.fit(X, y)
    assert len(trial_results(search)) == 3

    monkeypatch.undo()
    monkeypatch.setattr(
        CVExecutor,
        "evaluate",
        lambda *args, **kwargs: pytest.fail("committed trials must not be evaluated again"),
    )
    resumed = make_search(tmp_path)
    resumed.fit(X, y)

    assert resumed.best_params_ == {"C": 10.0}
    assert hasattr(resumed.best_estimator_, "coef_")


@pytest.mark.parametrize("run", range(20))
def test_post_commit_recovery_is_stable_over_twenty_runs(tmp_path, monkeypatch, run):
    X, y = make_classification(n_samples=20, n_features=4, random_state=run)
    path = tmp_path / str(run)
    path.mkdir()
    search = make_search(path, refit=False)
    monkeypatch.setattr(CVExecutor, "evaluate", deterministic_evaluate)
    original_commit = SQLiteStudyStore.commit_trial

    def commit_then_crash(self, *args, **kwargs):
        original_commit(self, *args, **kwargs)
        raise SimulatedCrash()

    monkeypatch.setattr(SQLiteStudyStore, "commit_trial", commit_then_crash)
    with pytest.raises(SimulatedCrash):
        search.fit(X, y)
    monkeypatch.undo()
    monkeypatch.setattr(CVExecutor, "evaluate", deterministic_evaluate)

    resumed = make_search(path, refit=False)
    resumed.fit(X, y)
    assert len(trial_results(resumed)) == 3
    assert resumed.best_params_ == {"C": 10.0}
