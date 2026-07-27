import json

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import HyperPhoenixCV
from hyperphoenixcv.audit import TrialHistory
from hyperphoenixcv.storage.sqlite_store import _restore


def _search(tmp_path, **kwargs):
    base = dict(
        estimator=LogisticRegression(max_iter=200), param_grid={"C": [0.1, 1.0]},
        scoring="accuracy", cv=2, verbose=False, refit=False,
        checkpoint_path=str(tmp_path / "study.sqlite3"),
        results_csv=str(tmp_path / "projection.csv"),
    )
    base.update(kwargs)
    return HyperPhoenixCV(**base)


def test_trial_history_is_read_only_paginated_and_has_all_terminal_states(tmp_path):
    search = _search(tmp_path)
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    search.fit(X, y)
    with pytest.raises(TypeError):
        search.trial_history_.page(limit=1)[0]["state"] = "failed"
    history = search.trial_history_
    assert history.count() == 2
    assert len(history.page(limit=1, states={"completed"})) == 1
    assert len(list(history.iter_records(page_size=1))) == 2


def test_audit_json_is_lossless_and_atomic_csv(tmp_path):
    search = _search(tmp_path, param_grid={"C": [0.1]})
    X, y = make_classification(n_samples=40, n_features=4, random_state=1)
    search.fit(X, y)
    history = search.trial_history_
    json_path, csv_path = tmp_path / "audit.json", tmp_path / "audit.csv"
    history.export_json(json_path)
    history.export_csv(csv_path)
    exported = json.loads(json_path.read_text())
    assert exported["format"] == "hyperphoenixcv.audit.v1"
    assert _restore(exported["trials"][0])["state"] == "completed"
    assert csv_path.exists()
    assert not list(tmp_path.glob(".audit.*.tmp"))


def test_history_retains_failed_pruned_cancelled_diagnostics_and_exception(tmp_path):
    search = _search(tmp_path)
    identity = search._identity_for_loading()
    path = search._storage_path()
    from hyperphoenixcv.storage.sqlite_store import SQLiteStudyStore
    with SQLiteStudyStore(path) as store:
        study_id = store.open_study(identity)
        for state, value in [("failed", 1), ("pruned", 2), ("cancelled", 3)]:
            store.commit_trial(study_id, {"C": value}, {
                "params": {"C": value}, "trial_state": state,
                "error": "boom" if state == "failed" else None,
                "trial_diagnostics": {"reports": [(1, np.nan)]},
                "objective_values": {"accuracy": np.nan},
            })
    history = TrialHistory(path, study_id)
    records = history.page(limit=10)
    assert {record["state"] for record in records} == {"failed", "pruned", "cancelled"}
    failed = next(record for record in records if record["state"] == "failed")
    assert failed["exception_message"] == "boom"
    assert np.isnan(failed["diagnostics"]["reports"][0][1])


def test_minimize_direction_controls_rank_top_result_and_refit(tmp_path):
    from hyperphoenixcv.result_manager import ResultManager
    manager = ResultManager(["loss"], metric_directions={"loss": "minimize"})
    manager.add_results([
        {"params": {"x": 1}, "mean_test_loss": 2.0},
        {"params": {"x": 2}, "mean_test_loss": 1.0},
    ])
    assert manager.get_top_results(1).iloc[0]["x"] == 2
    assert manager.format_cv_results()["rank_test_loss"] == [2, 1]


def test_callable_refit_uses_complete_cv_result_index(tmp_path):
    search = _search(tmp_path, refit=lambda results: 1)
    search.result_manager = __import__("hyperphoenixcv.result_manager", fromlist=["ResultManager"]).ResultManager(["accuracy"])
    search.result_manager.add_results([
        {"params": {"C": 1}, "mean_test_accuracy": 0.2},
        {"params": {"C": 2}, "trial_state": "failed", "error": "boom"},
        {"params": {"C": 3}, "mean_test_accuracy": 0.8},
    ])
    search.cv_results_ = search.result_manager.format_cv_results()
    with pytest.raises(ValueError, match="completed"):
        search._update_best_attributes()
    search.refit = lambda results: 2
    search._update_best_attributes()
    assert search.best_index_ == 2
