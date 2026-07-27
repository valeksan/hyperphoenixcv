from __future__ import annotations

from hyperphoenixcv.storage import SQLiteStudyStore, StudyStore
from hyperphoenixcv.study_identity import StudyIdentity
from sklearn.linear_model import LogisticRegression


def assert_study_store_contract(store: StudyStore, identity: StudyIdentity) -> None:
    study_id = store.open_study(identity, resume="auto")
    assert store.results(study_id) == []
    assert store.completed_param_keys(study_id) == set()

    state = store.study_state(study_id)
    state["contract"] = True
    store.update_study_state(study_id, state)
    assert store.study_state(study_id)["contract"] is True

    result = {"params": {"C": 1.0}, "mean_test_accuracy": 0.9}
    assert store.commit_trial(study_id, result["params"], result) is True
    assert store.commit_trial(study_id, result["params"], result) is False
    assert store.results(study_id) == [result]
    assert len(store.completed_param_keys(study_id)) == 1


def test_sqlite_study_store_contract(tmp_path):
    identity = StudyIdentity.create(
        estimator=LogisticRegression(), param_grid={"C": [1.0]}, scoring="accuracy",
        cv=2, random_state=7, dataset_id="store-contract-v1", scorer_id=None, cv_id=None,
    )
    store = SQLiteStudyStore(tmp_path / "study.sqlite3")
    try:
        assert_study_store_contract(store, identity)
    finally:
        store.close()
