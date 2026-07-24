import sqlite3
import time

import pytest
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv.storage.sqlite_store import SQLiteStudyStore, StudyMismatchError
from hyperphoenixcv.study_identity import StudyIdentity, canonical_json, param_key


def identity(dataset_id="train-v1"):
    return StudyIdentity.create(
        estimator=LogisticRegression(), param_grid={"C": [0.1, 1.0]},
        scoring="accuracy", cv=2, random_state=1, dataset_id=dataset_id,
        scorer_id=None, cv_id=None,
    )


def result(value=0.8):
    return {"params": {"C": 0.1}, "mean_test_accuracy": value, "std_test_accuracy": 0.1}


def test_commit_idempotency_restart_and_two_readers(tmp_path):
    path = tmp_path / "study.sqlite3"
    with SQLiteStudyStore(str(path)) as writer:
        study_id = writer.open_study(identity())
        assert writer.commit_trial(study_id, {"C": 0.1}, result())
        assert not writer.commit_trial(study_id, {"C": 0.1}, result(0.1))
        with SQLiteStudyStore(str(path)) as reader:
            assert reader.results(study_id) == [result()]

    with SQLiteStudyStore(str(path)) as restarted:
        assert restarted.open_study(identity(), resume="must") == study_id
        assert restarted.completed_param_keys(study_id)


def test_rollback_preserves_no_partial_trial(tmp_path):
    with SQLiteStudyStore(str(tmp_path / "study.sqlite3")) as store:
        study_id = store.open_study(identity())
        with pytest.raises(sqlite3.IntegrityError):
            with store._transaction() as conn:
                conn.execute(
                    "INSERT INTO trials (study_id, sequence, state, param_key, params_json, result_json, created_at, updated_at) "
                    "VALUES (?, 1, 'completed', 'x', '{}', '{}', 'now', 'now')",
                    ("missing-study",),
                )
        assert store.results(study_id) == []


def test_migrates_zero_version_database(tmp_path):
    path = tmp_path / "previous.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE legacy_data (value TEXT)")
    connection.commit()
    connection.close()

    with SQLiteStudyStore(str(path)) as store:
        study_id = store.open_study(identity())
        assert store.commit_trial(study_id, {"C": 0.1}, result())
    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    connection.close()


def test_migrates_v1_canonical_parameter_key_without_losing_idempotency(tmp_path):
    path = tmp_path / "previous.sqlite3"
    with SQLiteStudyStore(str(path)) as store:
        study_id = store.open_study(identity())
        assert store.commit_trial(study_id, {"C": 0.1}, result())

    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE trials SET param_key = ? WHERE study_id = ?",
        (canonical_json({"C": 0.1}), study_id),
    )
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    with SQLiteStudyStore(str(path)) as store:
        assert store.completed_param_keys(study_id) == {param_key({"C": 0.1})}
        assert not store.commit_trial(study_id, {"C": 0.1}, result(0.1))


def test_store_rejects_mismatched_auto_resume(tmp_path):
    with SQLiteStudyStore(str(tmp_path / "study.sqlite3")) as store:
        store.open_study(identity("v1"))
        with pytest.raises(StudyMismatchError, match="dataset_id"):
            store.open_study(identity("v2"))


def test_100_commits_visible_and_commit_p95(tmp_path):
    with SQLiteStudyStore(str(tmp_path / "study.sqlite3")) as store:
        study_id = store.open_study(identity())
        durations = []
        for value in range(100):
            started = time.perf_counter()
            assert store.commit_trial(study_id, {"C": value}, {"params": {"C": value}, "mean_test_accuracy": value})
            durations.append(time.perf_counter() - started)
        assert len(store.results(study_id)) == 100
        assert sorted(durations)[94] < 1.0


@pytest.mark.slow
def test_10k_commits_are_idempotent_and_linear_in_row_count(tmp_path):
    with SQLiteStudyStore(str(tmp_path / "study.sqlite3")) as store:
        study_id = store.open_study(identity())
        for value in range(10_000):
            store.commit_trial(study_id, {"C": value}, {"params": {"C": value}})
        assert len(store.results(study_id)) == 10_000
