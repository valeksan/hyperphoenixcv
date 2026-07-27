import sqlite3

import pytest

import hyperphoenixcv.storage.sqlite_store as sqlite_store
from hyperphoenixcv.storage.sqlite_store import (
    SQLiteStudyStore, StorageCorruptionError, StorageDiskFullError,
    StorageLockedError, StoragePermissionError, StorageSchemaError,
)
from tests.test_sqlite_store import identity, result


def _populated(path):
    with SQLiteStudyStore(path) as store:
        study_id = store.open_study(identity())
        assert store.commit_trial(study_id, {"C": 0.1}, result())
    return study_id


def test_backup_restore_and_read_only_inspection_during_active_fit(tmp_path):
    path, backup, restored = tmp_path / "study.sqlite3", tmp_path / "backup.sqlite3", tmp_path / "restored.sqlite3"
    study_id = _populated(path)
    with SQLiteStudyStore(path) as writer:
        with SQLiteStudyStore(path) as reader:
            assert reader.integrity_check()["ok"]
            assert reader.results(study_id) == [result()]
        assert writer.backup_to(backup) == backup
    assert SQLiteStudyStore.restore_from(backup, restored) == restored
    with SQLiteStudyStore(restored) as store:
        assert store.results(study_id) == [result()]
        assert store.integrity_check()["ok"]


def test_integrity_reports_malformed_json_and_abandoned_study_cleanup(tmp_path):
    path = tmp_path / "study.sqlite3"
    study_id = _populated(path)
    with SQLiteStudyStore(path) as store:
        empty = store.open_study(identity("empty"), resume="never")
        assert store.prune_empty_studies() == 1
        assert store.connection.execute("SELECT 1 FROM studies WHERE study_id = ?", (empty,)).fetchone() is None
        store.connection.execute("UPDATE trials SET result_json = '{broken' WHERE study_id = ?", (study_id,))
        report = store.integrity_check()
        assert not report["ok"]
        assert "Malformed JSON" in report["errors"][0]
        with pytest.raises(StorageCorruptionError):
            store.results(study_id)


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_migrations_keep_terminal_trials_on_schema_copies(tmp_path, version):
    path = tmp_path / f"v{version}.sqlite3"
    study_id = _populated(path)
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA user_version = {version}")
    connection.commit()
    connection.close()
    with SQLiteStudyStore(path) as store:
        assert store.results(study_id) == [result()]
        assert store.connection.execute("PRAGMA user_version").fetchone()[0] == 5


def test_interrupted_migration_rolls_back_without_losing_terminal_trial(tmp_path):
    path = tmp_path / "interrupted.sqlite3"
    study_id = _populated(path)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA ignore_check_constraints = ON")
    connection.execute(
        """INSERT INTO trials (
            study_id, sequence, state, param_key, params_json, result_json,
            created_at, updated_at
        ) VALUES (?, 2, 'cancelled', 'injected', '{}', '{}', 'now', 'now')""",
        (study_id,),
    )
    connection.execute("PRAGMA user_version = 3")
    connection.commit()
    connection.close()
    with pytest.raises(Exception, match="CHECK constraint failed"):
        SQLiteStudyStore(path)
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT COUNT(*) FROM trials WHERE study_id = ?", (study_id,)).fetchone()[0] == 2
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    connection.execute("DELETE FROM trials WHERE param_key = 'injected'")
    connection.commit()
    connection.close()
    with SQLiteStudyStore(path) as store:
        assert store.results(study_id) == [result()]


def test_invalid_partial_schema_is_diagnosable(tmp_path):
    path = tmp_path / "partial.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE studies (study_id TEXT PRIMARY KEY)")
    connection.commit()
    connection.close()
    with pytest.raises(StorageSchemaError, match="user_version is 0"):
        SQLiteStudyStore(path)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (OSError(28, "No space left on device"), StorageDiskFullError),
        (PermissionError("Permission denied"), StoragePermissionError),
        (sqlite3.OperationalError("database is locked"), StorageLockedError),
    ],
)
def test_storage_fault_taxonomy(tmp_path, monkeypatch, error, expected):
    def fail_connect(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(sqlite_store.sqlite3, "connect", fail_connect)
    with pytest.raises(expected):
        SQLiteStudyStore(tmp_path / "study.sqlite3")
