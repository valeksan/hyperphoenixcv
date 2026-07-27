from hyperphoenixcv.storage import SQLiteStudyStore, StudyStore


def test_sqlite_store_implements_study_store_protocol(tmp_path):
    store = SQLiteStudyStore(tmp_path / "study.sqlite3")
    try:
        assert isinstance(store, StudyStore)
    finally:
        store.close()
