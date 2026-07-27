import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import HyperPhoenixCV
from hyperphoenixcv.legacy_pickle import LegacyCheckpointError, LegacyCheckpointTrustError


def make_search(tmp_path):
    return HyperPhoenixCV(
        estimator=LogisticRegression(),
        search_space={"C": [0.1, 1.0]},
        scoring="accuracy",
        cv=2,
        storage_path=str(tmp_path / "study.sqlite3"),
        dataset_id="legacy-import-v1",
        verbose=False,
    )


def test_legacy_import_requires_explicit_trust(tmp_path, monkeypatch):
    source = tmp_path / "legacy.pkl"
    joblib.dump([], source)
    search = make_search(tmp_path)
    monkeypatch.setattr(
        "hyperphoenixcv.legacy_pickle.joblib.load",
        lambda _: pytest.fail("untrusted import must not deserialize"),
    )

    with pytest.raises(LegacyCheckpointTrustError, match="trusted=True"):
        search.import_legacy_checkpoint(str(source))


def test_trusted_legacy_import_reports_validation_and_is_idempotent(tmp_path):
    source = tmp_path / "legacy.pkl"
    legacy = [
        {"params": {"C": 0.1}, "mean_test_accuracy": 0.7, "std_test_accuracy": 0.1},
        {"params": {"C": 1.0}, "mean_test_accuracy": 0.8, "std_test_accuracy": 0.1},
        {"params": {"C": 0.1}, "mean_test_accuracy": 0.1},
        {"params": "not-a-dict"},
        "not-a-result",
    ]
    joblib.dump(legacy, source)
    before = source.read_bytes()
    search = make_search(tmp_path)

    with pytest.warns(UserWarning, match="execute arbitrary code"):
        report = search.import_legacy_checkpoint(str(source), trusted=True)
    assert report["imported"] == 2
    assert report["skipped"] == 1
    assert report["failed"] == 2
    assert [failure["index"] for failure in report["failures"]] == [3, 4]
    assert source.read_bytes() == before
    assert len(search.load_results_from_checkpoint()) == 2

    with pytest.warns(UserWarning):
        repeated = search.import_legacy_checkpoint(str(source), trusted=True)
    assert repeated["imported"] == 0
    assert repeated["skipped"] == 3
    assert repeated["failed"] == 2


def test_legacy_import_rejects_non_list_top_level(tmp_path):
    source = tmp_path / "legacy.pkl"
    joblib.dump({"params": {}}, source)

    with pytest.warns(UserWarning), pytest.raises(LegacyCheckpointError, match="List\\[dict\\]"):
        make_search(tmp_path).import_legacy_checkpoint(str(source), trusted=True)


def test_normal_fit_does_not_load_checkpoint_pickle(tmp_path, monkeypatch):
    search = make_search(tmp_path)
    monkeypatch.setattr(
        "hyperphoenixcv.legacy_pickle.joblib.load",
        lambda _: pytest.fail("fit must not deserialize pickle"),
    )
    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    search.fit(X, y)
