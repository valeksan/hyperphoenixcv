import os

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import checkpoint as checkpoint_module
from hyperphoenixcv.checkpoint import CheckpointManager
from hyperphoenixcv.study_identity import StudyIdentity


def identity():
    return StudyIdentity.create(
        estimator=LogisticRegression(), param_grid={"C": [1.0]}, scoring="accuracy",
        cv=2, random_state=0, dataset_id="train-v1", scorer_id=None, cv_id=None,
    )


def committed_manager(path):
    manager = CheckpointManager(str(path), verbose=False)
    manager.save([{"params": {"C": 1.0}}], identity())
    return manager


def test_dump_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "study.pkl"
    manager = committed_manager(path)
    old = joblib.load(path)["results"]

    def fail_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(checkpoint_module.joblib, "dump", fail_dump)
    with pytest.raises(OSError, match="disk full"):
        manager.save([{"params": {"C": 2.0}}], identity())

    assert joblib.load(path)["results"] == old


def test_replace_failure_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "study.pkl"
    manager = committed_manager(path)
    old = joblib.load(path)["results"]

    def fail_replace(*args, **kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="permission denied"):
        manager.save([{"params": {"C": 2.0}}], identity())

    assert joblib.load(path)["results"] == old
    assert not list(tmp_path.glob(".study.pkl.*.tmp"))


def test_corrupt_checkpoint_never_becomes_empty_state(tmp_path):
    path = tmp_path / "study.pkl"
    path.write_bytes(b"not a joblib checkpoint")

    with pytest.raises(RuntimeError, match="Automatic pickle resume"):
        CheckpointManager(str(path), verbose=False).load_envelope(identity())


def test_stray_temp_file_is_not_a_checkpoint(tmp_path):
    path = tmp_path / "study.pkl"
    (tmp_path / ".study.pkl.crash.tmp").write_text("incomplete")

    with pytest.raises(RuntimeError, match="Automatic pickle resume"):
        CheckpointManager(str(path), verbose=False).load_envelope(identity())
