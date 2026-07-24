import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv.checkpoint import CheckpointManager
from hyperphoenixcv.study_identity import (
    CheckpointMismatchError,
    CheckpointSchemaError,
    StudyIdentity,
    UnsupportedIdentityValueError,
)


def make_identity(**overrides):
    config = {
        "estimator": LogisticRegression(C=1.0),
        "param_grid": {"C": [0.1, 1.0], "solver": ["lbfgs"]},
        "scoring": "accuracy",
        "cv": 2,
        "random_state": 7,
        "dataset_id": "train-v1",
        "scorer_id": None,
        "cv_id": None,
    }
    config.update(overrides)
    return StudyIdentity.create(**config)


def test_identity_is_stable_when_grid_dict_order_changes():
    first = make_identity(param_grid={"C": [0.1, 1.0], "solver": ["lbfgs"]})
    second = make_identity(param_grid={"solver": ["lbfgs"], "C": [0.1, 1.0]})

    assert first.space_digest == second.space_digest
    assert first.config_digest == second.config_digest


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"estimator": LogisticRegression(C=0.5)}, "estimator_digest"),
        ({"param_grid": {"C": [0.2]}}, "space_digest"),
        ({"scoring": "f1"}, "scorer_digest"),
        ({"cv": 3}, "cv_digest"),
        ({"dataset_id": "train-v2"}, "dataset_id"),
    ],
)
def test_resume_rejects_changed_identity(tmp_path, change, field):
    path = tmp_path / "study.pkl"
    first = make_identity()
    CheckpointManager(str(path), verbose=False).save([{"params": {"C": 1.0}}], first)

    with pytest.raises(CheckpointMismatchError, match=field):
        CheckpointManager(str(path), verbose=False).load_envelope(make_identity(**change))


def test_resume_policy_and_schema_validation(tmp_path):
    missing = tmp_path / "missing.pkl"
    identity = make_identity()
    manager = CheckpointManager(str(missing), verbose=False)

    with pytest.raises(FileNotFoundError):
        manager.load_envelope(identity, resume="must")
    assert manager.load_envelope(identity, resume="never") is None

    joblib.dump({"schema_version": 999}, missing)
    with pytest.raises(CheckpointSchemaError, match="missing fields"):
        manager.load_envelope(identity)


def test_callable_requires_explicit_stable_id():
    with pytest.raises(UnsupportedIdentityValueError, match="scorer_id"):
        make_identity(scoring=lambda estimator, X, y: 1.0)
    assert make_identity(scoring=lambda estimator, X, y: 1.0, scorer_id="metric-v1")
