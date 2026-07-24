import pytest
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv.study_identity import (
    StudyIdentity,
    UnsupportedIdentityValueError,
    param_key,
)
from hyperphoenixcv.storage.sqlite_store import SQLiteStudyStore, StudyMismatchError


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


def test_param_key_is_sha256_and_stable_when_dictionary_order_changes():
    first = param_key({"C": 1.0, "solver": "lbfgs"})
    second = param_key({"solver": "lbfgs", "C": 1.0})

    assert first == second
    assert len(first) == 64
    assert all(char in "0123456789abcdef" for char in first)


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
    path = tmp_path / "study.sqlite3"
    first = make_identity()
    with SQLiteStudyStore(str(path)) as store:
        store.open_study(first)

    with SQLiteStudyStore(str(path)) as store:
        with pytest.raises(StudyMismatchError, match=field):
            store.open_study(make_identity(**change))


def test_resume_policy_and_schema_validation(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    identity = make_identity()
    with SQLiteStudyStore(str(missing)) as store:
        with pytest.raises(FileNotFoundError):
            store.open_study(identity, resume="must")
        created = store.open_study(identity, resume="never")
        assert store.open_study(identity, resume="must") == created


def test_callable_requires_explicit_stable_id():
    with pytest.raises(UnsupportedIdentityValueError, match="scorer_id"):
        make_identity(scoring=lambda estimator, X, y: 1.0)
    assert make_identity(scoring=lambda estimator, X, y: 1.0, scorer_id="metric-v1")
