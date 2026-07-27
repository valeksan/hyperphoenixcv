import pytest
import warnings
import inspect
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import HyperPhoenixCV


def _canonical(tmp_path, **kwargs):
    config = dict(
        estimator=LogisticRegression(max_iter=100),
        search_space={"C": [0.1, 1.0]},
        strategy="grid",
        storage_path=str(tmp_path / "study.sqlite3"),
        scoring="accuracy",
        cv=2,
        dataset_id="api-migration",
        results_csv=str(tmp_path / "results.csv"),
    )
    config.update(kwargs)
    return HyperPhoenixCV(**config)


def test_canonical_signature_clone_get_and_set_params(tmp_path):
    search = _canonical(tmp_path)
    signature = inspect.signature(HyperPhoenixCV)
    assert {"search_space", "strategy", "n_trials", "storage_path", "resume"} <= set(signature.parameters)
    assert not {
        "param_grid", "random_search", "n_iter", "use_bayesian_optimization",
        "bayesian_optimizer", "checkpoint_path", "clear_checkpoint",
    } & set(signature.parameters)
    cloned = clone(search)
    assert cloned.get_params(deep=False)["search_space"] == {"C": [0.1, 1.0]}
    assert cloned.get_params(deep=False)["storage_path"].endswith("study.sqlite3")
    cloned.set_params(search_space={"C": [1.0]}, strategy="random", n_trials=1, resume="never")
    assert cloned.search_space == {"C": [1.0]}
    assert cloned.strategy == "random"
    assert cloned.n_trials == 1
    assert cloned.resume == "never"
    cloned.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert len(cloned.cv_results_["params"]) == 1


def test_canonical_random_uses_n_trials(tmp_path):
    search = _canonical(tmp_path, strategy="random", n_trials=1, random_state=7)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        search.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert not [warning for warning in captured if isinstance(warning.message, FutureWarning)]
    assert len(search.cv_results_["params"]) == 1


def test_invalid_canonical_config_fails_before_storage(tmp_path):
    path = tmp_path / "study.sqlite3"
    search = _canonical(tmp_path, strategy="random", n_trials=None)
    with pytest.raises(ValueError, match="require n_trials"):
        search.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert not path.exists()


def test_clear_storage_is_explicit(tmp_path):
    search = _canonical(tmp_path)
    search.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    path = tmp_path / "study.sqlite3"
    assert path.exists()
    search.clear_storage()
    assert not path.exists()
