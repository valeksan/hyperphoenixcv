import pytest
import warnings
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


def test_canonical_grid_api_fits_and_clones(tmp_path):
    search = _canonical(tmp_path)
    cloned = clone(search)
    assert cloned.get_params(deep=False)["search_space"] == {"C": [0.1, 1.0]}
    assert cloned.get_params(deep=False)["storage_path"].endswith("study.sqlite3")
    cloned.set_params(n_trials=2, resume="never")
    assert cloned.n_trials == 2
    assert cloned.resume == "never"


def test_canonical_random_uses_n_trials(tmp_path):
    search = _canonical(tmp_path, strategy="random", n_trials=1, random_state=7)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        search.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert not [warning for warning in captured if isinstance(warning.message, FutureWarning)]
    assert len(search.cv_results_["params"]) == 1


def test_canonical_and_legacy_spaces_conflict_before_storage(tmp_path):
    path = tmp_path / "study.sqlite3"
    search = _canonical(tmp_path, param_grid={"C": [1.0]})
    with pytest.raises(ValueError, match="conflicts"):
        search.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert not path.exists()
