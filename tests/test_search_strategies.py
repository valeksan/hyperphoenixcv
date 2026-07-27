import importlib.util
import random

import numpy as np
import pytest
from sklearn.model_selection import ParameterSampler

from hyperphoenixcv.search_strategies import (
    ExhaustiveSearchStrategy,
    RandomSearchStrategy,
    create_search_strategy,
)


def test_grid_strategy_is_lazy_and_resume_safe():
    strategy = ExhaustiveSearchStrategy({"a": [1, 2, 3]})
    strategy.restore([{"params": {"a": 1}}])
    assert strategy.ask(2) == [{"a": 2}, {"a": 3}]
    assert strategy.total_candidates() == 3


def test_random_strategy_replays_without_global_rng_mutation():
    random.seed(931)
    np.random.seed(931)
    expected = (random.random(), np.random.random())
    random.seed(931)
    np.random.seed(931)
    strategy = RandomSearchStrategy({"a": range(10_000_000)}, n_trials=3, random_state=42)
    assert len(list(strategy.iter_parameters())) == 3
    assert (random.random(), np.random.random()) == expected


def test_random_strategy_matches_sklearn_sampler():
    space = [{"kind": ["a"], "value": [1, 2]}, {"kind": ["b"], "depth": [3, 4]}]
    expected = list(ParameterSampler(space, n_iter=3, random_state=19))
    assert list(RandomSearchStrategy(space, n_trials=3, random_state=19).iter_parameters()) == expected


def test_factory_has_only_canonical_strategies():
    assert isinstance(create_search_strategy({"a": [1]}, "grid", None), ExhaustiveSearchStrategy)
    random_strategy = create_search_strategy({"a": [1, 2]}, "random", 1, random_state=7)
    assert isinstance(random_strategy, RandomSearchStrategy)
    assert random_strategy.n_trials == 1
    with pytest.raises(ValueError, match="grid.*random.*optuna"):
        create_search_strategy({"a": [1]}, "experimental_surrogate_ranking", None)


def test_factory_requires_trial_budget_for_random_and_optuna():
    with pytest.raises(ValueError, match="requires n_trials"):
        create_search_strategy({"a": [1]}, "random", None)
    if importlib.util.find_spec("optuna") is None:
        with pytest.raises(ImportError, match=r"hyperphoenixcv\[optuna\]"):
            create_search_strategy({"C": object()}, "optuna", 2)
