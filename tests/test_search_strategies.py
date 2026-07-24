"""
Unit tests for search strategies.
"""

import pytest
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor

from src.hyperphoenixcv.search_strategies import (
    SearchStrategy,
    ExhaustiveSearchStrategy,
    RandomSearchStrategy,
    BayesianSearchStrategy,
    create_search_strategy,
)


class TestExhaustiveSearchStrategy:
    """Test ExhaustiveSearchStrategy."""

    def test_generate_parameters(self):
        param_grid = {
            'a': [1, 2],
            'b': ['x', 'y'],
        }
        strategy = ExhaustiveSearchStrategy(param_grid)
        params = strategy.generate_parameters()
        assert len(params) == 4
        expected = [
            {'a': 1, 'b': 'x'},
            {'a': 1, 'b': 'y'},
            {'a': 2, 'b': 'x'},
            {'a': 2, 'b': 'y'},
        ]
        for p in expected:
            assert p in params

    def test_suggest_next(self):
        param_grid = {'a': [1, 2]}
        strategy = ExhaustiveSearchStrategy(param_grid)
        completed = [{'params': {'a': 1}}]
        suggested = strategy.suggest_next(completed)
        # Should return all parameters (no sorting)
        assert len(suggested) == 2
        assert {'a': 1} in suggested
        assert {'a': 2} in suggested

    def test_iter_parameters_is_lazy_for_large_grid(self):
        strategy = ExhaustiveSearchStrategy({"a": range(10_000_000)})

        params = strategy.iter_parameters()

        assert iter(params) is params
        assert strategy.total_candidates() == 10_000_000
        assert next(params) == {"a": 0}


class TestRandomSearchStrategy:
    """Test RandomSearchStrategy."""

    def test_generate_parameters_full(self):
        param_grid = {'a': [1, 2]}
        strategy = RandomSearchStrategy(param_grid, n_iter=5)
        params = strategy.generate_parameters()
        # Since total combinations = 2 <= n_iter, returns all
        assert len(params) == 2

    def test_generate_parameters_random(self):
        param_grid = {'a': [1, 2, 3, 4, 5]}
        strategy = RandomSearchStrategy(param_grid, n_iter=3, random_state=42)
        params = strategy.generate_parameters()
        assert len(params) == 3
        # Ensure all params are from the grid
        for p in params:
            assert p['a'] in [1, 2, 3, 4, 5]
        # With fixed random_state, the sample should be deterministic
        strategy2 = RandomSearchStrategy(param_grid, n_iter=3, random_state=42)
        params2 = strategy2.generate_parameters()
        assert params == params2

    def test_suggest_next(self):
        param_grid = {'a': [1, 2, 3]}
        strategy = RandomSearchStrategy(param_grid, n_iter=2)
        completed = [{'params': {'a': 1}}]
        suggested = strategy.suggest_next(completed)
        # Should return all generated parameters (no sorting)
        assert len(suggested) == 2
        # Since random sample may or may not include completed param
        # We just check that each suggested param is in grid
        for p in suggested:
            assert p['a'] in [1, 2, 3]

    def test_iter_parameters_does_not_mutate_global_rng(self):
        random.seed(931)
        np.random.seed(931)
        expected_python = random.random()
        expected_numpy = np.random.random()

        random.seed(931)
        np.random.seed(931)
        strategy = RandomSearchStrategy({"a": range(10_000_000)}, n_iter=3, random_state=42)

        params = list(strategy.iter_parameters())

        assert len(params) == 3
        assert len({item["a"] for item in params}) == 3
        assert random.random() == expected_python
        assert np.random.random() == expected_numpy

    def test_total_candidates_respects_n_iter(self):
        strategy = RandomSearchStrategy({"a": [1, 2]}, n_iter=10)
        assert strategy.total_candidates() == 2


class TestBayesianSearchStrategy:
    """Test BayesianSearchStrategy."""

    @pytest.fixture
    def param_grid(self):
        return {'a': [1, 2, 3], 'b': ['x', 'y']}

    @pytest.fixture
    def completed_results(self):
        return [
            {
                'params': {'a': 1, 'b': 'x'},
                'mean_test_f1': 0.8,
                'std_test_f1': 0.1,
            },
            {
                'params': {'a': 2, 'b': 'x'},
                'mean_test_f1': 0.9,
                'std_test_f1': 0.05,
            },
        ]

    def test_generate_parameters(self, param_grid):
        strategy = BayesianSearchStrategy(param_grid, scoring='f1')
        params = strategy.generate_parameters()
        assert len(params) == 6  # 3 * 2

    def test_suggest_next(self, param_grid, completed_results):
        strategy = BayesianSearchStrategy(param_grid, scoring='f1', random_state=42)
        suggested = strategy.suggest_next(completed_results)
        # Should return remaining parameters sorted by predicted score
        # Since we have only two completed, remaining = 4
        assert len(suggested) == 4
        # Ensure completed parameters are not in suggested
        completed_params = [r['params'] for r in completed_results]
        for p in suggested:
            assert p not in completed_params
        # Order should be deterministic given random_state
        # We can't assert exact order because model predictions may vary,
        # but we can assert that the list is not empty
        assert all(isinstance(p, dict) for p in suggested)

    def test_suggest_next_no_completed(self, param_grid):
        strategy = BayesianSearchStrategy(param_grid, scoring='f1')
        suggested = strategy.suggest_next([])
        # Should return all parameters
        assert len(suggested) == 6

    def test_suggest_next_all_completed(self, param_grid, completed_results):
        # Create enough results to cover all parameters
        extra = [
            {'params': {'a': 3, 'b': 'x'}, 'mean_test_f1': 0.7},
            {'params': {'a': 1, 'b': 'y'}, 'mean_test_f1': 0.6},
            {'params': {'a': 2, 'b': 'y'}, 'mean_test_f1': 0.5},
            {'params': {'a': 3, 'b': 'y'}, 'mean_test_f1': 0.4},
        ]
        all_results = completed_results + extra
        strategy = BayesianSearchStrategy(param_grid, scoring='f1')
        suggested = strategy.suggest_next(all_results)
        # No remaining parameters
        assert suggested == []

    def test_encode_params(self, param_grid):
        strategy = BayesianSearchStrategy(param_grid, scoring='f1')
        params_list = [
            {'a': 1, 'b': 'x'},
            {'a': 2, 'b': 'y'},
        ]
        encoded = strategy._encode_params(params_list)
        assert encoded.shape == (2, 2)
        # 'a' should be numeric, 'b' encoded as 0/1
        assert encoded[0, 0] == 1
        assert encoded[1, 0] == 2
        # 'b' encoding may be 0 for 'x' and 1 for 'y' (or vice versa)
        assert set(encoded[:, 1]) == {0, 1}


class TestCreateSearchStrategy:
    """Test factory function."""

    def test_exhaustive(self):
        param_grid = {'a': [1, 2]}
        strategy = create_search_strategy(param_grid)
        assert isinstance(strategy, ExhaustiveSearchStrategy)

    def test_random_search(self):
        param_grid = {'a': [1, 2]}
        strategy = create_search_strategy(param_grid, random_search=True, n_iter=5)
        assert isinstance(strategy, RandomSearchStrategy)
        assert strategy.n_iter == 5

    def test_bayesian(self):
        param_grid = {'a': [1, 2]}
        model = RandomForestRegressor()
        strategy = create_search_strategy(
            param_grid,
            use_bayesian_optimization=True,
            scoring='accuracy',
            bayesian_optimizer=model,
        )
        assert isinstance(strategy, BayesianSearchStrategy)
        assert strategy.scoring == 'accuracy'
        assert strategy.model is model

    def test_bayesian_default_model(self):
        param_grid = {'a': [1, 2]}
        strategy = create_search_strategy(
            param_grid,
            use_bayesian_optimization=True,
        )
        assert isinstance(strategy, BayesianSearchStrategy)
        assert strategy.scoring == 'f1'
        assert isinstance(strategy.model, RandomForestRegressor)
