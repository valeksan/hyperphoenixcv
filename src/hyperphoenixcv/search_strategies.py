"""
Search strategies for hyperparameter optimization.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import List, Dict, Any, Optional, Protocol
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from .study_identity import param_key


class SearchSpace(Protocol):
    """Finite or streaming parameter source."""

    def iter_parameters(self) -> Iterator[Dict[str, Any]]: ...
    def total_candidates(self) -> int: ...


class Sampler(Protocol):
    """Resume-safe proposal protocol used by search coordinator."""

    def restore(self, results: List[Dict[str, Any]]) -> None: ...
    def ask(self, n: int) -> List[Dict[str, Any]]: ...
    def tell(self, results: List[Dict[str, Any]]) -> None: ...


class Evaluator(Protocol):
    """One parameter assignment -> terminal trial result."""

    def evaluate(self, estimator, X, y, params: Dict[str, Any], groups=None) -> Dict[str, Any]: ...


class Scheduler(Protocol):
    """Coordinator capacity contract. Scheduler implementation lands in P1.4."""

    def free_workers(self) -> int: ...


class SearchStrategy(ABC, SearchSpace, Sampler):
    """
    Abstract base class for hyperparameter search strategies.
    """

    def __init__(self, param_grid: Mapping[str, Any] | List[Dict[str, Any]]):
        self.param_grid = param_grid
        self._proposal_iterator: Iterator[Dict[str, Any]] | None = None
        self._known_param_keys: set[str] = set()

    @abstractmethod
    def generate_parameters(self) -> List[Dict[str, Any]]:
        """
        Generate a list of parameter combinations to evaluate.
        """
        pass

    @abstractmethod
    def iter_parameters(self) -> Iterator[Dict[str, Any]]:
        """Yield proposals without materializing the whole search space."""
        pass

    @abstractmethod
    def total_candidates(self) -> int:
        """Return finite candidate count without creating candidates."""
        pass

    def suggest_next(self, completed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Suggest next parameters based on completed results.
        Default implementation returns all generated parameters (no sorting).
        """
        return self.generate_parameters()

    def restore(self, results: List[Dict[str, Any]]) -> None:
        """Start a resumable ask/tell session from committed trial history."""
        self._known_param_keys = {
            param_key(result["params"])
            for result in results
            if "params" in result
        }
        self._proposal_iterator = self.iter_parameters()

    def ask(self, n: int) -> List[Dict[str, Any]]:
        """Return up to ``n`` unseen proposals without materializing the space."""
        if n < 1:
            return []
        if self._proposal_iterator is None:
            self.restore([])

        proposals = []
        while len(proposals) < n:
            try:
                params = next(self._proposal_iterator)
            except StopIteration:
                break
            key = param_key(params)
            if key in self._known_param_keys:
                continue
            # Reserve now. This prevents duplicate proposals in one batch; tell
            # keeps terminal results reserved across later batches.
            self._known_param_keys.add(key)
            proposals.append(params)
        return proposals

    def tell(self, results: List[Dict[str, Any]]) -> None:
        """Accept terminal results. Future adaptive samplers override this hook."""
        for result in results:
            if "params" in result:
                self._known_param_keys.add(param_key(result["params"]))


class ExhaustiveSearchStrategy(SearchStrategy):
    """
    Exhaustive grid search (ParameterGrid).
    """

    def generate_parameters(self) -> List[Dict[str, Any]]:
        """Compatibility API. New engine code must use ``iter_parameters``."""
        return list(self.iter_parameters())

    def iter_parameters(self) -> Iterator[Dict[str, Any]]:
        return iter(ParameterGrid(self.param_grid))

    def total_candidates(self) -> int:
        return len(ParameterGrid(self.param_grid))


class RandomSearchStrategy(SearchStrategy):
    """
    Random search strategy.
    """

    def __init__(
        self,
        param_grid: Mapping[str, Any] | List[Dict[str, Any]],
        n_iter: int = 10,
        random_state: Optional[int] = None,
    ):
        super().__init__(param_grid)
        self.n_iter = n_iter
        self.random_state = random_state

    def generate_parameters(self) -> List[Dict[str, Any]]:
        """Compatibility API. New engine code must use ``iter_parameters``."""
        return list(self.iter_parameters())

    def iter_parameters(self) -> Iterator[Dict[str, Any]]:
        """Match sklearn ``ParameterSampler`` order without candidate materialization.

        A persisted effective seed makes replay deterministic after a crash.  For
        distribution-valued spaces sklearn samples with replacement; ``n_iter``
        is therefore both proposal and resume budget.
        """
        return iter(ParameterSampler(
            self.param_grid,
            n_iter=max(self.n_iter, 0),
            random_state=self.random_state,
        ))

    def total_candidates(self) -> int:
        has_distribution = any(
            hasattr(values, "rvs")
            for branch in (self.param_grid if isinstance(self.param_grid, list) else [self.param_grid])
            for values in branch.values()
        )
        if has_distribution:
            return max(self.n_iter, 0)
        return min(max(self.n_iter, 0), len(ParameterGrid(self.param_grid)))


class ExperimentalSurrogateRankingStrategy(SearchStrategy):
    """
    Experimental surrogate-ranking strategy. This is not Bayesian optimization:
    it has no acquisition function or sequential Bayesian sampler.
    """

    def __init__(
        self,
        param_grid: Dict[str, Any],
        scoring: str,  # primary metric to optimize
        model=None,
        random_state: Optional[int] = None,
    ):
        super().__init__(param_grid)
        self.scoring = scoring
        if model is None:
            self.model = RandomForestRegressor(n_estimators=20, random_state=42)
        else:
            self.model = model
        self.random_state = random_state
        self.label_encoders = {}
        self._fit_label_encoders()

    def _fit_label_encoders(self):
        """Pre‑fit label encoders on all possible categorical values from param_grid."""
        # Collect all possible values for each parameter
        param_values = {}
        for param, values in self.param_grid.items():
            param_values[param] = values

        # Create a DataFrame with all possible combinations (could be huge, but we only need unique values per column)
        # Instead, we iterate over each parameter and collect unique values.
        for param, values in param_values.items():
            # Determine if the parameter is categorical (contains non‑numeric values)
            # We'll treat any value that is not int or float as categorical.
            categorical = any(
                not isinstance(v, (int, float, np.integer, np.floating))
                for v in values
            )
            if categorical:
                le = LabelEncoder()
                # Convert all values to strings for consistent encoding
                unique_vals = list(set(str(v) for v in values))
                le.fit(unique_vals)
                self.label_encoders[param] = le

    def generate_parameters(self) -> List[Dict[str, Any]]:
        return list(self.iter_parameters())

    def iter_parameters(self) -> Iterator[Dict[str, Any]]:
        return iter(ParameterGrid(self.param_grid))

    def total_candidates(self) -> int:
        return len(ParameterGrid(self.param_grid))

    def suggest_next(self, completed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not completed_results:
            return self.generate_parameters()

        # Extract parameters and scores
        completed_params = [r['params'] for r in completed_results]
        scoring_key = f'mean_test_{self.scoring}'
        completed_scores = [r.get(scoring_key, 0.0) for r in completed_results]

        # Encode parameters
        X_train = self._encode_params(completed_params)
        y_train = np.array(completed_scores)

        # Train surrogate model
        self.model.fit(X_train, y_train)

        # Generate all possible parameters and filter out completed ones
        all_params = self.generate_parameters()
        remaining = [p for p in all_params if p not in completed_params]
        if not remaining:
            return []

        # Predict scores for remaining parameters
        X_remaining = self._encode_params(remaining)
        predicted_scores = self.model.predict(X_remaining)

        # Sort by predicted score (descending)
        sorted_indices = np.argsort(predicted_scores)[::-1]
        return [remaining[i] for i in sorted_indices]

    def update_model(self, new_results: List[Dict[str, Any]]):
        """
        Incrementally update the surrogate model with new results.
        For simplicity, we just retrain on all data when suggest_next is called.
        """
        # This is a placeholder; actual incremental learning could be implemented.
        pass

    def _encode_params(self, params_list: List[Dict[str, Any]]) -> np.ndarray:
        """Encode categorical parameters into numeric matrix."""
        if not params_list:
            return np.array([]).reshape(0, -1)
        df = pd.DataFrame(params_list)
        X = df.copy()
        for col in X.columns:
            if col in self.label_encoders:
                # Categorical column with pre‑fitted encoder
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
            else:
                # Numeric column
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        return X.values


# Compatibility alias. New code must use the explicit experimental name.
BayesianSearchStrategy = ExperimentalSurrogateRankingStrategy


def create_search_strategy(
    param_grid: Mapping[str, Any] | List[Dict[str, Any]],
    random_search: bool = False,
    use_bayesian_optimization: bool = False,
    n_iter: int = 10,
    random_state: Optional[int] = None,
    bayesian_optimizer = None,
    scoring: str = 'f1',
) -> SearchStrategy:
    """
    Factory function to create a search strategy based on configuration.
    Maintains backward compatibility with HyperPhoenixCV parameters.
    """
    if use_bayesian_optimization:
        warnings.warn(
            "use_bayesian_optimization is deprecated: current surrogate mode is not "
            "Bayesian optimization. Use random_search until the Optuna backend lands.",
            FutureWarning,
            stacklevel=2,
        )
        return ExperimentalSurrogateRankingStrategy(
            param_grid=param_grid,
            scoring=scoring,
            model=bayesian_optimizer,
            random_state=random_state,
        )
    elif random_search:
        return RandomSearchStrategy(
            param_grid=param_grid,
            n_iter=n_iter,
            random_state=random_state,
        )
    else:
        return ExhaustiveSearchStrategy(param_grid=param_grid)
