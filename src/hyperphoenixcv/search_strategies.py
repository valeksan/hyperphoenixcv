"""Compatibility facade and factory for concrete search strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Any, Callable, Dict, List, Optional
import warnings

from .search_protocols import Evaluator, Sampler, SearchSpace
from .study_identity import param_key


class SearchStrategy(ABC, SearchSpace, Sampler):
    """Shared lazy/resume-safe behavior for concrete strategies."""

    def __init__(self, param_grid: Mapping[str, Any] | List[Dict[str, Any]]):
        self.param_grid = param_grid
        self._proposal_iterator: Iterator[Dict[str, Any]] | None = None
        self._known_param_keys: set[str] = set()

    @abstractmethod
    def generate_parameters(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def iter_parameters(self) -> Iterator[Dict[str, Any]]: ...

    @abstractmethod
    def total_candidates(self) -> int: ...

    def suggest_next(self, completed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.generate_parameters()

    def restore(self, results: List[Dict[str, Any]]) -> None:
        self._known_param_keys = {param_key(result["params"]) for result in results if "params" in result}
        self._proposal_iterator = self.iter_parameters()

    def ask(self, n: int) -> List[Dict[str, Any]]:
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
            if key not in self._known_param_keys:
                self._known_param_keys.add(key)
                proposals.append(params)
        return proposals

    def tell(self, results: List[Dict[str, Any]]) -> None:
        for result in results:
            if "params" in result:
                self._known_param_keys.add(param_key(result["params"]))


# Imports follow base-class definition to avoid concrete-module cycles while
# preserving historical ``hyperphoenixcv.search_strategies`` imports.
from .search_grid import ExhaustiveSearchStrategy
from .search_optuna import OptunaSearchStrategy
from .search_random import RandomSearchStrategy
from .search_surrogate import ExperimentalSurrogateRankingStrategy

BayesianSearchStrategy = ExperimentalSurrogateRankingStrategy


def create_search_strategy(
    param_grid: Mapping[str, Any] | List[Dict[str, Any]] | None,
    random_search: bool = False,
    use_bayesian_optimization: bool = False,
    n_iter: int = 10,
    random_state: Optional[int] = None,
    bayesian_optimizer=None,
    scoring: str = "f1",
    *,
    strategy: str | None = None,
    search_space: Mapping[str, Any] | Callable[[Any], Dict[str, Any]] | None = None,
    n_trials: int | None = None,
    optuna_warmup_trials: int = 10,
    optuna_directions: Mapping[str, str] | None = None,
) -> SearchStrategy:
    """Build a strategy while preserving legacy configuration validation."""
    allowed = {"grid", "random", "optuna", "experimental_surrogate_ranking"}
    if strategy is not None and strategy not in allowed:
        raise ValueError("strategy must be 'grid', 'random', 'optuna', or 'experimental_surrogate_ranking'")
    if strategy == "optuna":
        if random_search or use_bayesian_optimization or bayesian_optimizer is not None:
            raise ValueError("strategy='optuna' conflicts with legacy search settings")
        if search_space is None:
            raise ValueError("strategy='optuna' requires search_space")
        return OptunaSearchStrategy(
            search_space=search_space, n_trials=n_iter if n_trials is None else n_trials,
            random_state=random_state, warmup_trials=optuna_warmup_trials,
            directions=optuna_directions,
        )
    if search_space is not None:
        raise ValueError("search_space is supported only with strategy='optuna'")
    if param_grid is None:
        raise ValueError("param_grid is required unless strategy='optuna'")
    if strategy == "random":
        if use_bayesian_optimization or bayesian_optimizer is not None:
            raise ValueError("strategy='random' conflicts with Bayesian compatibility settings")
        if n_trials is not None:
            if n_iter != 10 and n_iter != n_trials:
                raise ValueError("n_trials conflicts with legacy n_iter")
            n_iter = n_trials
        random_search = True
    if strategy == "grid" and (random_search or use_bayesian_optimization or bayesian_optimizer is not None):
        raise ValueError("strategy='grid' conflicts with legacy search settings")
    if strategy == "experimental_surrogate_ranking":
        if random_search:
            raise ValueError("strategy='experimental_surrogate_ranking' conflicts with random_search")
        use_bayesian_optimization = True
    if random_search and strategy is None:
        warnings.warn("random_search is deprecated; use strategy='random' and n_trials instead.", FutureWarning, stacklevel=2)
    if use_bayesian_optimization:
        warnings.warn(
            "use_bayesian_optimization is deprecated: current surrogate mode is not Bayesian optimization. "
            "Use random_search until the Optuna backend lands.", FutureWarning, stacklevel=2,
        )
        return ExperimentalSurrogateRankingStrategy(param_grid=param_grid, scoring=scoring,
                                                     model=bayesian_optimizer, random_state=random_state)
    if random_search:
        return RandomSearchStrategy(param_grid=param_grid, n_iter=n_iter, random_state=random_state)
    return ExhaustiveSearchStrategy(param_grid=param_grid)
