"""Compatibility facade and factory for concrete search strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Any, Callable, Dict, List, Optional

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
def create_search_strategy(
    search_space: Mapping[str, Any] | List[Dict[str, Any]] | Callable[[Any], Dict[str, Any]],
    strategy: str,
    n_trials: int | None,
    random_state: Optional[int] = None,
    optuna_warmup_trials: int = 10,
    optuna_directions: Mapping[str, str] | None = None,
) -> SearchStrategy:
    """Build a canonical grid, random, or Optuna strategy."""
    allowed = {"grid", "random", "optuna"}
    if strategy not in allowed:
        raise ValueError("strategy must be 'grid', 'random', or 'optuna'")
    if strategy == "optuna":
        if n_trials is None:
            raise ValueError("strategy='optuna' requires n_trials")
        return OptunaSearchStrategy(
            search_space=search_space, n_trials=n_trials,
            random_state=random_state, warmup_trials=optuna_warmup_trials,
            directions=optuna_directions,
        )
    if strategy == "random":
        if n_trials is None:
            raise ValueError("strategy='random' requires n_trials")
        return RandomSearchStrategy(param_grid=search_space, n_trials=n_trials, random_state=random_state)
    if n_trials is not None:
        raise ValueError("n_trials is unsupported for strategy='grid'")
    return ExhaustiveSearchStrategy(param_grid=search_space)
