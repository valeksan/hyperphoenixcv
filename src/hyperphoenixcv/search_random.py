"""Resume-safe lazy random strategy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Dict, List, Optional

from sklearn.model_selection import ParameterGrid, ParameterSampler

from .search_strategies import SearchStrategy


class RandomSearchStrategy(SearchStrategy):
    def __init__(
        self,
        param_grid: Mapping[str, Any] | List[Dict[str, Any]],
        n_trials: int,
        random_state: Optional[int] = None,
    ):
        super().__init__(param_grid)
        self.n_trials = n_trials
        self.random_state = random_state

    def generate_parameters(self) -> List[Dict[str, Any]]:
        return list(self.iter_parameters())

    def iter_parameters(self) -> Iterator[Dict[str, Any]]:
        return iter(ParameterSampler(
            self.param_grid, n_iter=max(self.n_trials, 0), random_state=self.random_state,
        ))

    def total_candidates(self) -> int:
        has_distribution = any(
            hasattr(values, "rvs")
            for branch in (self.param_grid if isinstance(self.param_grid, list) else [self.param_grid])
            for values in branch.values()
        )
        if has_distribution:
            return max(self.n_trials, 0)
        return min(max(self.n_trials, 0), len(ParameterGrid(self.param_grid)))
