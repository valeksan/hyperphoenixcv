"""Lazy exhaustive sklearn-compatible grid strategy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Dict, List

from sklearn.model_selection import ParameterGrid

from .search_strategies import SearchStrategy


class ExhaustiveSearchStrategy(SearchStrategy):
    """Exhaustive grid search without candidate materialization."""

    def generate_parameters(self) -> List[Dict[str, Any]]:
        return list(self.iter_parameters())

    def iter_parameters(self) -> Iterator[Dict[str, Any]]:
        return iter(ParameterGrid(self.param_grid))

    def total_candidates(self) -> int:
        return len(ParameterGrid(self.param_grid))
