"""Search-layer contracts, independent of concrete proposal strategies."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Dict, List, Protocol


class SearchSpace(Protocol):
    """Finite or streaming parameter source."""

    def iter_parameters(self) -> Iterator[Dict[str, Any]]: ...
    def total_candidates(self) -> int: ...


class Sampler(Protocol):
    """Resume-safe proposal protocol used by study engine."""

    def restore(self, results: List[Dict[str, Any]]) -> None: ...
    def ask(self, n: int) -> List[Dict[str, Any]]: ...
    def tell(self, results: List[Dict[str, Any]]) -> None: ...


class Evaluator(Protocol):
    """One parameter assignment -> terminal trial result."""

    def evaluate(self, estimator, X, y, params: Dict[str, Any], groups=None) -> Dict[str, Any]: ...
