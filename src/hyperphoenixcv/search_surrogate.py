"""Experimental surrogate-ranking compatibility strategy.

This is not Bayesian optimization: it has no acquisition function or
sequential Bayesian sampler.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder

from .search_strategies import SearchStrategy


class ExperimentalSurrogateRankingStrategy(SearchStrategy):
    """Legacy experimental ranker; retained only for compatibility."""

    def __init__(self, param_grid: Dict[str, Any], scoring: str, model=None,
                 random_state: Optional[int] = None):
        super().__init__(param_grid)
        self.scoring = scoring
        self.model = model if model is not None else RandomForestRegressor(n_estimators=20, random_state=42)
        self.random_state = random_state
        self.label_encoders = {}
        self._fit_label_encoders()

    def _fit_label_encoders(self):
        for param, values in self.param_grid.items():
            categorical = any(not isinstance(value, (int, float, np.integer, np.floating)) for value in values)
            if categorical:
                encoder = LabelEncoder()
                encoder.fit(list(set(str(value) for value in values)))
                self.label_encoders[param] = encoder

    def generate_parameters(self) -> List[Dict[str, Any]]:
        return list(self.iter_parameters())

    def iter_parameters(self) -> Iterator[Dict[str, Any]]:
        return iter(ParameterGrid(self.param_grid))

    def total_candidates(self) -> int:
        return len(ParameterGrid(self.param_grid))

    def suggest_next(self, completed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not completed_results:
            return self.generate_parameters()
        completed_params = [result["params"] for result in completed_results]
        scores = [result.get(f"mean_test_{self.scoring}", 0.0) for result in completed_results]
        self.model.fit(self._encode_params(completed_params), np.array(scores))
        remaining = [params for params in self.generate_parameters() if params not in completed_params]
        if not remaining:
            return []
        predicted = self.model.predict(self._encode_params(remaining))
        return [remaining[index] for index in np.argsort(predicted)[::-1]]

    def update_model(self, new_results: List[Dict[str, Any]]):
        """Compatibility no-op; ranking retrains in ``suggest_next``."""

    def _encode_params(self, params_list: List[Dict[str, Any]]) -> np.ndarray:
        if not params_list:
            return np.array([]).reshape(0, -1)
        values = pd.DataFrame(params_list).copy()
        for column in values.columns:
            if column in self.label_encoders:
                values[column] = self.label_encoders[column].transform(values[column].astype(str))
            else:
                values[column] = pd.to_numeric(values[column], errors="coerce").fillna(0)
        return values.values
