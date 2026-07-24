"""
Result manager for storing, sorting, and exporting hyperparameter search results.
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional

from .study_identity import param_key


class ResultManager:
    """
    Manages hyperparameter search results.
    """

    def __init__(
        self,
        scoring: List[str],
        results_csv: str = "hyperphoenix_results.csv",
    ):
        self.scoring = scoring
        self.results_csv = results_csv
        self.results = []
        self._param_keys = set()

    @staticmethod
    def param_key(params: Dict[str, Any]) -> str:
        """Stable key used to make one study's result projection idempotent."""
        return param_key(params)

    def add_result(self, result: Dict[str, Any]):
        """
        Add a single result to the internal list.
        """
        key = self.param_key(result.get("params", {}))
        if key not in self._param_keys:
            self.results.append(result)
            self._param_keys.add(key)

    def add_results(self, results: List[Dict[str, Any]]):
        """
        Add multiple results at once.
        """
        for result in results:
            self.add_result(result)

    def load_from_checkpoint(self, checkpoint_path: str) -> List[Dict[str, Any]]:
        """
        Removed implicit pickle loading path.
        """
        raise RuntimeError(
            "Implicit pickle loading was removed. Use "
            "HyperPhoenixCV.import_legacy_checkpoint(path, trusted=True)."
        )

    def clear_results(self):
        """
        Clear the internal results list.
        """
        self.results.clear()
        self._param_keys.clear()

    def get_top_results(self, n: int = 10) -> pd.DataFrame:
        """
        Return top‑N results sorted by the first scoring metric.

        Returns:
            DataFrame with columns: parameters + mean_test_* + std_test_*.
        """
        if not self.results:
            return pd.DataFrame()

        # Filter out error results
        valid = [r for r in self.results if 'error' not in r]
        if not valid:
            return pd.DataFrame()

        rows = []
        for r in valid:
            row = {}
            row.update(r['params'])
            for metric in self.scoring:
                mean_key = f'mean_test_{metric}'
                std_key = f'std_test_{metric}'
                if mean_key in r:
                    row[mean_key] = r[mean_key]
                if std_key in r:
                    row[std_key] = r[std_key]
            rows.append(row)

        df = pd.DataFrame(rows)
        if self.scoring and f'mean_test_{self.scoring[0]}' in df.columns:
            df = df.sort_values(
                f'mean_test_{self.scoring[0]}',
                ascending=False,
            )
        return df.head(n)

    def save_to_csv(self, path: Optional[str] = None):
        """
        Save all valid results to a CSV file.

        Args:
            path: Optional custom path; if None, uses self.results_csv.
        """
        csv_path = path or self.results_csv
        df = self.get_top_results(n=len(self.results))  # get all valid results
        df.to_csv(csv_path, index=False)

    def format_cv_results(self) -> Dict[str, Any]:
        """
        Format results in a GridSearchCV‑compatible dictionary.

        Returns:
            Dictionary with keys 'params', 'mean_test_*', 'std_test_*', etc.
        """
        if not self.results:
            return {}

        results = self.results
        cv_results = {'params': [r['params'] for r in results]}
        param_names = sorted({name for result in results for name in result['params']})
        for name in param_names:
            values = [result['params'].get(name) for result in results]
            mask = [name not in result['params'] for result in results]
            cv_results[f'param_{name}'] = np.ma.array(values, mask=mask, dtype=object)

        for key in ('fit_time', 'score_time'):
            cv_results[f'mean_{key}'] = [result.get(f'mean_{key}', np.nan) for result in results]
            cv_results[f'std_{key}'] = [result.get(f'std_{key}', np.nan) for result in results]
        for metric in self.scoring:
            mean_key = f'mean_test_{metric}'
            std_key = f'std_test_{metric}'
            means = [result.get(mean_key, np.nan) for result in results]
            cv_results[mean_key] = means
            cv_results[std_key] = [result.get(std_key, np.nan) for result in results]
            fold_count = max((len(result.get(f'scores_{metric}', [])) for result in results), default=0)
            for fold in range(fold_count):
                cv_results[f'split{fold}_test_{metric}'] = [
                    result.get(f'scores_{metric}', [np.nan] * fold_count)[fold]
                    if fold < len(result.get(f'scores_{metric}', [])) else np.nan
                    for result in results
                ]
            cv_results[f'rank_test_{metric}'] = self._rank_descending(means)
        return cv_results

    @staticmethod
    def _rank_descending(values: list[float]) -> list[int]:
        """Sklearn-like ordinal ranks; NaN receives worst rank."""
        finite = [(index, value) for index, value in enumerate(values) if not pd.isna(value)]
        ranks = [len(values) + 1] * len(values)
        last_value = None
        rank = 0
        for position, (index, value) in enumerate(sorted(finite, key=lambda item: item[1], reverse=True), 1):
            if last_value is None or value != last_value:
                rank = position
                last_value = value
            ranks[index] = rank
        return ranks
