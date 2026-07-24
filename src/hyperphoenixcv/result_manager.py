"""
Result manager for storing, sorting, and exporting hyperparameter search results.
"""

import pandas as pd
import joblib
import os
import json
from typing import List, Dict, Any, Optional


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
        try:
            return json.dumps(params, sort_keys=True, separators=(",", ":"), default=repr)
        except (TypeError, ValueError):
            return repr(sorted(params.items()))

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
        Load results from a checkpoint file and add them to the internal list.

        Returns:
            Loaded results (same as returned by CheckpointManager.load()).
        """
        if not os.path.exists(checkpoint_path):
            return []
        loaded = joblib.load(checkpoint_path)
        self.add_results(loaded)
        return loaded

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
        valid = [r for r in self.results if 'error' not in r]
        if not valid:
            return {}

        cv_results = {'params': [r['params'] for r in valid]}
        for metric in self.scoring:
            mean_key = f'mean_test_{metric}'
            std_key = f'std_test_{metric}'
            cv_results[mean_key] = [r[mean_key] for r in valid]
            cv_results[std_key] = [r[std_key] for r in valid]
        return cv_results
