"""
CVExecutor performs cross‑validation for a given parameter set.
"""

import numpy as np
import logging
from collections.abc import Mapping
from sklearn.base import clone, is_classifier
from sklearn.model_selection import check_cv, cross_validate
from typing import Dict, Any, List, Union


logger = logging.getLogger(__name__)


class SklearnCVEvaluator:
    """
    Executes cross‑validation for a single hyperparameter combination.

    Parameters
    ----------
    cv : int or CV splitter, default=5
        Number of folds or a cross‑validation splitter object.
    scoring : str or list of str, default='f1'
        Metric(s) to evaluate.
    n_jobs : int, default=1
        Number of parallel jobs.
    verbose : bool, default=True
        Whether to print progress and errors.
    pre_dispatch : str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. See `sklearn.model_selection.cross_validate`.
    error_score : 'raise' or numeric, default='raise'
        Value to assign to the score if an error occurs in the estimator.
        If 'raise', the error is raised.
    """

    def __init__(
        self,
        cv: Union[int, object] = 5,
        scoring: Union[str, List[str]] = 'f1',
        n_jobs: int = 1,
        verbose: bool = True,
        pre_dispatch: str = '2*n_jobs',
        error_score: Union[str, float] = 'raise',
    ):
        self.cv = cv
        self.scoring = scoring if isinstance(scoring, list) else [scoring]
        self.metric_names, self._scoring_spec = self._normalize_scoring(scoring)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self._splits = None

    @staticmethod
    def _normalize_scoring(scoring):
        """Normalize supported sklearn scorer forms to named multi-metric spec."""
        if isinstance(scoring, Mapping):
            if not scoring:
                raise ValueError("scoring mapping must not be empty")
            return list(scoring), dict(scoring)
        if callable(scoring):
            return ["score"], {"score": scoring}
        if isinstance(scoring, str):
            return [scoring], {scoring: scoring}
        if isinstance(scoring, list) and all(isinstance(item, str) for item in scoring):
            if not scoring:
                raise ValueError("scoring list must not be empty")
            return list(scoring), {item: item for item in scoring}
        raise ValueError("scoring must be a string, callable, list of strings, or mapping")

    def _resolve_splits(self, estimator, X, y, groups):
        """Resolve sklearn splitter once per evaluator/study execution."""
        if self._splits is None:
            splitter = check_cv(self.cv, y=y, classifier=is_classifier(estimator))
            self._splits = list(splitter.split(X, y, groups))
        return self._splits

    def evaluate(
        self,
        estimator,
        X,
        y,
        params: Dict[str, Any],
        groups=None,
        fit_params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a parameter set via cross‑validation.

        Args:
            estimator: sklearn estimator (not fitted).
            X: Feature matrix.
            y: Target vector.
            params: Hyperparameters to set on the estimator.
            groups: Optional group labels for group‑based CV.

        Returns:
            Dictionary with keys:
                - 'params': the input params
                - 'mean_test_<metric>', 'std_test_<metric>' for each metric
                - 'scores_<metric>' (list of per‑fold scores) for each metric
                - 'error': only present if an exception occurred
        """
        estimator_with_params = clone(estimator).set_params(**params)

        try:
            cv_splitter = self._resolve_splits(estimator, X, y, groups)
            scores = cross_validate(
                estimator_with_params,
                X,
                y,
                cv=cv_splitter,
                scoring=self._scoring_spec,
                n_jobs=self.n_jobs,
                groups=groups,
                params=fit_params,
                return_train_score=False,
                pre_dispatch=self.pre_dispatch,
                error_score=self.error_score,
            )
            result = {
                'params': params,
                **self._format_scores(scores),
            }
        except Exception as e:
            if self.error_score == 'raise':
                raise
            if self.verbose:
                logger.warning("CV evaluation failed: %s", type(e).__name__)
            failed_scores = [float(self.error_score)] * len(self._resolve_splits(estimator, X, y, groups))
            result = {
                'params': params,
                'error': str(e),
                'error_type': type(e).__name__,
                'fold_errors': [str(e)] * len(failed_scores),
            }
            for metric in self.metric_names:
                result[f'mean_test_{metric}'] = float(np.mean(failed_scores))
                result[f'std_test_{metric}'] = float(np.std(failed_scores))
                result[f'scores_{metric}'] = failed_scores
        return result

    def _format_scores(self, scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Convert raw cross_validate output to a flat dictionary.
        """
        formatted = {}
        for timing in ('fit_time', 'score_time'):
            if timing not in scores:
                continue
            values = scores[timing]
            formatted[f'mean_{timing}'] = float(values.mean())
            formatted[f'std_{timing}'] = float(values.std())
            formatted[timing] = values.tolist()
        for metric in self.metric_names:
            test_metric = f'test_{metric}'
            if test_metric in scores:
                formatted[f'mean_test_{metric}'] = float(scores[test_metric].mean())
                formatted[f'std_test_{metric}'] = float(scores[test_metric].std())
                formatted[f'scores_{metric}'] = scores[test_metric].tolist()
        return formatted


# Existing public import path remains valid during P1 migration.
CVExecutor = SklearnCVEvaluator
