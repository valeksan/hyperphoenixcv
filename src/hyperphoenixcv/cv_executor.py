"""
CVExecutor performs cross‑validation for a given parameter set.
"""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from typing import Dict, Any, List, Union


class CVExecutor:
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
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score

    def evaluate(
        self,
        estimator,
        X,
        y,
        params: Dict[str, Any],
        groups=None,
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

        # Determine CV splitter
        if isinstance(self.cv, int):
            classification_metrics = {
                'accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall',
                'f1_macro', 'f1_micro', 'f1_weighted', 'precision_macro',
                'precision_micro', 'precision_weighted', 'recall_macro',
                'recall_micro', 'recall_weighted', 'jaccard', 'roc_auc'
            }
            if any(m in classification_metrics for m in self.scoring):
                cv_splitter = StratifiedKFold(
                    n_splits=self.cv,
                    shuffle=True,
                    random_state=42,
                )
            else:
                cv_splitter = KFold(
                    n_splits=self.cv,
                    shuffle=True,
                    random_state=42,
                )
        else:
            cv_splitter = self.cv

        try:
            scores = cross_validate(
                estimator_with_params,
                X,
                y,
                cv=cv_splitter,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                groups=groups,
                return_train_score=False,
                pre_dispatch=self.pre_dispatch,
                error_score=self.error_score,
            )
            result = {
                'params': params,
                **self._format_scores(scores),
            }
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error during CV for params {params}: {e}")
            result = {
                'params': params,
                'error': str(e),
            }
        return result

    def _format_scores(self, scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Convert raw cross_validate output to a flat dictionary.
        """
        formatted = {}
        for metric in self.scoring:
            test_metric = f'test_{metric}'
            if test_metric in scores:
                formatted[f'mean_test_{metric}'] = float(scores[test_metric].mean())
                formatted[f'std_test_{metric}'] = float(scores[test_metric].std())
                formatted[f'scores_{metric}'] = scores[test_metric].tolist()
        return formatted
