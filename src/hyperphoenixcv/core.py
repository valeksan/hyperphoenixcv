from __future__ import annotations

"""
HyperPhoenixCV - Resumable hyperparameter search with checkpoint support.

This module provides the HyperPhoenixCV class, which extends the functionality
of scikit-learn's GridSearchCV by adding checkpoint support, random search,
and Bayesian optimization to accelerate the search for optimal hyperparameters.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError

from .search_strategies import create_search_strategy
from .legacy_pickle import load_legacy_results, validate_legacy_result
from .result_manager import ResultManager
from .cv_executor import CVExecutor
from .study_identity import StudyIdentity
from .storage import SQLiteStudyStore


def _early_stop_from_results(results: list[dict], metric: str) -> tuple[float, int]:
    """Rebuild early-stop counters from terminal trial order."""
    best_score = -float("inf")
    no_improvement_count = 0
    score_key = f"mean_test_{metric}"
    for result in results:
        if "error" in result:
            no_improvement_count += 1
            continue
        score = result.get(score_key, -float("inf"))
        if score > best_score + 1e-9:
            best_score = score
            no_improvement_count = 0
        else:
            no_improvement_count += 1
    return best_score, no_improvement_count


class HyperPhoenixCV(BaseEstimator):
    """
    Resumable hyperparameter search with checkpoint support and Bayesian optimization.
    Supports exhaustive grid search, random search, and Bayesian optimization.

    Example usage:
    # Create an instance
    hp = HyperPhoenixCV(
        estimator=combat_pipeline,
        param_grid={
            'tfidf__max_features': [8000, 12000, 15000],
            'tfidf__ngram_range': [(1,1), (1,2)],
            'clf__C': [0.001, 0.01, 0.1],
            'clf__penalty': ['l1','l2'],
            'clf__solver': ['liblinear', 'saga'],
            'clf__class_weight': [None, 'balanced']
        },
        scoring=['f1', 'accuracy'],
        cv=5,
        n_jobs=-2,
        checkpoint_path="experiment_checkpoint.sqlite3",
        results_csv="experiment_results.csv",
        verbose=True
    )

    # Start the search
    hp.fit(X, y)

    # If the process was interrupted, run again with the same checkpoint_path:
    hp.fit(X, y)  # Will continue from the last saved point!

    # Get results
    print("Best parameters:", hp.best_params_)
    print("Best score:", hp.best_score_)

    # Top-10 results
    top_10 = hp.get_top_results(10)
    print(top_10)

    # Manually delete checkpoint
    hp.clear_checkpoint_file()
    """

    def __init__(
        self,
        estimator,
        param_grid: dict,
        scoring: str | list[str] = 'f1',
        cv: int = 5,
        n_jobs: int = 1,
        checkpoint_path: str = "hyperphoenix_checkpoint.sqlite3",
        results_csv: str = "hyperphoenix_results.csv",
        verbose: bool = True,
        clear_checkpoint: bool = False,
        random_search: bool = False,
        n_iter: int = 10,
        random_state: int | None = None,
        use_bayesian_optimization: bool = False,
        bayesian_optimizer = None,
        refit: bool = True,
        pre_dispatch: str = '2*n_jobs',
        error_score: Union[str, float] = 'raise',
        early_stopping_patience: int | None = None,
        dataset_id: str | None = None,
        resume: str = "auto",
        scorer_id: str | None = None,
        cv_id: str | None = None,
        storage_path: str | None = None,
    ):
        """
        Initializes HyperPhoenixCV.

        Parameters:
        -----------
        estimator : sklearn estimator
            Model/pipeline for hyperparameter tuning
        param_grid : dict
            Dictionary of parameters to search over
        scoring : str or list of str
            Metrics for evaluation (e.g., 'f1', 'accuracy' or ['f1', 'accuracy'])
        cv : int
            Number of folds for cross-validation
        n_jobs : int
            Number of processes for parallel computation
        checkpoint_path : str
            SQLite study-store path. A legacy-looking suffix is converted to
            ``.sqlite3`` for backward-compatible path selection.
        results_csv : str
            Path to CSV file for results
        verbose : bool
            Whether to print progress
        clear_checkpoint : bool
            Whether to delete existing checkpoint at the start of fit
        random_search : bool
            Whether to use random search instead of exhaustive grid search
        n_iter : int
            Number of random combinations (if random_search=True)
        random_state : int, optional
            Random seed for reproducibility
        use_bayesian_optimization : bool
            Whether to use Bayesian optimization (predictive parameter selection)
        bayesian_optimizer : sklearn regressor, optional
            Model that predicts which parameters will perform better
            (defaults to RandomForestRegressor)
        refit : bool, default=True
            Whether to refit the best model on the entire dataset after search.
            If True, after hyperparameter search completes, `best_estimator_.fit(X, y)` will be called.
        pre_dispatch : str, default='2*n_jobs'
            Controls the number of jobs that get dispatched during parallel
            execution. See `sklearn.model_selection.cross_validate`.
        error_score : 'raise' or numeric, default='raise'
            Value to assign to the score if an error occurs in the estimator.
            If 'raise', the error is raised.
        early_stopping_patience : int, optional
            If set, stop the search after this many iterations without improvement
            in the primary metric (scoring[0]). Useful for random search and
            Bayesian optimization to avoid unnecessary evaluations.
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.checkpoint_path = checkpoint_path
        self.results_csv = results_csv
        self.verbose = verbose
        self.clear_checkpoint = clear_checkpoint
        self.random_search = random_search
        self.n_iter = n_iter
        self.random_state = random_state
        self.use_bayesian_optimization = use_bayesian_optimization
        self.bayesian_optimizer = bayesian_optimizer
        self.refit = refit
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.early_stopping_patience = early_stopping_patience
        self.dataset_id = dataset_id
        self.resume = resume
        self.scorer_id = scorer_id
        self.cv_id = cv_id
        self.storage_path = storage_path

    @property
    def _scoring(self):
        """Normalized scoring used internally; public constructor value stays intact."""
        return self.scoring if isinstance(self.scoring, list) else [self.scoring]

    def _create_runtime_components(self):
        """Create per-fit collaborators. Constructor must stay side-effect free."""
        self.search_strategy = create_search_strategy(
            param_grid=self.param_grid,
            random_search=self.random_search,
            use_bayesian_optimization=self.use_bayesian_optimization,
            n_iter=self.n_iter,
            random_state=self.random_state,
            bayesian_optimizer=self.bayesian_optimizer,
            scoring=self._scoring[0] if self._scoring else 'f1',
        )
        self.result_manager = ResultManager(
            scoring=self._scoring,
            results_csv=self.results_csv,
        )
        self.cv_executor = CVExecutor(
            cv=self.cv,
            scoring=self._scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
        )

    def _reset_fit_state(self):
        """Discard runtime and fitted state from a previous fit attempt."""
        for name in (
            "search_strategy", "study_store", "study_id", "result_manager", "cv_executor",
            "_study_identity", "best_params_", "best_score_", "best_estimator_", "cv_results_", "best_index_",
        ):
            self.__dict__.pop(name, None)

    def _format_metric_string(self, result: dict) -> str:
        """
        Format metrics from a result dictionary into a readable string.

        Parameters:
        -----------
        result : dict
            Result dictionary containing mean_test_* and std_test_* keys.

        Returns:
        --------
        str
            Formatted string like "f1: 0.85 ± 0.02 | accuracy: 0.90 ± 0.01"
        """
        metrics = []
        for metric in self._scoring:
            mean_key = f'mean_test_{metric}'
            std_key = f'std_test_{metric}'
            if mean_key in result:
                metrics.append(
                    f"{metric}: {result[mean_key]:.4f} ± {result[std_key]:.4f}"
                )
        return " | ".join(metrics)

    def _compute_best_metrics(self) -> str:
        """
        Compute the best metric values across all valid results.

        Returns:
        --------
        str
            Formatted string like "f1: 0.92 | accuracy: 0.95"
        """
        valid_results = [r for r in self.result_manager.results if 'error' not in r]
        if not valid_results:
            return ""

        best_metrics = []
        for metric in self._scoring:
            metric_key = f'mean_test_{metric}'
            best_val = max(
                r[metric_key] for r in valid_results
                if metric_key in r
            )
            best_metrics.append(f"{metric}: {best_val:.4f}")
        return " | ".join(best_metrics)

    def fit(self, X, y, groups=None):
        """Run search with SQLite as transactional source of truth."""
        try:
            return self._fit_impl(X, y, groups)
        finally:
            store = self.__dict__.get("study_store")
            if store is not None:
                store.close()

    def _storage_path(self) -> str:
        if self.storage_path is not None:
            return self.storage_path
        return str(Path(self.checkpoint_path).with_suffix(".sqlite3"))

    def _strategy_identity_config(self) -> dict[str, object]:
        """Choices that alter proposal order or stopping semantics."""
        return {
            "random_search": self.random_search,
            "use_bayesian_optimization": self.use_bayesian_optimization,
            "n_iter": self.n_iter,
            "early_stopping_patience": self.early_stopping_patience,
        }

    def _fit_impl(self, X, y, groups=None):
        """
        Performs hyperparameter tuning with intermediate result saving.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        groups : array-like of shape (n_samples,), default=None
            Groups for group cross-validation (if used).

        Returns:
        --------
        self : object
            Returns the instance.
        """
        self._reset_fit_state()
        if self.dataset_id is None:
            warnings.warn(
                "dataset_id=None disables dataset-level checkpoint identity; pass a stable dataset_id for safe resume.",
                UserWarning,
                stacklevel=2,
            )
        self._study_identity = StudyIdentity.create(
            estimator=self.estimator,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            random_state=self.random_state,
            dataset_id=self.dataset_id,
            scorer_id=self.scorer_id,
            cv_id=self.cv_id,
            strategy_config=self._strategy_identity_config(),
        )
        self._create_runtime_components()
        self.study_store = SQLiteStudyStore(self._storage_path())
        if self.clear_checkpoint:
            self.study_store.clear()
            self.study_store = SQLiteStudyStore(self._storage_path())

        self.study_id = self.study_store.open_study(self._study_identity, resume=self.resume)
        checkpoint_results = self.study_store.results(self.study_id)
        self.result_manager.add_results(checkpoint_results)

        # Keep candidate generation lazy. Large grids must not consume RAM before
        # their first trial starts.
        total_candidates = self.search_strategy.total_candidates()
        if self.verbose:
            print(f"Total combinations: {total_candidates}")

        # Exclude already processed without a materialized ``remaining_params``.
        completed_keys = self.study_store.completed_param_keys(self.study_id)
        remaining_params = (
            p for p in self.search_strategy.iter_parameters()
            if self.result_manager.param_key(p) not in completed_keys
        )
        if self.verbose:
            print(f"Completed trials: {len(completed_keys)}")

        # If Bayesian optimization is used, sort remaining parameters by prediction
        if self.use_bayesian_optimization:
            remaining_params = self.search_strategy.suggest_next(checkpoint_results)
            if self.verbose:
                print("Remaining parameters sorted by predicted metric.")

        # Early stopping needs a meaningful proposal order. Exhaustive grid has
        # none, so legacy patience is ignored there rather than silently making
        # grid results order-dependent.
        early_stopping_enabled = self.early_stopping_patience is not None and (
            self.random_search or self.use_bayesian_optimization
        )
        if self.early_stopping_patience is not None and not early_stopping_enabled:
            warnings.warn(
                "early_stopping_patience applies only to random or adaptive search; ignoring it for grid search.",
                UserWarning,
                stacklevel=2,
            )

        # Early stopping tracking. Rebuild from committed trials if state write
        # was interrupted after a trial commit.
        primary_metric = self._scoring[0]
        best_score, no_improvement_count = _early_stop_from_results(
            checkpoint_results, primary_metric
        )
        if early_stopping_enabled:
            saved_state = self.study_store.study_state(self.study_id).get("early_stopping", {})
            if (
                saved_state.get("metric") == primary_metric
                and saved_state.get("processed_trial_count") == len(checkpoint_results)
            ):
                best_score = saved_state["best_score"]
                no_improvement_count = saved_state["no_improvement_count"]

        if early_stopping_enabled and no_improvement_count >= self.early_stopping_patience:
            remaining_params = ()

        # Iterate over remaining parameters
        for i, params in enumerate(remaining_params, start=1):
            if self.verbose:
                print(f"\n[{i}/{total_candidates}] Testing: {params}")

            result = self.cv_executor.evaluate(
                estimator=self.estimator,
                X=X,
                y=y,
                params=params,
                groups=groups,
            )
            if self.study_store.commit_trial(self.study_id, params, result):
                self.result_manager.add_result(result)

            if self.verbose and 'error' not in result:
                current_str = self._format_metric_string(result)
                best_str = self._compute_best_metrics()
                print(f"Saved. Current: {current_str} | Best: {best_str}")

            # Early stopping logic
            if early_stopping_enabled:
                if 'error' not in result:
                    current_score = result.get(f'mean_test_{primary_metric}', -float('inf'))
                    if current_score > best_score + 1e-9:  # improvement
                        best_score = current_score
                        no_improvement_count = 0
                        if self.verbose:
                            print(f"🎯 Improvement detected (new best: {best_score:.4f})")
                    else:
                        no_improvement_count += 1
                        if self.verbose:
                            print(f"⏳ No improvement ({no_improvement_count}/{self.early_stopping_patience})")
                else:
                    # Error counts as no improvement
                    no_improvement_count += 1

                if no_improvement_count >= self.early_stopping_patience:
                    self.study_store.update_study_state(
                        self.study_id,
                        {
                            "early_stopping": {
                                "metric": primary_metric,
                                "best_score": best_score,
                                "no_improvement_count": no_improvement_count,
                                "processed_trial_count": len(self.result_manager.results),
                                "stop_reason": "patience_exhausted",
                            }
                        },
                    )
                    if self.verbose:
                        print(f"🛑 Early stopping triggered after {i} iterations (no improvement for {self.early_stopping_patience} consecutive trials).")
                    break

                self.study_store.update_study_state(
                    self.study_id,
                    {
                        "early_stopping": {
                            "metric": primary_metric,
                            "best_score": best_score,
                            "no_improvement_count": no_improvement_count,
                            "processed_trial_count": len(self.result_manager.results),
                            "stop_reason": None,
                        }
                    },
                )

        # Save results to CSV
        self.result_manager.save_to_csv()

        # Update attributes for compatibility with GridSearchCV
        self.cv_results_ = self.result_manager.format_cv_results()
        self._update_best_attributes()

        # Refit the best estimator on the whole dataset
        if self.refit and hasattr(self, "best_params_") and self.best_params_:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

        if self.verbose:
            print(f"\nAll results saved to {self.results_csv}")
            if hasattr(self, "best_score_"):
                print(f"Best result ({self._scoring[0]}): {self.best_score_:.4f}")
            if self.random_search:
                print(
                    f"Random search used: {total_candidates} candidates."
                )

        return self

    def _update_best_attributes(self):
        """Set best_params_, best_score_, and best_index_ from result_manager."""
        valid_results = [r for r in self.result_manager.results if 'error' not in r]
        if not valid_results:
            return

        # Sort by the first metric
        scoring_key = f'mean_test_{self._scoring[0]}'
        best_result = max(valid_results, key=lambda x: x.get(scoring_key, float('-inf')))
        self.best_params_ = best_result['params']
        self.best_score_ = best_result.get(scoring_key, 0.0)

        # Find index in cv_results_['params']
        if self.cv_results_ and 'params' in self.cv_results_:
            params_list = self.cv_results_['params']
            for idx, param_dict in enumerate(params_list):
                if param_dict == self.best_params_:
                    self.best_index_ = idx
                    break
            else:
                self.best_index_ = None
        else:
            self.best_index_ = None

    def predict(self, X):
        """
        Predictions using the best model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data for prediction.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        self._check_refitted()
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Class probabilities (if the best model supports predict_proba).

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data for prediction.

        Returns:
        --------
        y_proba : array-like of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_refitted()
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        """
        Evaluate the best model on data X, y.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Data for evaluation.
        y : array-like of shape (n_samples,)
            True values.

        Returns:
        --------
        score : float
            Metric value (default uses scoring[0]).
        """
        self._check_refitted()
        return self.best_estimator_.score(X, y)

    def get_top_results(self, n: int = 10) -> pd.DataFrame:
        """
        Returns top‑N results.

        Parameters:
        -----------
        n : int
            Number of top results to return.

        Returns:
        --------
        pd.DataFrame: Top‑N results.
        """
        return self.result_manager.get_top_results(n)

    def _check_refitted(self):
        if not hasattr(self, "best_estimator_") or self.best_estimator_ is None:
            raise NotFittedError(
                "HyperPhoenixCV is not refitted. Call fit(..., refit=True) first."
            )

    def clear_checkpoint_file(self):
        """
        Deletes the checkpoint file explicitly.

        `clear_checkpoint` is a sklearn constructor parameter, so it cannot
        also be an instance method.
        """
        SQLiteStudyStore(self._storage_path()).clear()

    def load_results_from_checkpoint(self, n: int = 10) -> pd.DataFrame:
        """
        Loads results from SQLite storage and returns top‑N.
        Useful when fit() was interrupted and CSV was not created.

        Parameters:
        -----------
        n : int
            Number of top results to return

        Returns:
        --------
        pd.DataFrame
            Top‑N results from the checkpoint
        """
        with SQLiteStudyStore(self._storage_path()) as store:
            study_id = store.open_study(self._identity_for_loading(), resume="must")
            checkpoint_results = store.results(study_id)
        # Create a temporary ResultManager to format results
        temp_manager = ResultManager(scoring=self._scoring)
        temp_manager.add_results(checkpoint_results)
        return temp_manager.get_top_results(n)

    def import_legacy_checkpoint(self, path: str, *, trusted: bool = False) -> dict:
        """Import trusted legacy ``List[dict]`` pickle data into SQLite.

        Pickle/joblib deserialization can execute arbitrary code. Set
        ``trusted=True`` only after verifying source and provenance of ``path``.
        Normal ``fit`` and resume never read pickle checkpoints. Repeating this
        import is idempotent: existing parameter combinations are skipped.
        """
        legacy_results = load_legacy_results(path, trusted=trusted)
        report = {
            "imported": 0,
            "skipped": 0,
            "failed": 0,
            "failures": [],
            "skipped_indices": [],
        }
        with SQLiteStudyStore(self._storage_path()) as store:
            study_id = store.open_study(self._identity_for_loading(), resume="auto")
            report["study_id"] = study_id
            for index, result in enumerate(legacy_results):
                problem = validate_legacy_result(result)
                if problem is not None:
                    report["failed"] += 1
                    report["failures"].append({"index": index, "reason": problem})
                    continue
                try:
                    if store.commit_trial(study_id, result["params"], result):
                        report["imported"] += 1
                    else:
                        report["skipped"] += 1
                        report["skipped_indices"].append(index)
                except Exception as exc:
                    report["failed"] += 1
                    report["failures"].append({"index": index, "reason": str(exc)})
        return report

    def _identity_for_loading(self):
        if hasattr(self, "_study_identity"):
            return self._study_identity
        return StudyIdentity.create(
            estimator=self.estimator, param_grid=self.param_grid, scoring=self.scoring,
            cv=self.cv, random_state=self.random_state, dataset_id=self.dataset_id,
            scorer_id=self.scorer_id, cv_id=self.cv_id,
            strategy_config=self._strategy_identity_config(),
        )
    def _load_checkpoint(self):
        """
        Private compatibility method. Returns SQLite trial projection.
        """
        with SQLiteStudyStore(self._storage_path()) as store:
            study_id = store.open_study(self._identity_for_loading(), resume="must")
            return store.results(study_id)

    def _save_checkpoint(self, results):
        """
        Removed private pickle persistence path.
        """
        raise RuntimeError(
            "Pickle checkpoint persistence was removed. SQLite is the source of truth."
        )
