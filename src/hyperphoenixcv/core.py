from __future__ import annotations

"""
HyperPhoenixCV - Resumable hyperparameter search with checkpoint support.

This module provides the HyperPhoenixCV class, which extends the functionality
of scikit-learn's GridSearchCV by adding checkpoint support, random search,
an optional Optuna backend, and experimental surrogate-ranking compatibility.
"""

import warnings
import secrets
from pathlib import Path
from collections.abc import Mapping
import numpy as np
import pandas as pd
from typing import Union
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError

from .search_strategies import create_search_strategy
from .legacy_pickle import load_legacy_results, validate_legacy_result
from .result_manager import ResultManager
from .cv_executor import SklearnCVEvaluator
from .study_identity import StudyIdentity
from .storage import SQLiteStudyStore
from .study_engine import StudyEngine, StudySpec
from .scheduler import SchedulerSpec, TrialScheduler


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
    Resumable hyperparameter search with checkpoint support.
    Supports grid/random search, optional Optuna search, and experimental surrogate ranking.

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
        param_grid: dict | list[dict] | None = None,
        scoring: str | list[str] | Mapping[str, object] | object = 'f1',
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
        parallelism: str = "trials",
        inner_max_num_threads: int | None = None,
        strategy: str | None = None,
        search_space=None,
        search_space_id: str | None = None,
        n_trials: int | None = None,
        optuna_warmup_trials: int = 10,
        optuna_directions: Mapping[str, str] | None = None,
        intermediate_evaluator=None,
        trial_timeout: float | None = None,
        cancel_callback=None,
        memmap_max_nbytes: int | str | None = "1M",
        memmap_temp_folder: str | None = None,
        joblib_batch_size: int | str = "auto",
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
            Deprecated compatibility flag for experimental surrogate ranking;
            this is not Bayesian optimization.
        bayesian_optimizer : sklearn regressor, optional
            Deprecated model argument for experimental surrogate ranking.
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
            experimental surrogate ranking to avoid unnecessary evaluations.
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
        self.parallelism = parallelism
        self.inner_max_num_threads = inner_max_num_threads
        self.strategy = strategy
        self.search_space = search_space
        self.search_space_id = search_space_id
        self.n_trials = n_trials
        self.optuna_warmup_trials = optuna_warmup_trials
        self.optuna_directions = optuna_directions
        self.intermediate_evaluator = intermediate_evaluator
        self.trial_timeout = trial_timeout
        self.cancel_callback = cancel_callback
        self.memmap_max_nbytes = memmap_max_nbytes
        self.memmap_temp_folder = memmap_temp_folder
        self.joblib_batch_size = joblib_batch_size

    @property
    def _scoring(self):
        """Normalized scoring used internally; public constructor value stays intact."""
        if isinstance(self.scoring, Mapping):
            return list(self.scoring)
        if callable(self.scoring):
            return ["score"]
        return self.scoring if isinstance(self.scoring, list) else [self.scoring]

    def _create_runtime_components(self, *, sampler_random_state: int | None = None):
        """Create per-fit collaborators. Constructor must stay side-effect free."""
        self.search_strategy = create_search_strategy(
            param_grid=self.param_grid,
            random_search=self.random_search,
            use_bayesian_optimization=self.use_bayesian_optimization,
            n_iter=self.n_iter,
            random_state=sampler_random_state if sampler_random_state is not None else self.random_state,
            bayesian_optimizer=self.bayesian_optimizer,
            scoring=self._scoring[0] if self._scoring else 'f1',
            strategy=self.strategy,
            search_space=self.search_space,
            n_trials=self.n_trials,
            optuna_warmup_trials=self.optuna_warmup_trials,
            optuna_directions=(self.optuna_directions or {self._scoring[0]: "maximize"}),
        )
        self.result_manager = ResultManager(
            scoring=self._scoring,
            results_csv=self.results_csv,
        )
        self.cv_executor = SklearnCVEvaluator(
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=1 if self.parallelism == "trials" else self.n_jobs,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
        )

    def _reset_fit_state(self):
        """Discard runtime and fitted state from a previous fit attempt."""
        for name in (
            "search_strategy", "study_store", "study_id", "study_engine", "trial_scheduler", "result_manager", "cv_executor",
            "_study_identity", "best_params_", "best_score_", "best_estimator_", "cv_results_", "best_index_",
        ):
            self.__dict__.pop(name, None)

    def _is_multi_objective(self) -> bool:
        return self.strategy == "optuna" and self.optuna_directions is not None and len(self.optuna_directions) > 1

    def _validate_refit(self) -> None:
        """Validate supported sklearn refit forms before creating a study."""
        metrics = self._scoring
        if isinstance(self.refit, bool):
            if self._is_multi_objective() and self.refit:
                raise ValueError("Multi-objective Optuna search requires refit=False, a metric name, or a callable")
            if self.refit and len(metrics) > 1:
                raise ValueError(
                    "For multi-metric scoring, refit must be a scorer name, callable, or False."
                )
            return
        if isinstance(self.refit, str):
            if self.refit not in metrics:
                raise ValueError(f"refit={self.refit!r} is not a scoring metric")
            return
        if callable(self.refit):
            return
        raise ValueError("refit must be bool, scorer name, or callable")

    def _validate_strategy(self) -> None:
        if self.optuna_directions is not None:
            if self.strategy != "optuna":
                raise ValueError("optuna_directions requires strategy='optuna'")
            if not isinstance(self.optuna_directions, Mapping):
                raise TypeError("optuna_directions must be a mapping")
            if set(self.optuna_directions) != set(self._scoring):
                raise ValueError("optuna_directions keys must exactly match scoring metric names")
            if any(value not in {"maximize", "minimize"} for value in self.optuna_directions.values()):
                raise ValueError("optuna_directions values must be 'maximize' or 'minimize'")
        elif self.strategy == "optuna" and len(self._scoring) != 1:
            raise ValueError("multi-metric Optuna requires optuna_directions")
        if self.intermediate_evaluator is not None and not callable(self.intermediate_evaluator):
            raise TypeError("intermediate_evaluator must be callable or None")
        if self.intermediate_evaluator is not None and self.strategy != "optuna":
            raise ValueError("intermediate_evaluator requires strategy='optuna'")

    def _worker_count(self) -> int:
        return self.trial_scheduler.worker_count()

    def _validate_scheduler(self) -> None:
        self.trial_scheduler = TrialScheduler(SchedulerSpec(
            parallelism=self.parallelism, n_jobs=self.n_jobs,
            inner_max_num_threads=self.inner_max_num_threads,
            trial_timeout=self.trial_timeout,
            memmap_max_nbytes=self.memmap_max_nbytes,
            memmap_temp_folder=self.memmap_temp_folder,
            joblib_batch_size=self.joblib_batch_size,
        ))
        if self.cancel_callback is not None and not callable(self.cancel_callback):
            raise ValueError("cancel_callback must be callable or None")

    def _update_study_state(self, patch: dict) -> None:
        state = self.study_store.study_state(self.study_id)
        state.update(patch)
        self.study_store.update_study_state(self.study_id, state)

    def _evaluate_batch(self, proposals, X, y, groups, fit_params):
        if self.intermediate_evaluator is not None:
            # Cooperative path only. sklearn cross_validate has no honest
            # mid-fit signal, hence plain CV never receives a prune request.
            output = []
            for params in proposals:
                reporter = self.search_strategy.intermediate_reporter(params)
                try:
                    result = self.intermediate_evaluator(
                        self.estimator, X, y, params, reporter, groups, fit_params or {}
                    )
                    if not isinstance(result, dict):
                        raise TypeError("intermediate_evaluator must return a result dictionary")
                    result.setdefault("params", params)
                    diagnostics = result.setdefault("trial_diagnostics", {})
                    if not isinstance(diagnostics, dict):
                        raise TypeError("trial_diagnostics must be a dictionary")
                    diagnostics.setdefault("intermediate_reports", reporter.diagnostics)
                except Exception as exc:
                    result = {"params": params, "error": str(exc), "error_type": type(exc).__name__,
                              "trial_diagnostics": {"intermediate_reports": reporter.diagnostics}}
                output.append(result)
            return output
        kwargs = [
            {
                "estimator": self.estimator, "X": X, "y": y,
                "params": params, "groups": groups,
                **({"fit_params": fit_params} if fit_params else {}),
            }
            for params in proposals
        ]
        return self.trial_scheduler.evaluate(self.cv_executor, kwargs)

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

    def fit(self, X, y, groups=None, **fit_params):
        """Run search with SQLite as transactional source of truth."""
        try:
            return self._fit_impl(X, y, groups, fit_params)
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
            "strategy": self.strategy,
            "n_iter": self.n_iter,
            "n_trials": self.n_trials,
            "optuna_warmup_trials": self.optuna_warmup_trials,
            "optuna_directions": dict(self.optuna_directions) if self.optuna_directions is not None else None,
            "early_stopping_patience": self.early_stopping_patience,
            "parallelism": self.parallelism,
            "trial_timeout": self.trial_timeout,
            "memmap_max_nbytes": self.memmap_max_nbytes,
            "joblib_batch_size": self.joblib_batch_size,
        }

    def _identity_search_space(self):
        """Stable identity projection for optional Optuna spaces."""
        if self.search_space is None:
            return self.param_grid
        if callable(self.search_space):
            if self.search_space_id is None:
                raise ValueError(
                    "Callable search_space requires search_space_id for safe checkpoint resume"
                )
            return {"optuna_callable": self.search_space_id}
        return {str(name): repr(distribution) for name, distribution in self.search_space.items()}

    def _fit_impl(self, X, y, groups=None, fit_params: dict | None = None):
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
        self._validate_refit()
        self._validate_strategy()
        self._validate_scheduler()
        if self.clear_checkpoint:
            warnings.warn(
                "clear_checkpoint=True is deprecated and will be removed in 0.6; "
                "call clear_checkpoint_file() explicitly before fit() instead.",
                FutureWarning,
                stacklevel=2,
            )
        if self.dataset_id is None:
            warnings.warn(
                "dataset_id=None disables dataset-level checkpoint identity; pass a stable dataset_id for safe resume.",
                UserWarning,
                stacklevel=2,
            )
        self._study_identity = StudyIdentity.create(
            estimator=self.estimator,
            param_grid=self._identity_search_space(),
            scoring=self.scoring,
            cv=self.cv,
            random_state=self.random_state,
            dataset_id=self.dataset_id,
            scorer_id=self.scorer_id,
            cv_id=self.cv_id,
            strategy_config=self._strategy_identity_config(),
        )
        self.study_store = SQLiteStudyStore(self._storage_path())
        if self.clear_checkpoint:
            self.study_store.clear()
            self.study_store = SQLiteStudyStore(self._storage_path())

        self.study_id = self.study_store.open_study(self._study_identity, resume=self.resume)
        existing_scheduler = self.study_store.study_state(self.study_id).get("scheduler", {})
        self._update_study_state({
            "scheduler": {
                **existing_scheduler,
                "parallelism": self.parallelism,
                "n_jobs": self.n_jobs,
                "worker_count": self._worker_count(),
                "inner_max_num_threads": self.inner_max_num_threads,
                "trial_timeout": self.trial_timeout,
                "memmap_max_nbytes": self.memmap_max_nbytes,
                "memmap_temp_folder": self.memmap_temp_folder,
                "joblib_batch_size": self.joblib_batch_size,
                "attempts": existing_scheduler.get("attempts", len(self.study_store.results(self.study_id))),
                "cancellation_reason": existing_scheduler.get("cancellation_reason"),
            }
        })
        # ``ParameterSampler`` must replay exactly after a process crash.  A
        # caller seed already has that property; for ``None`` allocate one once
        # and persist it before asking for any proposal.
        sampler_random_state = self.random_state
        if self.random_search and sampler_random_state is None:
            state = self.study_store.study_state(self.study_id)
            sampler_random_state = state.get("sampler_random_state")
            if sampler_random_state is None:
                sampler_random_state = secrets.randbits(32)
                state["sampler_random_state"] = sampler_random_state
                self.study_store.update_study_state(self.study_id, state)
        self._create_runtime_components(sampler_random_state=sampler_random_state)
        checkpoint_results = self.study_store.results(self.study_id)
        self.result_manager.add_results(checkpoint_results)

        total_candidates = self.search_strategy.total_candidates()
        completed_keys = self.study_store.completed_param_keys(self.study_id)
        if self.verbose:
            print(f"Total combinations: {total_candidates}")
            print(f"Completed trials: {len(completed_keys)}")
        adaptive_search = self.random_search or self.use_bayesian_optimization
        if self.early_stopping_patience is not None and not adaptive_search:
            warnings.warn(
                "early_stopping_patience applies only to random or adaptive search; ignoring it for grid search.",
                UserWarning, stacklevel=2,
            )

        def on_trial(index, params, result):
            if not self.verbose:
                return
            print(f"\n[{index}/{total_candidates}] Testing: {params}")
            if "error" not in result:
                print(f"Saved. Current: {self._format_metric_string(result)} | Best: {self._compute_best_metrics()}")

        spec = StudySpec(
            scoring=tuple(self._scoring), strategy=self.strategy,
            random_search=self.random_search, adaptive_search=adaptive_search,
            early_stopping_patience=self.early_stopping_patience,
            batch_size=self._worker_count() if self.parallelism == "trials" else 1,
            total_candidates=total_candidates,
            optuna_directions=dict(self.optuna_directions) if self.optuna_directions is not None else None,
            cancel_callback=self.cancel_callback,
            timeout_enabled=self.trial_timeout is not None,
        )
        self.study_engine = StudyEngine(
            spec=spec, store=self.study_store, study_id=self.study_id,
            strategy=self.search_strategy, result_manager=self.result_manager,
            evaluate_batch=lambda proposals: self._evaluate_batch(proposals, X, y, groups, fit_params),
            on_trial=on_trial,
        )
        self.study_engine.run(checkpoint_results)

        # Save results to CSV
        self.result_manager.save_to_csv()

        # Update attributes for compatibility with GridSearchCV
        self.cv_results_ = self.result_manager.format_cv_results()
        self.pareto_front_ = self._pareto_front() if self._is_multi_objective() else []
        self._update_best_attributes()

        # Refit the best estimator on the whole dataset
        if self.refit and hasattr(self, "best_params_") and self.best_params_:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
            self.best_estimator_.fit(X, y, **(fit_params or {}))

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
        if self._is_multi_objective() and self.refit is False:
            return
        valid_results = [r for r in self.result_manager.results if 'error' not in r]
        if not valid_results:
            return

        if callable(self.refit):
            best_index = self.refit(self.cv_results_)
            if not isinstance(best_index, int) or not 0 <= best_index < len(valid_results):
                raise ValueError("refit callable must return a valid cv_results_ index")
            best_result = valid_results[best_index]
            self.best_index_ = best_index
            scoring_key = None
        else:
            metric = self.refit if isinstance(self.refit, str) else self._scoring[0]
            scoring_key = f'mean_test_{metric}'
            best_result = max(valid_results, key=lambda x: x.get(scoring_key, float('-inf')))
            self.best_index_ = valid_results.index(best_result)
        self.best_params_ = best_result['params']
        if scoring_key is not None:
            self.best_score_ = best_result.get(scoring_key, 0.0)

        if not self.cv_results_ or 'params' not in self.cv_results_:
            self.best_index_ = None

    def _pareto_front(self) -> list[dict]:
        """Completed non-dominated Optuna trials in durable trial order."""
        directions = self.optuna_directions or {}
        candidates = [
            (index, result) for index, result in enumerate(self.result_manager.results)
            if result.get("trial_state", "completed") == "completed"
            and "error" not in result and "objective_values" in result
        ]
        front = []
        for index, result in candidates:
            values = result["objective_values"]
            dominated = False
            for other_index, other in candidates:
                if other_index == index:
                    continue
                other_values = other["objective_values"]
                no_worse = all(
                    other_values[name] >= values[name] if direction == "maximize" else other_values[name] <= values[name]
                    for name, direction in directions.items()
                )
                better = any(
                    other_values[name] > values[name] if direction == "maximize" else other_values[name] < values[name]
                    for name, direction in directions.items()
                )
                if no_worse and better:
                    dominated = True
                    break
            if not dominated:
                front.append({"trial_index": index, "params": result["params"], "objective_values": values})
        return front

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
            estimator=self.estimator, param_grid=self._identity_search_space(), scoring=self.scoring,
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
