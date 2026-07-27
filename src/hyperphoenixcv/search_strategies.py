"""
Search strategies for hyperparameter optimization.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Callable, List, Dict, Any, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from .study_identity import param_key
from .search_protocols import Evaluator, Sampler, SearchSpace


class SearchStrategy(ABC, SearchSpace, Sampler):
    """
    Abstract base class for hyperparameter search strategies.
    """

    def __init__(self, param_grid: Mapping[str, Any] | List[Dict[str, Any]]):
        self.param_grid = param_grid
        self._proposal_iterator: Iterator[Dict[str, Any]] | None = None
        self._known_param_keys: set[str] = set()

    @abstractmethod
    def generate_parameters(self) -> List[Dict[str, Any]]:
        """
        Generate a list of parameter combinations to evaluate.
        """
        pass

    @abstractmethod
    def iter_parameters(self) -> Iterator[Dict[str, Any]]:
        """Yield proposals without materializing the whole search space."""
        pass

    @abstractmethod
    def total_candidates(self) -> int:
        """Return finite candidate count without creating candidates."""
        pass

    def suggest_next(self, completed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Suggest next parameters based on completed results.
        Default implementation returns all generated parameters (no sorting).
        """
        return self.generate_parameters()

    def restore(self, results: List[Dict[str, Any]]) -> None:
        """Start a resumable ask/tell session from committed trial history."""
        self._known_param_keys = {
            param_key(result["params"])
            for result in results
            if "params" in result
        }
        self._proposal_iterator = self.iter_parameters()

    def ask(self, n: int) -> List[Dict[str, Any]]:
        """Return up to ``n`` unseen proposals without materializing the space."""
        if n < 1:
            return []
        if self._proposal_iterator is None:
            self.restore([])

        proposals = []
        while len(proposals) < n:
            try:
                params = next(self._proposal_iterator)
            except StopIteration:
                break
            key = param_key(params)
            if key in self._known_param_keys:
                continue
            # Reserve now. This prevents duplicate proposals in one batch; tell
            # keeps terminal results reserved across later batches.
            self._known_param_keys.add(key)
            proposals.append(params)
        return proposals

    def tell(self, results: List[Dict[str, Any]]) -> None:
        """Accept terminal results. Future adaptive samplers override this hook."""
        for result in results:
            if "params" in result:
                self._known_param_keys.add(param_key(result["params"]))


from .search_grid import ExhaustiveSearchStrategy
from .search_random import RandomSearchStrategy


class OptunaSearchStrategy(SearchStrategy):
    """Optional genuine Optuna ask/tell adapter.

    ``search_space`` is either a mapping of Optuna distributions or a callable
    receiving an Optuna ``Trial`` and returning estimator parameters. Callable
    spaces support conditional suggestions. SQLite remains source of truth: a
    fresh in-memory Optuna study is rebuilt from committed terminal trials on
    every resume.
    """

    def __init__(
        self,
        search_space: Mapping[str, Any] | Callable[[Any], Dict[str, Any]],
        n_trials: int,
        random_state: Optional[int] = None,
        warmup_trials: int = 10,
        directions: Mapping[str, str] | None = None,
    ):
        super().__init__({})
        if n_trials < 0:
            raise ValueError("n_trials must be non-negative")
        if warmup_trials < 0:
            raise ValueError("optuna_warmup_trials must be non-negative")
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Optuna strategy requires optional dependency. Install with "
                "`pip install hyperphoenixcv[optuna]`."
            ) from exc
        if not isinstance(search_space, Mapping) and not callable(search_space):
            raise TypeError("search_space must be an Optuna distribution mapping or callable")
        if isinstance(search_space, Mapping):
            base = optuna.distributions.BaseDistribution
            invalid = [name for name, value in search_space.items() if not isinstance(value, base)]
            if invalid:
                raise TypeError(
                    "Optuna mapping values must be optuna distributions; invalid: "
                    + ", ".join(map(str, invalid))
                )
        self.optuna = optuna
        self.search_space = search_space
        self.n_trials = n_trials
        self.random_state = random_state
        self.warmup_trials = warmup_trials
        self.directions = dict(directions or {"score": "maximize"})
        if not self.directions or any(value not in {"maximize", "minimize"} for value in self.directions.values()):
            raise ValueError("optuna directions must map objectives to 'maximize' or 'minimize'")
        self.study = None
        self._trials_by_key: dict[str, Any] = {}
        self._terminal_count = 0

    def _new_study(self):
        sampler = self.optuna.samplers.TPESampler(
            seed=self.random_state, n_startup_trials=self.warmup_trials,
        )
        values = list(self.directions.values())
        if len(values) == 1:
            return self.optuna.create_study(direction=values[0], sampler=sampler)
        return self.optuna.create_study(directions=values, sampler=sampler)

    def _objective_values(self, result: Dict[str, Any]):
        values = result.get("objective_values")
        if values is not None:
            return [float(values[name]) for name in self.directions]
        return [float(result.get(f"mean_test_{name}", result.get("mean_test_score", float("nan"))))
                for name in self.directions]

    def _suggest(self, trial) -> Dict[str, Any]:
        if callable(self.search_space):
            params = self.search_space(trial)
            if not isinstance(params, dict):
                raise TypeError("Optuna search_space callable must return dict[str, Any]")
            return params
        return {
            name: trial._suggest(name, distribution)
            for name, distribution in self.search_space.items()
        }

    def generate_parameters(self) -> List[Dict[str, Any]]:
        self.restore([])
        return self.ask(self.n_trials)

    def iter_parameters(self) -> Iterator[Dict[str, Any]]:
        self.restore([])
        while params := self.ask(1):
            yield params[0]

    def total_candidates(self) -> int:
        return self.n_trials

    def restore(self, results: List[Dict[str, Any]]) -> None:
        self._known_param_keys = set()
        self._trials_by_key = {}
        self._terminal_count = len(results)
        self.study = self._new_study()
        trial_module = self.optuna.trial
        for result in results:
            params = result.get("params")
            if params is None:
                continue
            distributions = result.get("optuna_distributions")
            if distributions is not None:
                distributions = {
                    name: self.optuna.distributions.json_to_distribution(value)
                    for name, value in distributions.items()
                }
            elif isinstance(self.search_space, Mapping):
                distributions = dict(self.search_space)
            else:
                raise ValueError(
                    "Cannot resume callable Optuna search_space trial without persisted distributions"
                )
            state = str(result.get("trial_state", "failed" if "error" in result else "completed")).lower()
            if state == "pruned":
                frozen = trial_module.create_trial(
                    params=params, distributions=distributions, state=trial_module.TrialState.PRUNED,
                )
            elif state in {"failed", "cancelled"} or "error" in result:
                frozen = trial_module.create_trial(
                    params=params, distributions=distributions, state=trial_module.TrialState.FAIL,
                )
            else:
                values = self._objective_values(result)
                frozen = trial_module.create_trial(params=params, distributions=distributions,
                    **({"value": values[0]} if len(values) == 1 else {"values": values}))
            self.study.add_trial(frozen)
            self._known_param_keys.add(param_key(params))

    def ask(self, n: int) -> List[Dict[str, Any]]:
        if n < 1 or self._terminal_count + len(self._trials_by_key) >= self.n_trials:
            return []
        if self.study is None:
            self.restore([])
        proposals = []
        # Finite categorical spaces can repeat. Bound retries so caller cannot
        # hang when SQLite's unique parameter contract exhausts a space.
        attempts_left = max(100, (self.n_trials - self._terminal_count) * 20)
        while len(proposals) < n and self._terminal_count + len(self._trials_by_key) < self.n_trials and attempts_left:
            attempts_left -= 1
            trial = self.study.ask()
            params = self._suggest(trial)
            key = param_key(params)
            if key in self._known_param_keys:
                self.study.tell(trial, state=self.optuna.trial.TrialState.PRUNED)
                continue
            self._known_param_keys.add(key)
            self._trials_by_key[key] = trial
            proposals.append(params)
        return proposals

    def result_metadata(self, params: Dict[str, Any]) -> dict[str, Any]:
        """JSON-safe Optuna trial data needed for deterministic replay."""
        trial = self._trials_by_key.get(param_key(params))
        if trial is None:
            return {}
        return {
            "optuna_distributions": {
                name: self.optuna.distributions.distribution_to_json(distribution)
                for name, distribution in trial.distributions.items()
            }
        }

    def intermediate_reporter(self, params: Dict[str, Any]):
        """Return ``report(step, value) -> should_prune`` for one live trial.

        Optuna itself has no multi-objective intermediate-value API, so this is
        intentionally available only for scalar studies.
        """
        trial = self._trials_by_key.get(param_key(params))
        if trial is None:
            raise RuntimeError("No live Optuna trial for intermediate report")
        if len(self.directions) != 1:
            raise ValueError("Optuna pruning is available only for scalar objectives")
        last_step = -1
        reports = []
        def report(step: int, value: float) -> bool:
            nonlocal last_step
            if not isinstance(step, (int, np.integer)) or step <= last_step:
                raise ValueError("intermediate report step must be monotonically increasing")
            last_step = int(step)
            value = float(value)
            trial.report(value, last_step)
            should_prune = bool(trial.should_prune())
            reports.append({"step": last_step, "value": value, "should_prune": should_prune})
            return should_prune
        report.diagnostics = reports
        return report

    def tell(self, results: List[Dict[str, Any]]) -> None:
        for result in results:
            params = result.get("params")
            if params is None:
                continue
            key = param_key(params)
            trial = self._trials_by_key.pop(key, None)
            if trial is None:
                continue
            state = str(result.get("trial_state", "failed" if "error" in result else "completed")).lower()
            if state == "pruned":
                self.study.tell(trial, state=self.optuna.trial.TrialState.PRUNED)
            elif state in {"failed", "cancelled"} or "error" in result:
                self.study.tell(trial, state=self.optuna.trial.TrialState.FAIL)
            else:
                values = self._objective_values(result)
                self.study.tell(trial, values[0] if len(values) == 1 else values)
            self._terminal_count += 1


class ExperimentalSurrogateRankingStrategy(SearchStrategy):
    """
    Experimental surrogate-ranking strategy. This is not Bayesian optimization:
    it has no acquisition function or sequential Bayesian sampler.
    """

    def __init__(
        self,
        param_grid: Dict[str, Any],
        scoring: str,  # primary metric to optimize
        model=None,
        random_state: Optional[int] = None,
    ):
        super().__init__(param_grid)
        self.scoring = scoring
        if model is None:
            self.model = RandomForestRegressor(n_estimators=20, random_state=42)
        else:
            self.model = model
        self.random_state = random_state
        self.label_encoders = {}
        self._fit_label_encoders()

    def _fit_label_encoders(self):
        """Pre‑fit label encoders on all possible categorical values from param_grid."""
        # Collect all possible values for each parameter
        param_values = {}
        for param, values in self.param_grid.items():
            param_values[param] = values

        # Create a DataFrame with all possible combinations (could be huge, but we only need unique values per column)
        # Instead, we iterate over each parameter and collect unique values.
        for param, values in param_values.items():
            # Determine if the parameter is categorical (contains non‑numeric values)
            # We'll treat any value that is not int or float as categorical.
            categorical = any(
                not isinstance(v, (int, float, np.integer, np.floating))
                for v in values
            )
            if categorical:
                le = LabelEncoder()
                # Convert all values to strings for consistent encoding
                unique_vals = list(set(str(v) for v in values))
                le.fit(unique_vals)
                self.label_encoders[param] = le

    def generate_parameters(self) -> List[Dict[str, Any]]:
        return list(self.iter_parameters())

    def iter_parameters(self) -> Iterator[Dict[str, Any]]:
        return iter(ParameterGrid(self.param_grid))

    def total_candidates(self) -> int:
        return len(ParameterGrid(self.param_grid))

    def suggest_next(self, completed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not completed_results:
            return self.generate_parameters()

        # Extract parameters and scores
        completed_params = [r['params'] for r in completed_results]
        scoring_key = f'mean_test_{self.scoring}'
        completed_scores = [r.get(scoring_key, 0.0) for r in completed_results]

        # Encode parameters
        X_train = self._encode_params(completed_params)
        y_train = np.array(completed_scores)

        # Train surrogate model
        self.model.fit(X_train, y_train)

        # Generate all possible parameters and filter out completed ones
        all_params = self.generate_parameters()
        remaining = [p for p in all_params if p not in completed_params]
        if not remaining:
            return []

        # Predict scores for remaining parameters
        X_remaining = self._encode_params(remaining)
        predicted_scores = self.model.predict(X_remaining)

        # Sort by predicted score (descending)
        sorted_indices = np.argsort(predicted_scores)[::-1]
        return [remaining[i] for i in sorted_indices]

    def update_model(self, new_results: List[Dict[str, Any]]):
        """
        Incrementally update the surrogate model with new results.
        For simplicity, we just retrain on all data when suggest_next is called.
        """
        # This is a placeholder; actual incremental learning could be implemented.
        pass

    def _encode_params(self, params_list: List[Dict[str, Any]]) -> np.ndarray:
        """Encode categorical parameters into numeric matrix."""
        if not params_list:
            return np.array([]).reshape(0, -1)
        df = pd.DataFrame(params_list)
        X = df.copy()
        for col in X.columns:
            if col in self.label_encoders:
                # Categorical column with pre‑fitted encoder
                X[col] = self.label_encoders[col].transform(X[col].astype(str))
            else:
                # Numeric column
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        return X.values


# Compatibility alias. New code must use the explicit experimental name.
BayesianSearchStrategy = ExperimentalSurrogateRankingStrategy


def create_search_strategy(
    param_grid: Mapping[str, Any] | List[Dict[str, Any]] | None,
    random_search: bool = False,
    use_bayesian_optimization: bool = False,
    n_iter: int = 10,
    random_state: Optional[int] = None,
    bayesian_optimizer = None,
    scoring: str = 'f1',
    *,
    strategy: str | None = None,
    search_space: Mapping[str, Any] | Callable[[Any], Dict[str, Any]] | None = None,
    n_trials: int | None = None,
    optuna_warmup_trials: int = 10,
    optuna_directions: Mapping[str, str] | None = None,
) -> SearchStrategy:
    """
    Factory function to create a search strategy based on configuration.
    Maintains backward compatibility with HyperPhoenixCV parameters.
    """
    if strategy is not None and strategy not in {
        "grid", "random", "optuna", "experimental_surrogate_ranking",
    }:
        raise ValueError("strategy must be 'grid', 'random', 'optuna', or 'experimental_surrogate_ranking'")
    if strategy == "optuna":
        if random_search or use_bayesian_optimization or bayesian_optimizer is not None:
            raise ValueError("strategy='optuna' conflicts with legacy search settings")
        if search_space is None:
            raise ValueError("strategy='optuna' requires search_space")
        return OptunaSearchStrategy(
            search_space=search_space,
            n_trials=n_iter if n_trials is None else n_trials,
            random_state=random_state,
            warmup_trials=optuna_warmup_trials,
            directions=optuna_directions,
        )
    if search_space is not None:
        raise ValueError("search_space is supported only with strategy='optuna'")
    if param_grid is None:
        raise ValueError("param_grid is required unless strategy='optuna'")
    if strategy == "random":
        if use_bayesian_optimization or bayesian_optimizer is not None:
            raise ValueError("strategy='random' conflicts with Bayesian compatibility settings")
        if n_trials is not None:
            if n_iter != 10 and n_iter != n_trials:
                raise ValueError("n_trials conflicts with legacy n_iter")
            n_iter = n_trials
        random_search = True
    if strategy == "grid":
        if random_search or use_bayesian_optimization or bayesian_optimizer is not None:
            raise ValueError("strategy='grid' conflicts with legacy search settings")
    if strategy == "experimental_surrogate_ranking":
        if random_search:
            raise ValueError("strategy='experimental_surrogate_ranking' conflicts with random_search")
        use_bayesian_optimization = True
    if random_search and strategy is None:
        warnings.warn(
            "random_search is deprecated; use strategy='random' and n_trials instead.",
            FutureWarning,
            stacklevel=2,
        )
    if use_bayesian_optimization:
        warnings.warn(
            "use_bayesian_optimization is deprecated: current surrogate mode is not "
            "Bayesian optimization. Use random_search until the Optuna backend lands.",
            FutureWarning,
            stacklevel=2,
        )
        return ExperimentalSurrogateRankingStrategy(
            param_grid=param_grid,
            scoring=scoring,
            model=bayesian_optimizer,
            random_state=random_state,
        )
    elif random_search:
        return RandomSearchStrategy(
            param_grid=param_grid,
            n_iter=n_iter,
            random_state=random_state,
        )
    else:
        return ExhaustiveSearchStrategy(param_grid=param_grid)
