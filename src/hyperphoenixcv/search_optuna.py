"""Optional genuine Optuna ask/tell strategy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .search_strategies import SearchStrategy
from .study_identity import param_key


class OptunaSearchStrategy(SearchStrategy):
    """Rebuild in-memory Optuna state from committed SQLite terminal trials."""

    def __init__(self, search_space: Mapping[str, Any] | Callable[[Any], Dict[str, Any]], n_trials: int,
                 random_state: Optional[int] = None, warmup_trials: int = 10,
                 directions: Mapping[str, str] | None = None):
        super().__init__({})
        if n_trials < 0:
            raise ValueError("n_trials must be non-negative")
        if warmup_trials < 0:
            raise ValueError("optuna_warmup_trials must be non-negative")
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("Optuna strategy requires optional dependency. Install with `pip install hyperphoenixcv[optuna]`.") from exc
        if not isinstance(search_space, Mapping) and not callable(search_space):
            raise TypeError("search_space must be an Optuna distribution mapping or callable")
        if isinstance(search_space, Mapping):
            invalid = [name for name, value in search_space.items()
                       if not isinstance(value, optuna.distributions.BaseDistribution)]
            if invalid:
                raise TypeError("Optuna mapping values must be optuna distributions; invalid: " + ", ".join(map(str, invalid)))
        self.optuna, self.search_space, self.n_trials = optuna, search_space, n_trials
        self.random_state, self.warmup_trials = random_state, warmup_trials
        self.directions = dict(directions or {"score": "maximize"})
        if not self.directions or any(value not in {"maximize", "minimize"} for value in self.directions.values()):
            raise ValueError("optuna directions must map objectives to 'maximize' or 'minimize'")
        self.study = None
        self._trials_by_key: dict[str, Any] = {}
        self._terminal_count = 0

    def _new_study(self):
        sampler = self.optuna.samplers.TPESampler(seed=self.random_state, n_startup_trials=self.warmup_trials)
        directions = list(self.directions.values())
        return (self.optuna.create_study(direction=directions[0], sampler=sampler) if len(directions) == 1
                else self.optuna.create_study(directions=directions, sampler=sampler))

    def _objective_values(self, result):
        values = result.get("objective_values")
        return ([float(values[name]) for name in self.directions] if values is not None else
                [float(result.get(f"mean_test_{name}", result.get("mean_test_score", float("nan")))) for name in self.directions])

    def _suggest(self, trial):
        if callable(self.search_space):
            params = self.search_space(trial)
            if not isinstance(params, dict):
                raise TypeError("Optuna search_space callable must return dict[str, Any]")
            return params
        return {name: trial._suggest(name, distribution) for name, distribution in self.search_space.items()}

    def generate_parameters(self):
        self.restore([])
        return self.ask(self.n_trials)

    def iter_parameters(self) -> Iterator[Dict[str, Any]]:
        self.restore([])
        while params := self.ask(1):
            yield params[0]

    def total_candidates(self):
        return self.n_trials

    def restore(self, results):
        self._known_param_keys, self._trials_by_key = set(), {}
        self._terminal_count, self.study = len(results), self._new_study()
        for result in results:
            params = result.get("params")
            if params is None:
                continue
            raw = result.get("optuna_distributions")
            distributions = ({name: self.optuna.distributions.json_to_distribution(value) for name, value in raw.items()}
                             if raw is not None else dict(self.search_space) if isinstance(self.search_space, Mapping) else None)
            if distributions is None:
                raise ValueError("Cannot resume callable Optuna search_space trial without persisted distributions")
            state = str(result.get("trial_state", "failed" if "error" in result else "completed")).lower()
            trial_state = self.optuna.trial.TrialState
            if state == "pruned":
                frozen = self.optuna.trial.create_trial(params=params, distributions=distributions, state=trial_state.PRUNED)
            elif state in {"failed", "cancelled"} or "error" in result:
                frozen = self.optuna.trial.create_trial(params=params, distributions=distributions, state=trial_state.FAIL)
            else:
                values = self._objective_values(result)
                frozen = self.optuna.trial.create_trial(params=params, distributions=distributions,
                    **({"value": values[0]} if len(values) == 1 else {"values": values}))
            self.study.add_trial(frozen)
            self._known_param_keys.add(param_key(params))

    def ask(self, n):
        if n < 1 or self._terminal_count + len(self._trials_by_key) >= self.n_trials:
            return []
        if self.study is None:
            self.restore([])
        proposals, attempts_left = [], max(100, (self.n_trials - self._terminal_count) * 20)
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

    def result_metadata(self, params):
        trial = self._trials_by_key.get(param_key(params))
        return {} if trial is None else {"optuna_distributions": {
            name: self.optuna.distributions.distribution_to_json(distribution)
            for name, distribution in trial.distributions.items()}}

    def intermediate_reporter(self, params):
        trial = self._trials_by_key.get(param_key(params))
        if trial is None:
            raise RuntimeError("No live Optuna trial for intermediate report")
        if len(self.directions) != 1:
            raise ValueError("Optuna pruning is available only for scalar objectives")
        last_step, reports = -1, []
        def report(step, value):
            nonlocal last_step
            if not isinstance(step, (int, np.integer)) or step <= last_step:
                raise ValueError("intermediate report step must be monotonically increasing")
            last_step, value = int(step), float(value)
            trial.report(value, last_step)
            should_prune = bool(trial.should_prune())
            reports.append({"step": last_step, "value": value, "should_prune": should_prune})
            return should_prune
        report.diagnostics = reports
        return report

    def tell(self, results):
        for result in results:
            params = result.get("params")
            if params is None:
                continue
            trial = self._trials_by_key.pop(param_key(params), None)
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
