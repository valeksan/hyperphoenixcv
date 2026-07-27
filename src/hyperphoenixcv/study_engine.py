"""Storage-backed ask → evaluate → commit → tell coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .storage.protocols import StudyStore
from .events import (
    EventPublisher, StudyCompleted, StudyResumed, StudyStarted, StudyStopped,
    TrialCancelled, TrialCompleted, TrialFailed, TrialPruned, TrialStarted,
)


@dataclass(frozen=True)
class StudySpec:
    """Immutable execution settings, separated from sklearn facade state."""

    scoring: tuple[str, ...]
    strategy: str | None
    random_search: bool
    adaptive_search: bool
    early_stopping_patience: int | None
    batch_size: int
    total_candidates: int
    optuna_directions: dict[str, str] | None
    cancel_callback: Callable[[], str | bool | None] | None = None
    timeout_enabled: bool = False


@dataclass(frozen=True)
class StudyRun:
    """Terminal coordinator state needed by facade projection/refit."""

    stopped_reason: str | None
    attempts: int


def _early_stop_from_results(results: list[dict[str, Any]], metric: str) -> tuple[float, int]:
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


class StudyEngine:
    """Drive terminal trial lifecycle; commit always precedes sampler tell."""

    def __init__(
        self,
        *,
        spec: StudySpec,
        store: StudyStore,
        study_id: str,
        strategy: Any,
        result_manager: Any,
        evaluate_batch: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
        on_trial: Callable[[int, dict[str, Any], dict[str, Any]], None] | None = None,
        event_publisher: EventPublisher | None = None,
    ) -> None:
        self.spec = spec
        self.store = store
        self.study_id = study_id
        self.strategy = strategy
        self.result_manager = result_manager
        self.evaluate_batch = evaluate_batch
        self.on_trial = on_trial
        self.event_publisher = event_publisher or EventPublisher()

    def _update_state(self, patch: dict[str, Any]) -> None:
        state = self.store.study_state(self.study_id)
        state.update(patch)
        self.store.update_study_state(self.study_id, state)

    def run(self, checkpoint_results: list[dict[str, Any]]) -> StudyRun:
        """Run until exhausted, cancelled, or early-stopping condition fires."""
        self.strategy.restore(checkpoint_results)
        self.event_publisher.emit(StudyStarted(self.study_id, self.spec.total_candidates))
        if checkpoint_results:
            self.event_publisher.emit(StudyResumed(self.study_id, len(checkpoint_results)))
        early_enabled = self.spec.early_stopping_patience is not None and self.spec.adaptive_search
        primary_metric = self.spec.scoring[0]
        best_score, no_improvement_count = _early_stop_from_results(checkpoint_results, primary_metric)
        if early_enabled or self.spec.timeout_enabled or self.spec.cancel_callback is not None:
            saved = self.store.study_state(self.study_id).get("early_stopping", {})
            if saved.get("metric") == primary_metric and saved.get("processed_trial_count") == len(checkpoint_results):
                best_score = saved["best_score"]
                no_improvement_count = saved["no_improvement_count"]

        proposals_available = not (
            early_enabled and no_improvement_count >= self.spec.early_stopping_patience
        )
        batch_size = 1 if early_enabled else self.spec.batch_size
        attempts = 0
        stopped_reason = None
        while proposals_available:
            if self.spec.cancel_callback is not None:
                cancellation = self.spec.cancel_callback()
                if cancellation:
                    stopped_reason = cancellation if isinstance(cancellation, str) else "cancel_callback"
                    scheduler = self.store.study_state(self.study_id).get("scheduler", {})
                    self._update_state({"scheduler": {**scheduler, "cancellation_reason": stopped_reason}})
                    break
            proposals = self.strategy.ask(batch_size)
            if not proposals:
                break
            for offset, params in enumerate(proposals, 1):
                self.event_publisher.emit(TrialStarted(self.study_id, attempts + offset, params))
            for params, result in zip(proposals, self.evaluate_batch(proposals)):
                attempts += 1
                metadata_fn = getattr(self.strategy, "result_metadata", None)
                if metadata_fn is not None:
                    result.update(metadata_fn(params))
                if (
                    self.spec.strategy == "optuna" and "error" not in result
                    and result.get("trial_state") != "pruned" and "objective_values" not in result
                ):
                    directions = self.spec.optuna_directions or {primary_metric: "maximize"}
                    result["objective_values"] = {
                        name: result[f"mean_test_{name}"] for name in directions
                    }
                committed = self.store.commit_trial(self.study_id, params, result)
                if committed:
                    self.result_manager.add_result(result)
                    self.strategy.tell([result])
                    scheduler = self.store.study_state(self.study_id).get("scheduler", {})
                    self._update_state({"scheduler": {
                        **scheduler,
                        "attempts": scheduler.get("attempts", 0) + 1,
                        "cancellation_reason": result.get("cancellation_reason"),
                    }})
                if self.on_trial is not None:
                    self.on_trial(attempts, params, result)
                state = result.get("trial_state")
                if state == "pruned":
                    self.event_publisher.emit(TrialPruned(self.study_id, attempts, params, result))
                elif state == "cancelled" or result.get("cancellation_reason"):
                    self.event_publisher.emit(TrialCancelled(
                        self.study_id, attempts, params, result.get("cancellation_reason")
                    ))
                elif "error" in result:
                    self.event_publisher.emit(TrialFailed(
                        self.study_id, attempts, params, result.get("error_type"), result.get("error")
                    ))
                else:
                    self.event_publisher.emit(TrialCompleted(self.study_id, attempts, params, result))
                if early_enabled:
                    if "error" not in result:
                        score = result.get(f"mean_test_{primary_metric}", -float("inf"))
                        if score > best_score + 1e-9:
                            best_score, no_improvement_count = score, 0
                        else:
                            no_improvement_count += 1
                    else:
                        no_improvement_count += 1
                    if no_improvement_count >= self.spec.early_stopping_patience:
                        stopped_reason = "patience_exhausted"
                        proposals_available = False
                    self._update_state({"early_stopping": {
                        "metric": primary_metric,
                        "best_score": best_score,
                        "no_improvement_count": no_improvement_count,
                        "processed_trial_count": len(self.result_manager.results),
                        "stop_reason": stopped_reason,
                    }})
                    if not proposals_available:
                        break
        if stopped_reason is not None and stopped_reason != "cancel_callback":
            self.event_publisher.emit(StudyStopped(self.study_id, stopped_reason))
        elif stopped_reason is None:
            self.event_publisher.emit(StudyCompleted(self.study_id, len(self.result_manager.results)))
        return StudyRun(stopped_reason=stopped_reason, attempts=attempts)
