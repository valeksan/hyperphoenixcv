"""Runtime-only study events and callback dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeAlias


@dataclass(frozen=True)
class StudyEvent:
    """Base runtime event. Events are not persisted in SQLite."""

    study_id: str


@dataclass(frozen=True)
class StudyStarted(StudyEvent):
    total_candidates: int


@dataclass(frozen=True)
class StudyResumed(StudyEvent):
    completed_trials: int


@dataclass(frozen=True)
class TrialStarted(StudyEvent):
    trial_index: int
    params: dict[str, Any]


@dataclass(frozen=True)
class TrialCompleted(StudyEvent):
    trial_index: int
    params: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class TrialFailed(StudyEvent):
    trial_index: int
    params: dict[str, Any]
    error_type: str | None
    error: str | None


@dataclass(frozen=True)
class TrialPruned(StudyEvent):
    trial_index: int
    params: dict[str, Any]
    result: dict[str, Any]


@dataclass(frozen=True)
class TrialCancelled(StudyEvent):
    trial_index: int
    params: dict[str, Any]
    reason: str | None


@dataclass(frozen=True)
class StudyStopped(StudyEvent):
    reason: str


@dataclass(frozen=True)
class StudyCompleted(StudyEvent):
    completed_trials: int


@dataclass(frozen=True)
class ExportFailed(StudyEvent):
    path: str
    error_type: str
    error: str


@dataclass(frozen=True)
class RefitFailed(StudyEvent):
    error_type: str
    error: str


@dataclass(frozen=True)
class GPUDeviceAssigned(StudyEvent):
    """G1 runtime assignment; estimator configuration remains caller-owned."""

    device_index: int
    device_uuid: str
    device_name: str


@dataclass(frozen=True)
class GPUResourceFailure(StudyEvent):
    error_type: str
    error: str


@dataclass(frozen=True)
class GPUOutOfMemory(StudyEvent):
    trial_index: int
    error: str


AnyStudyEvent: TypeAlias = (
    StudyStarted | StudyResumed | TrialStarted | TrialCompleted | TrialFailed
    | TrialPruned | TrialCancelled | StudyStopped | StudyCompleted
    | ExportFailed | RefitFailed | GPUDeviceAssigned | GPUResourceFailure | GPUOutOfMemory
)
StudyCallback: TypeAlias = Callable[[AnyStudyEvent], None]


class EventPublisher:
    """Synchronous dispatcher. Callback exceptions fail current fit by design."""

    def __init__(self, callbacks: tuple[StudyCallback, ...] = ()) -> None:
        self.callbacks = callbacks

    def emit(self, event: AnyStudyEvent) -> None:
        for callback in self.callbacks:
            callback(event)
