import os

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import (
    HyperPhoenixCV, StudyCompleted, StudyResumed, StudyStarted, StudyStopped,
    TrialCompleted, TrialFailed, TrialStarted,
)
from hyperphoenixcv.study_engine import StudyEngine
from hyperphoenixcv.events import EventPublisher
from tests.test_study_engine import FakeResults, FakeStore, FakeStrategy, spec


def test_engine_emits_typed_events_in_terminal_order():
    events = []
    engine = StudyEngine(
        spec=spec(), store=FakeStore(), study_id="study",
        strategy=FakeStrategy([{"x": 1}, {"x": 2}]), result_manager=FakeResults(),
        evaluate_batch=lambda proposals: [
            {"params": proposals[0], "mean_test_score": 1.0},
            {"params": proposals[1], "error": "bad", "error_type": "ValueError"},
        ],
        event_publisher=EventPublisher((events.append,)),
    )

    engine.run([])

    assert [type(event) for event in events] == [
        StudyStarted, TrialStarted, TrialStarted, TrialCompleted, TrialFailed, StudyCompleted,
    ]


def test_resume_emits_resume_event_before_new_trials():
    events = []
    engine = StudyEngine(
        spec=spec(total_candidates=1), store=FakeStore(), study_id="study",
        strategy=FakeStrategy([]), result_manager=FakeResults(), evaluate_batch=lambda _: [],
        event_publisher=EventPublisher((events.append,)),
    )
    engine.run([{"params": {"x": 0}, "mean_test_score": 0.0}])
    assert [type(event) for event in events] == [StudyStarted, StudyResumed, StudyCompleted]


def test_callback_failure_is_fail_fast(tmp_path):
    X, y = make_classification(n_samples=20, n_features=4, random_state=0)
    search = HyperPhoenixCV(
        LogisticRegression(max_iter=100), {"C": [1.0]}, scoring="accuracy", cv=2,
        checkpoint_path=str(tmp_path / "study.sqlite3"), results_csv=str(tmp_path / "results.csv"),
        dataset_id="events-fail-fast", refit=False,
        callbacks=[lambda event: (_ for _ in ()).throw(RuntimeError("callback failed"))],
    )
    with pytest.raises(RuntimeError, match="callback failed"):
        search.fit(X, y)


def test_callbacks_run_in_coordinator_with_trial_parallelism(tmp_path):
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    pids = []
    search = HyperPhoenixCV(
        LogisticRegression(max_iter=100), {"C": [0.1, 1.0]}, scoring="accuracy", cv=2,
        n_jobs=2, checkpoint_path=str(tmp_path / "study.sqlite3"),
        results_csv=str(tmp_path / "results.csv"), dataset_id="events-parallel", refit=False,
        callbacks=[lambda event: pids.append(os.getpid()) if isinstance(event, TrialCompleted) else None],
    )
    search.fit(X, y)
    assert pids == [os.getpid(), os.getpid()]


def test_cancel_callback_emits_stopped(tmp_path):
    X, y = make_classification(n_samples=20, n_features=4, random_state=0)
    events = []
    search = HyperPhoenixCV(
        LogisticRegression(max_iter=100), {"C": [1.0]}, scoring="accuracy", cv=2,
        checkpoint_path=str(tmp_path / "study.sqlite3"), results_csv=str(tmp_path / "results.csv"),
        dataset_id="events-cancel", refit=False, cancel_callback=lambda: "cancelled",
        callbacks=[events.append],
    )
    search.fit(X, y)
    assert [type(event) for event in events] == [StudyStarted, StudyStopped]
