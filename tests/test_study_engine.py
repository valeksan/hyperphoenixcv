from __future__ import annotations

from hyperphoenixcv.study_engine import StudyEngine, StudySpec


class FakeStore:
    def __init__(self):
        self.state = {"scheduler": {}}
        self.committed = []
        self.events = []

    def study_state(self, study_id):
        return self.state

    def update_study_state(self, study_id, state):
        self.state = state

    def commit_trial(self, study_id, params, result):
        self.events.append(("commit", params))
        self.committed.append(result)
        return True


class FakeStrategy:
    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.told = []
        self.restored = None
        self.events = []

    def restore(self, results):
        self.restored = list(results)

    def ask(self, n):
        proposals, self.proposals = self.proposals[:n], self.proposals[n:]
        return proposals

    def tell(self, results):
        self.events.append(("tell", results[0]["params"]))
        self.told.extend(results)


class FakeResults:
    def __init__(self):
        self.results = []

    def add_result(self, result):
        self.results.append(result)


def spec(**overrides):
    values = dict(
        scoring=("score",), strategy=None, random_search=False,
        adaptive_search=False, early_stopping_patience=None, batch_size=2,
        total_candidates=2, optuna_directions=None,
    )
    values.update(overrides)
    return StudySpec(**values)


def test_engine_runs_without_sklearn_facade_and_commits_before_tell():
    store = FakeStore()
    strategy = FakeStrategy([{"x": 1}, {"x": 2}])
    results = FakeResults()

    engine = StudyEngine(
        spec=spec(), store=store, study_id="study", strategy=strategy,
        result_manager=results,
        evaluate_batch=lambda proposals: [
            {"params": params, "mean_test_score": float(params["x"])}
            for params in proposals
        ],
    )
    run = engine.run([])

    assert strategy.restored == []
    assert [result["params"] for result in store.committed] == [{"x": 1}, {"x": 2}]
    assert strategy.told == store.committed == results.results
    assert store.events == [("commit", {"x": 1}), ("commit", {"x": 2})]
    assert strategy.events == [("tell", {"x": 1}), ("tell", {"x": 2})]
    assert run.attempts == 2


def test_engine_cancellation_persists_reason_without_evaluation():
    store = FakeStore()
    strategy = FakeStrategy([{"x": 1}])
    results = FakeResults()
    engine = StudyEngine(
        spec=spec(cancel_callback=lambda: "stop-now"), store=store,
        study_id="study", strategy=strategy, result_manager=results,
        evaluate_batch=lambda proposals: (_ for _ in ()).throw(AssertionError("must not evaluate")),
    )

    run = engine.run([])

    assert run.stopped_reason == "stop-now"
    assert store.committed == []
    assert store.state["scheduler"]["cancellation_reason"] == "stop-now"
