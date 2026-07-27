# Honest Optuna pruning

Ordinary sklearn CV does not expose trustworthy intermediate progress. It
always runs a complete trial. HyperPhoenixCV never labels a heuristic or
timeout as Optuna pruning.

Pruning requires `strategy="optuna"` and an `intermediate_evaluator`. Evaluator
owns training loop, calls `report(step, value)` with strictly increasing integer
steps, then stops its own work when `report` returns `True`:

```python
def evaluator(estimator, X, y, params, report, groups, fit_params):
    model = clone(estimator).set_params(**params)
    for epoch in range(50):
        model.partial_fit(X, y, classes=classes)
        score = validation_score(model)
        if report(epoch, score):
            return {"params": params, "trial_state": "pruned"}
    return {"params": params, "mean_test_accuracy": score}
```

Reported values must be real intermediate validation measurements from current
trial. Do not report training loss as validation score, invent steps, or claim
ordinary `cross_validate` can be interrupted mid-fit. Pruned state, reports,
and diagnostics persist in SQLite. Resume replays committed history; proposal
order also depends on Optuna version, seed, and batch/parallelism settings.
