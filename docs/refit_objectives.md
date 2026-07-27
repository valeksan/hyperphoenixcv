# Refit: scalar and multi-objective searches

For scalar scoring, `refit=True` selects best completed trial, clones source
estimator, then fits clone on all supplied data.

For multi-metric sklearn scoring, select metric explicitly:

```python
search = HyperPhoenixCV(
    estimator=model,
    search_space={"C": [0.1, 1.0]},
    strategy="grid",
    scoring={"accuracy": "accuracy", "f1": "f1"},
    refit="f1",
)
```

Use `metric_directions={"loss": "minimize"}` for custom scalar metrics whose
smaller value is better. Unspecified sklearn scores rank as maximize.

Optuna multi-objective search has no implicit single best trial. Supply all
directions, inspect `pareto_front_`, then choose intentionally:

```python
search = HyperPhoenixCV(
    estimator=model,
    strategy="optuna",
    search_space=space,
    n_trials=40,
    scoring=["accuracy", "neg_log_loss"],
    optuna_directions={"accuracy": "maximize", "neg_log_loss": "maximize"},
    refit=False,
)
```

`refit=True` with multi-objective Optuna raises `ValueError`. Use `refit=False`,
a metric name, or selector callable. Callable refit receives complete
`cv_results_`, including failed/pruned rows; therefore it requires an
unbounded or sufficiently large `max_cv_results`.
