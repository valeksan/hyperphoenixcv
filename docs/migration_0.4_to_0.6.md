# 0.4 to 0.6 API migration

## Deadline

Version 0.5 accepts legacy arguments with `FutureWarning`. Version 0.6 removes
them. `clear_checkpoint=True` is removed in 0.6; clear storage explicitly
before `fit()`.

## Canonical API

| Feature | Legacy 0.4 | Canonical 0.5/0.6 | Conflict policy |
| --- | --- | --- | --- |
| Grid space | `param_grid` | `search_space`, `strategy="grid"` | Both spaces: error |
| Random search | `random_search=True`, `n_iter=N` | `strategy="random"`, `n_trials=N` | Explicit strategy plus legacy flags: error |
| Optuna | `param_grid=None`, `strategy="optuna"` | `search_space`, `strategy="optuna"`, `n_trials=N` | Legacy flags/model: error |
| SQLite path | `checkpoint_path` | `storage_path` | Use only `storage_path`; it takes precedence during 0.5 |
| Resume | implicit same path | `resume="auto"`, `"must"`, or `"never"` | Invalid mode: error |
| Clear study | `clear_checkpoint=True` | `clear_checkpoint_file()` before `fit()` | Legacy flag removed in 0.6 |
| Surrogate ranking | `use_bayesian_optimization`, `bayesian_optimizer` | Random or Optuna | Deprecated; removed in 0.6 |

`experimental_surrogate_ranking` is compatibility-only. It is not Bayesian
optimization and is removed in 0.6.

## Grid search

```python
# 0.4
HyperPhoenixCV(estimator=model, param_grid={"C": [0.1, 1]},
               checkpoint_path="study.sqlite3").fit(X, y)

# 0.5/0.6
HyperPhoenixCV(estimator=model, search_space={"C": [0.1, 1]},
               strategy="grid", storage_path="study.sqlite3",
               resume="auto").fit(X, y)
```

## Random search

```python
# 0.4
HyperPhoenixCV(estimator=model, param_grid=space, random_search=True,
               n_iter=20, checkpoint_path="study.sqlite3").fit(X, y)

# 0.5/0.6
HyperPhoenixCV(estimator=model, search_space=space, strategy="random",
               n_trials=20, storage_path="study.sqlite3",
               resume="auto").fit(X, y)
```

## Legacy pickle/checkpoint writer

`checkpoint.py` is internal legacy-import/test support, not normal resume API.
For one-time migration of trusted data use
`import_legacy_checkpoint(path, trusted=True)`. Then use SQLite storage and
`TrialHistory` exports.
