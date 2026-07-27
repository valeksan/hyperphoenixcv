# API reference

This reference follows public signatures/docstrings in `src/hyperphoenixcv`.
Run `help(HyperPhoenixCV)` for runtime introspection.

## `HyperPhoenixCV`

```python
HyperPhoenixCV(
    estimator, search_space, strategy="grid", n_trials=None,
    storage_path="hyperphoenix_checkpoint.sqlite3", scoring="f1", cv=5,
    n_jobs=1, results_csv="hyperphoenix_results.csv", verbose=True,
    random_state=None, refit=True, pre_dispatch="2*n_jobs", error_score="raise",
    early_stopping_patience=None, dataset_id=None, resume="auto", scorer_id=None,
    cv_id=None, parallelism="trials", inner_max_num_threads=None,
    search_space_id=None, optuna_warmup_trials=10, optuna_directions=None,
    metric_directions=None, intermediate_evaluator=None, trial_timeout=None,
    cancel_callback=None, memmap_max_nbytes="1M", memmap_temp_folder=None,
    joblib_batch_size="auto", callbacks=None, max_cv_results=10_000,
    compute="cpu", gpu_devices=(0,), gpu_slots_per_device=1,
)
```

`estimator` is sklearn-compatible. `search_space` is ParameterGrid syntax for
grid/random, Optuna distributions or callable for Optuna. `strategy` is
`"grid"`, `"random"`, or `"optuna"`; random/Optuna require positive
`n_trials`.

`storage_path`, `dataset_id`, `resume`, `scorer_id`, `cv_id`, and
`search_space_id` define safe resume; see [resume and storage](resume_and_storage.md).
`parallelism` and joblib settings control resources; see
[parallelism](parallelism.md). `metric_directions`, `optuna_directions`, and
`refit` control selection; see [refit objectives](refit_objectives.md).
`intermediate_evaluator` enables cooperative Optuna pruning only; see
[pruning](pruning.md). `callbacks` receives runtime events; see
[audit and events](audit_and_events.md).
`compute="gpu"` is G1 single-NVIDIA-device validation/diagnostics. It requires
one `gpu_devices` entry, `gpu_slots_per_device=1`, and `n_jobs=1`; device
preflight occurs before SQLite mutation. Estimator GPU configuration remains
caller-owned.

### Methods

| Method | Purpose |
| --- | --- |
| `fit(X, y, groups=None, **fit_params)` | Run/resume study; return fitted self. |
| `get_top_results(n=10)` | Return ranked top-N DataFrame. |
| `load_results_from_checkpoint(n=10)` | Read top-N from matching SQLite study. |
| `load_trial_history()` | Open read-only `TrialHistory` for matching study. |
| `clear_storage()` | Irreversibly delete SQLite store and sidecars. |
| `import_legacy_checkpoint(path, trusted=True)` | One-time explicit trusted pickle import. |

After successful fit: `best_params_`, `best_score_`, `best_index_`,
`best_estimator_` (when refit selected), `cv_results_`, `trial_history_`, and
`pareto_front_` (multi-objective Optuna). `cv_results_` is empty when history
exceeds `max_cv_results`; use `trial_history_` then.

## `TrialHistory`

| Method | Purpose |
| --- | --- |
| `count(states=None)` | Count terminal audit records. |
| `page(offset=0, limit=100, states=None)` | Immutable paginated records. |
| `iter_records(page_size=1000, states=None)` | Stream records from SQLite. |
| `export_json(path)` | Atomic lossless tagged JSON export. |
| `export_csv(path)` | Atomic flat convenience export. |
| `export_parquet(path)` | Atomic Parquet export; needs `hyperphoenixcv[parquet]`. |

Allowed terminal states: `completed`, `failed`, `pruned`, `cancelled`.

## Runtime event types

`StudyStarted`, `StudyResumed`, `TrialStarted`, `TrialCompleted`,
`TrialFailed`, `TrialPruned`, `TrialCancelled`, `StudyStopped`,
`StudyCompleted`, `ExportFailed`, and `RefitFailed` are public event classes.
`GPUDeviceAssigned`, `GPUResourceFailure`, and `GPUOutOfMemory` are GPU
diagnostic event classes.
Each has `study_id`; trial events also include trial index and terminal context.
