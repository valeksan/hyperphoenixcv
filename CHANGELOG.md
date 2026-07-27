# Changelog

## 0.4.3 — 2026-07-27

### Added

- Runtime study events and public callbacks for lifecycle, trial, export, and
  refit outcomes; callback failures are deterministic and fail fast.
- Durable, read-only `TrialHistory` audit access with pagination, streaming,
  lossless JSON export, flat CSV export, and optional Parquet export.
- Native Optuna search support, including scalar cooperative pruning and
  multi-objective directions/Pareto projection.
- Trial/fold scheduling controls, runtime serialization coverage, and expanded
  CI compatibility, wheel-smoke, and release checks.

### Changed

- SQLite remains the terminal-trial source of truth; checkpoint restore and
  audit access now stream records instead of retaining result dictionaries.
- `cv_results_` materialization is bounded by `max_cv_results=10_000` by
  default. Large studies retain paginated audit history, best-refit, and top-N
  access without unbounded sklearn projection memory.
- Metric directions support both maximize and minimize ranking outside Optuna.
- Documentation now covers local-only SQLite policy, events, audit exports,
  Optuna workflows, and TestPyPI installation.

### Fixed

- Audit JSON export preserves terminal failed/pruned/cancelled states,
  diagnostics, objectives, exceptions, tuples, sets, and non-finite floats.
- Callable refit validates indexes against complete materialized
  `cv_results_`, including failed rows.

## 0.4.2 — 2026-07-27

### Changed

- Extracted the local study engine, scheduler, storage contract, and concrete
  search strategies while preserving public strategy imports and SQLite schema.
- Declared the dependency/platform baseline in CI and project metadata.

### Fixed

- Raised the minimum Joblib version to 1.3 because the local scheduler uses
  `parallel_config`.

## 0.4.1 — 2026-07-24

### Deprecated

- `clear_checkpoint=True` is deprecated and will be removed in 0.6. Call
  `clear_checkpoint_file()` explicitly before `fit()` instead.

### Platform policy

- P0 supports SQLite stores on a local filesystem only. Windows locking and
  durability behavior is not CI-validated yet; Windows is therefore not a
  supported P0 target. Do not place a store on a network or synced filesystem.

## 0.4.0 — 2026-07-24

### Changed

- SQLite is the source of truth for normal search persistence. Trials commit
  transactionally and CSV is an export projection.
- Default `checkpoint_path` is now `hyperphoenix_checkpoint.sqlite3`.
- Supported Python versions are 3.10–3.12.

### Breaking changes

- Normal `fit()` and resume no longer load pickle/joblib checkpoints.
- `ResultManager.load_from_checkpoint()` and legacy `CheckpointManager` load
  operations reject implicit pickle loading.
- Use `clear_checkpoint_file()` to delete a SQLite study store; the former
  `clear_checkpoint()` instance method is unavailable because
  `clear_checkpoint` is a constructor parameter.

### Migration and security

- Migrate old trusted `List[dict]` pickle data with
  `HyperPhoenixCV.import_legacy_checkpoint(path, trusted=True)`.
- Pickle/joblib deserialization can execute arbitrary code. Import only files
  whose source and contents are trusted.
