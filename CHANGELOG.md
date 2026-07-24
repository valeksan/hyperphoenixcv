# Changelog

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
