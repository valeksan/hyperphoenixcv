# Resume identity and local storage

SQLite terminal trials are source of truth. One `storage_path` supports one
active local coordinator. Do not put it on NFS, SMB/CIFS, cloud-sync folders,
or any shared filesystem.

## Stable identity

Resume opens only a matching `StudyIdentity`. It includes `dataset_id`,
estimator, search space, CV, scorer, random seed, and strategy settings. Give
every real dataset a stable, caller-owned ID:

```python
search = HyperPhoenixCV(
    estimator=model,
    search_space={"C": [0.1, 1.0]},
    strategy="grid",
    storage_path="studies/customer-churn.sqlite3",
    dataset_id="customer-churn-2026-07-27-clean-v3",
)
search.fit(X, y)
```

`dataset_id=None` only proves shapes and dtypes poorly; it emits a warning.
Use an immutable dataset version, content digest, or managed-data revision.
Do not reuse an ID after changing rows, labels, feature processing, or split
policy.

`resume="auto"` resumes an exact match or creates a study. `"must"` requires
an existing exact match. `"never"` creates a new study. An identity mismatch
fails instead of mixing trial history.

Callable scorer, CV splitter, or Optuna search space needs respectively
`scorer_id`, `cv_id`, or `search_space_id` for safe resume.

## Backup and recovery

Before upgrades or filesystem maintenance: stop fit, check store, make backup,
then test upgrade against copy.

```bash
hyperphoenixcv-storage check study.sqlite3
hyperphoenixcv-storage backup study.sqlite3 backups/study.sqlite3
hyperphoenixcv-storage restore backups/study.sqlite3 study.sqlite3
```

`check` validates SQLite integrity plus HyperPhoenixCV JSON, sequence, and
parameter-key invariants. Use CLI backup; never copy SQLite file or WAL/SHM
sidecars while a connection is open. Committed terminal trials survive a crash;
uncommitted evaluated trial may execute again after resume.

See [storage recovery](storage_recovery.md) for error taxonomy, WAL/SHM policy,
and irreversible cleanup details.
