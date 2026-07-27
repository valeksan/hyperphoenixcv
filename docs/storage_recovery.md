# Local SQLite storage: backup and recovery

`storage_path` is one local SQLite file, one active HyperPhoenixCV coordinator.
Do not use NFS, SMB/CIFS, cloud-sync folders, or shared volumes. Linux warns
for recognized network mounts; unrecognized mounts remain unsupported.

Stop active fit before `restore`, `prune-empty`, or manual file handling.
Read-only `check` and `backup` may run while fit is active.

```bash
hyperphoenixcv-storage check study.sqlite3
hyperphoenixcv-storage backup study.sqlite3 backups/study.sqlite3
hyperphoenixcv-storage restore backups/study.sqlite3 study.sqlite3
hyperphoenixcv-storage prune-empty study.sqlite3
```

`check` runs `PRAGMA integrity_check`, foreign-key validation, JSON decoding,
trial sequence, and parameter-key invariants. Exit `0` = clean, `1` = bad data,
`2` = storage operation failure.

| Error | Recovery |
|---|---|
| `StorageSchemaError` | Preserve copy; restore backup or migrate clean copy. |
| `StorageCorruptionError` | Stop fit; run `check`; restore verified backup. |
| `StorageUnavailableError` | Verify path/parent mount; retry same file. |
| `StoragePermissionError` | Fix ownership/mode; resume. |
| `StorageDiskFullError` | Free space; resume same study. |
| `StorageLockedError` | Stop other writer; one local coordinator only. |

Committed terminal trials remain source of truth. Uncommitted evaluated trial
may run again after recovery.

WAL/SHM sidecars are live files: never copy/delete/move them while any fit or
inspection connection is open. `backup` replaces manual copies.
`clear_storage()` removes database plus sidecars irreversibly. `prune-empty`
removes only studies without terminal trials.

Before upgrade: stop fit, run `check`, make `backup`, then open upgraded
library against a copy first. Migrations are forward-only and preserve terminal
trials transactionally. If migration errors, retain original and restore backup.
