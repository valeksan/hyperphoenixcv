"""Read-only, paginated audit projection over durable SQLite trials."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping

import pandas as pd

from .storage.sqlite_store import SQLiteStudyStore, _json_value

TERMINAL_STATES = frozenset({"completed", "failed", "pruned", "cancelled"})


def _atomic_write(path: str | Path, writer) -> None:
    """Write temp file, fsync it, then atomically replace destination."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class TrialHistory:
    """Public read-only audit API. Results stay SQLite-backed until requested."""

    def __init__(self, storage_path: str, study_id: str):
        self.storage_path = str(storage_path)
        self.study_id = study_id

    @staticmethod
    def _states(states: set[str] | None) -> set[str] | None:
        if states is None:
            return None
        unknown = set(states) - TERMINAL_STATES
        if unknown:
            raise ValueError(f"unknown terminal state(s): {', '.join(sorted(unknown))}")
        return set(states)

    def count(self, *, states: set[str] | None = None) -> int:
        with SQLiteStudyStore(self.storage_path) as store:
            return store.trial_count(self.study_id, self._states(states))

    def page(
        self, *, offset: int = 0, limit: int = 100,
        states: set[str] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return immutable audit records; use offset/limit for large studies."""
        with SQLiteStudyStore(self.storage_path) as store:
            records = list(store.iter_trials(
                self.study_id, offset=offset, limit=limit, states=self._states(states),
            ))
        return tuple(_freeze(record) for record in records)

    def iter_records(
        self, *, page_size: int = 1000, states: set[str] | None = None,
    ) -> Iterator[Mapping[str, Any]]:
        if page_size <= 0:
            raise ValueError("page_size must be > 0")
        offset = 0
        while True:
            page = self.page(offset=offset, limit=page_size, states=states)
            yield from page
            if len(page) < page_size:
                return
            offset += len(page)

    def export_json(self, path: str | Path) -> None:
        """Lossless JSON audit export; tagged values preserve NaN, tuples, sets."""
        def write(handle) -> None:
            handle.write(b'{"format":"hyperphoenixcv.audit.v1","trials":[')
            first = True
            for record in self.iter_records():
                if not first:
                    handle.write(b",")
                first = False
                handle.write(json.dumps(_json_value(dict(record)), sort_keys=True,
                                        separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
            handle.write(b"]}")
        _atomic_write(path, write)

    def export_csv(self, path: str | Path) -> None:
        """Flat convenience export. Nested params/diagnostics are JSON strings."""
        rows = []
        for record in self.iter_records():
            row = {key: value for key, value in record.items() if key not in {"params", "result", "diagnostics", "objective_values"}}
            row["params"] = json.dumps(_json_value(record["params"]), sort_keys=True)
            row["result"] = json.dumps(_json_value(record["result"]), sort_keys=True)
            row["objectives"] = json.dumps(_json_value(record["objective_values"]), sort_keys=True)
            row["diagnostics"] = json.dumps(_json_value(record["diagnostics"]), sort_keys=True)
            rows.append(row)
        def write(handle) -> None:
            handle.write(pd.DataFrame(rows).to_csv(index=False).encode("utf-8"))
        _atomic_write(path, write)

    def export_parquet(self, path: str | Path) -> None:
        """Optional export; requires pandas Parquet engine (pyarrow/fastparquet)."""
        rows = [dict(record) for record in self.iter_records()]
        def write(handle) -> None:
            try:
                pd.DataFrame(rows).to_parquet(handle, index=False)
            except ImportError as exc:
                raise ImportError("Parquet export requires hyperphoenixcv[parquet] (pyarrow).") from exc
        _atomic_write(path, write)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        from types import MappingProxyType
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value
