"""SQLite source of truth for HyperPhoenixCV studies and trials."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterator
from uuid import uuid4

from ..study_identity import StudyIdentity, canonicalize, mismatch_fields, param_key


SCHEMA_VERSION = 5


class StudyStoreError(ValueError):
    """SQLite study storage cannot satisfy requested operation."""


class StudyMismatchError(StudyStoreError):
    """Storage path already belongs to another study."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    """Canonical JSON value, including non-finite *result* scores."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return {"__tuple__": [_json_value(item) for item in value]}
    if isinstance(value, set):
        return {"__set__": sorted((_json_value(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))}
    return canonicalize(value)


def _json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _restore(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            return {"nan": float("nan"), "inf": float("inf"), "-inf": -float("inf")}[value["__float__"]]
        if set(value) == {"__tuple__"}:
            return tuple(_restore(item) for item in value["__tuple__"])
        if set(value) == {"__set__"}:
            return set(_restore(item) for item in value["__set__"])
        return {key: _restore(item) for key, item in value.items()}
    return value


class SQLiteStudyStore:
    """One-process local SQLite store. No shared-filesystem locking guarantee."""

    def __init__(self, path: str):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._migrate()

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self) -> "SQLiteStudyStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def _migrate(self) -> None:
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise StudyStoreError(
                f"Unsupported SQLite schema version {version}; expected <= {SCHEMA_VERSION}"
            )
        if version == 0:
            self.connection.executescript(
                f"""
                    CREATE TABLE IF NOT EXISTS studies (
                        study_id TEXT PRIMARY KEY,
                        dataset_id TEXT,
                        estimator_digest TEXT NOT NULL,
                        space_digest TEXT NOT NULL,
                        cv_digest TEXT NOT NULL,
                        scorer_digest TEXT NOT NULL,
                        seed INTEGER,
                        config_digest TEXT NOT NULL,
                        state_json TEXT NOT NULL DEFAULT '{{}}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS studies_config_digest_idx
                        ON studies(config_digest);
                    CREATE TABLE IF NOT EXISTS trials (
                        trial_id INTEGER PRIMARY KEY,
                        study_id TEXT NOT NULL REFERENCES studies(study_id) ON DELETE CASCADE,
                        sequence INTEGER NOT NULL,
                        state TEXT NOT NULL CHECK(state IN ('completed', 'failed', 'pruned', 'cancelled')),
                        param_key TEXT NOT NULL,
                        params_json TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        objective_values_json TEXT,
                        diagnostics_json TEXT,
                        exception_type TEXT,
                        exception_message TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(study_id, param_key),
                        UNIQUE(study_id, sequence)
                    );
                    CREATE INDEX IF NOT EXISTS trials_study_sequence_idx
                        ON trials(study_id, sequence);
                    PRAGMA user_version = {SCHEMA_VERSION};
                """
            )
        if version == 1:
            # Version 1 stored canonical JSON in ``param_key``. Convert it to
            # a compact SHA-256 key before new resume logic reads the rows.
            with self._transaction() as conn:
                rows = conn.execute("SELECT trial_id, params_json FROM trials").fetchall()
                for row in rows:
                    params = _restore(json.loads(row["params_json"]))
                    conn.execute(
                        "UPDATE trials SET param_key = ? WHERE trial_id = ?",
                        (param_key(params), row["trial_id"]),
                    )
                conn.execute("PRAGMA user_version = 2")
            version = 2
        if version == 2:
            with self._transaction() as conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(studies)")}
                if "state_json" not in columns:
                    conn.execute("ALTER TABLE studies ADD COLUMN state_json TEXT NOT NULL DEFAULT '{}'")
                conn.execute("PRAGMA user_version = 3")
            version = 3
        if version == 3:
            # SQLite cannot alter a CHECK constraint in place. Preserve all
            # committed rows while adding Optuna's terminal PRUNED state.
            with self._transaction() as conn:
                conn.execute("""
                    CREATE TABLE trials_new (
                        trial_id INTEGER PRIMARY KEY,
                        study_id TEXT NOT NULL REFERENCES studies(study_id) ON DELETE CASCADE,
                        sequence INTEGER NOT NULL,
                        state TEXT NOT NULL CHECK(state IN ('completed', 'failed', 'pruned')),
                        param_key TEXT NOT NULL,
                        params_json TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        exception_type TEXT,
                        exception_message TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(study_id, param_key),
                        UNIQUE(study_id, sequence)
                    )
                """)
                conn.execute("""INSERT INTO trials_new (
                    trial_id, study_id, sequence, state, param_key, params_json,
                    result_json, exception_type, exception_message, created_at, updated_at
                ) SELECT trial_id, study_id, sequence, state, param_key, params_json,
                    result_json, exception_type, exception_message, created_at, updated_at FROM trials""")
                conn.execute("DROP TABLE trials")
                conn.execute("ALTER TABLE trials_new RENAME TO trials")
                conn.execute("CREATE INDEX trials_study_sequence_idx ON trials(study_id, sequence)")
                conn.execute("PRAGMA user_version = 4")
            version = 4
        if version == 4:
            # Vector objectives and pruning/cancellation diagnostics are
            # queryable fields; result_json remains old-reader compatible.
            with self._transaction() as conn:
                conn.execute("""
                    CREATE TABLE trials_new (
                        trial_id INTEGER PRIMARY KEY,
                        study_id TEXT NOT NULL REFERENCES studies(study_id) ON DELETE CASCADE,
                        sequence INTEGER NOT NULL,
                        state TEXT NOT NULL CHECK(state IN ('completed', 'failed', 'pruned', 'cancelled')),
                        param_key TEXT NOT NULL, params_json TEXT NOT NULL, result_json TEXT NOT NULL,
                        objective_values_json TEXT, diagnostics_json TEXT,
                        exception_type TEXT, exception_message TEXT,
                        created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                        UNIQUE(study_id, param_key), UNIQUE(study_id, sequence)
                    )
                """)
                conn.execute("""INSERT INTO trials_new (
                    trial_id, study_id, sequence, state, param_key, params_json, result_json,
                    exception_type, exception_message, created_at, updated_at
                ) SELECT trial_id, study_id, sequence, state, param_key, params_json, result_json,
                    exception_type, exception_message, created_at, updated_at FROM trials""")
                conn.execute("DROP TABLE trials")
                conn.execute("ALTER TABLE trials_new RENAME TO trials")
                conn.execute("CREATE INDEX trials_study_sequence_idx ON trials(study_id, sequence)")
                conn.execute("PRAGMA user_version = 5")

    @staticmethod
    def _identity(row: sqlite3.Row) -> StudyIdentity:
        return StudyIdentity(
            dataset_id=row["dataset_id"], estimator_digest=row["estimator_digest"],
            space_digest=row["space_digest"], cv_digest=row["cv_digest"],
            scorer_digest=row["scorer_digest"], seed=row["seed"],
            config_digest=row["config_digest"],
        )

    def open_study(self, identity: StudyIdentity, resume: str = "auto") -> str:
        if resume not in {"auto", "must", "never"}:
            raise ValueError("resume must be one of: 'auto', 'must', 'never'")
        rows = self.connection.execute(
            "SELECT * FROM studies ORDER BY created_at DESC"
        ).fetchall()
        exact = next((row for row in rows if row["config_digest"] == identity.config_digest), None)
        if resume != "never" and exact is not None:
            return exact["study_id"]
        if resume == "must":
            if not rows:
                raise FileNotFoundError(f"Study required by resume='must' does not exist: {self.path}")
            changed = mismatch_fields(identity, self._identity(rows[0]))
            raise StudyMismatchError(
                f"SQLite store {self.path} belongs to a different study; changed: {', '.join(changed)}. "
                "Use a new storage_path or resume='never'."
            )
        if resume == "auto" and rows:
            changed = mismatch_fields(identity, self._identity(rows[0]))
            raise StudyMismatchError(
                f"SQLite store {self.path} belongs to a different study; changed: {', '.join(changed)}. "
                "Use a new storage_path or resume='never'."
            )
        study_id, now = str(uuid4()), _now()
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO studies (
                    study_id, dataset_id, estimator_digest, space_digest, cv_digest,
                    scorer_digest, seed, config_digest, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (study_id, identity.dataset_id, identity.estimator_digest, identity.space_digest,
                 identity.cv_digest, identity.scorer_digest, identity.seed,
                 identity.config_digest, now, now),
            )
        return study_id

    def completed_param_keys(self, study_id: str) -> set[str]:
        return {
            row[0] for row in self.connection.execute(
                "SELECT param_key FROM trials WHERE study_id = ?", (study_id,)
            )
        }

    def study_state(self, study_id: str) -> dict[str, Any]:
        """Return durable orchestration state for a study."""
        row = self.connection.execute(
            "SELECT state_json FROM studies WHERE study_id = ?", (study_id,)
        ).fetchone()
        if row is None:
            raise StudyStoreError(f"Unknown study: {study_id}")
        return _restore(json.loads(row["state_json"]))

    def update_study_state(self, study_id: str, state: dict[str, Any]) -> None:
        """Atomically replace small orchestration state after a committed trial."""
        now = _now()
        with self._transaction() as conn:
            updated = conn.execute(
                "UPDATE studies SET state_json = ?, updated_at = ? WHERE study_id = ?",
                (_json(state), now, study_id),
            )
            if updated.rowcount != 1:
                raise StudyStoreError(f"Unknown study: {study_id}")

    def results(self, study_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT result_json, state, objective_values_json, diagnostics_json "
            "FROM trials WHERE study_id = ? ORDER BY sequence", (study_id,)
        ).fetchall()
        results = []
        for row in rows:
            result = _restore(json.loads(row["result_json"]))
            if row["objective_values_json"] is not None:
                result.setdefault("objective_values", _restore(json.loads(row["objective_values_json"])))
            if row["diagnostics_json"] is not None:
                result.setdefault("trial_diagnostics", _restore(json.loads(row["diagnostics_json"])))
            results.append(result)
        return results

    def commit_trial(self, study_id: str, params: dict[str, Any], result: dict[str, Any]) -> bool:
        """Atomically store terminal trial. False means same param already committed."""
        params_key, now = param_key(params), _now()
        state = str(result.get("trial_state", "failed" if "error" in result else "completed")).lower()
        if state not in {"completed", "failed", "pruned", "cancelled"}:
            raise ValueError("trial_state must be 'completed', 'failed', 'pruned', or 'cancelled'")
        with self._transaction() as conn:
            exists = conn.execute(
                "SELECT 1 FROM trials WHERE study_id = ? AND param_key = ?", (study_id, params_key)
            ).fetchone()
            if exists:
                return False
            sequence = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM trials WHERE study_id = ?", (study_id,)
            ).fetchone()[0]
            conn.execute(
                """INSERT INTO trials (
                    study_id, sequence, state, param_key, params_json, result_json,
                    objective_values_json, diagnostics_json, exception_type, exception_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (study_id, sequence, state, params_key, _json(params), _json(result),
                 _json(result["objective_values"]) if "objective_values" in result else None,
                 _json(result.get("trial_diagnostics")) if result.get("trial_diagnostics") is not None else None,
                 type(result.get("error")).__name__ if "error" in result else None,
                 str(result["error"]) if "error" in result else None, now, now),
            )
            conn.execute("UPDATE studies SET updated_at = ? WHERE study_id = ?", (now, study_id))
        return True

    def clear(self) -> None:
        self.close()
        for suffix in ("", "-wal", "-shm"):
            target = Path(f"{self.path}{suffix}")
            if target.exists():
                target.unlink()
