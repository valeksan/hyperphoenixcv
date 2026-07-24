"""Stable study identity and versioned checkpoint envelope."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping
from uuid import uuid4


SCHEMA_VERSION = 1
LIBRARY_VERSION = "0.4.1"


class UnsupportedIdentityValueError(ValueError):
    """A value has no safe, stable identity representation."""


class CheckpointSchemaError(ValueError):
    """Checkpoint is malformed or uses an unsupported schema."""


class CheckpointMismatchError(ValueError):
    """Checkpoint belongs to a different experiment."""


def _class_path(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def canonicalize(value: Any) -> Any:
    """Convert supported configuration values to deterministic JSON data."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, type):
        return {"__type__": _class_path(value)}
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise UnsupportedIdentityValueError("NaN and infinity are not stable config values")
        return value
    if isinstance(value, Mapping):
        return {str(key): canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    if isinstance(value, tuple):
        return {"__tuple__": [canonicalize(item) for item in value]}
    if isinstance(value, set):
        items = [canonicalize(item) for item in value]
        return {"__set__": sorted(items, key=canonical_json)}
    if hasattr(value, "item") and callable(value.item):
        try:
            return canonicalize(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "get_params") and callable(value.get_params):
        return {
            "__estimator__": _class_path(value),
            "params": canonicalize(value.get_params(deep=True)),
        }
    if callable(value):
        raise UnsupportedIdentityValueError(
            f"Callable {_class_path(value)} needs an explicit stable ID"
        )
    raise UnsupportedIdentityValueError(
        f"Unsupported config value {_class_path(value)}; supply a stable ID"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(canonicalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def param_key(params: Mapping[str, Any]) -> str:
    """Stable, compact identifier for one canonical parameter assignment."""
    return digest(params)


def _scorer_config(scoring: Any, stable_id: str | None) -> Any:
    values = scoring if isinstance(scoring, list) else [scoring]
    if any(callable(value) for value in values):
        if stable_id is None:
            raise UnsupportedIdentityValueError(
                "Callable scoring requires scorer_id for checkpoint resume"
            )
        return {"stable_id": stable_id}
    return {"scoring": canonicalize(values)}


def _cv_config(cv: Any, stable_id: str | None) -> Any:
    if isinstance(cv, int):
        return {"n_splits": cv}
    if stable_id is not None:
        return {"stable_id": stable_id}
    if hasattr(cv, "get_params") and callable(cv.get_params):
        return canonicalize(cv)
    raise UnsupportedIdentityValueError("CV splitter requires cv_id for checkpoint resume")


@dataclass(frozen=True)
class StudyIdentity:
    dataset_id: str | None
    estimator_digest: str
    space_digest: str
    cv_digest: str
    scorer_digest: str
    seed: int | None
    config_digest: str

    @classmethod
    def create(
        cls,
        *,
        estimator: Any,
        param_grid: Mapping[str, Any],
        scoring: Any,
        cv: Any,
        random_state: int | None,
        dataset_id: str | None,
        scorer_id: str | None,
        cv_id: str | None,
        strategy_config: Mapping[str, Any] | None = None,
    ) -> "StudyIdentity":
        values = {
            "dataset_id": dataset_id,
            "estimator_digest": digest(canonicalize(estimator)),
            "space_digest": digest(canonicalize(param_grid)),
            "cv_digest": digest(_cv_config(cv, cv_id)),
            "scorer_digest": digest(_scorer_config(scoring, scorer_id)),
            "seed": random_state,
        }
        config = {**values, "strategy": canonicalize(strategy_config or {})}
        return cls(config_digest=digest(config), **values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "estimator_digest": self.estimator_digest,
            "space_digest": self.space_digest,
            "cv_digest": self.cv_digest,
            "scorer_digest": self.scorer_digest,
            "seed": self.seed,
            "config_digest": self.config_digest,
        }


@dataclass
class CheckpointEnvelope:
    study_id: str
    identity: StudyIdentity
    results: list[dict[str, Any]]
    created_at: str
    updated_at: str
    schema_version: int = SCHEMA_VERSION
    library_version: str = LIBRARY_VERSION

    @classmethod
    def new(cls, identity: StudyIdentity, results: list[dict[str, Any]]) -> "CheckpointEnvelope":
        now = datetime.now(timezone.utc).isoformat()
        return cls(str(uuid4()), identity, results, now, now)

    def with_results(self, results: list[dict[str, Any]]) -> "CheckpointEnvelope":
        return CheckpointEnvelope(
            study_id=self.study_id,
            identity=self.identity,
            results=results,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "library_version": self.library_version,
            "study_id": self.study_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            **self.identity.as_dict(),
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CheckpointEnvelope":
        if not isinstance(value, dict):
            raise CheckpointSchemaError("Checkpoint must be a versioned envelope dictionary")
        required = {
            "schema_version", "library_version", "study_id", "created_at", "updated_at",
            "config_digest", "dataset_id", "estimator_digest", "space_digest", "cv_digest",
            "scorer_digest", "seed", "results",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise CheckpointSchemaError(f"Checkpoint envelope missing fields: {', '.join(missing)}")
        if value["schema_version"] != SCHEMA_VERSION:
            raise CheckpointSchemaError(
                f"Unsupported checkpoint schema version {value['schema_version']}; expected {SCHEMA_VERSION}"
            )
        if not isinstance(value["results"], list):
            raise CheckpointSchemaError("Checkpoint envelope results must be a list")
        identity = StudyIdentity(
            dataset_id=value["dataset_id"],
            estimator_digest=value["estimator_digest"],
            space_digest=value["space_digest"],
            cv_digest=value["cv_digest"],
            scorer_digest=value["scorer_digest"],
            seed=value["seed"],
            config_digest=value["config_digest"],
        )
        return cls(
            study_id=value["study_id"], identity=identity, results=value["results"],
            created_at=value["created_at"], updated_at=value["updated_at"],
            schema_version=value["schema_version"], library_version=value["library_version"],
        )


def mismatch_fields(expected: StudyIdentity, actual: StudyIdentity) -> list[str]:
    return [
        field for field in expected.as_dict()
        if getattr(expected, field) != getattr(actual, field)
    ]
