"""Explicit, trusted-only import for pre-SQLite pickle checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import joblib


class LegacyCheckpointError(ValueError):
    """Legacy checkpoint cannot be safely imported."""


class LegacyCheckpointTrustError(PermissionError):
    """Caller did not explicitly approve unpickling legacy data."""


def load_legacy_results(path: str | Path, *, trusted: bool = False) -> list[dict[str, Any]]:
    """Load a legacy checkpoint after caller explicitly accepts pickle risk.

    Pickle/joblib deserialization can execute arbitrary code. Pass
    ``trusted=True`` only for a file whose provenance and contents you trust.
    This function never modifies the source file.
    """
    if trusted is not True:
        raise LegacyCheckpointTrustError(
            "Legacy pickle import is disabled by default. Pass trusted=True only "
            "for a checkpoint you trust; unpickling can execute arbitrary code."
        )

    warnings.warn(
        "Loading trusted legacy pickle checkpoint. Unpickling can execute arbitrary code; "
        "only import files from a trusted source.",
        UserWarning,
        stacklevel=2,
    )
    try:
        loaded = joblib.load(path)
    except Exception as exc:
        raise LegacyCheckpointError(f"Cannot load legacy checkpoint: {path}") from exc
    if not isinstance(loaded, list):
        raise LegacyCheckpointError(
            "Invalid legacy checkpoint: expected List[dict] at top level."
        )
    return loaded


def validate_legacy_result(value: Any) -> str | None:
    """Return validation error, or ``None`` when legacy trial can be imported."""
    if not isinstance(value, dict):
        return "expected dict"
    if not isinstance(value.get("params"), dict):
        return "missing dict 'params'"
    return None
