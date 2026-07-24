"""
Checkpoint manager for saving and loading intermediate results.
"""

import os
from pathlib import Path
import tempfile
import joblib
from typing import List, Dict, Any

from .study_identity import (
    CheckpointEnvelope,
    CheckpointMismatchError,
    StudyIdentity,
    mismatch_fields,
)


class CheckpointManager:
    """
    Manages checkpoint files for hyperparameter search.
    """

    def __init__(self, checkpoint_path: str, verbose: bool = True):
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.envelope: CheckpointEnvelope | None = None

    def _load_raw(self) -> Any:
        try:
            return joblib.load(self.checkpoint_path)
        except Exception as exc:
            raise CheckpointCorruptionError(
                f"Cannot read checkpoint {self.checkpoint_path}. It may be corrupt; "
                "restore a known-good checkpoint or start with resume='never'."
            ) from exc

    def _atomic_dump(self, value: Any) -> None:
        target = Path(self.checkpoint_path)
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=parent,
        )
        os.close(descriptor)
        try:
            joblib.dump(value, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            if os.name == "posix":
                directory_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_envelope(
        self,
        identity: StudyIdentity,
        resume: str = "auto",
    ) -> CheckpointEnvelope | None:
        if resume not in {"auto", "must", "never"}:
            raise ValueError("resume must be one of: 'auto', 'must', 'never'")
        if resume == "never":
            return None
        if not os.path.exists(self.checkpoint_path):
            if resume == "must":
                raise FileNotFoundError(
                    f"Checkpoint required by resume='must' does not exist: {self.checkpoint_path}"
                )
            if self.verbose:
                print(f"No checkpoint found at {self.checkpoint_path}.")
            return None

        envelope = CheckpointEnvelope.from_dict(self._load_raw())
        changed = mismatch_fields(identity, envelope.identity)
        if changed:
            raise CheckpointMismatchError(
                f"Checkpoint {self.checkpoint_path} belongs to a different study; "
                f"changed: {', '.join(changed)}. Use a new checkpoint_path or resume='never'."
            )
        self.envelope = envelope
        if self.verbose:
            print(f"Loaded {len(envelope.results)} completed combinations from checkpoint.")
        return envelope

    def load(self) -> List[Dict[str, Any]]:
        """
        Load results from checkpoint file.

        Returns:
            List of results (each result is a dict with at least 'params' key).
        """
        if os.path.exists(self.checkpoint_path):
            loaded = self._load_raw()
            if isinstance(loaded, dict):
                envelope = CheckpointEnvelope.from_dict(loaded)
                self.envelope = envelope
                if self.verbose:
                    print(f"Loaded {len(envelope.results)} completed combinations from checkpoint.")
                return envelope.results
            if isinstance(loaded, list):
                if self.verbose:
                    print(f"Loaded {len(loaded)} completed combinations from checkpoint.")
                return loaded
            raise ValueError(f"Invalid legacy checkpoint at {self.checkpoint_path}")
        if self.verbose:
            print(f"No checkpoint found at {self.checkpoint_path}.")
        return []

    def save(
        self,
        results: List[Dict[str, Any]],
        identity: StudyIdentity | None = None,
    ):
        """
        Save results to checkpoint file.

        Args:
            results: List of results to save.
        """
        if identity is not None:
            if self.envelope is None:
                envelope = CheckpointEnvelope.new(identity, results)
            else:
                envelope = self.envelope.with_results(results)
            self._atomic_dump(envelope.as_dict())
            self.envelope = envelope
        else:
            self._atomic_dump(results)
        if self.verbose:
            print(f"Checkpoint saved to {self.checkpoint_path}")

    def clear(self):
        """
        Delete the checkpoint file if it exists.
        """
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            if self.verbose:
                print(f"Deleted checkpoint: {self.checkpoint_path}")
        elif self.verbose:
            print(f"Checkpoint {self.checkpoint_path} does not exist.")


class CheckpointCorruptionError(ValueError):
    """Checkpoint cannot be decoded; it must never be treated as empty state."""
