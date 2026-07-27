"""
Checkpoint manager for saving and loading intermediate results.
"""

import os
from pathlib import Path
import tempfile
import joblib
import logging
from typing import List, Dict, Any

from .study_identity import (
    CheckpointEnvelope,
    StudyIdentity,
)


logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages checkpoint files for hyperparameter search.
    """

    def __init__(self, checkpoint_path: str, verbose: bool = True):
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.envelope: CheckpointEnvelope | None = None

    def _load_raw(self) -> Any:
        raise RuntimeError(
            "Implicit pickle loading was removed. Use "
            "HyperPhoenixCV.import_legacy_checkpoint(path, trusted=True)."
        )

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
        raise RuntimeError(
            "Automatic pickle resume was removed. Use SQLite resume or explicit "
            "HyperPhoenixCV.import_legacy_checkpoint(path, trusted=True)."
        )

    def load(self) -> List[Dict[str, Any]]:
        """
        Load results from checkpoint file.

        Returns:
            List of results (each result is a dict with at least 'params' key).
        """
        raise RuntimeError(
            "Implicit pickle loading was removed. Use "
            "HyperPhoenixCV.import_legacy_checkpoint(path, trusted=True)."
        )

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
            logger.info("Checkpoint saved")

    def clear(self):
        """
        Delete the checkpoint file if it exists.
        """
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            if self.verbose:
                logger.info("Checkpoint deleted")
        elif self.verbose:
            logger.info("Checkpoint does not exist")


class CheckpointCorruptionError(ValueError):
    """Checkpoint cannot be decoded; it must never be treated as empty state."""
