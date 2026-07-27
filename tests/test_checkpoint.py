"""
Unit tests for CheckpointManager.
"""

import os
import tempfile
import pytest

from src.hyperphoenixcv.checkpoint import CheckpointManager


class TestCheckpointManager:
    """Test CheckpointManager."""

    @pytest.fixture
    def temp_file(self):
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def sample_results(self):
        return [
            {'params': {'a': 1}, 'mean_test_f1': 0.8},
            {'params': {'a': 2}, 'mean_test_f1': 0.9},
        ]

    def test_load_no_file(self, temp_file):
        # Ensure file does not exist
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        manager = CheckpointManager(temp_file, verbose=False)
        with pytest.raises(RuntimeError, match="Implicit pickle loading"):
            manager.load()

    def test_save_and_load(self, temp_file, sample_results):
        manager = CheckpointManager(temp_file, verbose=False)
        manager.save(sample_results)
        assert os.path.exists(temp_file)
        with pytest.raises(RuntimeError, match="Implicit pickle loading"):
            manager.load()

    def test_save_overwrite(self, temp_file, sample_results):
        manager = CheckpointManager(temp_file, verbose=False)
        manager.save(sample_results)
        new_results = [{'params': {'b': 3}}]
        manager.save(new_results)
        with pytest.raises(RuntimeError, match="Implicit pickle loading"):
            manager.load()

    def test_clear_existing(self, temp_file, sample_results):
        manager = CheckpointManager(temp_file, verbose=False)
        manager.save(sample_results)
        assert os.path.exists(temp_file)
        manager.clear()
        assert not os.path.exists(temp_file)

    def test_clear_nonexistent(self, temp_file):
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        manager = CheckpointManager(temp_file, verbose=False)
        # Should not raise an error
        manager.clear()
        assert not os.path.exists(temp_file)

    def test_verbose(self, temp_file, sample_results, caplog):
        manager = CheckpointManager(temp_file, verbose=True)
        with caplog.at_level("INFO"):
            manager.save(sample_results)
            with pytest.raises(RuntimeError, match="Implicit pickle loading"):
                manager.load()
            manager.clear()
        assert "Checkpoint saved" in caplog.text
        assert "Checkpoint deleted" in caplog.text or "Checkpoint does not exist" in caplog.text
