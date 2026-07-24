"""
Unit tests for ResultManager.
"""

import pandas as pd
import tempfile
import pytest

from src.hyperphoenixcv.result_manager import ResultManager


class TestResultManager:
    """Test ResultManager."""

    @pytest.fixture
    def scoring(self):
        return ['f1', 'accuracy']

    @pytest.fixture
    def manager(self, scoring):
        return ResultManager(scoring=scoring)

    @pytest.fixture
    def sample_results(self):
        return [
            {
                'params': {'a': 1, 'b': 'x'},
                'mean_test_f1': 0.8,
                'std_test_f1': 0.1,
                'mean_test_accuracy': 0.85,
                'std_test_accuracy': 0.05,
            },
            {
                'params': {'a': 2, 'b': 'y'},
                'mean_test_f1': 0.9,
                'std_test_f1': 0.08,
                'mean_test_accuracy': 0.92,
                'std_test_accuracy': 0.04,
            },
            {
                'params': {'a': 3, 'b': 'z'},
                'error': 'Some error',
            },
        ]

    def test_add_result(self, manager):
        result = {'params': {'a': 1}, 'mean_test_f1': 0.5}
        manager.add_result(result)
        assert len(manager.results) == 1
        assert manager.results[0] == result

    def test_add_results(self, manager, sample_results):
        manager.add_results(sample_results)
        assert len(manager.results) == 3

    def test_clear_results(self, manager, sample_results):
        manager.add_results(sample_results)
        assert len(manager.results) == 3
        manager.clear_results()
        assert len(manager.results) == 0

    def test_get_top_results(self, manager, sample_results):
        manager.add_results(sample_results)
        top = manager.get_top_results(n=5)
        # Should have 2 valid results (excluding error)
        assert isinstance(top, pd.DataFrame)
        assert len(top) == 2
        # Should be sorted by f1 (first scoring metric) descending
        assert top.iloc[0]['mean_test_f1'] == 0.9
        assert top.iloc[1]['mean_test_f1'] == 0.8
        # Columns include parameters and metrics
        assert 'a' in top.columns
        assert 'b' in top.columns
        assert 'mean_test_f1' in top.columns
        assert 'std_test_accuracy' in top.columns

    def test_get_top_results_empty(self, manager):
        top = manager.get_top_results()
        assert top.empty

    def test_get_top_results_only_errors(self, manager):
        manager.add_result({'params': {}, 'error': 'error'})
        top = manager.get_top_results()
        assert top.empty

    def test_save_to_csv(self, manager, sample_results, tmp_path):
        csv_path = tmp_path / "results.csv"
        manager.add_results(sample_results)
        manager.save_to_csv(str(csv_path))
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 2  # two valid rows
        assert 'mean_test_f1' in df.columns

    def test_save_to_csv_default_path(self, manager, sample_results, tmp_path, monkeypatch):
        # Change working directory to tmp_path to avoid polluting project
        monkeypatch.chdir(tmp_path)
        manager.add_results(sample_results)
        manager.save_to_csv()  # uses default 'hyperphoenix_results.csv'
        assert (tmp_path / "hyperphoenix_results.csv").exists()

    def test_format_cv_results(self, manager, sample_results):
        manager.add_results(sample_results)
        cv_results = manager.format_cv_results()
        assert 'params' in cv_results
        assert len(cv_results['params']) == 2
        assert 'mean_test_f1' in cv_results
        assert 'std_test_f1' in cv_results
        assert 'mean_test_accuracy' in cv_results
        assert 'std_test_accuracy' in cv_results
        assert cv_results['mean_test_f1'] == [0.8, 0.9]

    def test_format_cv_results_empty(self, manager):
        cv_results = manager.format_cv_results()
        assert cv_results == {}

    def test_load_from_checkpoint(self, manager, tmp_path):
        checkpoint_path = tmp_path / "checkpoint.pkl"
        with pytest.raises(RuntimeError, match="Implicit pickle loading"):
            manager.load_from_checkpoint(str(checkpoint_path))
        assert manager.results == []

    def test_load_from_checkpoint_no_file(self, manager, tmp_path):
        checkpoint_path = tmp_path / "nonexistent.pkl"
        with pytest.raises(RuntimeError, match="Implicit pickle loading"):
            manager.load_from_checkpoint(str(checkpoint_path))
        assert manager.results == []
