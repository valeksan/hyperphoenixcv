"""
Unit tests for CVExecutor.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.metrics import accuracy_score, make_scorer

from src.hyperphoenixcv.cv_executor import CVExecutor


class TestCVExecutor:
    """Test CVExecutor."""

    @pytest.fixture
    def data(self):
        X, y = make_classification(n_samples=50, n_features=5, random_state=42)
        return X, y

    @pytest.fixture
    def estimator(self):
        return LogisticRegression(random_state=42)

    def test_init_default(self):
        executor = CVExecutor()
        assert executor.cv == 5
        assert executor.scoring == ['f1']
        assert executor.n_jobs == 1
        assert executor.verbose is True
        assert executor.pre_dispatch == '2*n_jobs'
        assert executor.error_score == 'raise'

    def test_init_custom(self):
        executor = CVExecutor(
            cv=3,
            scoring=['accuracy', 'f1'],
            n_jobs=-1,
            verbose=False,
            pre_dispatch='n_jobs',
            error_score=np.nan
        )
        assert executor.cv == 3
        assert executor.scoring == ['accuracy', 'f1']
        assert executor.n_jobs == -1
        assert executor.verbose is False
        assert executor.pre_dispatch == 'n_jobs'
        assert np.isnan(executor.error_score)

    def test_evaluate_success(self, data, estimator):
        X, y = data
        executor = CVExecutor(cv=3, scoring='accuracy', verbose=False)
        params = {'C': 1.0}
        result = executor.evaluate(estimator, X, y, params)
        assert 'params' in result
        assert result['params'] == params
        assert 'mean_test_accuracy' in result
        assert 'std_test_accuracy' in result
        assert 'scores_accuracy' in result
        assert isinstance(result['mean_test_accuracy'], float)
        assert 0 <= result['mean_test_accuracy'] <= 1
        assert 'error' not in result

    def test_evaluate_multiple_scoring(self, data, estimator):
        X, y = data
        executor = CVExecutor(cv=3, scoring=['accuracy', 'f1'], verbose=False)
        params = {'C': 0.5}
        result = executor.evaluate(estimator, X, y, params)
        assert 'mean_test_accuracy' in result
        assert 'mean_test_f1' in result
        assert 'scores_accuracy' in result
        assert 'scores_f1' in result
        assert len(result['scores_accuracy']) == 3

    def test_evaluate_mapping_scoring(self, data, estimator):
        X, y = data
        executor = CVExecutor(
            cv=3, scoring={"acc": "accuracy", "f1_score": "f1"}, verbose=False
        )
        result = executor.evaluate(estimator, X, y, {"C": 0.5})

        assert {"mean_test_acc", "mean_test_f1_score"} <= set(result)

    def test_evaluate_callable_scoring(self, data, estimator):
        X, y = data
        executor = CVExecutor(cv=3, scoring=make_scorer(accuracy_score), verbose=False)
        result = executor.evaluate(estimator, X, y, {"C": 0.5})

        assert "mean_test_score" in result

    def test_evaluate_error(self, data, estimator):
        X, y = data
        executor = CVExecutor(verbose=False)
        # Invalid parameter that will cause an error
        params = {'C': -1.0}  # negative C may cause error in some solvers
        with pytest.raises(ValueError):
            executor.evaluate(estimator, X, y, params)

    def test_cv_stratified_for_classification(self, data, estimator):
        X, y = data
        executor = CVExecutor(cv=3, scoring='f1', verbose=False)
        # internal splitter should be StratifiedKFold because scoring is f1
        result = executor.evaluate(estimator, X, y, {'C': 1.0})
        assert 'mean_test_f1' in result

    def test_uses_sklearn_check_cv_splits(self, data, estimator):
        X, y = data
        executor = CVExecutor(cv=3, scoring='accuracy', verbose=False)
        executor.evaluate(estimator, X, y, {'C': 1.0})
        expected = list(check_cv(3, y=y, classifier=True).split(X, y))

        assert len(executor._splits) == len(expected)
        for actual, reference in zip(executor._splits, expected):
            assert np.array_equal(actual[0], reference[0])
            assert np.array_equal(actual[1], reference[1])

    def test_cv_kfold_for_regression_scoring(self, data):
        # Use a regression scorer (not classification)
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        X, y = data
        estimator = Ridge()
        # mean_squared_error is not a classification metric
        executor = CVExecutor(cv=3, scoring='neg_mean_squared_error', verbose=False)
        result = executor.evaluate(estimator, X, y, {'alpha': 1.0})
        assert 'mean_test_neg_mean_squared_error' in result

    def test_cv_custom_splitter(self, data, estimator):
        X, y = data
        custom_cv = KFold(n_splits=2, shuffle=True, random_state=123)
        executor = CVExecutor(cv=custom_cv, scoring='accuracy', verbose=False)
        result = executor.evaluate(estimator, X, y, {'C': 1.0})
        assert 'mean_test_accuracy' in result
        # 2 folds
        assert len(result['scores_accuracy']) == 2

    def test_format_scores(self):
        executor = CVExecutor(scoring=['f1', 'accuracy'], verbose=False)
        dummy_scores = {
            'test_f1': np.array([0.8, 0.9, 0.85]),
            'test_accuracy': np.array([0.7, 0.8, 0.75]),
        }
        formatted = executor._format_scores(dummy_scores)
        assert 'mean_test_f1' in formatted
        assert formatted['mean_test_f1'] == pytest.approx(0.85)
        assert 'std_test_f1' in formatted
        assert 'scores_f1' in formatted
        assert formatted['scores_f1'] == [0.8, 0.9, 0.85]
        assert 'mean_test_accuracy' in formatted
        assert formatted['mean_test_accuracy'] == pytest.approx(0.75)

    def test_evaluate_with_pre_dispatch(self, data, estimator):
        """Test that pre_dispatch parameter works."""
        X, y = data
        executor = CVExecutor(
            cv=2,
            scoring='accuracy',
            pre_dispatch='1',
            verbose=False
        )
        params = {'C': 1.0}
        result = executor.evaluate(estimator, X, y, params)
        assert 'mean_test_accuracy' in result
        assert 'error' not in result

    def test_evaluate_with_error_score_numeric(self, data, estimator):
        """Test that error_score numeric value works."""
        X, y = data
        executor = CVExecutor(
            cv=2,
            scoring='accuracy',
            error_score=np.nan,
            verbose=False
        )
        # Use a parameter that will cause an error (negative C)
        params = {'C': -1.0}
        result = executor.evaluate(estimator, X, y, params)
        assert 'params' in result
        assert 'error' in result
        assert np.isnan(result['mean_test_accuracy'])
        assert len(result['fold_errors']) == 2

    def test_result_includes_fold_timings(self, data, estimator):
        X, y = data
        result = CVExecutor(cv=3, scoring='accuracy', verbose=False).evaluate(
            estimator, X, y, {'C': 1.0}
        )

        assert len(result['fit_time']) == 3
        assert len(result['score_time']) == 3
        assert result['mean_fit_time'] >= 0
