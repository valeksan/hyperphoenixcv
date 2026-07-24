import os
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ParameterGrid
from hyperphoenixcv import HyperPhoenixCV

@pytest.fixture
def sample_data():
    """Creates synthetic data for tests."""
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    # For simplicity, convert numeric data to strings
    X_text = np.array([' '.join(map(str, row)) for row in X])
    return X_text, y

@pytest.fixture
def sample_pipeline():
    """Creates a simple pipeline for tests."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

@pytest.fixture
def sample_param_grid():
    """Example parameter grid for tests."""
    return {
        'tfidf__max_features': [10, 20],
        'clf__C': [0.1, 1.0]
    }

def test_hyperphoenixcv_initialization(sample_pipeline, sample_param_grid):
    """Tests HyperPhoenixCV initialization."""
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring='accuracy',
        cv=2,
        n_jobs=1,
        checkpoint_path="test_checkpoint.pkl",
        results_csv="test_results.csv",
        verbose=False,
        pre_dispatch='n_jobs',
        error_score=np.nan,
        early_stopping_patience=5
    )
    assert hp.estimator == sample_pipeline
    assert hp.param_grid == sample_param_grid
    assert hp.scoring == 'accuracy'
    assert hp.cv == 2
    assert hp.n_jobs == 1
    assert hp.pre_dispatch == 'n_jobs'
    assert np.isnan(hp.error_score)
    assert hp.early_stopping_patience == 5
    assert os.path.exists("test_checkpoint.pkl") is False  # Checkpoint is not created on initialization
    os.remove("test_checkpoint.pkl") if os.path.exists("test_checkpoint.pkl") else None
    os.remove("test_results.csv") if os.path.exists("test_results.csv") else None

def test_hyperphoenixcv_full_grid_search(sample_data, sample_pipeline, sample_param_grid):
    """Tests exhaustive grid search."""
    X, y = sample_data
    
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring='accuracy',
        cv=2,
        n_jobs=1,
        checkpoint_path="test_checkpoint.pkl",
        results_csv="test_results.csv",
        verbose=False
    )
    
    hp.fit(X, y)
    
    # Verify that all combinations were tested
    total_combinations = len(list(sample_param_grid['tfidf__max_features'])) * len(list(sample_param_grid['clf__C']))
    assert len(hp.cv_results_['params']) == total_combinations
    
    # Verify presence of best parameters and scores
    assert hp.best_params_ is not None
    assert hp.best_score_ > 0
    # Verify best_index_ attribute
    assert hp.best_index_ is not None
    assert 0 <= hp.best_index_ < total_combinations
    # Ensure best_index_ matches best_params_ in cv_results_
    assert hp.cv_results_['params'][hp.best_index_] == hp.best_params_
    
    # Verify existence of result files
    assert os.path.exists("test_checkpoint.pkl")
    assert os.path.exists("test_results.csv")
    
    # Cleanup
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_checkpoint_resume(sample_data, sample_pipeline, sample_param_grid):
    """Tests resuming after interruption."""
    X, y = sample_data
    
    # Create a checkpoint with partial results
    partial_results = [{
        'params': {'tfidf__max_features': 10, 'clf__C': 0.1},
        'mean_test_accuracy': 0.7,
        'std_test_accuracy': 0.05
    }]
    
    import joblib
    joblib.dump(partial_results, "test_checkpoint.pkl")
    
    # Create a new instance that should load the checkpoint
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring='accuracy',
        cv=2,
        n_jobs=1,
        checkpoint_path="test_checkpoint.pkl",
        results_csv="test_results.csv",
        verbose=False
    )
    
    hp.fit(X, y)
    
    # Verify that it continued from where it left off
    total_combinations = len(list(sample_param_grid['tfidf__max_features'])) * len(list(sample_param_grid['clf__C']))
    assert len(hp.cv_results_['params']) == total_combinations
    
    # Cleanup
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_random_search(sample_data, sample_pipeline, sample_param_grid):
    """Tests random parameter search."""
    X, y = sample_data
    
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring='accuracy',
        cv=2,
        n_jobs=1,
        random_search=True,
        n_iter=2,
        checkpoint_path="test_checkpoint.pkl",
        results_csv="test_results.csv",
        verbose=False
    )
    
    hp.fit(X, y)
    
    # Verify that n_iter combinations were tested
    assert len(hp.cv_results_['params']) == 2
    
    # Cleanup
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_multiple_metrics(sample_data, sample_pipeline, sample_param_grid):
    """Tests working with multiple metrics."""
    X, y = sample_data
    
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring=['accuracy', 'f1'],
        cv=2,
        n_jobs=1,
        checkpoint_path="test_checkpoint.pkl",
        results_csv="test_results.csv",
        verbose=False
    )
    
    hp.fit(X, y)
    
    # Verify presence of both metrics in results
    assert 'mean_test_accuracy' in hp.cv_results_
    assert 'mean_test_f1' in hp.cv_results_
    
    # Cleanup
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_bayesian_optimization(sample_data, sample_pipeline, sample_param_grid):
    """Tests Bayesian optimization."""
    X, y = sample_data
    
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring='accuracy',
        cv=2,
        n_jobs=1,
        use_bayesian_optimization=True,
        checkpoint_path="test_checkpoint.pkl",
        results_csv="test_results.csv",
        verbose=False
    )
    
    hp.fit(X, y)
    
    # Verify that results exist
    assert len(hp.cv_results_['params']) > 0
    
    # Cleanup
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_get_top_results(sample_data, sample_pipeline, sample_param_grid):
    """Tests the get_top_results method."""
    X, y = sample_data
    
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring='accuracy',
        cv=2,
        n_jobs=1,
        checkpoint_path="test_checkpoint.pkl",
        results_csv="test_results.csv",
        verbose=False
    )
    
    hp.fit(X, y)
    
    # Verify retrieval of top results
    top_results = hp.get_top_results(2)
    assert len(top_results) == 2
    assert 'mean_test_accuracy' in top_results.columns
    
    # Cleanup
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_load_from_checkpoint(sample_data, sample_pipeline, sample_param_grid):
    """Tests loading results from a checkpoint."""
    X, y = sample_data
    
    # Create a checkpoint with partial results
    partial_results = [
        {
            'params': {'tfidf__max_features': 10, 'clf__C': 0.1},
            'mean_test_accuracy': 0.7,
            'std_test_accuracy': 0.05
        },
        {
            'params': {'tfidf__max_features': 20, 'clf__C': 1.0},
            'mean_test_accuracy': 0.8,
            'std_test_accuracy': 0.03
        }
    ]
    
    import joblib
    joblib.dump(partial_results, "test_checkpoint.pkl")
    
    # Create an instance for loading
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring='accuracy',
        cv=2,
        checkpoint_path="test_checkpoint.pkl",
        verbose=False
    )
    
    # Load results
    top_results = hp.load_results_from_checkpoint(2)
    
    # Verify that correct results were loaded
    assert len(top_results) == 2
    assert top_results.iloc[0]['mean_test_accuracy'] == 0.8  # Best result should be first
    
    # Cleanup
    os.remove("test_checkpoint.pkl")

def test_hyperphoenixcv_error_handling(sample_data, sample_pipeline, sample_param_grid):
    """Tests error handling during search."""
    X, y = sample_data
    
    # Create parameters that will cause an error
    invalid_param_grid = {
        'tfidf__max_features': [10, 20],
        'clf__C': ['invalid', 1.0]  # Invalid value
    }
    
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=invalid_param_grid,
        scoring='accuracy',
        cv=2,
        n_jobs=1,
        checkpoint_path="error_checkpoint.pkl",
        verbose=False
    )
    
    hp.fit(X, y)
    
    # Verify that the error was recorded
    assert len(hp.cv_results_['params']) < len(list(ParameterGrid(invalid_param_grid)))
    assert any('error' in r for r in hp._load_checkpoint())
    
    # Cleanup
    os.remove("error_checkpoint.pkl")

def test_hyperphoenixcv_final_fit(sample_data, sample_pipeline, sample_param_grid):
    """Tests training the best model on the entire dataset."""
    X, y = sample_data
    
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring='accuracy',
        cv=2,
        refit=True,
        verbose=False
    )
    
    hp.fit(X, y)
    
    # Verify that the model is trained
    assert hasattr(hp, 'best_estimator_')
    assert hp.best_estimator_.named_steps['clf'].coef_ is not None
    
    # Verify predictions
    predictions = hp.best_estimator_.predict(X)
    assert len(predictions) == len(y)


def test_hyperphoenixcv_early_stopping(sample_data, sample_pipeline, sample_param_grid):
    """Tests early stopping functionality."""
    X, y = sample_data
    
    # Use random search with small n_iter and early_stopping_patience
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring='accuracy',
        cv=2,
        n_jobs=1,
        random_search=True,
        n_iter=10,
        early_stopping_patience=2,
        checkpoint_path="early_stop_checkpoint.pkl",
        results_csv="early_stop_results.csv",
        verbose=False
    )
    
    hp.fit(X, y)
    
    # Early stopping may cause fewer than n_iter evaluations
    # We just verify that the attribute is set and no errors
    assert hp.early_stopping_patience == 2
    # Ensure results exist
    assert len(hp.cv_results_['params']) > 0
    
    # Cleanup
    os.remove("early_stop_checkpoint.pkl") if os.path.exists("early_stop_checkpoint.pkl") else None
    os.remove("early_stop_results.csv") if os.path.exists("early_stop_results.csv") else None
