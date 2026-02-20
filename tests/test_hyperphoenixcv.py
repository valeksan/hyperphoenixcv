import os
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from hyperphoenixcv import HyperPhoenixCV

@pytest.fixture
def sample_data():
    """Создает синтетические данные для тестов."""
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    # Для простоты преобразуем числовые данные в строки
    X_text = np.array([' '.join(map(str, row)) for row in X])
    return X_text, y

@pytest.fixture
def sample_pipeline():
    """Создает простой пайплайн для тестов."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50)),
        ('clf', LogisticRegression(max_iter=1000))
    ])

@pytest.fixture
def sample_param_grid():
    """Пример сетки параметров для тестов."""
    return {
        'tfidf__max_features': [10, 20],
        'clf__C': [0.1, 1.0]
    }

def test_hyperphoenixcv_initialization(sample_pipeline, sample_param_grid):
    """Тестирует инициализацию HyperPhoenixCV."""
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
    assert hp.estimator == sample_pipeline
    assert hp.param_grid == sample_param_grid
    assert hp.scoring == ['accuracy']
    assert hp.cv == 2
    assert hp.n_jobs == 1
    assert os.path.exists("test_checkpoint.pkl") is False  # Чекпоинт не создается при инициализации
    os.remove("test_checkpoint.pkl") if os.path.exists("test_checkpoint.pkl") else None
    os.remove("test_results.csv") if os.path.exists("test_results.csv") else None

def test_hyperphoenixcv_full_grid_search(sample_data, sample_pipeline, sample_param_grid):
    """Тестирует полный перебор параметров."""
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
    
    # Проверяем, что все комбинации были протестированы
    total_combinations = len(list(sample_param_grid['tfidf__max_features'])) * len(list(sample_param_grid['clf__C']))
    assert len(hp.cv_results_['params']) == total_combinations
    
    # Проверяем наличие лучших параметров и скоров
    assert hp.best_params_ is not None
    assert hp.best_score_ > 0
    
    # Проверяем существование файлов результатов
    assert os.path.exists("test_checkpoint.pkl")
    assert os.path.exists("test_results.csv")
    
    # Очистка
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_checkpoint_resume(sample_data, sample_pipeline, sample_param_grid):
    """Тестирует возобновление после прерывания."""
    X, y = sample_data
    
    # Создаем чекпоинт с частичными результатами
    partial_results = [{
        'params': {'tfidf__max_features': 10, 'clf__C': 0.1},
        'mean_test_accuracy': 0.7,
        'std_test_accuracy': 0.05
    }]
    
    import joblib
    joblib.dump(partial_results, "test_checkpoint.pkl")
    
    # Создаем новый экземпляр, который должен загрузить чекпоинт
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
    
    # Проверяем, что продолжилось с того места, где остановилось
    total_combinations = len(list(sample_param_grid['tfidf__max_features'])) * len(list(sample_param_grid['clf__C']))
    assert len(hp.cv_results_['params']) == total_combinations
    
    # Очистка
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_random_search(sample_data, sample_pipeline, sample_param_grid):
    """Тестирует случайный перебор параметров."""
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
    
    # Проверяем, что было протестировано n_iter комбинаций
    assert len(hp.cv_results_['params']) == 2
    
    # Очистка
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_multiple_metrics(sample_data, sample_pipeline, sample_param_grid):
    """Тестирует работу с несколькими метриками."""
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
    
    # Проверяем наличие обеих метрик в результатах
    assert 'mean_test_accuracy' in hp.cv_results_
    assert 'mean_test_f1' in hp.cv_results_
    
    # Очистка
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_bayesian_optimization(sample_data, sample_pipeline, sample_param_grid):
    """Тестирует работу с байесовской оптимизацией."""
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
    
    # Проверяем, что результаты существуют
    assert len(hp.cv_results_['params']) > 0
    
    # Очистка
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_get_top_results(sample_data, sample_pipeline, sample_param_grid):
    """Тестирует метод get_top_results."""
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
    
    # Проверяем получение топ-результатов
    top_results = hp.get_top_results(2)
    assert len(top_results) == 2
    assert 'mean_test_accuracy' in top_results.columns
    
    # Очистка
    os.remove("test_checkpoint.pkl")
    os.remove("test_results.csv")

def test_hyperphoenixcv_load_from_checkpoint(sample_data, sample_pipeline, sample_param_grid):
    """Тестирует загрузку результатов из чекпоинта."""
    X, y = sample_data
    
    # Создаем чекпоинт с частичными результатами
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
    
    # Создаем экземпляр для загрузки
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring='accuracy',
        cv=2,
        checkpoint_path="test_checkpoint.pkl",
        verbose=False
    )
    
    # Загружаем результаты
    top_results = hp.load_results_from_checkpoint(2)
    
    # Проверяем, что загрузились правильные результаты
    assert len(top_results) == 2
    assert top_results.iloc[0]['mean_test_accuracy'] == 0.8  # Лучший результат должен быть первым
    
    # Очистка
    os.remove("test_checkpoint.pkl")

def test_hyperphoenixcv_error_handling(sample_data, sample_pipeline, sample_param_grid):
    """Тестирует обработку ошибок во время поиска."""
    X, y = sample_data
    
    # Создаем параметры, которые вызовут ошибку
    invalid_param_grid = {
        'tfidf__max_features': [10, 20],
        'clf__C': ['invalid', 1.0]  # Невалидное значение
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
    
    # Проверяем, что ошибка была зафиксирована
    assert len(hp.cv_results_['params']) < len(list(ParameterGrid(invalid_param_grid)))
    assert any('error' in r for r in hp._load_checkpoint())
    
    # Очистка
    os.remove("error_checkpoint.pkl")

def test_hyperphoenixcv_final_fit(sample_data, sample_pipeline, sample_param_grid):
    """Тестирует обучение лучшей модели на всем датасете."""
    X, y = sample_data
    
    hp = HyperPhoenixCV(
        estimator=sample_pipeline,
        param_grid=sample_param_grid,
        scoring='accuracy',
        cv=2,
        finaly_fit_best_model=True,
        verbose=False
    )
    
    hp.fit(X, y)
    
    # Проверяем, что модель обучена
    assert hasattr(hp, 'best_estimator_')
    assert hp.best_estimator_.named_steps['clf'].coef_ is not None
    
    # Проверяем предсказания
    predictions = hp.best_estimator_.predict(X)
    assert len(predictions) == len(y)
