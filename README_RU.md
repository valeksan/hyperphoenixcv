# HyperPhoenixCV 🐦‍🔥

> *"Возрождайтесь из пепла прерванных экспериментов"*

**Другие языки:** [English](README.md)

HyperPhoenixCV - это умный инструмент для подбора гиперпараметров, который, как мифический феникс, **возрождается после прерываний** и продолжает поиск оптимальных решений. Никогда больше не теряйте часы вычислений из-за неожиданных остановок!

## Установка

```bash
pip install hyperphoenixcv
```

Или установите из исходников:

```bash
git clone https://github.com/valeksan/hyperphoenixcv.git
cd hyperphoenixcv
pip install -e .
```

## Почему HyperPhoenixCV?

Название **HyperPhoenixCV** отсылает к мифическому фениксу — птице, которая возрождается из пепла. Точно так же ваш поиск гиперпараметров может "возродиться" после прерывания, продолжая с последней сохраненной точки, а не начиная всё сначала.
CV в названии указывает на связь с кросс-валидацией, подчеркивая специализацию библиотеки в области машинного обучения.

## Отличия от простого GridSearchCV:

### **Возобновляемость**
Прервались из-за нехватки времени или ресурсов? Просто запустите снова — продолжится с последней точки!
```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    checkpoint_path="my_experiment.pkl"  # Ключевой параметр!
)
hp.fit(X, y)  # Прервали? Запустите снова с тем же путем
```

### **Умная оптимизация**
Байесовская оптимизация помогает быстрее находить лучшие параметры.
```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    use_bayesian_optimization=True,
    verbose=True
)
```

### **Гибкость** 
Полный перебор, случайный поиск или предсказательная оптимизация — выбирайте подход. Множественные метрики.
```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    random_search=True,
    n_iter=50  # Количество случайных комбинаций
)
```
```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    scoring=['f1', 'accuracy', 'precision']
)
```

### **Автоматическое сохранение**
Все результаты сохраняются в чекпоинты и CSV.
```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    results_csv="experiment_results.csv"  # Результаты сохраняются в CSV
)
```

### **Совместимость**
Полностью совместим с scikit-learn API (а если нет то поправим).

## 🚀 Быстрый старт
```python
from hyperphoenixcv import HyperPhoenixCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Создаем пайплайн
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Определяем сетку параметров
param_grid = {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.001, 0.01, 0.1, 1.0]
}

# Создаем и запускаем HyperPhoenixCV
hp = HyperPhoenixCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    checkpoint_path="nlp_experiment.pkl",
    verbose=True
)

# Первый запуск (может занять много времени)
hp.fit(X_train, y_train)

# Если процесс был прерван, просто запустите снова:
hp.fit(X_train, y_train)  # Продолжит с последней сохраненной точки!

# Получаем результаты
print("Лучшие параметры:", hp.best_params_)
print("Лучший F1-скор:", hp.best_score_)

# Получаем топ-5 результатов
top_results = hp.get_top_results(5)
print(top_results)
```

## 🙏 Благодарности

Спасибо сообществу scikit-learn за основу, на которой построена эта библиотека.
