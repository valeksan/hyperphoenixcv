# HyperPhoenixCV 🐦‍🔥

![CI](https://github.com/valeksan/hyperphoenixcv/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/pypi/v/hyperphoenixcv)

> *"Возрождайтесь из пепла прерванных экспериментов"*

HyperPhoenixCV — это умная библиотека для подбора гиперпараметров, которая, подобно мифическому фениксу, **возрождается после прерываний** и продолжает поиск оптимальных решений. Никогда больше не теряйте часы вычислений из-за неожиданных остановок!

**Другие языки:** [English](README.md)

## ✨ Возможности

- **🔄 Возобновляемый поиск** — Продолжайте с последнего чекпоинта после любого прерывания.
- **🧠 Байесовская оптимизация** — Находите лучшие параметры быстрее с помощью интеллектуального поиска.
- **🎯 Несколько стратегий поиска** — Полный перебор, случайный поиск или предсказательная оптимизация.
- **📊 Оценка по нескольким метрикам** — Одновременное использование нескольких метрик (F1, accuracy, precision и др.).
- **💾 Автоматическое сохранение** — Результаты автоматически сохраняются в pickle-файлы и CSV.
- **🔌 Совместимость с Scikit‑learn** — Бесшовная интеграция с экосистемой scikit‑learn.

## 🚀 Установка

Установите из PyPI:

```bash
pip install hyperphoenixcv
```

Или установите последнюю версию из исходного кода:

```bash
git clone https://github.com/valeksan/hyperphoenixcv.git
cd hyperphoenixcv
pip install -e .
```

## 📖 Почему HyperPhoenixCV?

Название **HyperPhoenixCV** отсылает к мифическому фениксу — птице, которая возрождается из пепла. Точно так же ваш поиск гиперпараметров может «возродиться» после прерывания, продолжая с последней сохранённой точки, а не начиная всё сначала.

«CV» в названии подчёркивает фокус библиотеки на кросс‑валидации и рабочих процессах машинного обучения.

### Чем отличается от обычного `GridSearchCV`

| Возможность | `GridSearchCV` | `HyperPhoenixCV` |
|-------------|----------------|------------------|
| **Возобновляемость** | Начинает заново после прерывания | ✅ Продолжает с чекпоинта |
| **Оптимизация** | Только полный перебор | ✅ Байесовская, случайная или полная |
| **Мультиметричность** | Одна метрика за раз | ✅ Несколько метрик одновременно |
| **Сохранение результатов** | Требуется ручное сохранение | ✅ Автоматическое сохранение в pickle и CSV |
| **Отслеживание прогресса** | Ограничено | ✅ Подробные логи и промежуточные результаты |

## 🛠️ Быстрый старт

Вот минимальный пример, демонстрирующий основной workflow:

```python
from hyperphoenixcv import HyperPhoenixCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Создаём простой датасет
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Определяем модель и сетку параметров
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Создаём экземпляр HyperPhoenixCV с чекпоинтингом
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    checkpoint_path='my_experiment.pkl',
    verbose=True
)

# Запускаем поиск (автоматически возобновляется при прерывании)
hp.fit(X, y)

print("Лучшие параметры:", hp.best_params_)
print("Лучшая точность:", hp.best_score_)

# Получаем топ‑5 результатов
top_results = hp.get_top_results(5)
print(top_results)
```

### 🔁 Возобновление прерванного поиска

Если процесс был остановлен (например, из‑за ограничения по времени), просто запустите тот же скрипт снова — он загрузит чекпоинт и продолжит с того же места:

```python
hp.fit(X, y)  # Автоматически возобновляется из 'my_experiment.pkl'
```

## 📚 Расширенное использование

### Байесовская оптимизация

Включите байесовскую оптимизацию, чтобы сократить количество оценок:

```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    use_bayesian_optimization=True,
    n_iter=30,          # Количество итераций байесовской оптимизации
    verbose=True
)
```

### Случайный поиск

Выполните случайный поиск по пространству параметров:

```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    random_search=True,
    n_iter=50           # Количество случайных комбинаций
)
```

### Несколько метрик

Оценивайте с использованием нескольких метрик одновременно:

```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    scoring=['f1', 'accuracy', 'precision']
)
```

### Экспорт результатов

Сохраняйте все результаты в CSV‑файл для дальнейшего анализа:

```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    results_csv='experiment_results.csv'
)
```

## 🤝 Участие в разработке

Мы приветствуем вклад! Не стесняйтесь отправлять Pull Request.

## 📄 Лицензия

Этот проект распространяется под лицензией MIT — подробности см. в файле [LICENSE](LICENSE).

## 🙏 Благодарности

Спасибо сообществу scikit‑learn за основу, на которой построена эта библиотека.
