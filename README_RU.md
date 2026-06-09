# HyperPhoenixCV 🐦‍🔥

![CI](https://github.com/valeksan/hyperphoenixcv/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/pypi/v/hyperphoenixcv?v=0.3.0)

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
- **⚡ Оптимизация производительности** — Параллельное выполнение с `pre_dispatch` и обработка ошибок через `error_score`.
- **⏱️ Ранняя остановка** — Остановить поиск досрочно, если улучшений нет заданное число итераций (`early_stopping_patience`).
- **📈 Атрибут best_index_** — Доступ к `best_index_` для совместимости с `GridSearchCV`.

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
![HyperPhoenixCV Workflow](img_1773657268389.png)

*Диаграмма, иллюстрирующая процесс возобновляемого поиска.*

### Чем отличается от обычного `GridSearchCV`

| Возможность | `GridSearchCV` | `HyperPhoenixCV` |
|-------------|----------------|------------------|
| **Возобновляемость** | Начинает заново после прерывания | ✅ Продолжает с чекпоинта |
| **Оптимизация** | Только полный перебор | ✅ Байесовская, случайная или полная |
| **Мультиметричность** | Одна метрика за раз | ✅ Несколько метрик одновременно |
| **Сохранение результатов** | Требуется ручное сохранение | ✅ Автоматическое сохранение в pickle и CSV |
| **Отслеживание прогресса** | Ограничено | ✅ Подробные логи и промежуточные результаты |
| **Ранняя остановка** | Не поддерживается | ✅ Настраиваемый patience |
| **Обработка ошибок** | Выбрасывает исключение | ✅ Настраиваемый `error_score` (например, `np.nan`) |
| **Управление параллелизмом** | Базовое | ✅ `pre_dispatch` для лучшего управления ресурсами |

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
print("Индекс лучшего кандидата:", hp.best_index_)

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

### Производительность и обработка ошибок

Управляйте параллельным выполнением и поведением при ошибках:

```python
import numpy as np

hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    n_jobs=4,                # Использовать 4 ядра CPU
    pre_dispatch='2*n_jobs', # Ограничить количество одновременно запущенных задач
    error_score=np.nan,      # Присваивать NaN при ошибках вместо исключения
    verbose=True
)
```

### Ранняя остановка

Остановите поиск досрочно, если улучшений не наблюдается заданное число итераций:

```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    early_stopping_patience=5,  # Остановить после 5 итераций без улучшений
    verbose=True
)
```

### Пользовательские сплиттеры кросс‑валидации

HyperPhoenixCV поддерживает любой сплиттер кросс‑валидации, совместимый с scikit‑learn (например, `TimeSeriesSplit`, `GroupKFold`, `StratifiedKFold`). Вы можете передать объект сплиттера напрямую в параметр `cv`:

```python
from sklearn.model_selection import TimeSeriesSplit, GroupKFold

# Кросс‑валидация для временных рядов
ts_cv = TimeSeriesSplit(n_splits=5)
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    cv=ts_cv,          # Используется объект сплиттера
    scoring='accuracy'
)

# Групповая кросс‑валидация
group_cv = GroupKFold(n_splits=5)
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    cv=group_cv,
    scoring='accuracy'
)
# Затем вызовите fit с параметром groups
hp.fit(X, y, groups=groups)
```

Полный пример: [examples/custom_cv_example.py](examples/custom_cv_example.py)

## 📖 Справка по API

### HyperPhoenixCV

Основной класс для поиска гиперпараметров.

**Параметры** (наиболее важные):

- `estimator`: scikit‑learn совместимый estimator.
- `param_grid`: dict или list of dicts, определяющий пространство поиска.
- `scoring`: метрика(и) для оценки (строка, функция, список или словарь).
- `cv`: int, сплиттер кросс‑валидации или итерируемый объект (по умолчанию=5).
- `n_jobs`: количество параллельных jobs (по умолчанию=1).
- `pre_dispatch`: управляет количеством одновременно запускаемых jobs (по умолчанию='2*n_jobs').
- `error_score`: значение, присваиваемое при ошибке (по умолчанию=np.nan).
- `early_stopping_patience`: количество итераций без улучшений для досрочной остановки (по умолчанию=None, отключено).
- `checkpoint_path`: путь к pickle‑файлу для чекпоинтинга (по умолчанию=None).
- `results_csv`: путь к CSV‑файлу для сохранения результатов (по умолчанию=None).
- `verbose`: уровень детализации (по умолчанию=False).

**Атрибуты после обучения**:

- `best_params_`: dict лучших параметров.
- `best_score_`: лучшее значение кросс‑валидационной метрики.
- `best_index_`: индекс лучшего кандидата в результатах.
- `cv_results_`: dict с детальными результатами (как в `GridSearchCV`).
- `top_results_`: DataFrame с топ‑N результатами.

**Методы**:

- `fit(X, y, **fit_params)`: запустить поиск (возобновляет с чекпоинта, если доступен).
- `get_top_results(n=10)`: вернуть DataFrame с топ‑N кандидатами.

Полный список параметров и методов см. в исходном коде или используйте `help(HyperPhoenixCV)`.

## 🤝 Участие в разработке

Мы приветствуем вклад! Не стесняйтесь отправлять Pull Request.

## 📄 Лицензия

Этот проект распространяется под лицензией MIT — подробности см. в файле [LICENSE](LICENSE).

## 🙏 Благодарности

Спасибо сообществу scikit‑learn за основу, на которой построена эта библиотека.
