# HyperPhoenixCV 🐦‍🔥

![CI](https://github.com/valeksan/hyperphoenixcv/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![TestPyPI](https://img.shields.io/pypi/v/hyperphoenixcv?label=TestPyPI&pypiBaseUrl=https%3A%2F%2Ftest.pypi.org&cacheSeconds=300)](https://test.pypi.org/project/hyperphoenixcv/)

> *"Возрождайтесь из пепла прерванных экспериментов"*

HyperPhoenixCV — это умная библиотека для подбора гиперпараметров, которая, подобно мифическому фениксу, **возрождается после прерываний** и продолжает поиск оптимальных решений. Никогда больше не теряйте часы вычислений из-за неожиданных остановок!

**Другие языки:** [English](README.md)

## ✨ Возможности

- **🔄 Возобновляемый поиск** — Продолжайте с последнего чекпоинта после любого прерывания.
- **🎲 Режимы поиска** — Полный перебор или воспроизводимый случайный поиск.
- **🎯 Несколько стратегий поиска** — Полный перебор или случайный поиск.
- **📊 Оценка по нескольким метрикам** — Одновременное использование нескольких метрик (F1, accuracy, precision и др.).
- **💾 Транзакционное хранение** — Trials инкрементально сохраняются в локальный SQLite; CSV — только экспорт.
- **🔌 Совместимость с Scikit‑learn** — Бесшовная интеграция с экосистемой scikit‑learn.
- **⚡ Оптимизация производительности** — Параллельное выполнение с `pre_dispatch` и обработка ошибок через `error_score`.
- **⏱️ Ранняя остановка** — Остановить поиск досрочно, если улучшений нет заданное число итераций (`early_stopping_patience`).
- **📈 Атрибут best_index_** — Доступ к `best_index_` для совместимости с `GridSearchCV`.

## 🚀 Установка

Установите текущую версию из TestPyPI. PyPI указан как дополнительный индекс для зависимостей:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple hyperphoenixcv
```

Или установите последнюю версию из исходного кода:

```bash
git clone https://github.com/valeksan/hyperphoenixcv.git
cd hyperphoenixcv
pip install -e .
```

### Поддерживаемая среда

Python 3.10–3.12 на Linux. Минимальные зависимости: NumPy 1.21.6, pandas 1.5,
joblib 1.3, scikit-learn 1.4. Optuna 3.0+ — опциональная зависимость. Windows
и macOS пока не поддерживаются: нет CI-проверок надёжности хранения.

## 📖 Почему HyperPhoenixCV?

Название **HyperPhoenixCV** отсылает к мифическому фениксу — птице, которая возрождается из пепла. Точно так же ваш поиск гиперпараметров может «возродиться» после прерывания, продолжая с последней сохранённой точки, а не начиная всё сначала.

«CV» в названии подчёркивает фокус библиотеки на кросс‑валидации и рабочих процессах машинного обучения.
![HyperPhoenixCV Workflow](img_1773657268389.png)

*Диаграмма, иллюстрирующая процесс возобновляемого поиска.*

### Чем отличается от обычного `GridSearchCV`

| Возможность | `GridSearchCV` | `HyperPhoenixCV` |
|-------------|----------------|------------------|
| **Возобновляемость** | Начинает заново после прерывания | ✅ Продолжает с чекпоинта |
| **Оптимизация** | Только полный перебор | ✅ Случайная или полная |
| **Мультиметричность** | Одна метрика за раз | ✅ Несколько метрик одновременно |
| **Хранение результатов** | Требуется ручное сохранение | ✅ Транзакционный SQLite + экспорт CSV |
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

# Определяем модель и каноническое пространство поиска
model = RandomForestClassifier()
search_space = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Создаём экземпляр HyperPhoenixCV с чекпоинтингом
hp = HyperPhoenixCV(
    estimator=model,
    search_space=search_space,
    strategy='grid',
    scoring='accuracy',
    cv=5,
    storage_path='my_experiment.sqlite3',
    dataset_id='training-data-v1',
    verbose=True
)

# Запускаем поиск (автоматически возобновляется из SQLite при прерывании)
hp.fit(X, y)

print("Лучшие параметры:", hp.best_params_)
print("Лучшая точность:", hp.best_score_)
print("Индекс лучшего кандидата:", hp.best_index_)

# Получаем топ‑5 результатов
top_results = hp.get_top_results(5)
print(top_results)
```

### 🔁 Возобновление прерванного поиска

Если процесс был остановлен (например, из‑за ограничения по времени), снова запустите тот же скрипт с тем же путём study store и `dataset_id`:

```python
hp.fit(X, y)  # Автоматически возобновляется из 'my_experiment.sqlite3'
```

SQLite — локальное хранилище для одного coordinator. Это не backend для общей файловой системы или нескольких узлов. `resume`: `"auto"` (по умолчанию), `"must"`, `"never"`; несовпадающий identity study отклоняется, результаты не смешиваются.

### Миграция legacy pickle checkpoint

Обычные `fit()` и resume никогда не распаковывают pickle. Один раз явно импортируйте старый checkpoint в SQLite study:

```python
report = hp.import_legacy_checkpoint("old_experiment.pkl", trusted=True)
print(report)  # imported/skipped/failed и детали невалидных записей
```

**Предупреждение безопасности:** загрузка pickle/joblib может выполнить произвольный код. Указывайте `trusted=True` только для файла с проверенным происхождением и содержимым. Importer принимает legacy `List[dict]`, не меняет исходный файл и идемпотентен.

## 📚 Расширенное использование

### Стратегии поиска

Используйте seeded random search или настоящий Optuna ask/tell:

```python
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="random",
    n_trials=30,
    verbose=True
)
```

`results_csv` — flat convenience projection: сохраняет только completed rows
и flat score/parameter columns. SQLite остаётся source of truth. Для всех
terminal states, diagnostics, objectives, exceptions используйте paginated
audit history и atomic export:

```python
history = hp.trial_history_
history.export_json("audit.json")       # lossless tagged JSON
history.export_csv("audit.csv")         # flat convenience export
history.export_parquet("audit.parquet") # требует hyperphoenixcv[parquet]
```

Для больших study: `history.page(offset=0, limit=100)` или
`history.iter_records(page_size=1000)`. `metric_directions={"loss": "minimize"}`
задаёт ranking/non-Optuna scalar refit; default sklearn scores — maximize.
`cv_results_` по default materializes максимум `max_cv_results=10_000` trials;
`None` задавайте только при достаточном memory budget для полной sklearn projection.

Установите optional backend:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "hyperphoenixcv[optuna]"
```

```python
import optuna

hp = HyperPhoenixCV(
    estimator=model,
    strategy="optuna",
    search_space={
        "C": optuna.distributions.FloatDistribution(1e-4, 10, log=True),
        "penalty": optuna.distributions.CategoricalDistribution(["l1", "l2"]),
    },
    n_trials=30,
    optuna_warmup_trials=10,
    random_state=42,
)
```

Optuna использует настоящий `ask`/`tell`; terminal trials восстанавливаются из
SQLite. `n_trials` ограничивает trials, включая resume. Для conditional space
нужны `search_space(trial) -> dict` и стабильный `search_space_id`.

Multi-objective требует `optuna_directions`, публикует `pareto_front_`.
Используйте `refit=False`, имя метрики или callable; `refit=True` запрещён.
Обычный sklearn CV не делает mid-fit pruning. Cooperative
`intermediate_evaluator(estimator, X, y, params, report, groups, fit_params)`
вызывает `report(step, value)` с возрастающими шагами; `True` -> безопасно
остановить работу, вернуть `{"trial_state": "pruned"}`.

### Случайный поиск

Выполните случайный поиск по пространству параметров:

```python
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="random",
    n_trials=50         # Количество случайных комбинаций
)
```

### Несколько метрик

Оценивайте с использованием нескольких метрик одновременно:

```python
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="grid",
    scoring={'f1': 'f1', 'accuracy': 'accuracy', 'precision': 'precision'},
    refit='f1',
)
```

### Экспорт результатов

Сохраняйте все результаты в CSV‑файл для дальнейшего анализа:

```python
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="grid",
    results_csv='experiment_results.csv'
)
```

### Производительность и обработка ошибок

Управляйте параллельным выполнением и поведением при ошибках:

```python
import numpy as np

hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="grid",
    n_jobs=4,
    parallelism='trials',    # По умолчанию: до четырёх trials, CV однопоточный
    inner_max_num_threads=1, # Число native threads на процесс trial
    error_score=np.nan,
    verbose=True
)
```

`parallelism='folds'` запускает один trial за раз, распределяя `n_jobs` по
folds. Вложенный параллелизм trials × folds намеренно не поддерживается.

### Ранняя остановка

Остановите поиск досрочно, если улучшений не наблюдается заданное число итераций:

```python
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="random",
    n_trials=50,
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
    search_space=param_grid,
    strategy="grid",
    cv=ts_cv,          # Используется объект сплиттера
    scoring='accuracy'
)

# Групповая кросс‑валидация
group_cv = GroupKFold(n_splits=5)
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="grid",
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
- `search_space`: каноническое пространство поиска. Для `strategy="grid"` и
  `"random"` используется синтаксис `ParameterGrid`; для `"optuna"` —
  Optuna distributions или callable space.
- `strategy`: `"grid"`, `"random"` или `"optuna"`.
- `n_trials`: лимит trials для `"random"` и `"optuna"`.
- `scoring`: метрика(и) для оценки (строка, функция, список или словарь).
- `cv`: int, сплиттер кросс‑валидации или итерируемый объект (по умолчанию=5).
- `n_jobs`: количество параллельных jobs (по умолчанию=1).
- `parallelism`: `"trials"` (по умолчанию) или `"folds"`; `n_jobs` работает только по одной оси.
- `inner_max_num_threads`: опциональный лимит native threads для process-parallel trials.
- `trial_timeout`: optional timeout одного trial в секундах. Требует
  `parallelism="trials"` и `n_jobs >= 2`; timeout trial сохраняется как failed.
- `cancel_callback`: optional callable без аргументов. Верните `True` или строку
  причины, чтобы остановиться до следующего не начатого trial.
- `memmap_max_nbytes`, `memmap_temp_folder`, `joblib_batch_size`: настройки
  joblib process backend; массивы больше default `"1M"` используют read-only memmap.
- `pre_dispatch`: управляет количеством одновременно запускаемых jobs (по умолчанию='2*n_jobs').
- `error_score`: значение, присваиваемое при ошибке (по умолчанию=np.nan).
- `early_stopping_patience`: количество итераций без улучшений для досрочной остановки (по умолчанию=None, отключено).
- `storage_path`: канонический путь к локальному SQLite store; pickle никогда
  не используется для resume.
- `dataset_id`: стабильный идентификатор датасета. Нужен для сильной identity resume; `None` выдаёт warning.
- `resume`: `"auto"` (по умолчанию), `"must"` или `"never"`.
- `max_cv_results`: max trials в materialized `cv_results_` (по умолчанию=10,000);
  `None` включает unbounded projection.
- `clear_storage()`: явно удалить SQLite storage перед `fit()`.
- В 0.6 удалены legacy `param_grid`, `random_search`, `n_iter`,
  `use_bayesian_optimization`, `bayesian_optimizer`, `checkpoint_path` и
  `clear_checkpoint`. Подробнее: [migration 0.4 → 0.6](docs/migration_0.4_to_0.6.md).
- `results_csv`: путь к CSV‑файлу для сохранения результатов (по умолчанию=None).
- `verbose`: уровень детализации (по умолчанию=False).

**Атрибуты после обучения**:

- `best_params_`: dict лучших параметров.
- `best_score_`: лучшее значение кросс‑валидационной метрики.
- `best_index_`: индекс лучшего кандидата в результатах.
- `cv_results_`: dict с детальными результатами (как в `GridSearchCV`).
- `trial_history_`: read-only SQLite-backed terminal audit projection.
- `top_results_`: DataFrame с топ‑N результатами.

**Методы**:

- `fit(X, y)`: запустить поиск; при разрешении возобновляет matching SQLite study.
- `get_top_results(n=10)`: вернуть DataFrame с топ‑N кандидатами.
- `load_results_from_checkpoint(n=10)`: прочитать top results из SQLite после прерывания.
- `import_legacy_checkpoint(path, trusted=True)`: однократная trusted migration legacy pickle.
- `clear_storage()`: явно удалить SQLite store.
- `load_trial_history()`: открыть audit history matching existing study.

SQLite store поддержан только на локальной файловой системе. Windows locking и
durability не проверены в CI для P0, поэтому Windows пока не является
поддерживаемой платформой. Не используйте network/synced folders для store.

Полный список параметров и методов см. в исходном коде или используйте `help(HyperPhoenixCV)`.

## 🤝 Участие в разработке

Мы приветствуем вклад! Не стесняйтесь отправлять Pull Request.

## 📄 Лицензия

Этот проект распространяется под лицензией MIT — подробности см. в файле [LICENSE](LICENSE).

## 🙏 Благодарности

Спасибо сообществу scikit‑learn за основу, на которой построена эта библиотека.
