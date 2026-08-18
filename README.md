# HyperPhoenixCV 🐦‍🔥

![CI](https://github.com/valeksan/hyperphoenixcv/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![TestPyPI](https://img.shields.io/pypi/v/hyperphoenixcv?label=TestPyPI&pypiBaseUrl=https%3A%2F%2Ftest.pypi.org&cacheSeconds=300)](https://test.pypi.org/project/hyperphoenixcv/)

> *"Rise from the ashes of interrupted experiments"*

HyperPhoenixCV is a smart hyperparameter tuning library that, like the mythical phoenix, **resumes after interruptions** and continues searching for optimal solutions. Never lose hours of computation due to unexpected stops again!

**Other languages:** [Русский](README_RU.md)

## ✨ Features

- **🔄 Resumable searches** – Continue from the last checkpoint after any interruption.
- **🎲 Search modes** – Exhaustive grid, reproducible random, or adaptive Optuna TPE.
- **🎯 Model-guided optimization** – Optuna learns from completed trials to propose promising parameters.
- **📊 Multi‑metric evaluation** – Score using multiple metrics (F1, accuracy, precision, etc.) simultaneously.
- **💾 Transactional persistence** – Trials commit incrementally to local SQLite; CSV is an export.
- **🔌 Scikit‑learn compatible** – Seamlessly integrates with the scikit‑learn ecosystem.
- **⚡ Performance optimizations** – Parallel execution with `pre_dispatch` and graceful error handling with `error_score`.
- **⏱️ Early stopping** – Stop search early if no improvement for a given number of iterations (`early_stopping_patience`).
- **📈 Best index attribute** – Access `best_index_` for compatibility with `GridSearchCV`.

### Events and callbacks

Pass `callbacks=[callback]`; each callback receives typed runtime events such
as `StudyStarted`, `TrialCompleted`, `TrialFailed`, and `StudyCompleted`.
Callbacks run synchronously in coordinator process after each terminal trial
commit. Callback exceptions fail `fit` (fail-fast). Events are runtime-only:
they are not stored in SQLite and are not replayed on resume. `verbose=True`
uses standard `logging` progress events; configure Python logging to display
them. Default logs omit datasets, parameter values, and tracebacks.

### Guides

- [Resume identity and local storage](docs/resume_and_storage.md)
- [Scalar and multi-objective refit](docs/refit_objectives.md)
- [Honest Optuna pruning](docs/pruning.md)
- [Parallelism and resource tuning](docs/parallelism.md)
- [Audit exports and runtime events](docs/audit_and_events.md)
- [API reference](docs/api_reference.md)

## 🚀 Installation

Install current release from TestPyPI. Keep PyPI as extra index for dependencies:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple hyperphoenixcv
```

Or install the latest development version from source:

```bash
git clone --depth=1 https://github.com/valeksan/hyperphoenixcv.git
cd hyperphoenixcv
pip install -e .
```

### Supported runtime

Python 3.10–3.12 on Linux. Minimum dependencies: NumPy 1.21.6, pandas 1.5,
joblib 1.3, and scikit-learn 1.4. Optuna 3.0+ is optional. Windows and macOS
are not supported targets until durability CI coverage exists.

### Metadata routing

HyperPhoenixCV supports scikit-learn 1.4+ with metadata routing either enabled
or disabled. Pass CV groups as `fit(X, y, groups=groups)`. They are consumed
once to resolve splits and are not also sent to `cross_validate`. Other fit
metadata, such as `sample_weight`, is passed once through sklearn's `params`
path. With routing enabled, configure every receiving estimator/pipeline step
with sklearn's `set_fit_request(...)`; sklearn raises for metadata no step
requests. This matches sklearn routing rules.

## 📖 Why HyperPhoenixCV?

The name **HyperPhoenixCV** refers to the mythical phoenix – a bird that rises from its ashes. In the same way, your hyperparameter search can "rise again" after an interruption, continuing from the last saved checkpoint instead of starting over from scratch.

The "CV" in the name highlights the library's focus on cross‑validation and machine‑learning workflows.
![HyperPhoenixCV Workflow](img_1773657268389.png)

*Diagram illustrating the resumable search process.*

### How It Differs from Plain `GridSearchCV`

| Feature | `GridSearchCV` | `HyperPhoenixCV` |
|---------|----------------|------------------|
| **Resumability** | Starts over after interruption | ✅ Continues from checkpoint |
| **Optimization** | Exhaustive search only | ✅ Grid, random, or adaptive Optuna TPE |
| **Multi‑metric** | Single metric at a time | ✅ Multiple metrics simultaneously |
| **Persistence** | Manual saving required | ✅ Transactional SQLite + CSV export |
| **Progress tracking** | Limited | ✅ Verbose logs & intermediate results |
| **Early stopping** | Not supported | ✅ Configurable patience |
| **Error handling** | Raises exception | ✅ Configurable `error_score` (e.g., `np.nan`) |
| **Parallel dispatch** | Basic | ✅ `pre_dispatch` for better resource management |

## 🛠️ Quick Start

Here’s a minimal example that shows the core workflow:

```python
from hyperphoenixcv import HyperPhoenixCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Create a simple dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Define model and canonical search space
model = RandomForestClassifier()
search_space = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create a HyperPhoenixCV instance with checkpointing
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

# Run the search (resumes automatically from SQLite if interrupted)
hp.fit(X, y)

print("Best parameters:", hp.best_params_)
print("Best score:", hp.best_score_)
print("Best index:", hp.best_index_)

# Get top‑5 results
top_results = hp.get_top_results(5)
print(top_results)
```

### 🔁 Resuming an Interrupted Search

If the process is stopped (e.g., due to time limits), run the same script again with the same study-store path and `dataset_id`:

```python
hp.fit(X, y)  # Automatically resumes from 'my_experiment.sqlite3'
```

SQLite is local, single-coordinator storage. It is not a shared-filesystem or multi-node backend. `resume` accepts `"auto"` (default), `"must"`, or `"never"`; a changed study identity is rejected instead of mixing results.

## 📚 Advanced Usage

### Search strategies

Use seeded random search, or genuine Optuna ask/tell:

```python
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="random",
    n_trials=30,
    verbose=True
)
```

`results_csv` is a flat convenience projection: only completed rows and flat
score/parameter columns survive. SQLite remains source of truth. For every
terminal state, diagnostics, objectives, and exceptions, use paginated audit
history and its atomic exports:

```python
history = hp.trial_history_
history.export_json("audit.json")       # lossless tagged JSON
history.export_csv("audit.csv")         # flat convenience export
history.export_parquet("audit.parquet") # requires hyperphoenixcv[parquet]
```

Use `history.page(offset=0, limit=100)` or `history.iter_records(page_size=1000)`
for large studies. `metric_directions={"loss": "minimize"}` controls ranking
and non-Optuna scalar refit; unspecified sklearn scores default to maximize.
`cv_results_` materializes at most `max_cv_results=10_000` trials by default;
set it to `None` only when memory budget permits full sklearn projection.

#### Recommended: adaptive Optuna TPE

Use this when model evaluation is expensive and you cannot afford to exhaust
the whole space. The first trials are seeded warmup; afterwards TPE learns from
completed trials and proposes increasingly promising parameters. The SQLite
study remains resumable.

Install the optional backend first:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "hyperphoenixcv[optuna]"
```

```python
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier

from hyperphoenixcv import HyperPhoenixCV

X, y = load_breast_cancer(return_X_y=True)

hp = HyperPhoenixCV(
    estimator=HistGradientBoostingClassifier(random_state=42),
    strategy="optuna",
    search_space={
        "learning_rate": optuna.distributions.FloatDistribution(1e-3, 0.3, log=True),
        "max_leaf_nodes": optuna.distributions.IntDistribution(8, 64),
        "max_depth": optuna.distributions.IntDistribution(2, 12),
        "min_samples_leaf": optuna.distributions.IntDistribution(5, 50),
        "l2_regularization": optuna.distributions.FloatDistribution(1e-8, 10.0, log=True),
    },
    n_trials=30,
    optuna_warmup_trials=8,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    storage_path="optuna_tpe.sqlite3",
    dataset_id="breast-cancer-optuna-tpe-v1",
)

hp.fit(X, y)
print("Best parameters:", hp.best_params_)
print("Best ROC AUC:", hp.best_score_)
```

Full runnable version: [`examples/optuna_tpe_example.py`](examples/optuna_tpe_example.py).

Optuna trials use real `ask`/`tell`; terminal results replay from SQLite on
resume. `n_trials` caps terminal trials across resume. Conditional spaces need
`search_space(trial) -> dict` plus stable `search_space_id`.

Multi-objective mode requires explicit directions and exposes `pareto_front_`.
Use `refit=False`, metric name, or selector callable; `refit=True` is invalid.

```python
hp = HyperPhoenixCV(
    estimator=model, strategy="optuna", search_space=space,
    n_trials=40, scoring=["accuracy", "neg_log_loss"],
    optuna_directions={"accuracy": "maximize", "neg_log_loss": "maximize"},
    refit=False,
)
```

Plain sklearn CV never prunes mid-fit. Cooperative `intermediate_evaluator`
receives `(estimator, X, y, params, report, groups, fit_params)`. Call
`report(step, value)` with increasing integer steps; it returns prune request.
Evaluator stops safely then returns `{"trial_state": "pruned"}`. Replay means
same seed + committed history, not changed Optuna version or batch size.
Examples: `examples/optuna_multiobjective_example.py`,
`examples/optuna_cooperative_pruning_example.py`.

### Random Search

Perform a random search over the parameter space:

```python
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="random",
    n_trials=50         # Number of random combinations
)
```

### Multiple Metrics

Evaluate using several metrics at once:

```python
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="grid",
    scoring={'f1': 'f1', 'accuracy': 'accuracy', 'precision': 'precision'},
    refit='f1',
)
```

### Exporting Results

Save all results to a CSV file for further analysis:

```python
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="grid",
    results_csv='experiment_results.csv'
)
```

### Performance & Error Handling

Control parallel execution and error behavior:

```python
import numpy as np

hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="grid",
    n_jobs=4,
    parallelism='trials',    # Default: up to four trials, one CV worker each
    inner_max_num_threads=1, # Native threads per trial process
    error_score=np.nan,
    verbose=True
)
```

`parallelism='folds'` instead evaluates one trial at a time and uses `n_jobs`
for folds. Nested trial-and-fold parallelism is intentionally unsupported.

### Early Stopping

Stop the search early if no improvement is observed for a given number of iterations:

```python
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="random",
    n_trials=50,
    early_stopping_patience=5,  # Stop after 5 iterations without improvement
    verbose=True
)
```

### Custom Cross‑Validation Splitter

HyperPhoenixCV supports any cross‑validation splitter that follows the scikit‑learn interface (e.g., `TimeSeriesSplit`, `GroupKFold`, `StratifiedKFold`). You can pass a splitter object directly to the `cv` parameter:

```python
from sklearn.model_selection import TimeSeriesSplit, GroupKFold

# Time‑series cross‑validation
ts_cv = TimeSeriesSplit(n_splits=5)
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="grid",
    cv=ts_cv,          # Use the splitter object
    scoring='accuracy'
)

# Group‑aware cross‑validation
group_cv = GroupKFold(n_splits=5)
hp = HyperPhoenixCV(
    estimator=model,
    search_space=param_grid,
    strategy="grid",
    cv=group_cv,
    scoring='accuracy'
)
# Then call fit with groups parameter
hp.fit(X, y, groups=groups)
```

See the full example: [examples/custom_cv_example.py](examples/custom_cv_example.py)

## 📖 API Reference

### HyperPhoenixCV

Main class for hyperparameter search.

**Parameters** (most important):

- `estimator`: scikit‑learn compatible estimator.
- `search_space`: canonical search-space argument. For `strategy="grid"` and
  `"random"`, use `ParameterGrid` syntax; for `"optuna"`, use Optuna
  distributions or callable space.
- `strategy`: `"grid"`, `"random"`, or `"optuna"`.
- `n_trials`: trial cap for `"random"` and `"optuna"`.
- `scoring`: metric(s) to evaluate (string, callable, list, or dict).
- `cv`: int, cross‑validation splitter, or iterable (default=5).
- `n_jobs`: number of parallel jobs (default=1).
- `parallelism`: `"trials"` (default) or `"folds"`; exactly one axis uses `n_jobs`.
- `compute`: `"cpu"` (default) or G1 `"gpu"`. GPU mode validates one local
  NVIDIA device at `fit()` through `nvidia-smi`; it never installs CUDA or
  changes estimator parameters.
- `gpu_devices`: one physical NVIDIA index or UUID in G1 (default `(0,)` when
  `compute="gpu"`). Device ID is runtime diagnostics, not resume identity.
- `gpu_slots_per_device`: G1 requires `1`. G1 also requires sequential
  `n_jobs=1`; parallel GPU trials/folds are G2 work.
- `inner_max_num_threads`: optional native-thread cap for process-parallel trials.
- `trial_timeout`: optional per-trial timeout in seconds. Requires
  `parallelism="trials"` and `n_jobs >= 2`; timed-out trials are stored failed.
- `cancel_callback`: optional zero-argument callable. Return `True` or a reason
  string to stop before next unstarted trial.
- `memmap_max_nbytes`, `memmap_temp_folder`, `joblib_batch_size`: joblib
  process-backend settings; arrays over default `"1M"` use read-only memmap.
- `pre_dispatch`: controls number of dispatched jobs (default='2*n_jobs').
- `error_score`: `'raise'` or numeric value assigned when a trial error occurs
  (default=`'raise'`).
- `early_stopping_patience`: number of iterations without improvement to stop early (default=None, disabled).
- `storage_path`: canonical local SQLite store path.
- `dataset_id`: stable dataset identifier. Required for strong resume identity; `None` emits a warning.
- `resume`: `"auto"` (default), `"must"`, or `"never"`.
- `max_cv_results`: max trials materialized in `cv_results_` (default=10,000);
  `None` opts into unbounded projection.
- `clear_storage()`: explicitly delete SQLite study storage before `fit()`.
- `results_csv`: path to CSV file for saving results
  (default=`"hyperphoenix_results.csv"`).
- `verbose`: enable progress logging (default=True).

### G1 GPU use

GPU mode is resource validation and diagnostics, not automatic acceleration.
Install/configure GPU-capable estimator yourself, e.g. XGBoost with
`device="cuda"`. HyperPhoenixCV preserves estimator device parameters and runs
trial evaluation plus refit sequentially in same process context. Missing or
unselected NVIDIA device fails before SQLite study creation. Unknown estimator
GPU capability is reported as unverified. CPU-only estimators do not gain GPU
support from `compute="gpu"`.

**Attributes after fitting**:

- `best_params_`: dict of best parameters.
- `best_score_`: best cross‑validation score.
- `best_index_`: index of the best candidate in the results.
- `cv_results_`: dict of detailed results (like `GridSearchCV`).
- `trial_history_`: read-only, SQLite-backed terminal audit projection.
- `top_results_`: DataFrame with top‑N results.

**Methods**:

- `fit(X, y)`: run the search and resume from matching SQLite study when allowed.
- `get_top_results(n=10)`: return a DataFrame with top‑N candidates.
- `clear_storage()`: delete SQLite store explicitly.
- `load_trial_history()`: open audit history for an existing matching study.

SQLite storage is supported only on a local filesystem. Windows locking and
durability behavior is not CI-validated in P0, so Windows is not yet a
supported target. Do not use network or synced folders for a study store.
Backup, integrity checks, WAL/SHM cleanup, and recovery procedure: see
[local storage recovery](docs/storage_recovery.md).

For complete parameters and methods, see [API reference](docs/api_reference.md)
or `help(HyperPhoenixCV)`.

## Security and operations

SQLite paths are local-only: do not use network, shared, or sync folders. One
active coordinator may write a study. Use `hyperphoenixcv-storage backup`, not
manual database/WAL copying; see [storage recovery](docs/storage_recovery.md).

Callbacks execute synchronously in your process and are trusted application
code. Logging omits raw datasets, parameters, and full tracebacks by default;
avoid emitting sensitive values in custom callbacks.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Thanks to the scikit‑learn community for the foundation on which this library is built.
