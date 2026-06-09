# HyperPhoenixCV 🐦‍🔥

![CI](https://github.com/valeksan/hyperphoenixcv/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![PyPI](https://img.shields.io/pypi/v/hyperphoenixcv?v=0.3.0)

> *"Rise from the ashes of interrupted experiments"*

HyperPhoenixCV is a smart hyperparameter tuning library that, like the mythical phoenix, **resumes after interruptions** and continues searching for optimal solutions. Never lose hours of computation due to unexpected stops again!

**Other languages:** [Русский](README_RU.md)

## ✨ Features

- **🔄 Resumable searches** – Continue from the last checkpoint after any interruption.
- **🧠 Bayesian optimization** – Find better parameters faster with intelligent search.
- **🎯 Multiple search strategies** – Exhaustive grid search, random search, or predictive optimization.
- **📊 Multi‑metric evaluation** – Score using multiple metrics (F1, accuracy, precision, etc.) simultaneously.
- **💾 Automatic checkpointing** – Results are saved automatically to pickle files and CSV.
- **🔌 Scikit‑learn compatible** – Seamlessly integrates with the scikit‑learn ecosystem.
- **⚡ Performance optimizations** – Parallel execution with `pre_dispatch` and graceful error handling with `error_score`.
- **⏱️ Early stopping** – Stop search early if no improvement for a given number of iterations (`early_stopping_patience`).
- **📈 Best index attribute** – Access `best_index_` for compatibility with `GridSearchCV`.

## 🚀 Installation

Install from PyPI:

```bash
pip install hyperphoenixcv
```

Or install the latest development version from source:

```bash
git clone https://github.com/valeksan/hyperphoenixcv.git
cd hyperphoenixcv
pip install -e .
```

## 📖 Why HyperPhoenixCV?

The name **HyperPhoenixCV** refers to the mythical phoenix – a bird that rises from its ashes. In the same way, your hyperparameter search can "rise again" after an interruption, continuing from the last saved checkpoint instead of starting over from scratch.

The "CV" in the name highlights the library's focus on cross‑validation and machine‑learning workflows.
![HyperPhoenixCV Workflow](img_1773657268389.png)

*Diagram illustrating the resumable search process.*

### How It Differs from Plain `GridSearchCV`

| Feature | `GridSearchCV` | `HyperPhoenixCV` |
|---------|----------------|------------------|
| **Resumability** | Starts over after interruption | ✅ Continues from checkpoint |
| **Optimization** | Exhaustive search only | ✅ Bayesian, random, or exhaustive |
| **Multi‑metric** | Single metric at a time | ✅ Multiple metrics simultaneously |
| **Checkpointing** | Manual saving required | ✅ Automatic pickle & CSV export |
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

# Define the model and parameter grid
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Create a HyperPhoenixCV instance with checkpointing
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    checkpoint_path='my_experiment.pkl',
    verbose=True
)

# Run the search (resumes automatically if interrupted)
hp.fit(X, y)

print("Best parameters:", hp.best_params_)
print("Best score:", hp.best_score_)
print("Best index:", hp.best_index_)

# Get top‑5 results
top_results = hp.get_top_results(5)
print(top_results)
```

### 🔁 Resuming an Interrupted Search

If the process is stopped (e.g., due to time limits), simply run the same script again – it will load the checkpoint and continue where it left off:

```python
hp.fit(X, y)  # Automatically resumes from 'my_experiment.pkl'
```

## 📚 Advanced Usage

### Bayesian Optimization

Enable Bayesian optimization to reduce the number of evaluations:

```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    use_bayesian_optimization=True,
    n_iter=30,          # Number of Bayesian iterations
    verbose=True
)
```

### Random Search

Perform a random search over the parameter space:

```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    random_search=True,
    n_iter=50           # Number of random combinations
)
```

### Multiple Metrics

Evaluate using several metrics at once:

```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    scoring=['f1', 'accuracy', 'precision']
)
```

### Exporting Results

Save all results to a CSV file for further analysis:

```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    results_csv='experiment_results.csv'
)
```

### Performance & Error Handling

Control parallel execution and error behavior:

```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    n_jobs=4,                # Use 4 CPU cores
    pre_dispatch='2*n_jobs', # Limit number of simultaneously dispatched jobs
    error_score=np.nan,      # Assign NaN to failed evaluations instead of raising
    verbose=True
)
```

### Early Stopping

Stop the search early if no improvement is observed for a given number of iterations:

```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
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
    param_grid=param_grid,
    cv=ts_cv,          # Use the splitter object
    scoring='accuracy'
)

# Group‑aware cross‑validation
group_cv = GroupKFold(n_splits=5)
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
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
- `param_grid`: dict or list of dicts defining the search space.
- `scoring`: metric(s) to evaluate (string, callable, list, or dict).
- `cv`: int, cross‑validation splitter, or iterable (default=5).
- `n_jobs`: number of parallel jobs (default=1).
- `pre_dispatch`: controls number of dispatched jobs (default='2*n_jobs').
- `error_score`: value to assign when an error occurs (default=np.nan).
- `early_stopping_patience`: number of iterations without improvement to stop early (default=None, disabled).
- `checkpoint_path`: path to pickle file for checkpointing (default=None).
- `results_csv`: path to CSV file for saving results (default=None).
- `verbose`: verbosity level (default=False).

**Attributes after fitting**:

- `best_params_`: dict of best parameters.
- `best_score_`: best cross‑validation score.
- `best_index_`: index of the best candidate in the results.
- `cv_results_`: dict of detailed results (like `GridSearchCV`).
- `top_results_`: DataFrame with top‑N results.

**Methods**:

- `fit(X, y, **fit_params)`: run the search (resumes from checkpoint if available).
- `get_top_results(n=10)`: return a DataFrame with top‑N candidates.

For a complete list of parameters and methods, see the source code or use `help(HyperPhoenixCV)`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Thanks to the scikit‑learn community for the foundation on which this library is built.
