# HyperPhoenixCV 🐦‍🔥

> *"Rise from the ashes of interrupted experiments"*

HyperPhoenixCV is a smart hyperparameter tuning tool that, like the mythical phoenix, **resumes after interruptions** and continues searching for optimal solutions. Never lose hours of computation due to unexpected stops again!

**Other languages:** [Русский](README_RU.md)

## Installation

```bash
pip install hyperphoenixcv
```

Or install from source:

```bash
git clone https://github.com/valeksan/hyperphoenixcv.git
cd hyperphoenixcv
pip install -e .
```

## Why HyperPhoenixCV?

The name **HyperPhoenixCV** refers to the mythical phoenix – a bird that rises from its ashes. In the same way, your hyperparameter search can "rise again" after an interruption, continuing from the last saved checkpoint instead of starting over from scratch.
The "CV" in the name indicates the connection with cross‑validation, emphasizing the library's specialization in machine learning.

## Differences from plain GridSearchCV:

### **Resumability**
Ran out of time or resources? Just run it again – it will continue from the last checkpoint!
```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    checkpoint_path="my_experiment.pkl"  # Key parameter!
)
hp.fit(X, y)  # Interrupted? Run again with the same path
```

### **Smart optimization**
Bayesian optimization helps find better parameters faster.
```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    use_bayesian_optimization=True,
    verbose=True
)
```

### **Flexibility**
Exhaustive search, random search, or predictive optimization – choose your approach. Multiple metrics supported.
```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    random_search=True,
    n_iter=50  # Number of random combinations
)
```
```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    scoring=['f1', 'accuracy', 'precision']
)
```

### **Automatic saving**
All results are saved to checkpoints and CSV.
```python
hp = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    results_csv="experiment_results.csv"  # Results saved to CSV
)
```

### **Compatibility**
Fully compatible with the scikit‑learn API (if not, we'll fix it).

## 🚀 Quick Start
```python
from hyperphoenixcv import HyperPhoenixCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create a pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Define the parameter grid
param_grid = {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.001, 0.01, 0.1, 1.0]
}

# Create and run HyperPhoenixCV
hp = HyperPhoenixCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    checkpoint_path="nlp_experiment.pkl",
    verbose=True
)

# First run (may take a long time)
hp.fit(X_train, y_train)

# If the process was interrupted, just run again:
hp.fit(X_train, y_train)  # Continues from the last saved checkpoint!

# Get results
print("Best parameters:", hp.best_params_)
print("Best F1 score:", hp.best_score_)

# Get top‑5 results
top_results = hp.get_top_results(5)
print(top_results)
```

## 🙏 Acknowledgments

Thanks to the scikit‑learn community for the foundation on which this library is built.