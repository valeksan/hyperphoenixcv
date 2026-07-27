"""Recommended adaptive search with Optuna TPE and resumable SQLite state.

Install the optional backend first::

    pip install "hyperphoenixcv[optuna]"

The first trials are a seeded warmup. Afterwards TPE uses completed trial
history to propose increasingly promising hyperparameters. Running this file
again resumes the same study instead of starting over.
"""

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier

from hyperphoenixcv import HyperPhoenixCV


X, y = load_breast_cancer(return_X_y=True)

search = HyperPhoenixCV(
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

search.fit(X, y)
print("Best parameters:", search.best_params_)
print("Best ROC AUC:", search.best_score_)
