"""Resumable Optuna Pareto search. Requires ``hyperphoenixcv[optuna]``."""

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import HyperPhoenixCV


X, y = load_breast_cancer(return_X_y=True)
search = HyperPhoenixCV(
    estimator=LogisticRegression(max_iter=500),
    strategy="optuna",
    search_space={"C": optuna.distributions.FloatDistribution(1e-3, 10, log=True)},
    n_trials=20,
    scoring=["accuracy", "neg_log_loss"],
    optuna_directions={"accuracy": "maximize", "neg_log_loss": "maximize"},
    refit=False,
    random_state=42,
    dataset_id="breast-cancer-pareto-v1",
    storage_path="pareto.sqlite3",
)
search.fit(X, y)
for trial in search.pareto_front_:
    print(trial)
