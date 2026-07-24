"""Cooperative Optuna pruning; plain sklearn CV cannot prune mid-fit."""

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from hyperphoenixcv import HyperPhoenixCV


def evaluator(estimator, X, y, params, report, groups, fit_params):
    """Estimator owns fit loop, reports real intermediate validation scores."""
    model = clone(estimator).set_params(**params)
    classes = sorted(set(y))
    for epoch in range(20):
        model.partial_fit(X, y, classes=classes)
        if report(epoch, accuracy_score(y, model.predict(X))):
            return {"params": params, "trial_state": "pruned"}
    return {"params": params, "mean_test_accuracy": accuracy_score(y, model.predict(X))}


X, y = load_breast_cancer(return_X_y=True)
search = HyperPhoenixCV(
    estimator=SGDClassifier(random_state=42),
    param_grid=None,
    strategy="optuna",
    search_space={"alpha": optuna.distributions.FloatDistribution(1e-6, 1e-2, log=True)},
    n_trials=20,
    scoring="accuracy",
    intermediate_evaluator=evaluator,
    refit=False,
    dataset_id="breast-cancer-pruning-v1",
)
search.fit(X, y)
