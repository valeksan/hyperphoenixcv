"""Legacy filename; this project currently has no Bayesian optimizer.

Use seeded random search for a finite categorical space. The deprecated
``use_bayesian_optimization`` flag enables experimental surrogate ranking only;
it has no acquisition function or Bayesian sequential sampler.
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from hyperphoenixcv import HyperPhoenixCV


X, y = make_classification(n_samples=500, n_features=20, random_state=42)

search = HyperPhoenixCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    scoring="accuracy",
    cv=5,
    random_search=True,
    n_iter=12,
    random_state=42,
    parallelism="trials",
    n_jobs=2,
    checkpoint_path="random_checkpoint.sqlite3",
    dataset_id="random-search-example-v1",
    results_csv="random_results.csv",
)

search.fit(X, y)
print("Best parameters:", search.best_params_)
print("Best accuracy:", search.best_score_)
