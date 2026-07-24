"""
Example of HyperPhoenixCV with a custom cross‑validation splitter.

This example demonstrates:
- Using a custom CV splitter (e.g., TimeSeriesSplit, GroupKFold)
- How to pass a splitter object instead of an integer
- Group‑based cross‑validation with groups parameter
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GroupKFold
from hyperphoenixcv import HyperPhoenixCV

# Create a synthetic dataset with groups
X, y = make_classification(n_samples=200, n_features=20, random_state=42)

# Simulate groups (e.g., each group contains 5 samples)
groups = np.repeat(np.arange(40), 5)

print("Dataset shape:", X.shape)
print("Groups shape:", groups.shape)
print("Unique groups:", np.unique(groups).shape[0])

# Define model and parameter grid
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Example 1: TimeSeriesSplit (for time‑series data)
print("\n" + "="*60)
print("EXAMPLE 1: TimeSeriesSplit")
print("="*60)

ts_cv = TimeSeriesSplit(n_splits=5)
hp_ts = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    cv=ts_cv,  # Pass the splitter object directly
    scoring='accuracy',
    checkpoint_path='timeseries_checkpoint.sqlite3',
    dataset_id='custom-cv-synthetic-v1',
    verbose=True
)

# Fit with groups=None (TimeSeriesSplit doesn't use groups)
hp_ts.fit(X, y)

print(f"Best accuracy (TimeSeriesSplit): {hp_ts.best_score_:.4f}")
print(f"Best parameters: {hp_ts.best_params_}")

# Example 2: GroupKFold (group‑aware cross‑validation)
print("\n" + "="*60)
print("EXAMPLE 2: GroupKFold with groups parameter")
print("="*60)

group_cv = GroupKFold(n_splits=5)
hp_group = HyperPhoenixCV(
    estimator=model,
    param_grid=param_grid,
    cv=group_cv,
    scoring='accuracy',
    checkpoint_path='groupkfold_checkpoint.sqlite3',
    dataset_id='custom-cv-synthetic-v1',
    verbose=True
)

# Fit with groups parameter
hp_group.fit(X, y, groups=groups)

print(f"Best accuracy (GroupKFold): {hp_group.best_score_:.4f}")
print(f"Best parameters: {hp_group.best_params_}")

# Compare results
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"TimeSeriesSplit best score: {hp_ts.best_score_:.4f}")
print(f"GroupKFold best score:      {hp_group.best_score_:.4f}")

if hp_ts.best_score_ > hp_group.best_score_:
    print("TimeSeriesSplit performed better on this synthetic dataset.")
else:
    print("GroupKFold performed better on this synthetic dataset.")

# Clean up checkpoints
hp_ts.clear_checkpoint_file()
hp_group.clear_checkpoint_file()
print("\nSQLite study stores deleted.")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("HyperPhoenixCV supports any cross‑validation splitter that follows")
print("the scikit‑learn splitter interface (e.g., KFold, StratifiedKFold,")
print("TimeSeriesSplit, GroupKFold, etc.).")
print("\nTo use a custom splitter:")
print("1. Create the splitter object (e.g., `my_cv = CustomSplitter(...)`)")
print("2. Pass it as the `cv` parameter to HyperPhoenixCV")
print("3. If the splitter requires groups, pass them via `fit(groups=...)`")
print("\nThis flexibility allows HyperPhoenixCV to be used in a wide range")
print("of scenarios, including time‑series, grouped data, and more.")
