"""
Quick verification that HyperPhoenixCV works on synthetic data.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from hyperphoenixcv import HyperPhoenixCV
import os

# Generate synthetic data
X, y = make_classification(
    n_samples=200,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [10, 50],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5]
}

# Create HyperPhoenixCV instance with a temporary checkpoint file
checkpoint_file = "temp_checkpoint.pkl"
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

tuner = HyperPhoenixCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=1,
    verbose=1,
    checkpoint_path=checkpoint_file,
    clear_checkpoint=False,
    refit=True
)

print("Starting hyperparameter search...")
tuner.fit(X_train, y_train)

print(f"Best score: {tuner.best_score_:.4f}")
print(f"Best params: {tuner.best_params_}")

# Test prediction
y_pred = tuner.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Test accuracy: {accuracy:.4f}")

# Ensure best_estimator_ is fitted
assert hasattr(tuner, 'best_estimator_')
assert tuner.best_estimator_ is not None
print("Verification passed: library works correctly.")

# Clean up
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print(f"Cleaned up {checkpoint_file}")
if os.path.exists("hyperphoenix_results.csv"):
    os.remove("hyperphoenix_results.csv")