"""
Example of HyperPhoenixCV with random search for efficient hyperparameter tuning.

This example demonstrates:
- Using random search instead of full grid search
- Setting n_iter to control the number of combinations
- How random search can find good parameters faster than full grid search
- Getting results from random search
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from hyperphoenixcv import HyperPhoenixCV

# Load dataset
print("Loading data...")
categories = ['alt.atheism', 'comp.graphics']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
X, y = newsgroups_train.data, newsgroups_train.target

# Create a pipeline
print("Creating pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000, solver='saga', penalty='l1'))
])

# Define a larger parameter grid to demonstrate the advantage of random search
param_grid = {
    'tfidf__max_features': [500, 1000, 5000, 10000, 15000, 20000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__use_idf': [True, False],
    'tfidf__smooth_idf': [True, False],
    'tfidf__sublinear_tf': [True, False],
    'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear', 'saga']
}

# Calculate total combinations to show the advantage of random search
total_combinations = 1
for v in param_grid.values():
    total_combinations *= len(v)
print(f"\nTotal possible combinations: {total_combinations}")
print("Exhaustive search would take too long!")
print("Random search will test only a small fraction of them.\n")

# Create HyperPhoenixCV with random search
print("Configuring HyperPhoenixCV with random search...")
hp = HyperPhoenixCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    random_search=True,  # Enable random search
    n_iter=50,           # Number of random combinations to test
    random_state=42,     # For reproducibility
    checkpoint_path="random_search_checkpoint.sqlite3",
    dataset_id="20newsgroups-atheism-graphics-train-v1",
    results_csv="random_search_results.csv",
    verbose=True
)

# Run hyperparameter search
print("\nStarting random hyperparameter search...")
hp.fit(X, y)

# Print results
print("\n" + "="*50)
print("RANDOM SEARCH RESULTS")
print("="*50)
print(f"Tested {hp.n_iter} random combinations out of {total_combinations} possible")
print(f"That's only {hp.n_iter/total_combinations*100:.4f}% of exhaustive search!")
print("\nBest parameters:", hp.best_params_)
print("Best f1 score:", hp.best_score_)

# Get top 5 results
top_5 = hp.get_top_results(5)
print("\nTop-5 results:")
print(top_5)

# Compare with theoretical full grid search time
estimated_full_grid_time = hp.n_iter / total_combinations * 100 * 2  # Assume 2 minutes per combination
if estimated_full_grid_time > 60:
    hours = estimated_full_grid_time / 60
    time_str = f"{hours:.1f} hours"
else:
    time_str = f"{estimated_full_grid_time:.1f} minutes"

print("\n" + "="*50)
print(f"TIME SAVINGS")
print("="*50)
print(f"Exhaustive search of all combinations would take approximately {time_str}")
print(f"Random search completed in {hp.n_iter} combinations and found good parameters!")
print("="*50)

# Clean up checkpoints after successful run
hp.clear_checkpoint_file()
print("\nSQLite study store successfully deleted.")

# Tip for users
print("\nTip: For very large parameter spaces, start with random search,")
print("then use the found best parameters as a basis for more detailed search.")
