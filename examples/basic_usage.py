"""
Basic usage of HyperPhoenixCV for hyperparameter tuning with checkpointing.

This example demonstrates:
- Setting up a basic pipeline
- Defining a parameter grid
- Running HyperPhoenixCV with checkpointing
- Getting the best parameters and score
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from hyperphoenixcv import HyperPhoenixCV

# Load dataset
print("Loading data...")
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
X, y = newsgroups_train.data, newsgroups_train.target

# Create a pipeline
print("Creating pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
])

# Define parameter grid
param_grid = {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.001, 0.01, 0.1, 1.0],
    'clf__penalty': ['l1', 'l2']
}

# Create HyperPhoenixCV
print("Configuring HyperPhoenixCV...")
hp = HyperPhoenixCV(
    estimator=pipeline,
    search_space=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    storage_path="text_classification_checkpoint.sqlite3",
    dataset_id="20newsgroups-atheism-christian-train-v1",
    results_csv="text_classification_results.csv",
    verbose=True
)

# Run hyperparameter search
print("\nStarting hyperparameter search...")
hp.fit(X, y)

# Print results
print("\nBest parameters:", hp.best_params_)
print("Best f1 score:", hp.best_score_)

# Get top 5 results
top_5 = hp.get_top_results(5)
print("\nTop-5 results:")
print(top_5)

# Clean up checkpoints after successful run
hp.clear_storage()
print("\nSQLite study store successfully deleted.")
