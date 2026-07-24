"""
Example of HyperPhoenixCV with Bayesian optimization for smart hyperparameter tuning.

This example demonstrates:
- Using Bayesian optimization to prioritize promising parameters
- How the algorithm learns from previous iterations
- Comparison with random search and full grid search
- Getting results from Bayesian-optimized search
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from hyperphoenixcv import HyperPhoenixCV
from sklearn.ensemble import RandomForestRegressor

# Load dataset
print("Loading data...")
categories = ['alt.atheism', 'sci.space', 'comp.graphics', 'rec.sport.baseball']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
X, y = newsgroups_train.data, newsgroups_train.target

# Create a pipeline
print("Creating pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Define parameter grid
param_grid = {
    'tfidf__max_features': [500, 1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0],
    'clf__penalty': ['l1', 'l2']
}

# Custom Bayesian optimizer with more trees for better predictions
custom_optimizer = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

# Create HyperPhoenixCV with Bayesian optimization
print("\nConfiguring HyperPhoenixCV with Bayesian optimization...")
hp_bayesian = HyperPhoenixCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1,
    use_bayesian_optimization=True,
    bayesian_optimizer=custom_optimizer,
    checkpoint_path="bayesian_checkpoint.sqlite3",
    dataset_id="20newsgroups-bayesian-train-v1",
    results_csv="bayesian_results.csv",
    verbose=True
)

# Run Bayesian-optimized hyperparameter search
print("\n" + "="*60)
print("RUNNING SEARCH WITH BAYESIAN OPTIMIZATION")
print("Bayesian optimization analyzes previous results")
print("and predicts which parameters may yield better results")
print("="*60)
hp_bayesian.fit(X, y)

# Get results
bayesian_best_score = hp_bayesian.best_score_
bayesian_top_results = hp_bayesian.get_top_results(5)

print("\n" + "="*50)
print("BAYESIAN OPTIMIZATION RESULTS")
print("="*50)
print(f"Best f1_macro score: {bayesian_best_score:.4f}")
print("\nTop-5 parameter combinations:")
print(bayesian_top_results[['tfidf__max_features', 'tfidf__ngram_range', 
                           'clf__C', 'clf__penalty', 'mean_test_f1_macro']])

# For comparison, let's run random search with the same number of iterations
print("\n" + "="*50)
print("RUNNING RANDOM SEARCH FOR COMPARISON")
print(f"Will perform {len(hp_bayesian.cv_results_['params'])} iterations")
print("="*50)

hp_random = HyperPhoenixCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1,
    random_search=True,
    n_iter=len(hp_bayesian.cv_results_['params']),  # Same number of iterations
    random_state=42,
    verbose=True
)
hp_random.fit(X, y)

# Compare results
random_best_score = hp_random.best_score_

print("\n" + "="*50)
print("RESULTS COMPARISON")
print("="*50)
print(f"Bayesian optimization: {bayesian_best_score:.4f}")
print(f"Random search: {random_best_score:.4f}")

if bayesian_best_score > random_best_score:
    print("✅ Bayesian optimization outperformed random search!")
    print(f"  Improvement: {(bayesian_best_score - random_best_score) * 100:.2f} percentage points")
else:
    print("⚠️ Random search performed better in this run")
    print("  This can happen in early stages of optimization")

# Visualize the optimization process
print("\nCreating optimization progress plot...")
try:
    # Get scores in order of evaluation
    bayesian_scores = [r[f'mean_test_f1_macro'] for r in hp_bayesian.cv_results_['params']]
    random_scores = [r[f'mean_test_f1_macro'] for r in hp_random.cv_results_['params']]
    
    # Calculate cumulative best score at each step
    bayesian_cummax = np.maximum.accumulate(bayesian_scores)
    random_cummax = np.maximum.accumulate(random_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(bayesian_cummax, 'b-', label='Bayesian optimization', linewidth=2)
    plt.plot(random_cummax, 'r--', label='Random search', linewidth=2)
    
    plt.xlabel('Number of evaluated combinations')
    plt.ylabel('Best F1-macro score')
    plt.title('Hyperparameter search progress')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'optimization_progress.png'")
    
    # Show plot in notebook environment (optional)
    try:
        import IPython
        if IPython.get_ipython():
            plt.show()
    except:
        pass
        
except Exception as e:
    print(f"⚠️ Failed to create plot: {e}")

# Insights and recommendations
print("\n" + "="*50)
print("INSIGHTS AND RECOMMENDATIONS")
print("="*50)
print("How Bayesian optimization works in HyperPhoenixCV:")
print("1. Explores parameter space in early iterations")
print("2. Builds a model of parameter-metric relationship as data accumulates")
print("3. Uses this model to select the most promising parameters")
print("4. Over time focuses on the most promising regions of the space")

print("\nUsage recommendations:")
print("- Use Bayesian optimization when parameter space is large")
print("- For small parameter spaces, exhaustive search may suffice")
print("- Combine with checkpoints to resume search after interruptions")
print("- Customize bayesian_optimizer for your tasks (number of trees, etc.)")

# Clean up checkpoints
hp_bayesian.clear_checkpoint_file()
print("\nBayesian optimization SQLite study store successfully deleted.")

# Tip for users
print("\nTip: Bayesian optimization is especially effective when evaluating a single")
print("parameter combination takes a long time (e.g., training deep models).")
print("In such cases, saving even a few iterations can save hours of computation!")
