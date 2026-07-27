# Parallelism and resource tuning

Choose one parallelism axis:

| Setting | Concurrent trials | CV workers per trial |
| --- | ---: | ---: |
| `parallelism="trials"` | `n_jobs` | 1 |
| `parallelism="folds"` | 1 | `n_jobs` |

Nested trial-plus-fold parallelism is rejected to prevent oversubscription.
Start with `n_jobs` equal to physical-core count or lower. For estimators using
BLAS/OpenMP, set `inner_max_num_threads=1` under trial parallelism, then measure
with representative workload.

```python
search = HyperPhoenixCV(
    estimator=model,
    search_space=space,
    strategy="random",
    n_trials=100,
    parallelism="trials",
    n_jobs=4,
    inner_max_num_threads=1,
    pre_dispatch="2*n_jobs",
    memmap_max_nbytes="1M",
)
```

`pre_dispatch`, `memmap_max_nbytes`, `memmap_temp_folder`, and
`joblib_batch_size` tune joblib transport. Place memmap folder on local fast
storage with sufficient capacity. `trial_timeout` requires trial parallelism
with at least two workers; a timeout records terminal failure. Cancellation is
cooperative before unstarted trial/batch, not hard interruption of arbitrary
estimator code.

Measure before changing defaults. See `benchmarks/performance_benchmark.py`
for reproducible workload and resume-latency measurement.
