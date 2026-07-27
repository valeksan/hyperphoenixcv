# P2.7 performance gate

Run release baseline:

```bash
python benchmarks/p2_performance_benchmark.py --output benchmarks/p2_baseline.json
```

Five runs supply median/p95 values. Comparison flags only median slowdowns over
15%; benchmark exits successfully, so release owner inspects regression report
instead of failing on noisy host measurement. `--runs` rejects fewer than 3.

Fast smoke:

```bash
python benchmarks/p2_performance_benchmark.py --runs 3 --resume-sizes 1000
```

Harness records: wall time, throughput, proposal/store p50+p95, process peak
RSS (KiB on Linux), serialized SQLite JSON bytes, CPU utilization, 10^9-space
laziness, 10^3/10^4/10^5 resume latency, bounded projection check, cProfile.

Taxonomy uses deterministic sleeping estimator: cheap/medium/expensive trials
target about 5/30/150 ms at two folds. It measures coordination cost without
machine-specific model convergence noise. Harness compares
`parallelism="trials"` vs `"folds"` using `n_jobs=2`,
`inner_max_num_threads=1`. Never enable both axes; alter config only when
benchmarking stated deployment target.

Profile report is evidence for native decision. Record/profile review still
required before accepting an RFC or declaring native extension unnecessary.
