# P2.7 performance gate record

Recorded on 2026-07-27, Linux, Python 3.12.3, 12 logical CPUs. Command:

```bash
python benchmarks/p2_performance_benchmark.py \
  --output benchmarks/p2_baseline.json \
  --profile benchmarks/p2_profile.txt
```

The committed baseline uses five samples. Regression comparison uses medians
and a 15% threshold; three or more samples are mandatory, so one noisy run
never fails the gate.

| Workload | Median wall | Trials/s | Proposal p95 | SQLite commit p95 |
| --- | ---: | ---: | ---: | ---: |
| cheap (~5 ms/trial) | 0.167 s | 48.0 | 0.048 ms | 2.44 ms |
| medium (~30 ms/trial) | 0.390 s | 20.5 | 0.066 ms | 2.48 ms |
| expensive (~150 ms/trial) | 1.394 s | 5.74 | 0.051 ms | 2.33 ms |

Scale checks passed: a 10^9-combination grid counted and sampled its first ten
candidates in 0.29 ms without materialization. Resume restored and skipped
1k/10k/100k durable trials in 0.038/0.353/3.637 s. The P2.3 no-retention
projection processed 100k results while retaining zero result objects.

Parallel comparison with `n_jobs=2`, `inner_max_num_threads=1` showed trial
and fold axes at 2.081 s and 2.056 s respectively for this intentionally small
medium workload. Process startup dominates it; this is not a deployment
recommendation. Benchmark target workload before choosing an axis.

## Profile and decision

`benchmarks/p2_profile.txt` records cProfile for the medium end-to-end run.
The high inclusive paths are sklearn validation/scoring and deliberate trial
sleep; HyperPhoenixCV's `StudyEngine` only coordinates them. The largest
specific HyperPhoenixCV path is SQLite-store construction (0.090 s of 0.826 s,
10.9%), including filesystem detection. It is local SQLite/filesystem I/O, not
a CPU loop suitable for a native extension. Per-trial SQLite commit p95 is
2.48 ms. Direct component measurements record 100k `param_key` calls at
0.721 s, no-retention projection at 0.018 s, and 8 MiB joblib transport at
0.041 s; none is at least 10% of representative search runtime per trial.

No eligible Python-compute hotspot reaches the native gate. No optimization is
implemented: changing storage durability or adding native code would not be a
measured end-to-end win. Future profiling must rerun this command on a
representative estimator/dataset before reopening decision.

## RFC: native alternatives considered

1. Keep Python: current choice. Cost zero; resume/hash/projection already meet
   gate targets.
2. Vectorize/batch parameter hashing or projection with NumPy: only reconsider
   if an application has large in-memory batches and profile shows >=10%
   end-to-end cost.
3. Native extension (Rust/Cython/C++ batch API): require Python fallback,
   property parity, platform wheels, >=2x hotspot speedup, and >=10%
   representative end-to-end improvement.

Option 1 accepted. Options 2 and 3 remain rejected pending new evidence.
