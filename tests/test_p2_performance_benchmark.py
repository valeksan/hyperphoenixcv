"""Fast invariant coverage for P2.7 harness; full benchmark stays out of CI."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


BENCHMARK = Path(__file__).parents[1] / "benchmarks" / "p2_performance_benchmark.py"
spec = spec_from_file_location("p2_performance_benchmark", BENCHMARK)
assert spec and spec.loader
benchmark = module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_large_space_is_counted_and_sampled_without_materialization():
    result = benchmark.measure_large_space()
    assert result["candidate_count"] == 1_000_000_000
    assert result["first_ten_seconds"] >= 0


def test_workload_taxonomy_has_increasing_trial_costs():
    durations = [benchmark.WORKLOADS[name]["delay"] for name in ("cheap", "medium", "expensive")]
    assert durations == sorted(durations)


def test_resume_latency_skips_terminal_trials():
    result = benchmark.measure_resume_latency(10)
    assert result["trials"] == 10
    assert result["resume_seconds"] >= 0


def test_regression_comparison_uses_median_and_threshold():
    values = {key: {"median": 1.0, "p95": 1.0} for key in benchmark.REGRESSION_METRICS}
    baseline = {"workloads": {name: values for name in benchmark.WORKLOADS}}
    report = {"workloads": {name: values for name in benchmark.WORKLOADS}}
    assert benchmark.compare(baseline, report, 0.15)["regressions"] == {}
