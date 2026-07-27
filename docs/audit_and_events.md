# Audit exports and runtime events

`trial_history_` is read-only, SQLite-backed audit projection. It contains
completed, failed, pruned, and cancelled terminal trials, including diagnostics,
exceptions, objectives, and intermediate reports.

```python
history = search.trial_history_
for record in history.iter_records(page_size=1000):
    process(record)

history.export_json("audit.json")       # lossless tagged JSON
history.export_csv("audit.csv")         # flat convenience projection
history.export_parquet("audit.parquet") # requires hyperphoenixcv[parquet]
```

Use `page(offset=0, limit=100)` or `count(states={"failed"})` for large
studies. CSV and `results_csv` are not lossless: nested diagnostics are
flattened/stringified. JSON is durable audit export. Export writes temporary
file, fsyncs, then atomically replaces destination.

Callbacks receive typed runtime events after terminal commit:

```python
from hyperphoenixcv import TrialCompleted, StudyCompleted

def observe(event):
    if isinstance(event, TrialCompleted):
        metrics.publish("trials.completed", 1)
    elif isinstance(event, StudyCompleted):
        metrics.publish("studies.completed", 1)

search = HyperPhoenixCV(..., callbacks=[observe], verbose=True)
```

Events are synchronous, coordinator-local, runtime-only, and never replayed
on resume. Callback exception fails `fit` by design. `verbose=True` emits
standard logging events; configure handlers/levels in application. Default
events/logs omit raw datasets, parameters, and tracebacks. Treat callback code
as trusted in-process code.
