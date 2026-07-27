import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from hyperphoenixcv import GPUDeviceAssigned, GPUOutOfMemory, HyperPhoenixCV
from hyperphoenixcv.compute import (
    ComputeConfigurationError, DeviceInventory, GPUDevice, GPUResourceError,
)
from hyperphoenixcv.events import EventPublisher


def make_search(tmp_path, **kwargs):
    return HyperPhoenixCV(
        estimator=LogisticRegression(max_iter=200), search_space={"C": [1.0]},
        scoring="accuracy", cv=2, verbose=False,
        storage_path=str(tmp_path / "gpu.sqlite3"), compute="gpu", **kwargs,
    )


@pytest.mark.parametrize("kwargs, message", [
    ({"gpu_devices": ()}, "exactly one"),
    ({"gpu_devices": (0, 1)}, "exactly one"),
    ({"n_jobs": 2}, "n_jobs=1"),
    ({"gpu_slots_per_device": 2}, "gpu_slots_per_device=1"),
])
def test_g1_rejects_unsafe_or_ambiguous_resource_declaration(tmp_path, kwargs, message):
    with pytest.raises(ComputeConfigurationError, match=message):
        make_search(tmp_path, **kwargs).fit([[0], [1], [2], [3]], [0, 0, 1, 1])


def test_gpu_preflight_failure_happens_before_sqlite_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(DeviceInventory, "discover", classmethod(
        lambda cls: (_ for _ in ()).throw(GPUResourceError("no test GPU"))
    ))
    path = tmp_path / "gpu.sqlite3"
    with pytest.raises(GPUResourceError, match="no test GPU"):
        make_search(tmp_path).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert not path.exists()


def test_gpu_fake_inventory_records_assignment_without_changing_estimator(tmp_path, monkeypatch):
    monkeypatch.setattr(DeviceInventory, "discover", classmethod(
        lambda cls: (GPUDevice(0, "GPU-test", "Fake NVIDIA"),)
    ))
    events = []
    X, y = make_classification(n_samples=30, n_features=4, random_state=0)
    search = make_search(tmp_path, callbacks=[events.append]).fit(X, y)

    assert search.estimator.get_params()["C"] == 1.0
    assert search.gpu_assignment_.device_uuid == "GPU-test"
    assert search.compute_diagnostics_["estimator_gpu_verification"] == "not_attempted"
    assert any(isinstance(event, GPUDeviceAssigned) for event in events)
    record = search.trial_history_.page(limit=1)[0]
    assert record["diagnostics"]["gpu_assignment"]["device_uuid"] == "GPU-test"


def test_gpu_resume_identity_excludes_physical_device_id(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=30, n_features=4, random_state=0)
    monkeypatch.setattr(DeviceInventory, "discover", classmethod(
        lambda cls: (GPUDevice(0, "GPU-first", "Fake NVIDIA"),)
    ))
    first = make_search(tmp_path, gpu_devices=(0,)).fit(X, y)
    monkeypatch.setattr(DeviceInventory, "discover", classmethod(
        lambda cls: (GPUDevice(1, "GPU-second", "Fake NVIDIA"),)
    ))
    resumed = make_search(tmp_path, gpu_devices=(1,)).fit(X, y)
    assert resumed.study_id == first.study_id
    assert resumed.gpu_assignment_.device_uuid == "GPU-second"


def test_gpu_oom_emits_typed_diagnostic(tmp_path):
    search = make_search(tmp_path)
    search.gpu_assignment_ = object()
    events = []
    search.study_id = "study"
    search.event_publisher = EventPublisher((events.append,))
    search._on_trial(2, {}, {"error": "CUDA out of memory while fitting"})
    assert isinstance(events[0], GPUOutOfMemory)
