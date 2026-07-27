"""Local compute-resource declarations and NVIDIA preflight helpers.

Core deliberately does not import CUDA, NVML, or estimator frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess


class ComputeConfigurationError(ValueError):
    """Invalid CPU/GPU resource declaration."""


class GPUResourceError(RuntimeError):
    """Requested local GPU resource is unavailable."""


@dataclass(frozen=True)
class ComputeSpec:
    """Immutable resource request. Hardware is inspected only at fit time."""

    compute: str = "cpu"
    gpu_devices: tuple[int | str, ...] = (0,)
    gpu_slots_per_device: int = 1

    def validate(self, *, n_jobs: int, parallelism: str) -> None:
        if self.compute not in {"cpu", "gpu"}:
            raise ComputeConfigurationError("compute must be 'cpu' or 'gpu'")
        if not isinstance(self.gpu_devices, tuple):
            raise ComputeConfigurationError("gpu_devices must be a tuple")
        if not isinstance(self.gpu_slots_per_device, int) or self.gpu_slots_per_device < 1:
            raise ComputeConfigurationError("gpu_slots_per_device must be a positive integer")
        if self.compute == "cpu":
            return
        if len(self.gpu_devices) != 1:
            raise ComputeConfigurationError("compute='gpu' requires exactly one gpu_devices entry")
        if len(set(self.gpu_devices)) != len(self.gpu_devices):
            raise ComputeConfigurationError("gpu_devices must not contain duplicates")
        if self.gpu_slots_per_device != 1:
            raise ComputeConfigurationError("G1 supports gpu_slots_per_device=1 only")
        if n_jobs != 1:
            raise ComputeConfigurationError("G1 GPU execution requires n_jobs=1")
        if parallelism not in {"trials", "folds"}:
            # Scheduler provides fuller API error; keep ComputeSpec independently safe.
            raise ComputeConfigurationError("parallelism must be 'trials' or 'folds'")

    def identity_config(self) -> dict[str, object]:
        """Portable identity: physical device IDs are intentionally excluded."""
        if self.compute == "cpu":
            return {}
        return {
            "compute": self.compute,
            "gpu_device_count": len(self.gpu_devices),
            "gpu_slots_per_device": self.gpu_slots_per_device,
        }


@dataclass(frozen=True)
class GPUDevice:
    index: int
    uuid: str
    name: str

    def matches(self, requested: int | str) -> bool:
        return requested == self.index or requested == self.uuid


@dataclass(frozen=True)
class GPUAssignment:
    requested_device: int | str
    device_index: int
    device_uuid: str
    device_name: str

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_device": self.requested_device,
            "device_index": self.device_index,
            "device_uuid": self.device_uuid,
            "device_name": self.device_name,
        }


class DeviceInventory:
    """NVIDIA inventory through nvidia-smi; safe fake boundary for CPU-only tests."""

    @classmethod
    def discover(cls) -> tuple[GPUDevice, ...]:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            raise GPUResourceError("NVIDIA GPU preflight failed: nvidia-smi is unavailable")
        try:
            output = subprocess.check_output(
                [executable, "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                text=True, stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise GPUResourceError("NVIDIA GPU preflight failed: nvidia-smi could not query devices") from exc
        devices = []
        for line in output.splitlines():
            fields = [item.strip() for item in line.split(",", 2)]
            if len(fields) != 3:
                raise GPUResourceError("NVIDIA GPU preflight failed: malformed nvidia-smi output")
            try:
                devices.append(GPUDevice(index=int(fields[0]), uuid=fields[1], name=fields[2]))
            except ValueError as exc:
                raise GPUResourceError("NVIDIA GPU preflight failed: malformed device index") from exc
        if not devices:
            raise GPUResourceError("NVIDIA GPU preflight failed: no devices found")
        return tuple(devices)


def preflight_gpu(spec: ComputeSpec) -> GPUAssignment | None:
    """Resolve requested G1 device without changing CUDA visibility/env."""
    if spec.compute == "cpu":
        return None
    requested = spec.gpu_devices[0]
    for device in DeviceInventory.discover():
        if device.matches(requested):
            return GPUAssignment(requested, device.index, device.uuid, device.name)
    raise GPUResourceError(f"NVIDIA GPU preflight failed: requested device {requested!r} was not found")
