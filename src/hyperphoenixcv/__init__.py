from .core import HyperPhoenixCV
from .audit import TrialHistory
from .events import (
    ExportFailed, RefitFailed, StudyCompleted, StudyResumed, StudyStarted,
    StudyStopped, TrialCancelled, TrialCompleted, TrialFailed, TrialPruned,
    TrialStarted, GPUDeviceAssigned, GPUResourceFailure, GPUOutOfMemory,
)
from .compute import ComputeSpec, GPUResourceError

__all__ = [
    'HyperPhoenixCV', 'StudyStarted', 'StudyResumed', 'TrialStarted',
    'TrialCompleted', 'TrialFailed', 'TrialPruned', 'TrialCancelled',
    'StudyStopped', 'StudyCompleted', 'ExportFailed', 'RefitFailed', 'TrialHistory',
    'ComputeSpec', 'GPUResourceError', 'GPUDeviceAssigned', 'GPUResourceFailure', 'GPUOutOfMemory',
]
__version__ = '0.6.2'
