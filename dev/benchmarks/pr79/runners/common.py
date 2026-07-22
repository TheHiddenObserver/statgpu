"""Shared utilities for PR79 benchmark runners.

Provides:
- Case identity (case_id, method_config_id from canonical JSON hashing)
- Timing with GPU synchronization
- Result structuring (raw JSON format)
- Environment recording
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def make_case_id(canonical: Dict[str, Any]) -> str:
    """Generate a stable case_id from canonical case parameters."""
    raw = json.dumps(canonical, sort_keys=True, default=str)
    return "case-" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def make_method_config_id(canonical: Dict[str, Any]) -> str:
    """Generate a stable method_config_id from method parameters."""
    raw = json.dumps(canonical, sort_keys=True, default=str)
    return "method-" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def synchronized_time(fn, *args, _sync_before=True, _sync_after=True, **kwargs):
    """Time a function call with GPU synchronization."""
    _sync()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    _sync()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def _sync():
    """Synchronize all GPU streams."""
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def record_environment() -> Dict[str, Any]:
    """Record current environment for reproducibility."""
    info: Dict[str, Any] = {
        "python_version": "",
        "numpy_version": np.__version__,
    }
    import platform
    info["python_version"] = platform.python_version()

    try:
        import statgpu
        info["statgpu_path"] = statgpu.__file__
    except ImportError:
        info["statgpu_path"] = None

    try:
        import cupy as cp
        info["cupy"] = {
            "version": cp.__version__,
            "devices": int(cp.cuda.runtime.getDeviceCount()),
        }
    except ImportError:
        info["cupy"] = {"available": False}

    try:
        import torch
        info["torch"] = {
            "version": torch.__version__,
            "cuda": torch.cuda.is_available(),
        }
    except ImportError:
        info["torch"] = {"available": False}

    try:
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass

    try:
        import statsmodels
        info["statsmodels_version"] = statsmodels.__version__
    except ImportError:
        pass

    return info


def make_raw_run(
    run_key: str,
    case_id: str,
    method_config_id: str,
    model_id: str,
    framework: str,
    backend: str,
    parameters: Dict[str, Any],
    timing: Dict[str, float],
    results: Dict[str, Any],
    status: str = "success",
    error: Optional[str] = None,
    resources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a raw run record in PR79 benchmark source format."""
    return {
        "run_key": run_key,
        "case_id": case_id,
        "method_config_id": method_config_id,
        "model_id": model_id,
        "framework": framework,
        "backend": backend,
        "parameters": parameters,
        "status": status,
        "timing": timing,
        "results": results,
        "resources": resources or {},
        "error": error,
    }


def safe_run(fn, *args, **kwargs) -> Tuple[Any, Optional[str]]:
    """Run a function and return (result, error_string)."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
