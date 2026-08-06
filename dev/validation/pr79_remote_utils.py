#!/usr/bin/env python3
"""
PR79 GPU Validation - Remote Utilities.

This module is uploaded to the remote GPU server and provides
utility functions for environment recording, CUDA probing,
memory tracking, and structured result writing.

It is self-contained and does NOT import statgpu (to avoid
depending on which worktree is active). It uses only standard
library + numpy/scipy/cupy/torch/sklearn which are in myconda.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Environment Recording (Section 6)
# ============================================================================


def record_environment(result_dir: str) -> Dict[str, Any]:
    """Record full environment state. Returns a dict of all captured data.

    Parameters
    ----------
    result_dir : str
        Path to the environment/ subdirectory in results.

    Returns
    -------
    dict with keys: git_sha, git_status, python_version, packages,
                    nvidia_smi, cuda_probe
    """
    os.makedirs(result_dir, exist_ok=True)
    env_data = {}

    def _run(cmd, timeout=60):
        return subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout)

    def _save(filename, content):
        with open(os.path.join(result_dir, filename), "w") as f:
            f.write(content)

    # Git
    r = _run("git rev-parse HEAD")
    env_data["git_sha"] = r.stdout.strip()
    _save("git_sha.txt", r.stdout)

    r = _run("git status --short")
    env_data["git_dirty"] = bool(r.stdout.strip())
    _save("git_status.txt", r.stdout)

    # Python
    r = _run("python -V")
    env_data["python_version"] = r.stdout.strip()
    _save("python_version.txt", r.stdout)

    # Packages
    r = _run("python -m pip list --format=json", timeout=120)
    try:
        env_data["packages"] = json.loads(r.stdout)
    except json.JSONDecodeError:
        r2 = _run("python -m pip freeze", timeout=120)
        env_data["packages_raw"] = r2.stdout
    _save("pip_freeze.txt", r.stdout if r.stdout else r.stderr)

    r = _run("python -m pip check", timeout=60)
    env_data["pip_check"] = r.stdout.strip()
    _save("pip_check.txt", r.stdout)

    # NVIDIA
    r = _run("nvidia-smi -q", timeout=30)
    _save("nvidia_smi_q.txt", r.stdout)

    r = _run("nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader", timeout=30)
    env_data["gpu_summary"] = r.stdout.strip()
    _save("gpu_summary.csv", r.stdout)

    r = _run("nvcc --version 2>/dev/null || echo 'nvcc not found'", timeout=30)
    _save("nvcc_version.txt", r.stdout)

    # CUDA probe
    cuda = probe_cuda()
    env_data["cuda_probe"] = cuda
    with open(os.path.join(result_dir, "backend_probe.json"), "w") as f:
        json.dump(cuda, f, indent=2, default=str)

    # statgpu import
    try:
        import statgpu
        env_data["statgpu_path"] = statgpu.__file__
    except Exception as e:
        env_data["statgpu_path"] = f"ERROR: {e}"

    return env_data


def probe_cuda() -> Dict[str, Any]:
    """Run the minimum CUDA probe from Section 6.1 of the test plan.

    Returns a dict with CuPy and Torch CUDA status.
    """
    x_np = np.arange(16, dtype=np.float64).reshape(4, 4)
    probe = {}

    # CuPy probe
    try:
        import cupy as cp
        x_cp = cp.asarray(x_np)
        cp.cuda.Stream.null.synchronize()
        x_back = cp.asnumpy(x_cp)
        cupy_ok = bool(np.array_equal(x_back, x_np))
        probe["cupy"] = {
            "available": True,
            "version": cp.__version__,
            "device_count": int(cp.cuda.runtime.getDeviceCount()),
            "current_device": int(cp.cuda.runtime.getDevice()),
            "float64_roundtrip": cupy_ok,
        }
    except Exception as e:
        probe["cupy"] = {
            "available": False,
            "error": str(e),
        }

    # Torch probe
    try:
        import torch
        x_t = torch.as_tensor(x_np, device="cuda")
        torch.cuda.synchronize()
        x_t_back = x_t.cpu().numpy()
        torch_ok = bool(np.array_equal(x_t_back, x_np))
        probe["torch"] = {
            "available": torch.cuda.is_available(),
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else -1,
            "float64_roundtrip": torch_ok,
        }
    except Exception as e:
        probe["torch"] = {
            "available": False,
            "error": str(e),
        }

    # NumPy always available
    probe["numpy"] = {"version": np.__version__}

    # SciPy
    try:
        import scipy
        probe["scipy"] = {"version": scipy.__version__}
    except ImportError:
        probe["scipy"] = {"version": None}

    # sklearn
    try:
        import sklearn
        probe["sklearn"] = {"version": sklearn.__version__}
    except ImportError:
        probe["sklearn"] = {"version": None}

    # statsmodels
    try:
        import statsmodels
        probe["statsmodels"] = {"version": statsmodels.__version__}
    except ImportError:
        probe["statsmodels"] = {"version": None}

    return probe


# ============================================================================
# Memory Tracking (Section 13)
# ============================================================================


def record_memory_cupy() -> Dict[str, int]:
    """Record CuPy memory pool state."""
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        pinned = cp.get_default_pinned_memory_pool()
        free, total = cp.cuda.runtime.memGetInfo()
        return {
            "used_bytes": pool.used_bytes(),
            "total_bytes": pool.total_bytes(),
            "pinned_free_blocks": pinned.n_free_blocks(),
            "device_free_bytes": free,
            "device_total_bytes": total,
        }
    except Exception:
        return {}


def record_memory_torch() -> Dict[str, int]:
    """Record Torch CUDA memory state."""
    try:
        import torch
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
        }
    except Exception:
        return {}


def sync_all():
    """Synchronize both CuPy and Torch CUDA streams."""
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    try:
        import torch
        torch.cuda.synchronize()
    except Exception:
        pass


def reset_memory_tracking():
    """Reset memory tracking counters."""
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
    except Exception:
        pass


def measure_fit_memory(estimator, X, y, n_repeats=20, n_warmup=2):
    # type: (Any, Any, Any, int, int) -> List[Dict[str, Any]]
    """Run repeated fit cycles and record memory after each.

    Section 13: 20+ repeated fits with memory tracking.
    """
    import cupy as cp
    import torch

    records = []

    for i in range(n_warmup + n_repeats):
        sync_all()
        reset_memory_tracking()

        # Fit
        estimator.fit(X, y)

        sync_all()
        mem_cupy = record_memory_cupy()
        mem_torch = record_memory_torch()

        # Predict / transform if available
        try:
            if hasattr(estimator, "predict"):
                estimator.predict(X)
        except Exception:
            pass

        sync_all()

        records.append({
            "iteration": i,
            "phase": "warmup" if i < n_warmup else "measured",
            "cupy": mem_cupy,
            "torch": mem_torch,
        })

        # Cleanup
        del estimator
        gc.collect()
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return records


# ============================================================================
# Timing Utilities (Section 14)
# ============================================================================


def measure_fit_time(estimator_factory, X, y, n_warmup=2, n_measured=5):
    # type: (callable, Any, Any, int, int) -> Dict[str, Any]
    """Measure fit timing with warmup and synchronization.

    Section 14: warmup 2x, measure 5x, report median/IQR/min/max.
    """
    import cupy as cp
    import torch

    times = []

    for i in range(n_warmup + n_measured):
        est = estimator_factory()  # Fresh estimator each iteration

        sync_all()
        t0 = time.perf_counter()
        est.fit(X, y)
        sync_all()
        elapsed = time.perf_counter() - t0

        if i >= n_warmup:
            times.append(elapsed)

    times = sorted(times)
    n = len(times)
    median = times[n // 2]
    q1 = times[n // 4] if n >= 4 else times[0]
    q3 = times[3 * n // 4] if n >= 4 else times[-1]

    return {
        "median_s": round(median, 6),
        "iqr_s": round(q3 - q1, 6),
        "min_s": round(times[0], 6),
        "max_s": round(times[-1], 6),
        "n_measured": n,
        "times_s": [round(t, 6) for t in times],
    }


def compare_base_head(base_result, head_result):
    # type: (Dict, Dict) -> Dict[str, Any]
    """Compare base vs head timing results.

    Reports relative change and flags regressions.
    """
    comparison = {}

    for key in base_result:
        if key in head_result and isinstance(base_result[key], (int, float)):
            base_val = base_result[key]
            head_val = head_result[key]
            if base_val > 0:
                ratio = head_val / base_val
                comparison[key] = {
                    "base": base_val,
                    "head": head_val,
                    "ratio": round(ratio, 4),
                    "regression": ratio > 1.2,  # >20% slower
                }

    return comparison


# ============================================================================
# Device Purity Audit (Section 12)
# ============================================================================


def audit_host_transfers(estimator, X, y):
    # type: (Any, Any, Any) -> Dict[str, Any]
    """Dynamically audit host transfers during fit/predict.

    Monkeypatches cp.asnumpy, torch.Tensor.cpu, and backend to_numpy
    to record every transfer. Reports total bytes transferred and
    whether any full-array transfers occurred.

    Section 12.2: Dynamic transfer audit.
    """
    import cupy as cp
    import torch

    transfers = []
    original_funcs = {}

    def _record_transfer(name, shape, dtype_str, trace_lines):
        """Record a transfer event with call stack."""
        try:
            size_bytes = int(np.prod(shape)) * np.dtype(dtype_str).itemsize
        except Exception:
            size_bytes = 0
        transfers.append({
            "name": name,
            "shape": list(shape) if hasattr(shape, "__iter__") else str(shape),
            "dtype": dtype_str,
            "size_bytes": size_bytes,
            "call_stack": trace_lines,
        })

    # Patch cp.asnumpy
    try:
        original_asnumpy = cp.asnumpy
        def _patched_asnumpy(arr, *args, **kwargs):
            result = original_asnumpy(arr, *args, **kwargs)
            if hasattr(arr, "shape") and hasattr(arr, "dtype"):
                tb = traceback.extract_stack()[:-1]
                trace_lines = [f"{f.filename}:{f.lineno} in {f.name}" for f in tb[-6:]]
                _record_transfer("cp.asnumpy", arr.shape, str(arr.dtype), trace_lines)
            return result
        cp.asnumpy = _patched_asnumpy
        original_funcs["cp.asnumpy"] = original_asnumpy
    except Exception:
        pass

    # Patch torch.Tensor.cpu
    try:
        original_cpu = torch.Tensor.cpu
        def _patched_cpu(self, *args, **kwargs):
            result = original_cpu(self, *args, **kwargs)
            if hasattr(self, "shape") and hasattr(self, "dtype"):
                tb = traceback.extract_stack()[:-2]
                trace_lines = [f"{f.filename}:{f.lineno} in {f.name}" for f in tb[-6:]]
                _record_transfer("tensor.cpu()", self.shape, str(self.dtype), trace_lines)
            return result
        torch.Tensor.cpu = _patched_cpu
        original_funcs["torch.Tensor.cpu"] = original_cpu
    except Exception:
        pass

    # Run fit and predict
    try:
        estimator.fit(X, y)
    except Exception as e:
        transfers.append({"error_fit": str(e)})

    try:
        if hasattr(estimator, "predict"):
            estimator.predict(X)
    except Exception:
        pass

    # Restore original functions
    for name, func in original_funcs.items():
        if name == "cp.asnumpy":
            cp.asnumpy = func
        elif name == "torch.Tensor.cpu":
            torch.Tensor.cpu = func

    # Summarize
    total_bytes = sum(t["size_bytes"] for t in transfers if "size_bytes" in t)
    large_transfers = [t for t in transfers if t.get("size_bytes", 0) > 1024]  # > 1KB

    return {
        "total_transfers": len(transfers),
        "total_bytes": total_bytes,
        "large_transfers": len(large_transfers),
        "details": transfers,
    }


# ============================================================================
# Parity Matrix (Section 9)
# ============================================================================


def parity_matrix(model_class, params, X, y, backends=None):
    # type: (type, Dict, Any, Any, List[str]) -> Dict[str, Any]
    """Fit a model on numpy/cupy/torch and compare results.

    Parameters
    ----------
    model_class : type
        A statgpu estimator class.
    params : dict
        Parameters to pass to the constructor.
    X, y : ndarray
        NumPy arrays (will be converted to CuPy/Torch as needed).
    backends : list of str
        Which backends to test. Default: ["numpy", "cupy", "torch"].

    Returns
    -------
    dict with per-backend results and comparison.
    """
    import cupy as cp

    backends = backends or ["numpy", "cupy", "torch"]
    results = {}

    for backend in backends:
        X_b, y_b = X, y
        est_params = dict(params)

        if backend == "cupy":
            X_b = cp.asarray(X)
            y_b = cp.asarray(y)
            est_params["device"] = "cuda"
        elif backend == "torch":
            import torch
            X_b = torch.as_tensor(X, device="cuda")
            y_b = torch.as_tensor(y, device="cuda")
            est_params["device"] = "torch"

        try:
            est = model_class(**est_params)
            est.fit(X_b, y_b)

            # Extract results
            coef = _to_numpy_safe(getattr(est, "coef_", None))
            intercept = _to_numpy_safe(getattr(est, "intercept_", None))
            pred = _to_numpy_safe(est.predict(X_b)) if hasattr(est, "predict") else None

            results[backend] = {
                "success": True,
                "coef": coef.tolist() if coef is not None else None,
                "intercept": float(intercept) if intercept is not None else None,
                "prediction_mean": float(np.mean(pred)) if pred is not None else None,
                "n_iter_": int(getattr(est, "n_iter_", -1)),
            }
        except Exception as e:
            results[backend] = {
                "success": False,
                "error": str(e),
            }

    # Compare against numpy baseline
    comparison = {}
    if "numpy" in results and results["numpy"]["success"]:
        ref = results["numpy"]
        for backend in ["cupy", "torch"]:
            if backend in results and results[backend]["success"]:
                b = results[backend]
                diffs = {}
                if ref["coef"] is not None and b["coef"] is not None:
                    coef_diff = np.max(np.abs(
                        np.array(ref["coef"]) - np.array(b["coef"])
                    ))
                    diffs["coef_max_abs_diff"] = float(coef_diff)
                if ref["prediction_mean"] is not None and b["prediction_mean"] is not None:
                    diffs["pred_mean_abs_diff"] = abs(
                        ref["prediction_mean"] - b["prediction_mean"]
                    )
                comparison[backend] = diffs

    return {
        "model": model_class.__name__,
        "params": params,
        "backend_results": results,
        "comparison_vs_numpy": comparison,
    }


def _to_numpy_safe(arr):
    """Safely convert any backend array to numpy."""
    if arr is None:
        return None
    try:
        import cupy as cp
        if hasattr(arr, "get"):
            return cp.asnumpy(arr)
    except Exception:
        pass
    try:
        if hasattr(arr, "cpu") and hasattr(arr, "numpy"):
            return arr.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(arr)


# ============================================================================
# Result Writer
# ============================================================================


class ResultWriter:
    """Context manager for writing structured JSON results to a subdirectory."""

    def __init__(self, result_dir, subdir):
        self.dir = os.path.join(result_dir, subdir)
        self.files = {}

    def __enter__(self):
        os.makedirs(self.dir, exist_ok=True)
        return self

    def write(self, filename, data):
        """Write JSON data to a file in the result subdirectory."""
        path = os.path.join(self.dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        self.files[filename] = path
        print(f"  Wrote: {path}")

    def __exit__(self, *args):
        pass


# ============================================================================
# Main (for standalone testing on remote)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PR79 Remote Utilities")
    parser.add_argument("--probe", action="store_true",
                        help="Run CUDA probe and print JSON")
    parser.add_argument("--env", type=str, default=None,
                        help="Record environment to specified directory")

    args = parser.parse_args()

    if args.probe:
        result = probe_cuda()
        print(json.dumps(result, indent=2, default=str))

    if args.env:
        result = record_environment(args.env)
        print(json.dumps(result, indent=2, default=str))
