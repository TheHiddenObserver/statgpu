"""Distribution backend precision + performance benchmark.

Covers:
  - Precision: 15 distributions x all methods vs scipy.stats reference
  - Performance: 3 scales (10K, 1M, 100M) x 3 backends (numpy/cupy/torch)
  - Remote execution via paramiko on GPU server

Usage (local)::

    python dev/benchmarks/benchmark_distributions.py [--output results/dist_bench.json]

Usage (remote)::

    python dev/benchmarks/benchmark_distributions.py --remote --output results/dist_bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tarfile
import io
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Bypass statgpu.__init__ (which eagerly imports subpackages with optional
# deps like joblib).  Load only the modules we actually need.
def _load_bare_module(package_path, module_name):
    """Load a module from a file path without triggering __init__.py chains."""
    import importlib.util as ilu
    import sys
    spec = ilu.spec_from_file_location(module_name, str(package_path))
    assert spec is not None and spec.loader is not None
    mod = ilu.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bootstrap_statgpu_packages():
    """Inject minimal statgpu package stubs so internal imports work."""
    import sys
    from pathlib import Path
    statgpu_pkg = ROOT / "statgpu"

    # Inject statgpu as a bare package (no __init__ execution)
    if "statgpu" not in sys.modules:
        import types
        sys.modules["statgpu"] = types.ModuleType("statgpu")
        sys.modules["statgpu"].__path__ = [str(statgpu_pkg)]
        sys.modules["statgpu"].__package__ = "statgpu"

    # Inject statgpu.backends (from _base.py + _utils.py)
    if "statgpu.backends" not in sys.modules:
        _base_mod = _load_bare_module(statgpu_pkg / "backends" / "_base.py", "statgpu.backends._base")
        _utils_mod = _load_bare_module(statgpu_pkg / "backends" / "_utils.py", "statgpu.backends._utils")
        sys.modules["statgpu.backends"] = types.ModuleType("statgpu.backends")
        sys.modules["statgpu.backends"].__path__ = [str(statgpu_pkg / "backends")]
        sys.modules["statgpu.backends"].__package__ = "statgpu.backends"
        for _attr in dir(_base_mod):
            if not _attr.startswith("_") or _attr in ("_is_cupy_array", "_is_torch_array", "_resolve_backend"):
                setattr(sys.modules["statgpu.backends"], _attr, getattr(_base_mod, _attr))
        # Inject _get_torch_device_str from _utils.py
        if hasattr(_utils_mod, "_get_torch_device_str"):
            setattr(sys.modules["statgpu.backends"], "_get_torch_device_str", _utils_mod._get_torch_device_str)

    # Inject statgpu.inference
    if "statgpu.inference" not in sys.modules:
        sys.modules["statgpu.inference"] = types.ModuleType("statgpu.inference")
        sys.modules["statgpu.inference"].__path__ = [str(statgpu_pkg / "inference")]
        sys.modules["statgpu.inference"].__package__ = "statgpu.inference"


_bootstrap_statgpu_packages()
from statgpu.backends import _resolve_backend as _rb  # noqa: F401, E402

# Now load _distributions_backend
_distributions_mod = _load_bare_module(
    ROOT / "statgpu" / "inference" / "_distributions_backend.py",
    "statgpu.inference._distributions_backend",
)
get_distribution = _distributions_mod.get_distribution

# ── helpers ──────────────────────────────────────────────────────────────────


def _maybe_import_cupy():
    try:
        import cupy as cp
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            return None
        return cp
    except Exception:
        return None


def _maybe_import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def _bench(
    fn: Callable[[], Any],
    *,
    warmup: int = 3,
    repeats: int = 10,
    synchronize: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    for _ in range(max(0, warmup)):
        fn()
        if synchronize is not None:
            synchronize()

    times: List[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        if synchronize is not None:
            synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "get"):
        return x.get()
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_backend(xp, data):
    return xp.asarray(data, dtype=xp.float64)


# ── precision tests ──────────────────────────────────────────────────────────

_PRECISION_CASES = [
    # ── norm (6 methods, 1 param set) ─────────────────────────────────────────
    ("norm", "cdf", {}, lambda rng: np.linspace(-5, 5, 1000)),
    ("norm", "sf", {}, lambda rng: np.linspace(-5, 5, 1000)),
    ("norm", "ppf", {}, lambda rng: np.linspace(0.001, 0.999, 1000)),
    ("norm", "isf", {}, lambda rng: np.linspace(0.001, 0.999, 1000)),
    ("norm", "pdf", {}, lambda rng: np.linspace(-5, 5, 1000)),
    ("norm", "two_sided_pvalue", {}, lambda rng: np.linspace(0.01, 5, 500)),

    # ── t (6 methods × 3 df values) ──────────────────────────────────────────
    ("t", "cdf", {"df": 5}, lambda rng: np.linspace(-5, 5, 1000)),
    ("t", "cdf", {"df": 10}, lambda rng: np.linspace(-5, 5, 1000)),
    ("t", "cdf", {"df": 30}, lambda rng: np.linspace(-5, 5, 1000)),
    ("t", "sf", {"df": 5}, lambda rng: np.linspace(0.1, 5, 500)),
    ("t", "sf", {"df": 10}, lambda rng: np.linspace(0.1, 5, 500)),
    ("t", "sf", {"df": 30}, lambda rng: np.linspace(0.1, 5, 500)),
    ("t", "ppf", {"df": 5}, lambda rng: np.linspace(0.001, 0.999, 1000)),
    ("t", "ppf", {"df": 10}, lambda rng: np.linspace(0.001, 0.999, 1000)),
    ("t", "ppf", {"df": 30}, lambda rng: np.linspace(0.001, 0.999, 1000)),
    ("t", "isf", {"df": 5}, lambda rng: np.linspace(0.001, 0.999, 1000)),
    ("t", "isf", {"df": 10}, lambda rng: np.linspace(0.001, 0.999, 1000)),
    ("t", "isf", {"df": 30}, lambda rng: np.linspace(0.001, 0.999, 1000)),
    ("t", "pdf", {"df": 5}, lambda rng: np.linspace(-5, 5, 1000)),
    ("t", "pdf", {"df": 10}, lambda rng: np.linspace(-5, 5, 1000)),
    ("t", "pdf", {"df": 30}, lambda rng: np.linspace(-5, 5, 1000)),
    ("t", "two_sided_pvalue", {"df": 5}, lambda rng: np.linspace(0.1, 5, 500)),
    ("t", "two_sided_pvalue", {"df": 10}, lambda rng: np.linspace(0.1, 5, 500)),
    ("t", "two_sided_pvalue", {"df": 30}, lambda rng: np.linspace(0.1, 5, 500)),

    # ── chi2 (6 methods × 3 df values) ───────────────────────────────────────
    ("chi2", "cdf", {"df": 1}, lambda rng: np.linspace(0.001, 50, 500)),
    ("chi2", "cdf", {"df": 5}, lambda rng: np.linspace(0.001, 50, 500)),
    ("chi2", "cdf", {"df": 30}, lambda rng: np.linspace(0.001, 100, 500)),
    ("chi2", "sf", {"df": 1}, lambda rng: np.linspace(0.1, 50, 500)),
    ("chi2", "sf", {"df": 5}, lambda rng: np.linspace(0.1, 50, 500)),
    ("chi2", "sf", {"df": 30}, lambda rng: np.linspace(0.1, 100, 500)),
    ("chi2", "ppf", {"df": 1}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("chi2", "ppf", {"df": 5}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("chi2", "ppf", {"df": 30}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("chi2", "isf", {"df": 1}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("chi2", "isf", {"df": 5}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("chi2", "isf", {"df": 30}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("chi2", "pdf", {"df": 1}, lambda rng: np.linspace(0.001, 50, 500)),
    ("chi2", "pdf", {"df": 5}, lambda rng: np.linspace(0.001, 50, 500)),
    ("chi2", "pdf", {"df": 30}, lambda rng: np.linspace(0.001, 100, 500)),

    # ── gamma (6 methods × 3 shape values) ───────────────────────────────────
    ("gamma", "cdf", {"a": 1.0}, lambda rng: np.linspace(0.001, 20, 500)),
    ("gamma", "cdf", {"a": 2.0}, lambda rng: np.linspace(0.001, 20, 500)),
    ("gamma", "cdf", {"a": 5.0}, lambda rng: np.linspace(0.001, 30, 500)),
    ("gamma", "sf", {"a": 1.0}, lambda rng: np.linspace(0.1, 20, 500)),
    ("gamma", "sf", {"a": 2.0}, lambda rng: np.linspace(0.1, 20, 500)),
    ("gamma", "sf", {"a": 5.0}, lambda rng: np.linspace(0.1, 30, 500)),
    ("gamma", "ppf", {"a": 1.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("gamma", "ppf", {"a": 2.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("gamma", "ppf", {"a": 5.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("gamma", "isf", {"a": 1.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("gamma", "isf", {"a": 2.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("gamma", "isf", {"a": 5.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("gamma", "pdf", {"a": 1.0}, lambda rng: np.linspace(0.001, 20, 500)),
    ("gamma", "pdf", {"a": 2.0}, lambda rng: np.linspace(0.001, 20, 500)),
    ("gamma", "pdf", {"a": 5.0}, lambda rng: np.linspace(0.001, 30, 500)),

    # ── beta (6 methods × 2 param sets) ──────────────────────────────────────
    ("beta", "cdf", {"a": 2.0, "b": 3.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("beta", "cdf", {"a": 0.5, "b": 0.5}, lambda rng: np.linspace(0.01, 0.99, 500)),
    ("beta", "sf", {"a": 2.0, "b": 3.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("beta", "sf", {"a": 0.5, "b": 0.5}, lambda rng: np.linspace(0.01, 0.99, 500)),
    ("beta", "ppf", {"a": 2.0, "b": 3.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("beta", "ppf", {"a": 0.5, "b": 0.5}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("beta", "isf", {"a": 2.0, "b": 3.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("beta", "isf", {"a": 0.5, "b": 0.5}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("beta", "pdf", {"a": 2.0, "b": 3.0}, lambda rng: np.linspace(0.01, 0.99, 500)),
    ("beta", "pdf", {"a": 0.5, "b": 0.5}, lambda rng: np.linspace(0.01, 0.99, 500)),

    # ── f (6 methods × 2 param sets) ─────────────────────────────────────────
    ("f", "cdf", {"dfn": 3, "dfd": 100}, lambda rng: np.linspace(0.001, 10, 500)),
    ("f", "cdf", {"dfn": 5, "dfd": 10}, lambda rng: np.linspace(0.001, 10, 500)),
    ("f", "sf", {"dfn": 3, "dfd": 100}, lambda rng: np.linspace(0.1, 10, 500)),
    ("f", "sf", {"dfn": 5, "dfd": 10}, lambda rng: np.linspace(0.1, 10, 500)),
    ("f", "ppf", {"dfn": 3, "dfd": 100}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("f", "ppf", {"dfn": 5, "dfd": 10}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("f", "isf", {"dfn": 3, "dfd": 100}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("f", "isf", {"dfn": 5, "dfd": 10}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("f", "pdf", {"dfn": 3, "dfd": 100}, lambda rng: np.linspace(0.01, 10, 500)),
    ("f", "pdf", {"dfn": 5, "dfd": 10}, lambda rng: np.linspace(0.01, 10, 500)),

    # ── uniform (5 methods) ──────────────────────────────────────────────────
    ("uniform", "cdf", {}, lambda rng: np.linspace(-1, 3, 500)),
    ("uniform", "sf", {}, lambda rng: np.linspace(-1, 3, 500)),
    ("uniform", "ppf", {}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("uniform", "isf", {}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("uniform", "pdf", {}, lambda rng: np.linspace(-1, 3, 500)),

    # ── expon (5 methods) ────────────────────────────────────────────────────
    ("expon", "cdf", {}, lambda rng: np.linspace(0.001, 20, 500)),
    ("expon", "sf", {}, lambda rng: np.linspace(0.001, 20, 500)),
    ("expon", "ppf", {}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("expon", "isf", {}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("expon", "pdf", {}, lambda rng: np.linspace(0.001, 20, 500)),

    # ── cauchy (5 methods) ───────────────────────────────────────────────────
    ("cauchy", "cdf", {}, lambda rng: np.linspace(-10, 10, 500)),
    ("cauchy", "sf", {}, lambda rng: np.linspace(-10, 10, 500)),
    ("cauchy", "ppf", {}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("cauchy", "isf", {}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("cauchy", "pdf", {}, lambda rng: np.linspace(-10, 10, 500)),

    # ── laplace (5 methods) ──────────────────────────────────────────────────
    ("laplace", "cdf", {}, lambda rng: np.linspace(-10, 10, 500)),
    ("laplace", "sf", {}, lambda rng: np.linspace(-10, 10, 500)),
    ("laplace", "ppf", {}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("laplace", "isf", {}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("laplace", "pdf", {}, lambda rng: np.linspace(-10, 10, 500)),

    # ── logistic (5 methods) ─────────────────────────────────────────────────
    ("logistic", "cdf", {}, lambda rng: np.linspace(-10, 10, 500)),
    ("logistic", "sf", {}, lambda rng: np.linspace(-10, 10, 500)),
    ("logistic", "ppf", {}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("logistic", "isf", {}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("logistic", "pdf", {}, lambda rng: np.linspace(-10, 10, 500)),

    # ── weibull_min (5 methods × 2 shape values) ────────────────────────────
    ("weibull_min", "cdf", {"c": 1.5}, lambda rng: np.linspace(0.001, 10, 500)),
    ("weibull_min", "cdf", {"c": 2.0}, lambda rng: np.linspace(0.001, 10, 500)),
    ("weibull_min", "sf", {"c": 1.5}, lambda rng: np.linspace(0.001, 10, 500)),
    ("weibull_min", "sf", {"c": 2.0}, lambda rng: np.linspace(0.001, 10, 500)),
    ("weibull_min", "ppf", {"c": 1.5}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("weibull_min", "ppf", {"c": 2.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("weibull_min", "isf", {"c": 1.5}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("weibull_min", "isf", {"c": 2.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("weibull_min", "pdf", {"c": 1.5}, lambda rng: np.linspace(0.01, 10, 500)),
    ("weibull_min", "pdf", {"c": 2.0}, lambda rng: np.linspace(0.01, 10, 500)),

    # ── lognorm (5 methods × 2 shape values) ─────────────────────────────────
    ("lognorm", "cdf", {"s": 0.5}, lambda rng: np.linspace(0.001, 10, 500)),
    ("lognorm", "cdf", {"s": 1.0}, lambda rng: np.linspace(0.001, 20, 500)),
    ("lognorm", "sf", {"s": 0.5}, lambda rng: np.linspace(0.001, 10, 500)),
    ("lognorm", "sf", {"s": 1.0}, lambda rng: np.linspace(0.001, 20, 500)),
    ("lognorm", "ppf", {"s": 0.5}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("lognorm", "ppf", {"s": 1.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("lognorm", "isf", {"s": 0.5}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("lognorm", "isf", {"s": 1.0}, lambda rng: np.linspace(0.001, 0.999, 500)),
    ("lognorm", "pdf", {"s": 0.5}, lambda rng: np.linspace(0.01, 10, 500)),
    ("lognorm", "pdf", {"s": 1.0}, lambda rng: np.linspace(0.01, 20, 500)),

    # ── poisson (5 methods × 2 mu values) ────────────────────────────────────
    ("poisson", "cdf", {"mu": 5}, lambda rng: np.arange(0, 30)),
    ("poisson", "cdf", {"mu": 20}, lambda rng: np.arange(0, 60)),
    ("poisson", "sf", {"mu": 5}, lambda rng: np.arange(0, 30)),
    ("poisson", "sf", {"mu": 20}, lambda rng: np.arange(0, 60)),
    ("poisson", "ppf", {"mu": 5}, lambda rng: np.linspace(0.001, 0.999, 100)),
    ("poisson", "ppf", {"mu": 20}, lambda rng: np.linspace(0.001, 0.999, 100)),
    ("poisson", "isf", {"mu": 5}, lambda rng: np.linspace(0.001, 0.999, 100)),
    ("poisson", "isf", {"mu": 20}, lambda rng: np.linspace(0.001, 0.999, 100)),
    ("poisson", "pmf", {"mu": 5}, lambda rng: np.arange(0, 30)),
    ("poisson", "pmf", {"mu": 20}, lambda rng: np.arange(0, 60)),

    # ── binom (5 methods × 2 param sets) ─────────────────────────────────────
    ("binom", "cdf", {"n": 100, "p": 0.3}, lambda rng: np.arange(0, 101)),
    ("binom", "cdf", {"n": 50, "p": 0.7}, lambda rng: np.arange(0, 51)),
    ("binom", "sf", {"n": 100, "p": 0.3}, lambda rng: np.arange(0, 101)),
    ("binom", "sf", {"n": 50, "p": 0.7}, lambda rng: np.arange(0, 51)),
    ("binom", "ppf", {"n": 100, "p": 0.3}, lambda rng: np.linspace(0.001, 0.999, 100)),
    ("binom", "ppf", {"n": 50, "p": 0.7}, lambda rng: np.linspace(0.001, 0.999, 100)),
    ("binom", "isf", {"n": 100, "p": 0.3}, lambda rng: np.linspace(0.001, 0.999, 100)),
    ("binom", "isf", {"n": 50, "p": 0.7}, lambda rng: np.linspace(0.001, 0.999, 100)),
    ("binom", "pmf", {"n": 100, "p": 0.3}, lambda rng: np.arange(0, 101)),
    ("binom", "pmf", {"n": 50, "p": 0.7}, lambda rng: np.arange(0, 51)),
]

_SCIPY_METHOD_MAP = {
    "two_sided_pvalue": None,  # handled separately
}


def _run_precision_tests():
    """Run precision tests against scipy.stats reference."""
    from scipy import stats as sps

    results = []
    pass_count = 0
    warn_count = 0
    fail_count = 0

    for dist_name, method, extra_kwargs, x_factory in _PRECISION_CASES:
        d = get_distribution(dist_name, backend="numpy")
        x = x_factory(np.random.default_rng(42))

        # scipy.stats uses the same param names as statgpu for all distributions
        scipy_kwargs = dict(extra_kwargs)
        backend_kwargs = dict(extra_kwargs)

        # Compute statgpu result
        statgpu_method = getattr(d, method, None)
        if statgpu_method is None:
            continue

        try:
            statgpu_result = _to_numpy(statgpu_method(x, **backend_kwargs))
        except Exception as e:
            results.append({
                "dist": dist_name,
                "method": method,
                "extra_kwargs": extra_kwargs,
                "status": "ERROR",
                "error": str(e),
            })
            fail_count += 1
            continue

        # Compute scipy reference
        scipy_dist = getattr(sps, dist_name, None)
        if scipy_dist is None:
            results.append({
                "dist": dist_name,
                "method": method,
                "extra_kwargs": extra_kwargs,
                "status": "SKIP",
                "reason": "scipy dist not found",
            })
            continue

        if method == "two_sided_pvalue":
            # two_sided_pvalue = 2 * sf(|stat|)
            try:
                ref_dist = getattr(sps, dist_name, sps.norm)
                if dist_name == "t":
                    ref = 2.0 * sps.t.sf(x, df=extra_kwargs.get("df", 10))
                else:
                    ref = 2.0 * sps.norm.sf(x)
            except Exception:
                ref = None
        else:
            try:
                ref_method = getattr(scipy_dist, method, None)
                if ref_method is None:
                    ref = None
                else:
                    ref = np.asarray(ref_method(x, **scipy_kwargs), dtype=np.float64)
            except Exception:
                ref = None

        if ref is None:
            results.append({
                "dist": dist_name,
                "method": method,
                "extra_kwargs": extra_kwargs,
                "status": "SKIP",
                "reason": "scipy method not found",
            })
            continue

        # Align shapes
        statgpu_result = np.asarray(statgpu_result, dtype=np.float64).reshape(ref.shape)

        # Compute errors
        diff = np.abs(statgpu_result - ref)
        # Mask for valid reference values
        valid = np.isfinite(ref) & (np.abs(ref) > 0)
        if valid.any():
            rel_error = np.abs((statgpu_result[valid] - ref[valid]) / ref[valid])
            max_rel = float(np.max(rel_error)) if rel_error.size > 0 else 0.0
        else:
            max_rel = 0.0

        max_abs = float(np.max(diff))
        median_abs = float(np.median(diff))

        # Thresholds: CDF/SF < 1e-10, PPF/ISF < 1e-6 (inverse special functions
        # like gammaincinv/betaincinv have ~1e-7 precision), PDF/PMF < 1e-9
        is_inverse = method in ("ppf", "isf")
        is_cdf_sf = method in ("cdf", "sf", "two_sided_pvalue")
        if is_inverse:
            threshold = 1e-6
        elif is_cdf_sf:
            threshold = 1e-10
        else:
            threshold = 1e-9

        if max_abs < threshold:
            status = "PASS"
            pass_count += 1
        elif max_abs < threshold * 100:
            status = "WARN"
            warn_count += 1
        else:
            status = "FAIL"
            fail_count += 1

        results.append({
            "dist": dist_name,
            "method": method,
            "extra_kwargs": extra_kwargs,
            "status": status,
            "max_abs_error": max_abs,
            "median_abs_error": median_abs,
            "max_rel_error": max_rel,
            "n_points": int(len(x)),
        })

    return {
        "results": results,
        "summary": {
            "pass": pass_count,
            "warn": warn_count,
            "fail": fail_count,
            "total": pass_count + warn_count + fail_count,
        },
    }


# ── performance tests ────────────────────────────────────────────────────────

_SCALES = {
    "small_10k": 10_000,
    "medium_100k": 100_000,
    "large_1m": 1_000_000,
    "xlarge_100m": 100_000_000,  # use --large-only for remote (may cause SSH timeout)
}

_PERF_CASES = [
    # (dist_name, methods, kwargs, x_range_factory)
    ("norm", ["cdf", "sf", "ppf", "isf", "pdf"], {},
     lambda n, rng: rng.uniform(-5, 5, size=n)),
    ("t", ["cdf", "sf", "ppf", "isf", "pdf"], {"df": 10},
     lambda n, rng: rng.uniform(-5, 5, size=n)),
    ("t", ["cdf", "sf", "ppf", "isf"], {"df": 5},
     lambda n, rng: rng.uniform(-5, 5, size=n)),
    ("t", ["cdf", "sf", "ppf", "isf"], {"df": 30},
     lambda n, rng: rng.uniform(-5, 5, size=n)),
    ("chi2", ["cdf", "sf", "ppf", "isf", "pdf"], {"df": 5},
     lambda n, rng: np.abs(rng.normal(size=n)) * 5),
    ("chi2", ["cdf", "sf", "ppf", "isf"], {"df": 30},
     lambda n, rng: rng.chisquare(df=30, size=n)),
    ("gamma", ["cdf", "sf", "ppf", "isf", "pdf"], {"a": 2.0},
     lambda n, rng: rng.gamma(shape=2.0, size=n)),
    ("beta", ["cdf", "sf", "ppf", "isf", "pdf"], {"a": 2.0, "b": 3.0},
     lambda n, rng: rng.beta(2.0, 3.0, size=n)),
    ("f", ["cdf", "sf", "ppf", "isf", "pdf"], {"dfn": 3, "dfd": 100},
     lambda n, rng: rng.gamma(shape=3/2, size=n) / rng.gamma(shape=100/2, size=n) * (100/3)),
    ("uniform", ["cdf", "sf", "ppf", "isf", "pdf"], {},
     lambda n, rng: rng.uniform(-1, 3, size=n)),
    ("expon", ["cdf", "sf", "ppf", "isf", "pdf"], {},
     lambda n, rng: rng.exponential(scale=1.0, size=n)),
    ("cauchy", ["cdf", "sf", "ppf", "isf", "pdf"], {},
     lambda n, rng: (rng.standard_cauchy(size=n))),
    ("laplace", ["cdf", "sf", "ppf", "isf", "pdf"], {},
     lambda n, rng: rng.laplace(size=n)),
    ("logistic", ["cdf", "sf", "ppf", "isf", "pdf"], {},
     lambda n, rng: rng.logistic(size=n)),
    ("weibull_min", ["cdf", "sf", "ppf", "isf", "pdf"], {"c": 2.0},
     lambda n, rng: rng.weibull(a=2.0, size=n)),
    ("lognorm", ["cdf", "sf", "ppf", "isf", "pdf"], {"s": 1.0},
     lambda n, rng: rng.lognormal(mean=0, sigma=1.0, size=n)),
    ("poisson", ["cdf", "sf", "ppf", "isf", "pmf"], {"mu": 5},
     lambda n, rng: rng.poisson(lam=5, size=n)),
    ("binom", ["cdf", "sf", "ppf", "isf", "pmf"], {"n": 100, "p": 0.3},
     lambda n, rng: rng.binomial(n=100, p=0.3, size=n)),
]


def _run_performance_tests(
    cp_module,
    torch_module,
    rng: np.random.Generator,
    *,
    warmup: int = 3,
    repeats: int = 10,
    large_warmup: int = 1,
    large_repeats: int = 3,
    scales: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Run performance benchmarks across all backends and scales."""
    active_scales = scales if scales is not None else _SCALES
    benchmarks = {}

    for scale_name, n in active_scales.items():
        scale_results = {}

        # Adjust repeats for large scale to avoid OOM / long waits
        rp = large_repeats if n >= 100_000_000 else repeats
        wp = large_warmup if n >= 100_000_000 else warmup

        for dist_name, methods, extra_kwargs, x_factory in _PERF_CASES:
            key = dist_name
            if extra_kwargs:
                kw_str = ",".join(f"{k}={v}" for k, v in extra_kwargs.items())
                key = f"{dist_name}({kw_str})"

            x_np = x_factory(n, rng)

            # Numpy backend
            d_np = get_distribution(dist_name, backend="numpy")
            np_results = {}
            for method in methods:
                fn = getattr(d_np, method, None)
                if fn is None:
                    continue

                def _np_call(f=fn, data=x_np, kw=extra_kwargs):
                    return f(data, **kw)

                br = _bench(_np_call, warmup=wp, repeats=rp)
                np_results[method] = br

            scale_results[key] = {"numpy": np_results}

            # CuPy backend
            if cp_module is not None:
                try:
                    d_cp = get_distribution(dist_name, backend="cupy")
                    x_cp = cp_module.asarray(x_np, dtype=cp_module.float64)
                    cp_results = {}
                    for method in methods:
                        fn = getattr(d_cp, method, None)
                        if fn is None:
                            continue

                        def _cp_call(f=fn, data=x_cp, kw=extra_kwargs):
                            return f(data, **kw)

                        br = _bench(
                            _cp_call, warmup=wp, repeats=rp,
                            synchronize=cp_module.cuda.runtime.deviceSynchronize,
                        )
                        cp_results[method] = br
                    scale_results[key]["cupy"] = cp_results
                except Exception as e:
                    scale_results[key]["cupy"] = {"error": str(e)}

            # Torch backend
            if torch_module is not None:
                try:
                    d_torch = get_distribution(dist_name, backend="torch")
                    x_torch = torch_module.as_tensor(
                        x_np, dtype=torch_module.float64,
                        device="cuda" if torch_module.cuda.is_available() else "cpu",
                    )
                    torch_results = {}
                    for method in methods:
                        fn = getattr(d_torch, method, None)
                        if fn is None:
                            continue

                        def _torch_call(f=fn, data=x_torch, kw=extra_kwargs):
                            return f(data, **kw)

                        sync_fn = None
                        if x_torch.device.type == "cuda":
                            sync_fn = torch_module.cuda.synchronize

                        br = _bench(_torch_call, warmup=wp, repeats=rp, synchronize=sync_fn)
                        torch_results[method] = br
                    scale_results[key]["torch"] = torch_results
                except Exception as e:
                    scale_results[key]["torch"] = {"error": str(e)}

        benchmarks[scale_name] = scale_results

    return benchmarks


def _compute_speedup(perf: Dict[str, Any]) -> Dict[str, Any]:
    """Add speedup ratios to performance results."""
    speedups = {}
    for scale_name, scale_data in perf.items():
        speedups[scale_name] = {}
        for dist_key, dist_data in scale_data.items():
            speedups[scale_name][dist_key] = {}
            np_data = dist_data.get("numpy")
            if np_data is None:
                continue

            for backend in ("cupy", "torch"):
                bd = dist_data.get(backend)
                if bd is None or "error" in (bd or {}):
                    continue
                speedups[scale_name][dist_key][backend] = {}
                for method in np_data:
                    if method in bd:
                        np_ms = np_data[method].get("mean_ms", 0)
                        gp_ms = bd[method].get("mean_ms", 0)
                        if gp_ms > 0:
                            speedups[scale_name][dist_key][backend][method] = (
                                round(np_ms / gp_ms, 2)
                            )
    return speedups


# ── report formatting ────────────────────────────────────────────────────────

def _format_precision_report(precision: Dict[str, Any]) -> str:
    lines = [
        "=" * 70,
        "Distribution Precision Report",
        "=" * 70,
        "",
        f"{'Distribution':<18} {'Method':<22} {'Status':<8} {'MaxAbsErr':<16} {'MedianAbsErr':<16}",
        "-" * 70,
    ]
    for r in precision["results"]:
        kw = r.get("extra_kwargs", {})
        kw_str = ""
        if kw:
            kw_str = "(" + ",".join(f"{k}={v}" for k, v in kw.items()) + ")"
        dist_label = f"{r['dist']}{kw_str}"
        status = r.get("status", "?")
        max_abs = r.get("max_abs_error", "N/A")
        median_abs = r.get("median_abs_error", "N/A")
        if isinstance(max_abs, float):
            max_abs = f"{max_abs:.2e}"
        if isinstance(median_abs, float):
            median_abs = f"{median_abs:.2e}"
        lines.append(
            f"{dist_label:<18} {r['method']:<22} {status:<8} {str(max_abs):<16} {str(median_abs):<16}"
        )

    lines.append("-" * 70)
    s = precision["summary"]
    lines.append(
        f"Summary: {s['pass']}/{s['total']} PASS, {s['warn']} WARN, {s['fail']} FAIL"
    )
    lines.append("")
    return "\n".join(lines)


def _format_performance_report(
    perf: Dict[str, Any], speedups: Dict[str, Any],
) -> str:
    lines = [
        "=" * 80,
        "Distribution Performance Report",
        "=" * 80,
    ]

    for scale_name, scale_data in perf.items():
        scale_speedups = speedups.get(scale_name, {})
        n_pts = _SCALES.get(scale_name, "?")
        lines.append("")
        lines.append(f"Scale: {scale_name} ({n_pts:,} points)")
        lines.append("-" * 80)
        lines.append(
            f"{'Distribution':<25} {'Method':<8} {'numpy(ms)':<12} "
            f"{'cupy(ms)':<12} {'torch(ms)':<12} {'cupy/s':<8} {'torch/s':<8}"
        )
        lines.append("-" * 80)

        for dist_key, dist_data in scale_data.items():
            np_data = dist_data.get("numpy", {})
            cp_data = dist_data.get("cupy", {}) or {}
            torch_data = dist_data.get("torch", {}) or {}
            sp = scale_speedups.get(dist_key, {})
            sp_cp = sp.get("cupy", {})
            sp_torch = sp.get("torch", {})

            all_methods = set(np_data.keys())
            if not all_methods:
                continue

            first = True
            for method in sorted(all_methods):
                np_ms = np_data.get(method, {}).get("mean_ms", "N/A")
                cp_ms = cp_data.get(method, {}).get("mean_ms", "N/A")
                torch_ms = torch_data.get(method, {}).get("mean_ms", "N/A")
                cp_speedup = sp_cp.get(method, "-")
                torch_speedup = sp_torch.get(method, "-")

                np_str = f"{np_ms:.4f}" if isinstance(np_ms, float) else str(np_ms)
                cp_str = f"{cp_ms:.4f}" if isinstance(cp_ms, float) else str(cp_ms)
                torch_str = f"{torch_ms:.4f}" if isinstance(torch_ms, float) else str(torch_ms)

                label = dist_key if first else ""
                first = False
                lines.append(
                    f"{label:<25} {method:<8} {np_str:<12} {cp_str:<12} "
                    f"{torch_str:<12} {str(cp_speedup):<8} {str(torch_speedup):<8}"
                )

    lines.append("")
    return "\n".join(lines)


# ── remote execution ─────────────────────────────────────────────────────────

REMOTE_HOST = "hz-4.matpool.com"
REMOTE_PORT = 28838
REMOTE_USER = "root"
REMOTE_PASS = "q06qj[{K8[[gj5yB"
REMOTE_DIR = "/tmp/statgpu_dist_bench"


def _run_on_remote(
    output: str,
) -> None:
    """Upload project to remote GPU server and run benchmark there."""
    import paramiko

    print(f"Connecting to {REMOTE_HOST}:{REMOTE_PORT} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASS)
    # Enable keepalive on the transport channel
    try:
        ssh.get_transport().set_keepalive(30)
    except Exception:
        pass

    project_root = ROOT
    remote_tar = f"{REMOTE_DIR}.tar.gz"

    print("Uploading project ...")
    # Create tar in-memory
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(project_root / "statgpu"), arcname="statgpu")
        tar.add(str(project_root / "dev"), arcname="dev")
    buf.seek(0)

    sftp = ssh.open_sftp()
    with sftp.file(remote_tar, "w") as f:
        f.write(buf.read())
    sftp.close()

    # Extract and run
    cmd = (
        "source /root/miniconda3/etc/profile.d/conda.sh && "
        "conda activate myconda && "
        f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR} && "
        f"cd {REMOTE_DIR} && "
        f"tar xzf {remote_tar} && "
        f"cd {REMOTE_DIR} && "
        f"PYTHONPATH={REMOTE_DIR} python dev/benchmarks/benchmark_distributions.py "
        f"--output {REMOTE_DIR}/results/dist_bench.json"
    )
    print("Running benchmark on remote server ...")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=3600)
    out = stdout.read().decode()
    err = stderr.read().decode()
    exit_status = stdout.channel.recv_exit_status()

    print(out)
    if err:
        print("STDERR:", err)

    if exit_status != 0:
        print(f"Remote command failed with exit status {exit_status}")
        ssh.close()
        sys.exit(exit_status)

    # Download results
    local_output = Path(output)
    local_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        sftp = ssh.open_sftp()
        remote_json = f"{REMOTE_DIR}/results/dist_bench.json"
        sftp.get(remote_json, str(local_output))
        sftp.close()
        print(f"Results downloaded to {local_output}")
    except Exception as e:
        print(f"Warning: could not download results: {e}")

    # Cleanup
    ssh.exec_command(f"rm -rf {REMOTE_DIR} {remote_tar}")
    ssh.close()
    print("Done.")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distribution backend precision + performance benchmark",
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="Run on remote GPU server via paramiko",
    )
    parser.add_argument(
        "--output", default="",
        help="Output JSON path (default: results/distribution_bench_YYYY-MM-DD.json)",
    )
    parser.add_argument(
        "--no-precision", action="store_true",
        help="Skip precision tests",
    )
    parser.add_argument(
        "--no-performance", action="store_true",
        help="Skip performance tests",
    )
    parser.add_argument(
        "--large-only", action="store_true",
        help="Only run large-scale (100M) performance tests",
    )
    parser.add_argument(
        "--skip-large", action="store_true",
        help="Skip large-scale (100M) performance tests",
    )
    args = parser.parse_args()

    if args.remote:
        out = args.output or f"results/distribution_bench_{date.today()}.json"
        _run_on_remote(out)
        return

    # Local execution
    rng = np.random.default_rng(42)
    cp_module = _maybe_import_cupy()
    torch_module = _maybe_import_torch()

    payload = {
        "date": str(date.today()),
        "environment": {
            "cupy_available": cp_module is not None,
            "torch_available": torch_module is not None,
        },
    }

    if not args.no_precision:
        print("Running precision tests ...")
        precision = _run_precision_tests()
        payload["precision"] = precision
        print(_format_precision_report(precision))

    if not args.no_performance:
        print("Running performance tests ...")
        if args.large_only:
            scales_to_run = {"xlarge_100m": _SCALES["xlarge_100m"]}
        elif args.skip_large:
            scales_to_run = {k: v for k, v in _SCALES.items() if k != "xlarge_100m"}
        else:
            scales_to_run = _SCALES

        perf = _run_performance_tests(cp_module, torch_module, rng, scales=scales_to_run)
        speedups = _compute_speedup(perf)

        payload["performance"] = perf
        payload["speedups"] = speedups
        print(_format_performance_report(perf, speedups))

    # Write output
    if not args.output:
        args.output = f"results/distribution_bench_{date.today()}.json"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to {out_path}")

    # Also write human-readable report
    txt_path = out_path.with_suffix(".txt")
    report_parts = []
    if not args.no_precision and "precision" in payload:
        report_parts.append(_format_precision_report(payload["precision"]))
    if not args.no_performance and "performance" in payload:
        report_parts.append(
            _format_performance_report(payload["performance"], payload.get("speedups", {}))
        )
    txt_path.write_text("\n".join(report_parts), encoding="utf-8")
    print(f"Text report written to {txt_path}")


if __name__ == "__main__":
    main()
