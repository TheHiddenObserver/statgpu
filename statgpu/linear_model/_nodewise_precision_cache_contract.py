"""Numerically safe cache for the standardized node-wise precision contract.

The old cache keyed parent-estimator tolerance and used backend-specific design
hashes.  This cache is keyed by the actual precision problem: canonical working
Gram, resolved node-wise alpha, the internal node-wise solver contract, and the
concrete backend/device.  Cached values are numerical precision matrices only;
request-specific provenance is rebuilt on every call.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
import threading

import numpy as np

from statgpu.linear_model import _nodewise_alpha_inference_contract as _runtime
from statgpu.linear_model.penalized import _nodewise_precision as _precision

_MARKER = "__statgpu_nodewise_precision_cache__"
_CACHE_MAXSIZE = int(os.getenv("STATGPU_LASSO_DEBIASED_M_CACHE_SIZE", "16"))
_CACHE = OrderedDict()
_LOCK = threading.Lock()

_ORIGINAL_NUMPY = _runtime.build_nodewise_precision_numpy
_ORIGINAL_CUPY = _runtime.build_nodewise_precision_cupy
_ORIGINAL_TORCH = _runtime.build_nodewise_precision_torch


def _resolved(requested, p, effective_n):
    return _precision.resolve_nodewise_alpha(
        requested,
        p=int(p),
        effective_n=float(effective_n),
    )


def _key_from_gram(gram_np, *, backend_device, alpha):
    arr = np.ascontiguousarray(np.asarray(gram_np, dtype=np.float64))
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(arr.view(np.uint8).tobytes())
    h.update(str(backend_device).encode("utf-8"))
    h.update(str(_precision.NODEWISE_CONTRACT_VERSION).encode("utf-8"))
    h.update(str(_precision.NODEWISE_SOLVER).encode("utf-8"))
    h.update(str(_precision.NODEWISE_STOPPING).encode("utf-8"))
    h.update(
        np.asarray(
            [
                float(alpha),
                float(_precision.NODEWISE_TOL),
                float(_precision.NODEWISE_MAX_ITER),
                float(_precision.NODEWISE_KKT_TOL),
            ],
            dtype=np.float64,
        ).tobytes()
    )
    return h.hexdigest()


def _get(key):
    if _CACHE_MAXSIZE <= 0:
        return None
    with _LOCK:
        value = _CACHE.get(key)
        if value is not None:
            _CACHE.move_to_end(key)
            return value
    return None


def _put(key, M_np, max_kkt):
    if _CACHE_MAXSIZE <= 0:
        return
    payload = (np.asarray(M_np, dtype=np.float64).copy(), float(max_kkt))
    with _LOCK:
        _CACHE[key] = payload
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAXSIZE:
            _CACHE.popitem(last=False)


def _metadata(requested, resolved, source, rule, effective_n, weighted, max_kkt, hit):
    meta = _precision.precision_metadata(
        requested=requested,
        resolved=resolved,
        source=source,
        rule=rule,
        effective_n=effective_n,
        weighted=weighted,
        max_kkt=max_kkt,
    )
    meta["precision_cache_hit"] = bool(hit)
    return meta


def _numpy_cached(X_work, *, requested_alpha, effective_n, weighted):
    X = np.asarray(X_work, dtype=np.float64)
    if X.ndim != 2 or int(X.shape[1]) <= 1 or _CACHE_MAXSIZE <= 0:
        return _ORIGINAL_NUMPY(
            X_work,
            requested_alpha=requested_alpha,
            effective_n=effective_n,
            weighted=weighted,
        )
    alpha, source, rule = _resolved(requested_alpha, X.shape[1], effective_n)
    gram = X.T @ X / float(X.shape[0])
    key = _key_from_gram(gram, backend_device="numpy:cpu", alpha=alpha)
    cached = _get(key)
    if cached is not None:
        M, max_kkt = cached
        return M.copy(), float(alpha), _metadata(
            requested_alpha, alpha, source, rule, effective_n, weighted, max_kkt, True
        )
    M, resolved, meta = _ORIGINAL_NUMPY(
        X_work,
        requested_alpha=requested_alpha,
        effective_n=effective_n,
        weighted=weighted,
    )
    _put(key, M, meta.get("nodewise_max_kkt_residual", 0.0))
    meta = dict(meta)
    meta["precision_cache_hit"] = False
    return M, resolved, meta


def _cupy_cached(X_work, *, requested_alpha, effective_n, weighted):
    try:
        import cupy as cp
    except ImportError:
        return _ORIGINAL_CUPY(
            X_work,
            requested_alpha=requested_alpha,
            effective_n=effective_n,
            weighted=weighted,
        )
    X = cp.asarray(X_work, dtype=cp.float64)
    if X.ndim != 2 or int(X.shape[1]) <= 1 or _CACHE_MAXSIZE <= 0:
        return _ORIGINAL_CUPY(
            X_work,
            requested_alpha=requested_alpha,
            effective_n=effective_n,
            weighted=weighted,
        )
    alpha, source, rule = _resolved(requested_alpha, X.shape[1], effective_n)
    gram = X.T @ X / float(X.shape[0])
    key = _key_from_gram(
        cp.asnumpy(gram),
        backend_device=f"cupy:cuda:{int(X.device.id)}",
        alpha=alpha,
    )
    cached = _get(key)
    if cached is not None:
        M_np, max_kkt = cached
        M = cp.asarray(M_np, dtype=X.dtype)
        return M, float(alpha), _metadata(
            requested_alpha, alpha, source, rule, effective_n, weighted, max_kkt, True
        )
    M, resolved, meta = _ORIGINAL_CUPY(
        X_work,
        requested_alpha=requested_alpha,
        effective_n=effective_n,
        weighted=weighted,
    )
    _put(key, cp.asnumpy(M), meta.get("nodewise_max_kkt_residual", 0.0))
    meta = dict(meta)
    meta["precision_cache_hit"] = False
    return M, resolved, meta


def _torch_cached(X_work, *, requested_alpha, effective_n, weighted):
    try:
        import torch
    except ImportError:
        return _ORIGINAL_TORCH(
            X_work,
            requested_alpha=requested_alpha,
            effective_n=effective_n,
            weighted=weighted,
        )
    if not isinstance(X_work, torch.Tensor):
        return _ORIGINAL_TORCH(
            X_work,
            requested_alpha=requested_alpha,
            effective_n=effective_n,
            weighted=weighted,
        )
    X = X_work.to(dtype=torch.float64)
    if X.ndim != 2 or int(X.shape[1]) <= 1 or _CACHE_MAXSIZE <= 0:
        return _ORIGINAL_TORCH(
            X_work,
            requested_alpha=requested_alpha,
            effective_n=effective_n,
            weighted=weighted,
        )
    alpha, source, rule = _resolved(requested_alpha, X.shape[1], effective_n)
    gram = X.T @ X / float(X.shape[0])
    key = _key_from_gram(
        gram.detach().cpu().numpy(),
        backend_device=f"torch:{X.device}",
        alpha=alpha,
    )
    cached = _get(key)
    if cached is not None:
        M_np, max_kkt = cached
        M = torch.as_tensor(M_np, dtype=X.dtype, device=X.device)
        return M, float(alpha), _metadata(
            requested_alpha, alpha, source, rule, effective_n, weighted, max_kkt, True
        )
    M, resolved, meta = _ORIGINAL_TORCH(
        X_work,
        requested_alpha=requested_alpha,
        effective_n=effective_n,
        weighted=weighted,
    )
    _put(key, M.detach().cpu().numpy(), meta.get("nodewise_max_kkt_residual", 0.0))
    meta = dict(meta)
    meta["precision_cache_hit"] = False
    return M, resolved, meta


def install_nodewise_precision_cache_contract() -> None:
    if getattr(_runtime.build_nodewise_precision_numpy, _MARKER, False):
        return
    setattr(_numpy_cached, _MARKER, True)
    setattr(_cupy_cached, _MARKER, True)
    setattr(_torch_cached, _MARKER, True)
    _runtime.build_nodewise_precision_numpy = _numpy_cached
    _runtime.build_nodewise_precision_cupy = _cupy_cached
    _runtime.build_nodewise_precision_torch = _torch_cached


__all__ = ["install_nodewise_precision_cache_contract"]
