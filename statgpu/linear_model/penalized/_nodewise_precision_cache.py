"""Cache standardized node-wise precision without mutating runtime methods.

Cache identity is the actual precision problem: canonical working-design bytes,
resolved node-wise alpha, the internal node-wise solver contract, and the
concrete backend/device. Request-specific provenance is rebuilt on every call.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
import threading

import numpy as np

from . import _nodewise_precision as _precision

_CACHE_MAXSIZE = int(os.getenv("STATGPU_LASSO_DEBIASED_M_CACHE_SIZE", "16"))
_CACHE = OrderedDict()
_LOCK = threading.Lock()
_GPU_HASH_ROW_CHUNK = 1024


def _key(design_digest: bytes, *, backend_device: str, alpha: float) -> str:
    h = hashlib.blake2b(digest_size=32)
    h.update(design_digest)
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


def _digest_numpy(X) -> bytes:
    arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8).tobytes())
    return h.digest()


def _digest_cupy(X) -> bytes:
    import cupy as cp

    arr = cp.asarray(X, dtype=cp.float64)
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    h.update(str(arr.dtype).encode("utf-8"))
    chunk = max(1, min(int(arr.shape[0]), _GPU_HASH_ROW_CHUNK))
    for start in range(0, int(arr.shape[0]), chunk):
        h.update(cp.asnumpy(arr[start : start + chunk]).tobytes())
    return h.digest()


def _digest_torch(X) -> bytes:
    import torch

    if not isinstance(X, torch.Tensor):
        raise TypeError("Torch node-wise precision cache requires torch.Tensor input")
    arr = X.to(dtype=torch.float64)
    h = hashlib.blake2b(digest_size=32)
    h.update(np.asarray(tuple(arr.shape), dtype=np.int64).tobytes())
    h.update(str(arr.dtype).encode("utf-8"))
    chunk = max(1, min(int(arr.shape[0]), _GPU_HASH_ROW_CHUNK))
    for start in range(0, int(arr.shape[0]), chunk):
        h.update(arr[start : start + chunk].detach().cpu().numpy().tobytes())
    return h.digest()


def _get(key):
    if _CACHE_MAXSIZE <= 0:
        return None
    with _LOCK:
        value = _CACHE.get(key)
        if value is not None:
            _CACHE.move_to_end(key)
            M, max_kkt = value
            return M.copy(), float(max_kkt)
    return None


def _put(key, M_np, max_kkt):
    if _CACHE_MAXSIZE <= 0:
        return
    value = (np.asarray(M_np, dtype=np.float64).copy(), float(max_kkt))
    with _LOCK:
        _CACHE[key] = value
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


def _mark_uncached(result):
    """Keep cache provenance present even when a branch cannot use the cache."""
    M, resolved, meta = result
    meta = dict(meta)
    meta["precision_cache_hit"] = False
    return M, resolved, meta


def build_nodewise_precision_numpy(X_work, *, requested_alpha, effective_n, weighted):
    X = np.asarray(X_work, dtype=np.float64)
    if X.ndim != 2 or int(X.shape[1]) <= 1 or _CACHE_MAXSIZE <= 0:
        return _mark_uncached(
            _precision.build_nodewise_precision_numpy(
                X_work,
                requested_alpha=requested_alpha,
                effective_n=effective_n,
                weighted=weighted,
            )
        )
    alpha, source, rule = _precision.resolve_nodewise_alpha(
        requested_alpha, p=X.shape[1], effective_n=effective_n
    )
    key = _key(_digest_numpy(X), backend_device="numpy:cpu", alpha=alpha)
    cached = _get(key)
    if cached is not None:
        M, max_kkt = cached
        return M, float(alpha), _metadata(
            requested_alpha, alpha, source, rule, effective_n, weighted, max_kkt, True
        )
    M, resolved, meta = _precision.build_nodewise_precision_numpy(
        X_work,
        requested_alpha=requested_alpha,
        effective_n=effective_n,
        weighted=weighted,
    )
    _put(key, M, meta.get("nodewise_max_kkt_residual", 0.0))
    meta = dict(meta)
    meta["precision_cache_hit"] = False
    return M, resolved, meta


def build_nodewise_precision_cupy(X_work, *, requested_alpha, effective_n, weighted):
    import cupy as cp

    X = cp.asarray(X_work, dtype=cp.float64)
    if X.ndim != 2 or int(X.shape[1]) <= 1 or _CACHE_MAXSIZE <= 0:
        return _mark_uncached(
            _precision.build_nodewise_precision_cupy(
                X_work,
                requested_alpha=requested_alpha,
                effective_n=effective_n,
                weighted=weighted,
            )
        )
    alpha, source, rule = _precision.resolve_nodewise_alpha(
        requested_alpha, p=X.shape[1], effective_n=effective_n
    )
    key = _key(
        _digest_cupy(X),
        backend_device=f"cupy:cuda:{int(X.device.id)}",
        alpha=alpha,
    )
    cached = _get(key)
    if cached is not None:
        M_np, max_kkt = cached
        return cp.asarray(M_np, dtype=X.dtype), float(alpha), _metadata(
            requested_alpha, alpha, source, rule, effective_n, weighted, max_kkt, True
        )
    M, resolved, meta = _precision.build_nodewise_precision_cupy(
        X_work,
        requested_alpha=requested_alpha,
        effective_n=effective_n,
        weighted=weighted,
    )
    _put(key, cp.asnumpy(M), meta.get("nodewise_max_kkt_residual", 0.0))
    meta = dict(meta)
    meta["precision_cache_hit"] = False
    return M, resolved, meta


def build_nodewise_precision_torch(X_work, *, requested_alpha, effective_n, weighted):
    import torch

    if not isinstance(X_work, torch.Tensor):
        return _mark_uncached(
            _precision.build_nodewise_precision_torch(
                X_work,
                requested_alpha=requested_alpha,
                effective_n=effective_n,
                weighted=weighted,
            )
        )
    X = X_work.to(dtype=torch.float64)
    if X.ndim != 2 or int(X.shape[1]) <= 1 or _CACHE_MAXSIZE <= 0:
        return _mark_uncached(
            _precision.build_nodewise_precision_torch(
                X_work,
                requested_alpha=requested_alpha,
                effective_n=effective_n,
                weighted=weighted,
            )
        )
    alpha, source, rule = _precision.resolve_nodewise_alpha(
        requested_alpha, p=X.shape[1], effective_n=effective_n
    )
    key = _key(
        _digest_torch(X),
        backend_device=f"torch:{X.device}",
        alpha=alpha,
    )
    cached = _get(key)
    if cached is not None:
        M_np, max_kkt = cached
        return torch.as_tensor(M_np, dtype=X.dtype, device=X.device), float(alpha), _metadata(
            requested_alpha, alpha, source, rule, effective_n, weighted, max_kkt, True
        )
    M, resolved, meta = _precision.build_nodewise_precision_torch(
        X_work,
        requested_alpha=requested_alpha,
        effective_n=effective_n,
        weighted=weighted,
    )
    _put(key, M.detach().cpu().numpy(), meta.get("nodewise_max_kkt_residual", 0.0))
    meta = dict(meta)
    meta["precision_cache_hit"] = False
    return M, resolved, meta


__all__ = [
    "build_nodewise_precision_numpy",
    "build_nodewise_precision_cupy",
    "build_nodewise_precision_torch",
]
