"""Unified resampling engine for bootstrap and permutation testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


def _is_cupy_array(x: Any) -> bool:
    try:
        import cupy as cp

        return isinstance(x, cp.ndarray)
    except Exception:
        return False


def _resolve_backend(backend: str, *arrays) -> str:
    backend_name = str(backend).strip().lower()
    if backend_name not in ("auto", "numpy", "cupy"):
        raise ValueError("backend must be one of: 'auto', 'numpy', 'cupy'")
    if backend_name != "auto":
        return backend_name

    for arr in arrays:
        if arr is not None and _is_cupy_array(arr):
            return "cupy"
    return "numpy"


def _get_xp(backend_name: str):
    if backend_name == "numpy":
        return np
    if backend_name == "cupy":
        import cupy as cp

        return cp
    raise ValueError(f"Unsupported backend: {backend_name}")


def _to_numpy(x):
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def _to_float(x) -> float:
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)


def _inf_value_for_xp(xp):
    return np.inf if xp is np else xp.inf


def _coerce_sample_value(x, xp):
    """Convert statistic output to a scalar array value for samples without forcing host sync."""
    try:
        x_arr = xp.asarray(x)
    except Exception:
        return float(x)

    if x_arr.ndim != 0:
        raise ValueError("statistic must return a scalar value")
    return x_arr.astype(xp.float64)


def _coerce_vectorized_values(values, expected_size: int, xp):
    """Normalize vectorized statistic output to a 1D float64 array or return None."""
    try:
        arr = xp.asarray(values)
    except Exception:
        return None

    if arr.ndim == 0:
        return None

    if arr.ndim != 1:
        if int(arr.size) != int(expected_size):
            return None
        arr = arr.reshape(-1)

    if int(arr.shape[0]) != int(expected_size):
        return None

    return arr.astype(xp.float64, copy=False)


def _try_vectorized_statistic(statistic, expected_size: int, xp, *args):
    """Try vectorized statistic call and return normalized output when compatible."""
    try:
        out = statistic(*args)
    except Exception:
        return None
    return _coerce_vectorized_values(out, expected_size, xp)


def _validate_fastpath_hint(statistic_hint: Optional[str]) -> Optional[str]:
    if statistic_hint is None:
        return None
    hint = str(statistic_hint).strip().lower()
    if hint in ("", "none"):
        return None
    allowed = {"mean", "pearson_corr"}
    if hint not in allowed:
        raise ValueError("statistic_hint must be one of: None, 'mean', 'pearson_corr'")
    return hint


def _mean_batch_stat(samples_batch, xp):
    return xp.mean(samples_batch, axis=-1, dtype=xp.float64)


def _select_single_feature_vector(X, xp):
    X_arr = xp.asarray(X)
    if X_arr.ndim == 1:
        return X_arr.astype(xp.float64, copy=False)
    if X_arr.ndim == 2 and int(X_arr.shape[1]) == 1:
        return X_arr[:, 0].astype(xp.float64, copy=False)
    raise ValueError("statistic_hint='pearson_corr' requires X with shape (n,) or (n, 1)")


def _pearson_corr_with_y_batch(x_vec, y_batch, xp):
    x = xp.asarray(x_vec, dtype=xp.float64).reshape(-1)
    y = xp.asarray(y_batch, dtype=xp.float64)

    x_centered = x - xp.mean(x)
    x_norm_sq = xp.sum(x_centered * x_centered)

    if y.ndim == 1:
        y_centered = y - xp.mean(y)
        denom = xp.sqrt(x_norm_sq * xp.sum(y_centered * y_centered))
        denom_safe = xp.where(denom > 0.0, denom, _inf_value_for_xp(xp))
        numer = xp.sum(y_centered * x_centered)
        return numer / denom_safe

    if y.ndim != 2:
        raise ValueError("y must be 1D or 2D batch matrix for pearson_corr fastpath")

    y_centered = y - xp.mean(y, axis=1, keepdims=True)
    y_norm_sq = xp.sum(y_centered * y_centered, axis=1)
    denom = xp.sqrt(x_norm_sq * y_norm_sq)
    denom_safe = xp.where(denom > 0.0, denom, _inf_value_for_xp(xp))
    numer = xp.sum(y_centered * x_centered.reshape(1, -1), axis=1)
    return numer / denom_safe


def _rng_default(backend_name: str, random_state: Optional[int]):
    if backend_name == "numpy":
        return np.random.default_rng(random_state)
    import cupy as cp

    seed = 0 if random_state is None else int(random_state)
    return cp.random.RandomState(seed)


def _rng_integers(rng, low: int, high: int, size, backend_name: str):
    if backend_name == "numpy":
        return rng.integers(low, high, size=size, dtype=np.int64)
    if hasattr(rng, "integers"):
        try:
            return rng.integers(low, high, size=size, dtype=np.int64)
        except TypeError:
            # Some RNG implementations may not support dtype explicitly.
            return rng.integers(low, high, size=size)
    return rng.randint(low, high, size=size, dtype="int64")


def _rng_permutation(rng, n: int, backend_name: str):
    if backend_name == "numpy":
        return rng.permutation(n)
    return rng.permutation(n)


def _rng_random(rng, size, backend_name: str, dtype=None):
    if backend_name == "numpy":
        if dtype is None:
            return rng.random(size=size)
        return rng.random(size=size, dtype=dtype)

    if hasattr(rng, "random"):
        if dtype is None:
            return rng.random(size=size)
        try:
            return rng.random(size=size, dtype=dtype)
        except TypeError:
            out = rng.random(size=size)
            if hasattr(out, "astype"):
                return out.astype(dtype, copy=False)
            return out

    out = rng.random_sample(size)
    if dtype is not None and hasattr(out, "astype"):
        return out.astype(dtype, copy=False)
    return out


def _cupy_index_dtype_name(n: int) -> str:
    return "int32" if int(n) <= np.iinfo(np.int32).max else "int64"


def _recommend_cupy_batch_size(
    n: int,
    n_resamples: int,
    *,
    bytes_per_row: int,
    target_bytes: int,
    min_batch: int,
    max_batch: int,
) -> int:
    if n <= 0:
        return 1

    by_memory = max(1, target_bytes // max(1, bytes_per_row * n))
    batch = min(max_batch, max(min_batch, by_memory))
    return max(1, min(batch, int(n_resamples)))


def _iter_iid_bootstrap_index_batches(rng, n: int, n_resamples: int, backend_name: str):
    if backend_name == "numpy":
        batch_size = _recommend_cupy_batch_size(
            n,
            n_resamples,
            bytes_per_row=8,
            target_bytes=32 * 1024 * 1024,
            min_batch=8,
            max_batch=1024,
        )
        for start in range(0, n_resamples, batch_size):
            cur = min(batch_size, n_resamples - start)
            idx_batch = _rng_integers(rng, 0, n, size=(cur, n), backend_name=backend_name)
            yield idx_batch
        return

    # int64 index matrix; keep around ~64MB to balance throughput and memory.
    batch_size = _recommend_cupy_batch_size(
        n,
        n_resamples,
        bytes_per_row=8,
        target_bytes=64 * 1024 * 1024,
        min_batch=32,
        max_batch=2048,
    )
    index_dtype = _cupy_index_dtype_name(n)

    for start in range(0, n_resamples, batch_size):
        cur = min(batch_size, n_resamples - start)
        if hasattr(rng, "integers"):
            try:
                idx_batch = rng.integers(0, n, size=(cur, n), dtype=index_dtype)
            except TypeError:
                idx_batch = rng.integers(0, n, size=(cur, n))
        else:
            idx_batch = rng.randint(0, n, size=(cur, n), dtype=index_dtype)
        yield idx_batch


def _iter_iid_permutation_batches(rng, n: int, n_resamples: int, backend_name: str):
    if backend_name == "numpy":
        batch_size = _recommend_cupy_batch_size(
            n,
            n_resamples,
            bytes_per_row=12,
            target_bytes=24 * 1024 * 1024,
            min_batch=4,
            max_batch=256,
        )
        for start in range(0, n_resamples, batch_size):
            cur = min(batch_size, n_resamples - start)
            keys = _rng_random(rng, (cur, n), backend_name, dtype=np.float32)
            perm_batch = np.argsort(keys, axis=1)
            yield perm_batch
        return

    # Approx memory per row: float32 random keys + int64 permutation indices.
    batch_size = _recommend_cupy_batch_size(
        n,
        n_resamples,
        bytes_per_row=12,
        target_bytes=48 * 1024 * 1024,
        min_batch=16,
        max_batch=2048,
    )

    import cupy as cp

    for start in range(0, n_resamples, batch_size):
        cur = min(batch_size, n_resamples - start)
        # Generate keys in batch then argsort per row to obtain permutations.
        keys = _rng_random(rng, (cur, n), backend_name, dtype=cp.float32)
        perm_batch = cp.argsort(keys, axis=1)
        yield perm_batch


def _iter_stratified_bootstrap_index_batches(
    rng,
    state,
    n_resamples: int,
    backend_name: str,
    *,
    shuffle_rows: bool = True,
):
    xp = _get_xp(backend_name)
    strata_rows = state["strata_rows"]
    strata_rows_matrix = state.get("strata_rows_matrix")
    strata_uniform_size = state.get("strata_uniform_size")
    n = int(state["n_samples"])

    if backend_name == "numpy":
        target = 24 * 1024 * 1024
        min_batch = 4
        max_batch = 512
        key_dtype = np.float32
    else:
        target = 64 * 1024 * 1024
        min_batch = 16
        max_batch = 1024
        import cupy as cp

        key_dtype = cp.float32

    bytes_per_row = 8 * n + (4 * n if shuffle_rows else 0)
    batch_size = _recommend_cupy_batch_size(
        n,
        n_resamples,
        bytes_per_row=bytes_per_row,
        target_bytes=target,
        min_batch=min_batch,
        max_batch=max_batch,
    )

    for start in range(0, n_resamples, batch_size):
        cur = min(batch_size, n_resamples - start)
        if strata_rows_matrix is not None and strata_uniform_size is not None:
            n_strata = int(strata_rows_matrix.shape[0])
            m = int(strata_uniform_size)
            sampled_local = _rng_integers(
                rng,
                0,
                m,
                size=(cur, n_strata, m),
                backend_name=backend_name,
            )
            strata_ids = xp.arange(n_strata, dtype=xp.int64).reshape(1, n_strata, 1)
            idx_batch = strata_rows_matrix[strata_ids, sampled_local].reshape(cur, -1)
        else:
            idx_batch = xp.empty((cur, n), dtype=xp.int64)
            offset = 0
            for pos in strata_rows:
                m = int(pos.size)
                sampled_local = _rng_integers(rng, 0, m, size=(cur, m), backend_name=backend_name)
                idx_batch[:, offset : offset + m] = pos[sampled_local]
                offset += m

        if shuffle_rows:
            keys = _rng_random(rng, (cur, n), backend_name, dtype=key_dtype)
            perm = xp.argsort(keys, axis=1)
            idx_batch = xp.take_along_axis(idx_batch, perm, axis=1)

        yield idx_batch


def _iter_block_bootstrap_index_batches(
    rng,
    state,
    n_resamples: int,
    backend_name: str,
):
    xp = _get_xp(backend_name)
    n = int(state["n_samples"])
    b = int(state["block_size"])
    n_blocks = int(state["n_blocks"])
    max_start = int(state["max_start"])

    if backend_name == "numpy":
        target = 24 * 1024 * 1024
        min_batch = 4
        max_batch = 512
    else:
        target = 64 * 1024 * 1024
        min_batch = 16
        max_batch = 1024

    bytes_per_row = 8 * max(1, n)
    batch_size = _recommend_cupy_batch_size(
        max(1, n_blocks),
        n_resamples,
        bytes_per_row=bytes_per_row,
        target_bytes=target,
        min_batch=min_batch,
        max_batch=max_batch,
    )

    offsets = xp.arange(b, dtype=xp.int64).reshape(1, 1, b)
    for start in range(0, n_resamples, batch_size):
        cur = min(batch_size, n_resamples - start)
        starts = _rng_integers(rng, 0, max_start, size=(cur, n_blocks), backend_name=backend_name)
        idx_batch = (starts[:, :, None] + offsets).reshape(cur, -1)
        yield idx_batch[:, :n].astype(xp.int64)


def _iter_cluster_bootstrap_index_batches(
    rng,
    state,
    n_resamples: int,
    backend_name: str,
):
    """Batch cluster bootstrap index generation for uniform cluster sizes."""
    xp = _get_xp(backend_name)
    n = int(state["n_samples"])
    n_clusters = int(state["n_clusters"])
    rows_matrix = state.get("cluster_rows_matrix")
    uniform_size = state.get("cluster_uniform_size")

    if rows_matrix is None or uniform_size is None:
        raise ValueError("Batched cluster bootstrap requires uniform cluster sizes")

    m = int(uniform_size)
    draws = int(np.ceil(n / max(1, m)))
    total_len = draws * m

    if backend_name == "numpy":
        target = 24 * 1024 * 1024
        min_batch = 4
        max_batch = 512
    else:
        target = 64 * 1024 * 1024
        min_batch = 16
        max_batch = 1024

    batch_size = _recommend_cupy_batch_size(
        max(1, total_len),
        n_resamples,
        bytes_per_row=8,
        target_bytes=target,
        min_batch=min_batch,
        max_batch=max_batch,
    )

    for start in range(0, n_resamples, batch_size):
        cur = min(batch_size, n_resamples - start)
        cluster_ids = _rng_integers(rng, 0, n_clusters, size=(cur, draws), backend_name=backend_name)
        idx_batch = rows_matrix[cluster_ids].reshape(cur, -1)
        yield idx_batch[:, :n].astype(xp.int64)


def _iter_non_iid_bootstrap_index_batches(
    rng,
    state,
    n_resamples: int,
    backend_name: str,
    *,
    shuffle_rows: bool = True,
):
    strategy_n = state["strategy"]
    if strategy_n == "stratified":
        yield from _iter_stratified_bootstrap_index_batches(
            rng,
            state,
            n_resamples,
            backend_name,
            shuffle_rows=shuffle_rows,
        )
        return
    if strategy_n == "block":
        yield from _iter_block_bootstrap_index_batches(rng, state, n_resamples, backend_name)
        return
    if strategy_n == "cluster":
        yield from _iter_cluster_bootstrap_index_batches(rng, state, n_resamples, backend_name)
        return
    raise ValueError("Batched non-IID bootstrap supports only 'stratified', 'cluster', and 'block'")


def _iter_labelwise_permuted_y_batches(
    rng,
    y,
    state,
    n_resamples: int,
    backend_name: str,
):
    xp = _get_xp(backend_name)
    y_arr = xp.asarray(y)
    n = int(state["n_samples"])
    label_rows = state["label_rows"]
    dense_label_rows = state.get("dense_label_rows")
    dense_valid_mask = state.get("dense_valid_mask")
    dense_valid_flat = state.get("dense_valid_flat")
    dense_pos_valid = state.get("dense_pos_valid")
    label_sizes = state.get("label_sizes")

    use_dense = (
        dense_label_rows is not None
        and dense_valid_mask is not None
        and dense_valid_flat is not None
        and dense_pos_valid is not None
        and label_sizes is not None
    )

    if backend_name == "numpy":
        target = 24 * 1024 * 1024
        min_batch = 4
        max_batch = 512
        key_dtype = np.float32
    else:
        target = 64 * 1024 * 1024
        min_batch = 16
        max_batch = 1024
        import cupy as cp

        key_dtype = cp.float32

    if use_dense:
        n_labels = int(dense_label_rows.shape[0])
        max_label_size = int(dense_label_rows.shape[1])
        dense_elems = n_labels * max_label_size
        # Dense mode holds key and permutation tensors over (labels, max_label_size).
        bytes_per_row = max(8, y_arr.dtype.itemsize) * n + 12 * dense_elems
        size_for_batch = max(1, dense_elems)
    else:
        bytes_per_row = max(8, y_arr.dtype.itemsize) * n + 4 * n
        size_for_batch = n

    batch_size = _recommend_cupy_batch_size(
        size_for_batch,
        n_resamples,
        bytes_per_row=bytes_per_row,
        target_bytes=target,
        min_batch=min_batch,
        max_batch=max_batch,
    )

    for start in range(0, n_resamples, batch_size):
        cur = min(batch_size, n_resamples - start)
        y_batch = xp.empty((cur, n), dtype=y_arr.dtype)

        if use_dense:
            keys = _rng_random(
                rng,
                (cur, int(dense_label_rows.shape[0]), int(dense_label_rows.shape[1])),
                backend_name,
                dtype=key_dtype,
            )
            keys = xp.where(dense_valid_mask.reshape(1, *dense_valid_mask.shape), keys, _inf_value_for_xp(xp))
            perm_dense = xp.argsort(keys, axis=2)
            shuffled_dense = xp.take_along_axis(
                dense_label_rows.reshape(1, *dense_label_rows.shape),
                perm_dense,
                axis=2,
            )

            # Flatten valid entries once and write all groups in one vectorized assignment.
            shuffled_valid = shuffled_dense.reshape(cur, -1)[:, dense_valid_flat]
            y_batch[:, dense_pos_valid] = y_arr[shuffled_valid]

            yield y_batch
            continue

        for pos in label_rows:
            m = int(pos.size)
            if m == 1:
                y_batch[:, pos] = y_arr[pos]
                continue
            keys = _rng_random(rng, (cur, m), backend_name, dtype=key_dtype)
            perm = xp.argsort(keys, axis=1)
            y_batch[:, pos] = y_arr[pos][perm]

        yield y_batch


@dataclass
class BootstrapResult:
    """Result object for bootstrap-based statistics."""

    statistic_name: str
    strategy: str
    observed: float
    samples: Any
    confidence_interval: Tuple[float, float]
    confidence_level: float
    n_resamples: int
    random_state: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        samples_np = _to_numpy(self.samples)
        return {
            "statistic_name": self.statistic_name,
            "strategy": self.strategy,
            "observed": float(self.observed),
            "samples": samples_np.tolist(),
            "confidence_interval": [
                float(self.confidence_interval[0]),
                float(self.confidence_interval[1]),
            ],
            "confidence_level": float(self.confidence_level),
            "n_resamples": int(self.n_resamples),
            "random_state": self.random_state,
            "metadata": self.metadata,
        }

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe()") from exc

        samples_np = _to_numpy(self.samples)
        return pd.DataFrame(
            {
                "sample_index": np.arange(samples_np.size, dtype=int),
                "statistic": samples_np,
            }
        )


@dataclass
class PermutationTestResult:
    """Result object for permutation tests."""

    statistic_name: str
    strategy: str
    alternative: str
    observed: float
    samples: Any
    pvalue: float
    n_resamples: int
    random_state: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        samples_np = _to_numpy(self.samples)
        return {
            "statistic_name": self.statistic_name,
            "strategy": self.strategy,
            "alternative": self.alternative,
            "observed": float(self.observed),
            "samples": samples_np.tolist(),
            "pvalue": float(self.pvalue),
            "n_resamples": int(self.n_resamples),
            "random_state": self.random_state,
            "metadata": self.metadata,
        }

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for to_dataframe()") from exc

        samples_np = _to_numpy(self.samples)
        return pd.DataFrame(
            {
                "sample_index": np.arange(samples_np.size, dtype=int),
                "statistic": samples_np,
            }
        )


def _validate_confidence_level(confidence_level: float) -> float:
    level = float(confidence_level)
    if level <= 0.0 or level >= 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    return level


def _validate_n_resamples(n_resamples: int) -> int:
    n = int(n_resamples)
    if n <= 0:
        raise ValueError("n_resamples must be a positive integer")
    return n


def _ensure_same_first_dim(arrays: Sequence[Any]) -> int:
    if len(arrays) == 0:
        raise ValueError("At least one array is required")
    n = arrays[0].shape[0]
    for arr in arrays[1:]:
        if arr.shape[0] != n:
            raise ValueError("All arrays must have the same length in axis 0")
    return n


def _bootstrap_indices_iid(rng, n: int, backend_name: str):
    return _rng_integers(rng, 0, n, size=n, backend_name=backend_name)


def _prepare_bootstrap_state(
    n: int,
    strategy: str,
    strata,
    clusters,
    block_size: Optional[int],
    backend_name: str,
):
    xp = _get_xp(backend_name)
    strategy_n = str(strategy).strip().lower()

    if strategy_n == "iid":
        return {"strategy": strategy_n, "n_samples": int(n)}

    if strategy_n == "stratified":
        if strata is None:
            raise ValueError("strata is required when strategy='stratified'")
        strata_arr = xp.asarray(strata).reshape(-1)
        if int(strata_arr.shape[0]) != n:
            raise ValueError("strata must have the same length as arrays")
        labels = xp.unique(strata_arr)
        rows = tuple(xp.where(strata_arr == label)[0].astype(xp.int64) for label in labels)
        sizes = np.asarray([int(r.size) for r in rows], dtype=np.int64)
        uniform_size = int(sizes[0]) if sizes.size > 0 and np.all(sizes == sizes[0]) else None
        rows_matrix = None
        if uniform_size is not None:
            rows_matrix = xp.stack(rows, axis=0).astype(xp.int64)
        return {
            "strategy": strategy_n,
            "n_samples": int(n),
            "strata_rows": rows,
            "strata_sizes": tuple(int(s) for s in sizes.tolist()),
            "strata_uniform_size": uniform_size,
            "strata_rows_matrix": rows_matrix,
        }

    if strategy_n == "cluster":
        if clusters is None:
            raise ValueError("clusters is required when strategy='cluster'")
        clusters_arr = xp.asarray(clusters).reshape(-1)
        if int(clusters_arr.shape[0]) != n:
            raise ValueError("clusters must have the same length as arrays")
        labels = xp.unique(clusters_arr)
        rows = tuple(xp.where(clusters_arr == label)[0].astype(xp.int64) for label in labels)
        if len(rows) == 0:
            raise ValueError("clusters must contain at least one group")
        sizes = np.asarray([int(r.size) for r in rows], dtype=np.int64)
        avg_size = float(np.mean(sizes)) if sizes.size > 0 else 1.0
        avg_size = max(avg_size, 1.0)
        uniform_size = int(sizes[0]) if np.all(sizes == sizes[0]) else None
        rows_matrix = None
        if uniform_size is not None:
            # Uniform clusters can be assembled in dense batched form without padding/masking.
            rows_matrix = xp.stack(rows, axis=0).astype(xp.int64)
        return {
            "strategy": strategy_n,
            "n_samples": int(n),
            "cluster_rows": rows,
            "cluster_sizes": sizes,
            "n_clusters": len(rows),
            "avg_cluster_size": avg_size,
            "cluster_uniform_size": uniform_size,
            "cluster_rows_matrix": rows_matrix,
        }

    if strategy_n == "block":
        b = int(block_size) if block_size is not None else 0
        if b <= 0:
            raise ValueError("block_size must be a positive integer for block bootstrap")
        b_eff = min(b, n)
        n_blocks = int(np.ceil(n / b_eff))
        max_start = max(1, n - b_eff + 1)
        return {
            "strategy": strategy_n,
            "n_samples": int(n),
            "block_size": b_eff,
            "n_blocks": n_blocks,
            "max_start": max_start,
        }

    raise ValueError("strategy must be one of: 'iid', 'stratified', 'cluster', 'block'")


def _bootstrap_indices_stratified(
    rng,
    state,
    backend_name: str,
):
    xp = _get_xp(backend_name)
    chunks = []
    n = 0
    for pos in state["strata_rows"]:
        pos_n = int(pos.size)
        sampled_local = _rng_integers(rng, 0, pos_n, size=pos_n, backend_name=backend_name)
        chunks.append(pos[sampled_local])
        n += pos_n

    idx = xp.concatenate(chunks) if chunks else xp.empty((0,), dtype=xp.int64)
    if int(idx.size) != int(n):
        raise RuntimeError("Stratified bootstrap produced invalid sample size")

    perm = _rng_permutation(rng, int(idx.size), backend_name)
    return idx[perm].astype(xp.int64)


def _bootstrap_indices_cluster(
    rng,
    n: int,
    state,
    backend_name: str,
):
    xp = _get_xp(backend_name)
    cluster_rows = state["cluster_rows"]
    cluster_sizes = state["cluster_sizes"]
    n_clusters = int(state["n_clusters"])
    avg_size = float(state["avg_cluster_size"])

    # Sample cluster ids in batches to avoid scalar sync per sampled cluster.
    selected_ids = []
    total_size = 0
    batch = max(4, int(np.ceil(n / avg_size)))

    while total_size < n:
        ids = _rng_integers(rng, 0, n_clusters, size=batch, backend_name=backend_name)
        ids_np = _to_numpy(ids).astype(np.int64, copy=False)
        selected_ids.extend(ids_np.tolist())
        total_size += int(cluster_sizes[ids_np].sum())
        if total_size < n:
            remaining = n - total_size
            batch = max(1, int(np.ceil(remaining / avg_size)) + 1)

    chunks = []
    filled = 0
    for cid in selected_ids:
        rows = cluster_rows[int(cid)]
        chunks.append(rows)
        filled += int(rows.size)
        if filled >= n:
            break

    idx = xp.concatenate(chunks)[:n] if chunks else xp.empty((0,), dtype=xp.int64)
    return idx.astype(xp.int64)


def _bootstrap_indices_block(
    rng,
    n: int,
    state,
    backend_name: str,
):
    xp = _get_xp(backend_name)
    b = int(state["block_size"])
    n_blocks = int(state["n_blocks"])
    max_start = int(state["max_start"])

    starts = _rng_integers(rng, 0, max_start, size=n_blocks, backend_name=backend_name)
    offsets = xp.arange(b, dtype=xp.int64)
    idx = (starts.reshape(-1, 1) + offsets.reshape(1, -1)).reshape(-1)
    return idx[:n].astype(xp.int64)

def _build_bootstrap_indices(
    rng,
    n: int,
    state,
    backend_name: str,
):
    strategy_n = state["strategy"]
    if strategy_n == "iid":
        return _bootstrap_indices_iid(rng, n, backend_name)
    if strategy_n == "stratified":
        return _bootstrap_indices_stratified(rng, state, backend_name)
    if strategy_n == "cluster":
        return _bootstrap_indices_cluster(rng, n, state, backend_name)
    if strategy_n == "block":
        return _bootstrap_indices_block(rng, n, state, backend_name)
    raise ValueError("strategy must be one of: 'iid', 'stratified', 'cluster', 'block'")


def bootstrap_statistic(
    statistic: Callable[..., float],
    *arrays,
    n_resamples: int = 200,
    strategy: str = "iid",
    strata=None,
    clusters=None,
    block_size: Optional[int] = None,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    statistic_name: str = "statistic",
    backend: str = "auto",
    force_vectorized: bool = False,
    statistic_hint: Optional[str] = None,
) -> BootstrapResult:
    """
    Generic bootstrap engine over one or multiple aligned arrays.

    Parameters
    ----------
    statistic : callable
        A function receiving resampled arrays and returning a scalar.
        On CuPy IID paths, a vectorized callable is also supported:
        if called with batched samples and it returns a vector of length
        ``batch_size``, that vectorized output is used directly.
    *arrays : array-like
        One or more arrays with aligned first dimension.
    n_resamples : int, default=200
        Number of bootstrap resamples.
    strategy : {'iid', 'stratified', 'cluster', 'block'}, default='iid'
        Resampling strategy.
    strata : array-like, optional
        Strata labels used by stratified bootstrap.
    clusters : array-like, optional
        Cluster labels used by cluster bootstrap.
    block_size : int, optional
        Block size for block bootstrap.
    confidence_level : float, default=0.95
        Confidence level for percentile CI.
    random_state : int, optional
        Seed for reproducibility.
    statistic_name : str, default='statistic'
        Name to attach to the result object.
    backend : {'auto', 'numpy', 'cupy'}, default='auto'
        Backend selection. 'auto' infers from input arrays.
    force_vectorized : bool, default=False
        If True, require the statistic callable (or fastpath) to produce
        vectorized batch output on IID path; raises if unavailable.
    statistic_hint : {'mean', 'pearson_corr'} or None, default=None
        Optional built-in fastpath hint. For bootstrap, ``'mean'`` enables
        direct batch mean computation on IID path.

    Returns
    -------
    BootstrapResult
        Structured bootstrap result with samples and confidence interval.
    """
    n_boot = _validate_n_resamples(n_resamples)
    level = _validate_confidence_level(confidence_level)

    backend_name = _resolve_backend(backend, *arrays, strata, clusters)
    xp = _get_xp(backend_name)

    arrays_xp = [xp.asarray(a) for a in arrays]
    n = _ensure_same_first_dim(arrays_xp)
    if strata is not None and xp.asarray(strata).shape[0] != n:
        raise ValueError("strata must have the same length as arrays")
    if clusters is not None and xp.asarray(clusters).shape[0] != n:
        raise ValueError("clusters must have the same length as arrays")

    observed = _to_float(statistic(*arrays_xp))
    fastpath_hint = _validate_fastpath_hint(statistic_hint)
    bootstrap_state = _prepare_bootstrap_state(
        n,
        strategy,
        strata,
        clusters,
        block_size,
        backend_name,
    )

    rng = _rng_default(backend_name, random_state)
    samples = xp.empty(n_boot, dtype=xp.float64)
    strategy_n = bootstrap_state["strategy"]

    if strategy_n == "iid":
        vectorized_mode = None
        write_pos = 0
        for idx_batch in _iter_iid_bootstrap_index_batches(rng, n, n_boot, backend_name):
            cur = int(idx_batch.shape[0])

            if fastpath_hint == "mean":
                if len(arrays_xp) != 1:
                    raise ValueError("statistic_hint='mean' requires a single input array")
                sampled_batch = arrays_xp[0][idx_batch]
                samples[write_pos : write_pos + cur] = _mean_batch_stat(sampled_batch, xp)
                write_pos += cur
                continue

            if len(arrays_xp) == 1:
                sampled_batch = arrays_xp[0][idx_batch]
                if vectorized_mode is not False:
                    vec_values = _try_vectorized_statistic(statistic, cur, xp, sampled_batch)
                    if vec_values is not None:
                        samples[write_pos : write_pos + cur] = vec_values
                        vectorized_mode = True
                        write_pos += cur
                        continue
                    if vectorized_mode is None:
                        if force_vectorized:
                            raise ValueError(
                                "force_vectorized=True but statistic did not return "
                                "a vector of length batch_size"
                            )
                        vectorized_mode = False
                for j in range(cur):
                    samples[write_pos + j] = _coerce_sample_value(statistic(sampled_batch[j]), xp)
            else:
                sampled_args_batch = [arr[idx_batch] for arr in arrays_xp]
                if vectorized_mode is not False:
                    vec_values = _try_vectorized_statistic(
                        statistic,
                        cur,
                        xp,
                        *sampled_args_batch,
                    )
                    if vec_values is not None:
                        samples[write_pos : write_pos + cur] = vec_values
                        vectorized_mode = True
                        write_pos += cur
                        continue
                    if vectorized_mode is None:
                        if force_vectorized:
                            raise ValueError(
                                "force_vectorized=True but statistic did not return "
                                "a vector of length batch_size"
                            )
                        vectorized_mode = False
                for j in range(cur):
                    sampled_args = [arr[j] for arr in sampled_args_batch]
                    samples[write_pos + j] = _coerce_sample_value(statistic(*sampled_args), xp)
            write_pos += cur
    elif strategy_n in ("stratified", "block") or (
        strategy_n == "cluster" and bootstrap_state.get("cluster_rows_matrix") is not None
    ):
        vectorized_mode = None
        write_pos = 0
        shuffle_rows = not (fastpath_hint == "mean")

        for idx_batch in _iter_non_iid_bootstrap_index_batches(
            rng,
            bootstrap_state,
            n_boot,
            backend_name,
            shuffle_rows=shuffle_rows,
        ):
            cur = int(idx_batch.shape[0])

            if fastpath_hint == "mean":
                if len(arrays_xp) != 1:
                    raise ValueError("statistic_hint='mean' requires a single input array")
                sampled_batch = arrays_xp[0][idx_batch]
                samples[write_pos : write_pos + cur] = _mean_batch_stat(sampled_batch, xp)
                write_pos += cur
                continue

            if len(arrays_xp) == 1:
                sampled_batch = arrays_xp[0][idx_batch]
                if vectorized_mode is not False:
                    vec_values = _try_vectorized_statistic(statistic, cur, xp, sampled_batch)
                    if vec_values is not None:
                        samples[write_pos : write_pos + cur] = vec_values
                        vectorized_mode = True
                        write_pos += cur
                        continue
                    if vectorized_mode is None:
                        if force_vectorized:
                            raise ValueError(
                                "force_vectorized=True but statistic did not return "
                                "a vector of length batch_size"
                            )
                        vectorized_mode = False
                for j in range(cur):
                    samples[write_pos + j] = _coerce_sample_value(statistic(sampled_batch[j]), xp)
            else:
                sampled_args_batch = [arr[idx_batch] for arr in arrays_xp]
                if vectorized_mode is not False:
                    vec_values = _try_vectorized_statistic(
                        statistic,
                        cur,
                        xp,
                        *sampled_args_batch,
                    )
                    if vec_values is not None:
                        samples[write_pos : write_pos + cur] = vec_values
                        vectorized_mode = True
                        write_pos += cur
                        continue
                    if vectorized_mode is None:
                        if force_vectorized:
                            raise ValueError(
                                "force_vectorized=True but statistic did not return "
                                "a vector of length batch_size"
                            )
                        vectorized_mode = False
                for j in range(cur):
                    sampled_args = [arr[j] for arr in sampled_args_batch]
                    samples[write_pos + j] = _coerce_sample_value(statistic(*sampled_args), xp)
            write_pos += cur
    else:
        for i in range(n_boot):
            idx = _build_bootstrap_indices(
                rng,
                n,
                bootstrap_state,
                backend_name,
            )
            sampled_args = [arr[idx] for arr in arrays_xp]
            samples[i] = _coerce_sample_value(statistic(*sampled_args), xp)

    alpha = 1.0 - level
    ci = (
        _to_float(xp.quantile(samples, alpha / 2.0)),
        _to_float(xp.quantile(samples, 1.0 - alpha / 2.0)),
    )

    return BootstrapResult(
        statistic_name=str(statistic_name),
        strategy=str(strategy).lower(),
        observed=observed,
        samples=samples,
        confidence_interval=ci,
        confidence_level=level,
        n_resamples=n_boot,
        random_state=random_state,
        metadata={"n_samples": n, "backend": backend_name},
    )


def _permute_y(
    rng,
    y,
    state,
    backend_name: str,
):
    xp = _get_xp(backend_name)
    strategy_n = state["strategy"]
    y_arr = xp.asarray(y)

    if strategy_n == "iid":
        perm = _rng_permutation(rng, int(y_arr.shape[0]), backend_name)
        return y_arr[perm]

    if strategy_n in ("stratified", "grouped"):
        y_perm = y_arr.copy()
        for pos in state["label_rows"]:
            shuffled_pos = pos[_rng_permutation(rng, int(pos.size), backend_name)]
            y_perm[pos] = y_arr[shuffled_pos]
        return y_perm

    raise ValueError("strategy must be one of: 'iid', 'stratified', 'grouped'")


def _prepare_permutation_state(
    n: int,
    strategy: str,
    strata,
    groups,
    backend_name: str,
):
    xp = _get_xp(backend_name)
    strategy_n = str(strategy).strip().lower()

    if strategy_n == "iid":
        return {"strategy": strategy_n, "n_samples": int(n)}

    if strategy_n in ("stratified", "grouped"):
        labels = strata if strategy_n == "stratified" else groups
        if labels is None:
            key = "strata" if strategy_n == "stratified" else "groups"
            raise ValueError(f"{key} is required when strategy='{strategy_n}'")

        labels_arr = xp.asarray(labels).reshape(-1)
        if int(labels_arr.shape[0]) != n:
            raise ValueError("labels must have same length as y")

        unique_labels = xp.unique(labels_arr)
        label_rows = tuple(xp.where(labels_arr == label)[0].astype(xp.int64) for label in unique_labels)

        dense_label_rows = None
        dense_valid_mask = None
        dense_valid_flat = None
        dense_pos_valid = None
        label_sizes = tuple(int(pos.size) for pos in label_rows)

        # Build a dense label matrix for CuPy when groups are not too ragged.
        if backend_name == "cupy" and len(label_sizes) > 0:
            max_label_size = max(label_sizes)
            if max_label_size > 1:
                fill_ratio = float(n) / float(len(label_sizes) * max_label_size)
                if fill_ratio >= 0.60:
                    dense_label_rows = xp.full((len(label_rows), max_label_size), -1, dtype=xp.int64)
                    dense_valid_mask = xp.zeros((len(label_rows), max_label_size), dtype=bool)
                    for i, pos in enumerate(label_rows):
                        m = label_sizes[i]
                        dense_label_rows[i, :m] = pos
                        dense_valid_mask[i, :m] = True
                    dense_valid_flat = dense_valid_mask.reshape(-1)
                    dense_pos_valid = dense_label_rows.reshape(-1)[dense_valid_flat]

        return {
            "strategy": strategy_n,
            "n_samples": int(n),
            "label_rows": label_rows,
            "label_sizes": label_sizes,
            "dense_label_rows": dense_label_rows,
            "dense_valid_mask": dense_valid_mask,
            "dense_valid_flat": dense_valid_flat,
            "dense_pos_valid": dense_pos_valid,
        }

    raise ValueError("strategy must be one of: 'iid', 'stratified', 'grouped'")


def permutation_test(
    statistic: Callable[[Any, Any], float],
    X,
    y,
    n_resamples: int = 1000,
    strategy: str = "iid",
    strata=None,
    groups=None,
    alternative: str = "two-sided",
    random_state: Optional[int] = None,
    statistic_name: str = "statistic",
    backend: str = "auto",
    force_vectorized: bool = False,
    statistic_hint: Optional[str] = None,
) -> PermutationTestResult:
    """
    Generic permutation test for a supervised statistic ``statistic(X, y)``.

    Parameters
    ----------
    statistic : callable
        Function receiving ``(X, y)`` and returning a scalar.
        On CuPy IID paths, vectorized output is supported when ``y`` is a
        batch matrix and the callable returns a vector with one value per row.
    X : array-like
        Feature matrix.
    y : array-like
        Response vector.
    n_resamples : int, default=1000
        Number of permutation resamples.
    strategy : {'iid', 'stratified', 'grouped'}, default='iid'
        Permutation strategy. 'grouped' permutes within groups.
    strata : array-like, optional
        Strata labels used by strategy='stratified'.
    groups : array-like, optional
        Group labels used by strategy='grouped'.
    alternative : {'two-sided', 'greater', 'less'}, default='two-sided'
        Alternative hypothesis.
    random_state : int, optional
        Random seed.
    statistic_name : str, default='statistic'
        Name to attach to the result object.
    backend : {'auto', 'numpy', 'cupy'}, default='auto'
        Backend selection. 'auto' infers from input arrays.
    force_vectorized : bool, default=False
        If True, require vectorized batch output on IID path; raises if
        statistic is not vectorized-compatible.
    statistic_hint : {'mean', 'pearson_corr'} or None, default=None
        Optional built-in fastpath hint. For permutation, ``'pearson_corr'``
        computes Pearson correlation in vectorized batches for IID path.

    Returns
    -------
    PermutationTestResult
        Structured permutation test result with empirical p-value.
    """
    n_perm = _validate_n_resamples(n_resamples)
    alt = str(alternative).strip().lower()
    if alt not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be one of: 'two-sided', 'greater', 'less'")

    backend_name = _resolve_backend(backend, X, y, strata, groups)
    xp = _get_xp(backend_name)

    X_arr = xp.asarray(X)
    y_arr = xp.asarray(y).reshape(-1)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    observed = _to_float(statistic(X_arr, y_arr))
    fastpath_hint = _validate_fastpath_hint(statistic_hint)
    permutation_state = _prepare_permutation_state(
        int(y_arr.shape[0]),
        strategy,
        strata,
        groups,
        backend_name,
    )

    rng = _rng_default(backend_name, random_state)
    samples = xp.empty(n_perm, dtype=xp.float64)
    strategy_n = permutation_state["strategy"]

    x_vec_fast = None
    if fastpath_hint == "pearson_corr":
        x_vec_fast = _select_single_feature_vector(X_arr, xp)

    if strategy_n == "iid":

        vectorized_mode = None
        write_pos = 0
        for perm_batch in _iter_iid_permutation_batches(
            rng,
            int(y_arr.shape[0]),
            n_perm,
            backend_name,
        ):
            cur = int(perm_batch.shape[0])
            y_perm_batch = y_arr[perm_batch]

            if fastpath_hint == "pearson_corr":
                corr_batch = _pearson_corr_with_y_batch(x_vec_fast, y_perm_batch, xp)
                samples[write_pos : write_pos + cur] = _coerce_vectorized_values(corr_batch, cur, xp)
                write_pos += cur
                continue

            if vectorized_mode is not False:
                vec_values = _try_vectorized_statistic(
                    statistic,
                    cur,
                    xp,
                    X_arr,
                    y_perm_batch,
                )
                if vec_values is not None:
                    samples[write_pos : write_pos + cur] = vec_values
                    vectorized_mode = True
                    write_pos += cur
                    continue
                if vectorized_mode is None:
                    if force_vectorized:
                        raise ValueError(
                            "force_vectorized=True but statistic did not return "
                            "a vector of length batch_size"
                        )
                    vectorized_mode = False
            for j in range(cur):
                samples[write_pos + j] = _coerce_sample_value(statistic(X_arr, y_perm_batch[j]), xp)
            write_pos += cur
    else:
        vectorized_mode = None
        write_pos = 0
        for y_perm_batch in _iter_labelwise_permuted_y_batches(
            rng,
            y_arr,
            permutation_state,
            n_perm,
            backend_name,
        ):
            cur = int(y_perm_batch.shape[0])

            if fastpath_hint == "pearson_corr":
                corr_batch = _pearson_corr_with_y_batch(x_vec_fast, y_perm_batch, xp)
                samples[write_pos : write_pos + cur] = _coerce_vectorized_values(corr_batch, cur, xp)
                write_pos += cur
                continue

            if vectorized_mode is not False:
                vec_values = _try_vectorized_statistic(
                    statistic,
                    cur,
                    xp,
                    X_arr,
                    y_perm_batch,
                )
                if vec_values is not None:
                    samples[write_pos : write_pos + cur] = vec_values
                    vectorized_mode = True
                    write_pos += cur
                    continue
                if vectorized_mode is None:
                    if force_vectorized:
                        raise ValueError(
                            "force_vectorized=True but statistic did not return "
                            "a vector of length batch_size"
                        )
                    vectorized_mode = False

            for j in range(cur):
                samples[write_pos + j] = _coerce_sample_value(statistic(X_arr, y_perm_batch[j]), xp)
            write_pos += cur

    if alt == "two-sided":
        numerator = _to_float(xp.sum(xp.abs(samples) >= abs(observed)))
    elif alt == "greater":
        numerator = _to_float(xp.sum(samples >= observed))
    else:
        numerator = _to_float(xp.sum(samples <= observed))

    pvalue = float((numerator + 1.0) / (n_perm + 1.0))

    return PermutationTestResult(
        statistic_name=str(statistic_name),
        strategy=str(strategy).lower(),
        alternative=alt,
        observed=observed,
        samples=samples,
        pvalue=pvalue,
        n_resamples=n_perm,
        random_state=random_state,
        metadata={"n_samples": int(y_arr.shape[0]), "backend": backend_name},
    )
