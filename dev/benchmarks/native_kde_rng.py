"""Experimental CuPy-native RNG and fixed-factor KDE percentile bootstrap.

This developer-only wrapper does not change public APIs. It deliberately uses
inverse-CDF sampling, not CuPy RandomState.choice's O(size*n) weighted path.
"""
from __future__ import annotations

import math
import numbers
import operator

from dev.benchmarks.native_kde_bootstrap import (
    _positive_integer, native_kde_bootstrap, plan_workspace,
)


def validate_seed(seed):
    """Explicit portable uint32 seed; no implicit entropy or global RNG state."""
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError('seed must be an integer in [0, 2**32 - 1]')
    try:
        value = operator.index(seed)
    except TypeError as exc:
        raise ValueError('seed must be an integer in [0, 2**32 - 1]') from exc
    if not 0 <= value < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32 - 1]')
    return value


def plan_rng_workspace(n, q, b, budget, query_chunk=None):
    """Plan managed allocations including endogenous indices and RNG arrays.

    Full B*n sampling fixes RNG call shape independently of query chunk size.
    Reserve three B*n float64/int64 arrays conservatively (uniforms, search
    output and retained indices), plus normalized probabilities, CDF and
    validation temporaries. Inputs, CUDA context, cuRAND state and library
    scratch bypassing CuPy's allocator are excluded: not a total CUDA hard cap.
    """
    n, q, b, budget = (_positive_integer(v, k) for v, k in
        ((n, 'n'), (q, 'q'), (b, 'b'), (budget, 'budget')))
    reserve = 8 * (3 * n * b + 8 * n + 64)
    if budget <= reserve:
        raise MemoryError('workspace budget cannot hold native RNG buffers; no downscaling')
    numerical = plan_workspace(n, q, b, budget - reserve, query_chunk)
    return dict(numerical, budget_bytes=budget,
                estimated_bytes=numerical['estimated_bytes'] + reserve,
                numerical_budget_bytes=budget - reserve,
                rng_buffer_reserve_bytes=reserve, index_bytes=8 * n * b,
                rng_shape=[b, n], rng_batching='one full matrix; no automatic resampling downscale')


def _probability_cdf(weights, n, xp):
    """Build a representable CDF; xp=np is only for deterministic unit tests.

    Zero-mass plateaus are retained. A positive mass lost through normalization
    or cumulative rounding fails explicitly rather than silently disappearing.
    """
    if weights is None:
        probabilities = xp.full(n, 1.0 / n, dtype=xp.float64)
    else:
        if not bool(xp.all(xp.isfinite(weights))) or float(xp.min(weights)) < 0:
            raise ValueError('weights must be finite and nonnegative')
        total = xp.sum(weights)
        if not bool(xp.isfinite(total)) or float(total) <= 0:
            raise ValueError('weights must have finite positive sum')
        probabilities = weights / total
        if bool(xp.any((weights > 0) & (probabilities == 0))):
            raise ValueError('positive probability underflow is outside the RNG envelope')
    cdf = xp.cumsum(probabilities, dtype=xp.float64)
    total = cdf[-1]
    if not bool(xp.isfinite(total)) or float(total) <= 0:
        raise ValueError('CDF must have finite positive total')
    cdf /= total
    cdf[-1] = 1.0
    increments = xp.empty_like(cdf)
    increments[0] = cdf[0]
    increments[1:] = cdf[1:] - cdf[:-1]
    if bool(xp.any((probabilities > 0) & (increments <= 0))):
        raise ValueError('positive probability collapsed in float64 CDF; outside RNG envelope')
    return cdf


def _sample_indices(cdf, n_resamples, seed, xp):
    """Local cuRAND-backed generator, device uniforms and device binary search."""
    rng = xp.random.RandomState(seed)
    uniforms = rng.random_sample((n_resamples, cdf.size), dtype=xp.float64)
    # RandomState.random_sample is [0,1); right skips leading/interior zero mass
    # and assigns exact CDF boundaries to the following nonempty interval.
    indices = xp.searchsorted(cdf, uniforms, side='right')
    # Fail closed if an RNG/library contract changes (e.g. an endpoint of one).
    if int(xp.min(indices)) < 0 or int(xp.max(indices)) >= cdf.size:
        raise ValueError('RNG produced an index outside the inverse-CDF support')
    return indices.astype(xp.int64, copy=False)


def native_kde_bootstrap_rng(samples, points, *, weights=None, bandwidth=.4,
                             confidence_level=.95, n_resamples=100, seed=0,
                             workspace_bytes=64 * 1024**2, query_chunk=None,
                             return_indices=False):
    """Native random weighted row bootstrap; returns four device arrays.

    Optional ``return_indices=True`` returns ``(result, indices)`` for a separate
    untimed oracle run. Same seed/input/version/device gives repeatable samples;
    query chunking does not alter the RNG call. No bitstream equality with NumPy,
    cross-version/cross-device guarantee, native Torch or public default change.
    The seed creates a new local generator on each invocation; global state is
    neither read nor changed. Sampling probabilities and refit weights both use
    original weights, preserving the existing weighted statistical procedure.
    """
    seed = validate_seed(seed)
    n_resamples = _positive_integer(n_resamples, 'n_resamples')
    if not isinstance(return_indices, bool):
        raise TypeError('return_indices must be boolean')
    for value, name in ((bandwidth, 'bandwidth'), (confidence_level, 'confidence_level')):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(name + ' must be a finite real scalar')
    if bandwidth <= 0 or not math.isfinite(bandwidth * bandwidth) or bandwidth * bandwidth <= 0:
        raise ValueError('bandwidth squared must be finite and positive')
    if not 0 < confidence_level < 1:
        raise ValueError('confidence_level must lie strictly between zero and one')
    import cupy as cp
    arrays = ((samples, 'samples'), (points, 'points'))
    if weights is not None:
        arrays += ((weights, 'weights'),)
    for array, name in arrays:
        if not isinstance(array, cp.ndarray):
            raise TypeError(name + ' must be a CuPy array; no implicit host/device conversion')
        if array.device.id != samples.device.id:
            raise ValueError('all arrays must reside on the same CUDA device')
        if array.dtype != cp.float64:
            raise TypeError(name + ' must have dtype float64')
    for array, name in ((samples, 'samples'), (points, 'points')):
        if array.ndim != 1 and not (array.ndim == 2 and array.shape[1] == 1):
            raise ValueError(name + ' must be univariate')
    n, q = samples.size, points.size
    if n < 2 or q < 1:
        raise ValueError('require at least two samples and one query')
    if weights is not None and (weights.ndim != 1 or weights.size != n):
        raise ValueError('weights must be a float64 vector of length n')
    plan = plan_rng_workspace(n, q, n_resamples, workspace_bytes, query_chunk)
    with cp.cuda.Device(samples.device.id):
        if not bool(cp.all(cp.isfinite(samples))) or not bool(cp.all(cp.isfinite(points))):
            raise ValueError('samples and points must be finite')
        cdf = _probability_cdf(weights, n, cp)
        indices = _sample_indices(cdf, n_resamples, seed, cp)
        del cdf
        result = native_kde_bootstrap(samples, points, indices, weights=weights,
            bandwidth=bandwidth, confidence_level=confidence_level,
            workspace_bytes=plan['numerical_budget_bytes'], query_chunk=plan['query_chunk'])
    return (result, indices) if return_indices else result
