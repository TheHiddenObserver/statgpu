"""Device-native weighted KDE bootstrap implementation.

Host work is limited to argument validation, scalar control/diagnostics and a
four-word seed bridge. Numerical arrays stay on the selected device until the
public wrapper constructs its NumPy result. This module never imports dev code.
"""
from __future__ import annotations

from contextlib import nullcontext
import operator

import numpy as np

from statgpu.backends import xp_empty
from ._kernel_common import _get_xp, _to_float_scalar


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def plan_workspace(n, p, q, b, batch_size, workspace_bytes=None, bandwidth=None):
    """Conservative managed-buffer estimate, not a total allocator/VRAM cap.

    Includes retained bootstrap values, quantile sorting/copy allowance,
    one replicate's RNG/index/refit buffers and query-by-sample temporaries.
    Excludes caller inputs, fitted original model, allocator caching, library
    solver/eigensolver/RNG scratch. Selector array reserve includes the current
    8192-point weighted-resample bound and 2048-point FFT buffers.
    There is no implicit reduction in draws, samples or queries.
    """
    chunk = min(_positive_integer(batch_size, "batch_size"), max(1, q))
    selector = (8 * (12*n + 6*n*p + 8*p*p + 20*8192 + 20*2048)
                if isinstance(bandwidth, str) and bandwidth.strip().lower() not in ("scott", "silverman") else 0)
    fixed = 8 * (4*b*q + 12*n + 6*n*p + 8*p*p + 8*q + 64) + selector
    per_query = 8 * (8*n + 4*p + 16)
    if workspace_bytes is not None:
        budget = _positive_integer(workspace_bytes, "workspace_bytes")
        if fixed + per_query > budget:
            raise MemoryError("workspace_bytes cannot hold bootstrap managed buffers and one query; no downscaling")
        chunk = min(chunk, (budget-fixed)//per_query)
    return {"query_chunk": int(chunk), "estimated_managed_bytes": int(fixed + chunk*per_query),
            "workspace_bytes": workspace_bytes, "selector_reserve_bytes": int(selector),
            "workspace_scope": "bootstrap and selector managed buffers; excludes inputs, original model, library scratch, allocator cache",
            "rng_batching": "one n_samples vector per replicate; independent of query chunk"}


def _make_rng(random_state, xp, reference):
    # default_rng supplies the existing public accepted input/negative-seed
    # validation. A supplied Generator is advanced, not copied or hashed.
    bridge = np.random.default_rng(random_state)
    words = bridge.integers(0, 2**32, size=4, dtype=np.uint32)
    seed = int(np.random.SeedSequence(words).generate_state(1, dtype=np.uint64)[0])
    if xp.__name__ == "torch":
        rng = xp.Generator(device=reference.device)
        rng.manual_seed(seed)
    else:
        rng = xp.random.RandomState(seed)
    return rng


def _probability_cdf(weights, xp):
    cdf = xp.cumsum(weights, dim=0) if xp.__name__ == "torch" else xp.cumsum(weights, dtype=xp.float64)
    total = _to_float_scalar(cdf[-1])
    if not np.isfinite(total) or total <= 0:
        raise ValueError("bootstrap weights must have finite positive sum")
    cdf = cdf / cdf[-1]
    cdf[-1] = 1.0
    # Float64 cumulative summation may collapse tiny positive probabilities.
    # As with the previous NumPy sampler, representable CDF mass is sampled;
    # no additional public weight-domain restriction is imposed.
    return cdf


def _draw_indices(cdf, rng, n_samples, xp):
    """One replicate's native RNG and inverse-CDF search (oracle hook)."""
    if xp.__name__ == "torch":
        uniforms = xp.rand((n_samples,), dtype=xp.float64, device=cdf.device, generator=rng)
        indices = xp.searchsorted(cdf, uniforms, right=True)
    else:
        uniforms = rng.random_sample(n_samples, dtype=xp.float64)
        indices = xp.searchsorted(cdf, uniforms, side="right")
    if _to_float_scalar(xp.min(indices)) < 0 or _to_float_scalar(xp.max(indices)) >= n_samples:
        raise ValueError("native RNG returned an index outside bootstrap support")
    return indices


def _percentiles(draws, confidence_level, xp):
    if draws.shape[1] == 0:
        return (xp_empty((0,), xp.float64, xp, ref_arr=draws),
                xp_empty((0,), xp.float64, xp, ref_arr=draws))
    alpha = (1.0-confidence_level)/2.0
    if xp.__name__ == "torch":
        probs = xp.tensor([alpha, 1.0-alpha], dtype=xp.float64, device=draws.device)
        bounds = xp.quantile(draws, probs, dim=0)
    elif draws.shape[1] == 1:
        # CuPy 13.6 singleton-axis multi-quantile writeback regression.
        bounds = xp.quantile(draws[:, 0], [alpha, 1.0-alpha]).reshape(2, 1)
    else:
        bounds = xp.quantile(draws, [alpha, 1.0-alpha], axis=0)
    return bounds[0].copy() if xp.__name__ != "torch" else bounds[0].clone(), bounds[1].copy() if xp.__name__ != "torch" else bounds[1].clone()


def native_bootstrap(model, points_2d, *, bandwidth, kernel, n_resamples,
                     confidence_level, random_state, batch_size, workspace_bytes=None):
    """Evaluate native bootstrap; public wrapper alone exports result arrays."""
    from ._kde import fit_kde
    from ._native_bandwidth import projection_warning_diagnostics
    xp = _get_xp(model.backend_)
    n, p = model.samples_.shape
    q = int(points_2d.shape[0])
    plan = plan_workspace(n, p, q, n_resamples, batch_size, workspace_bytes, bandwidth)
    context = xp.cuda.Device(model.samples_.device.id) if model.backend_ == "cupy" else nullcontext()
    with context:
        rng = _make_rng(random_state, xp, model.samples_)
        cdf = _probability_cdf(model.weights_, xp)
        draws = xp_empty((n_resamples, q), xp.float64, xp, ref_arr=model.samples_)
        estimate = model.pdf(points_2d, batch_size=plan["query_chunk"])
        for i in range(n_resamples):
            indices = _draw_indices(cdf, rng, n, xp)
            sampled_weights = model.weights_[indices]
            sampled_weights = sampled_weights / xp.sum(sampled_weights)
            fitted = fit_kde(model.samples_[indices], weights=sampled_weights,
                             bandwidth=bandwidth, kernel=kernel, backend=model.backend_)
            draws[i] = fitted.pdf(points_2d, batch_size=plan["query_chunk"])
        if (not bool(_to_float_scalar(xp.all(xp.isfinite(draws))))
                or not bool(_to_float_scalar(xp.all(xp.isfinite(estimate))))):
            raise ValueError("native KDE bootstrap produced non-finite density values")
        lower, upper = _percentiles(draws, confidence_level, xp)
    metadata = dict(plan, execution="native", projection_diagnostics=projection_warning_diagnostics(), seed_bridge="four uint32 words from numpy.default_rng(random_state); local backend generator",
                    selector_refit="each replicate" if isinstance(bandwidth, str) else "fixed numeric covariance factor",
                    rng_probability_precision="float64 cumulative probabilities; sub-resolution positive masses may collapse",
                    host_transfers="input conversion and final NumPy results only; scalar control/diagnostics and seed bridge")
    return dict(estimate=estimate, lower=lower, upper=upper, draws=draws, metadata=metadata)
