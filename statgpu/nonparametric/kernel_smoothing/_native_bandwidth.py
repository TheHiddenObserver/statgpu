"""Device-resident KDE bandwidth selection for CuPy and Torch.

Binning and scalar searches use the NumPy selector formulas. Native weighted
resampling uses stable sample ties and native PCA uses a canonical vector sign;
these explicit conventions can differ materially from the legacy NumPy path's
unstable ties and arbitrary eigenvector sign. Only scalar decisions leave the device;
projection diagnostics keep their array on the input device. NumPy and kernel
regression continue to use the established selector implementation.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import math
import sys
import warnings

from statgpu.backends import xp_arange, xp_asarray, xp_astype, xp_maximum
from ._kernel_common import _bandwidth_factor, _to_float_scalar

_TINY = sys.float_info.min
_SQRT_PI = math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _scalar(value):
    return _to_float_scalar(value)


_PROJECTION_WARNING_STATE = ContextVar("kde_projection_warning_state", default=None)


def _empty_projection_warning_diagnostics():
    return {
        "status": "none",
        "count": 0,
        "exact_degeneracy_count": 0,
        "near_degeneracy_count": 0,
        "zero_scale_count": 0,
        "min_relative_eigengap": None,
    }


def projection_warning_diagnostics():
    """Snapshot ambiguity counts in the current public-call scope.

    Exact, near, and zero-scale counts are mutually exclusive. Outside a
    decorated call this returns an empty snapshot, not global warning history.
    """
    state = _PROJECTION_WARNING_STATE.get()
    return dict(state) if state is not None else _empty_projection_warning_diagnostics()


def _emit_projection_warning(count):
    # Python 3.9-compatible attribution: internal decorated public APIs and
    # contextmanager exit frames have different depths for fit/CI/bootstrap.
    # Point at the first external caller rather than contextlib or statgpu.
    frame = sys._getframe(1)
    stacklevel = 2
    try:
        while frame is not None:
            module = frame.f_globals["__name__"] if "__name__" in frame.f_globals else ""
            if not (module == "statgpu" or module.startswith("statgpu.") or module == "contextlib"):
                break
            stacklevel += 1
            frame = frame.f_back
    finally:
        del frame
    warnings.warn(
        f"KDE PCA bandwidth selection encountered {count} ambiguous projection(s) "
        "with a repeated or nearly repeated largest eigenvalue; the principal "
        "axis is backend-dependent. Native computation is retained; inspect "
        "the projection eigengap diagnostics.",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


@contextmanager
def _projection_warning_scope():
    parent = _PROJECTION_WARNING_STATE.get()
    if parent is not None:
        yield parent
        return
    state = _empty_projection_warning_diagnostics()
    token = _PROJECTION_WARNING_STATE.set(state)
    try:
        yield state
    finally:
        # Reset on success and failure. The post-finally warning is reached only
        # after successful completion, never for an aborted confidence interval.
        _PROJECTION_WARNING_STATE.reset(token)
    if state["count"]:
        _emit_projection_warning(state["count"])


def projection_warning_scope(function):
    """Aggregate ambiguous PCA warnings once per outer successful public call."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _projection_warning_scope():
            return function(*args, **kwargs)
    return wrapped


def _projection_eigengap_diagnostics(eigenvalues, dimension, xp):
    """Diagnose a repeated leading eigenspace without an absolute scale floor."""
    largest = _scalar(eigenvalues[-1])
    second = _scalar(eigenvalues[-2])
    scale = _scalar(xp.max(xp.abs(eigenvalues)))
    gap = max(largest - second, 0.0)
    relative_tolerance = 64.0 * sys.float_info.epsilon * int(dimension)
    relative_gap = gap / scale if scale > 0.0 else 0.0
    if scale == 0.0:
        classification = "zero_scale"
    elif gap == 0.0:
        classification = "exact"
    elif math.isfinite(relative_gap) and relative_gap <= relative_tolerance:
        classification = "near"
    else:
        classification = "distinct"
    return {
        "projection_eigengap": gap,
        "projection_eigenvalue_scale": scale,
        "projection_relative_eigengap": relative_gap,
        "projection_eigengap_relative_tolerance": relative_tolerance,
        "projection_eigengap_tolerance": relative_tolerance * scale,
        "projection_eigengap_classification": classification,
        "projection_eigengap_ambiguous": classification != "distinct",
    }


def _record_projection_ambiguity(details):
    if not details["projection_eigengap_ambiguous"]:
        return
    state = _PROJECTION_WARNING_STATE.get()
    if state is None:
        _emit_projection_warning(1)
        return
    state["status"] = "ambiguous_principal_eigenspace"
    state["count"] += 1
    kind = details["projection_eigengap_classification"]
    key = {
        "exact": "exact_degeneracy_count",
        "near": "near_degeneracy_count",
        "zero_scale": "zero_scale_count",
    }[kind]
    state[key] += 1
    gap = details["projection_relative_eigengap"]
    old_gap = state["min_relative_eigengap"]
    state["min_relative_eigengap"] = gap if old_gap is None else min(old_gap, gap)


def _std(x, xp, ddof=1):
    if xp.__name__ == "torch":
        return _scalar(xp.std(x, correction=ddof))
    return _scalar(xp.std(x, ddof=ddof))


def _quantiles(x, xp):
    q = xp_asarray([0.75, 0.25], dtype=xp.float64, xp=xp, ref_arr=x)
    return xp.quantile(x, q)


def _normalized_weights(weights, xp, ref_arr):
    w = xp_asarray(weights, dtype=xp.float64, xp=xp, ref_arr=ref_arr).reshape(-1)
    if int(w.shape[0]) == 0:
        raise ValueError("weights must not be empty")
    if not bool(_scalar(xp.all(xp.isfinite(w)))):
        raise ValueError("weights must be finite")
    if _scalar(xp.min(w)) < 0.0:
        raise ValueError("weights must be non-negative")
    total = _scalar(xp.sum(w))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return w / total


def _weighted_quantile_resample_1d(x, w, n_target, xp):
    """Linear ``np.interp`` semantics, including repeated CDF knots."""
    if n_target < 2:
        raise ValueError("n_target must be at least 2 for weighted resampling")
    order = xp.argsort(x, stable=True) if xp.__name__ == "torch" else xp.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cdf = xp.cumsum(ws, axis=0)
    cdf[-1] = 1.0
    u = (xp_arange(n_target, dtype=xp.float64, xp=xp, ref_arr=x) + 0.5) / n_target
    # Right insertion skips zero-weight plateaus at an exact CDF knot, just as
    # np.interp does. Endpoints use the first/last original sample, not a
    # categorical inverse CDF and not deletion of zero-weight observations.
    if xp.__name__ == "torch":
        pos = xp.searchsorted(cdf, u, right=True)
    else:
        pos = xp.searchsorted(cdf, u, side="right")
    right = xp.clip(pos, 1, int(x.shape[0]) - 1)
    left = right - 1
    span = cdf[right] - cdf[left]
    safe_span = xp.where(span > 0.0, span, xp.ones_like(span))
    values = xs[left] + (u - cdf[left]) * (xs[right] - xs[left]) / safe_span
    values = xp.where(pos == 0, xs[0], values)
    return xp.where(pos >= int(x.shape[0]), xs[-1], values)


def _pair_distance_counts_1d(x, xp, nb=1000):
    n = int(x.shape[0])
    if n < 2:
        raise ValueError("need at least 2 data points for automatic bandwidth")
    if not bool(_scalar(xp.all(xp.isfinite(x)))):
        raise ValueError("samples must be finite for automatic bandwidth")
    if nb <= 1:
        raise ValueError("nb must be greater than 1")
    xmin, xmax = _scalar(xp.min(x)), _scalar(xp.max(x))
    data_range = xmax - xmin
    if not math.isfinite(data_range) or data_range <= 0.0:
        raise ValueError("data are constant in automatic bandwidth calculation")
    d = 1.01 * data_range / nb
    if not math.isfinite(d) or d <= 0.0:
        raise ValueError("invalid bin width in automatic bandwidth calculation")
    idx = xp_astype(xp.floor((x - xmin) / d), xp.int64, xp)
    idx = xp.clip(idx, 0, nb - 1)
    hist = xp_astype(xp.bincount(idx, minlength=nb), xp.float64, xp)
    # Zero-padded FFT is the same linear autocorrelation as np.correlate.
    # Histogram counts are integers, so round away FFT floating-point noise.
    # This retains the original 1000-bin approximation, not a new binning rule.
    fft_size = 1 << (2 * nb - 1).bit_length()
    transformed = xp.fft.rfft(hist, n=fft_size)
    cnt = xp.fft.irfft(transformed * transformed.conj(), n=fft_size)[:nb]
    cnt = xp_maximum(xp.round(cnt), 0.0, xp)
    cnt[0] = xp.sum(hist * (hist - 1.0) * 0.5)
    return d, cnt


def _binned_functional(kind, n, d, cnt, h, xp):
    if not math.isfinite(h) or h <= 0.0:
        return float("inf") if kind in ("ucv", "bcv") else float("nan")
    idx = xp_arange(int(cnt.shape[0]), dtype=xp.float64, xp=xp, ref_arr=cnt)
    delta = (idx * d / h) ** 2
    mask = delta < 1000.0
    # Keep a fixed shape instead of copying masked vectors to the host.
    delta = xp.where(mask, delta, xp.zeros_like(delta))
    counts = xp.where(mask, cnt, xp.zeros_like(cnt))
    if kind == "ucv":
        term = xp.exp(-delta / 4.0) - math.sqrt(8.0) * xp.exp(-delta / 2.0)
        total = _scalar(xp.sum(term * counts))
        return (0.5 + total / n) / (n * h * _SQRT_PI)
    if kind == "bcv":
        term = xp.exp(-delta / 4.0) * (delta * delta - 12.0 * delta + 12.0)
        total = _scalar(xp.sum(term * counts))
        return (1.0 + total / (32.0 * n)) / (2.0 * n * h * _SQRT_PI)
    if kind == "phi4":
        term = xp.exp(-delta / 2.0) * (delta * delta - 6.0 * delta + 3.0)
        total = 2.0 * _scalar(xp.sum(term * counts)) + 3.0 * n
        denom = n * (n - 1) * h**5 * _SQRT_2PI
    else:
        term = xp.exp(-delta / 2.0) * (
            delta * delta * delta - 15.0 * delta * delta + 45.0 * delta - 15.0
        )
        total = 2.0 * _scalar(xp.sum(term * counts)) - 15.0 * n
        denom = n * (n - 1) * h**7 * _SQRT_2PI
    return total / denom if denom > 0.0 else float("nan")


def _nrd_factor(method, x, data_variance, n_eff, xp):
    x = x[xp.isfinite(x)]
    if int(x.shape[0]) < 2:
        raise ValueError("need at least 2 finite samples for 'nrd0'/'nrd' bandwidth")
    sd = _std(x, xp)
    quantiles = _quantiles(x, xp)
    robust = _scalar(quantiles[0] - quantiles[1]) / 1.34
    scale = min(sd, robust) if math.isfinite(robust) and robust > 0.0 else sd
    if not math.isfinite(scale) or scale <= 0.0:
        scale = _std(x, xp, ddof=0)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("unable to compute positive scale for 'nrd0'/'nrd' bandwidth")
    coeff = 0.9 if method == "nrd0" else 1.06
    bw_abs = coeff * scale * n_eff ** (-1.0 / 5.0)
    if not math.isfinite(bw_abs) or bw_abs <= 0.0:
        raise ValueError("automatic bandwidth rule produced a non-positive value")
    data_sd = math.sqrt(max(_scalar(data_variance), 0.0))
    if data_sd <= 0.0 or not math.isfinite(data_sd):
        data_sd = max(_TINY, sd)
    return bw_abs / data_sd


def _r_factor(method, x, w, data_variance, weighted, xp):
    from ._bandwidth_selection import _bisection_root, _golden_section_minimize

    method = "sj-ste" if method == "sj" else method
    if int(x.shape[0]) < 2:
        raise ValueError("need at least 2 samples for automatic bandwidth selection")
    if not bool(_scalar(xp.all(xp.isfinite(x)))):
        raise ValueError("samples must be finite for automatic bandwidth selection")
    if int(w.shape[0]) != int(x.shape[0]):
        raise ValueError("weights shape is incompatible with samples")
    if weighted:
        effective = 1.0 / _scalar(xp.sum(w * w))
        n_rep = round(max(128.0, min(8192.0, effective * 8.0)))
        x = _weighted_quantile_resample_1d(x, w, n_rep, xp)
    n = int(x.shape[0])
    d, cnt = _pair_distance_counts_1d(x, xp)
    sample_sd = _std(x, xp)
    if not math.isfinite(sample_sd) or sample_sd <= 0.0:
        raise ValueError("data are constant in automatic bandwidth calculation")
    quantiles = _quantiles(x, xp)
    robust = _scalar(quantiles[0] - quantiles[1]) / 1.349
    scale = min(sample_sd, robust) if math.isfinite(robust) and robust > 0.0 else sample_sd
    if not math.isfinite(scale) or scale <= 0.0:
        scale = sample_sd
    hmax = 1.144 * (sample_sd if method in ("ucv", "bcv") else scale) * n**(-1.0 / 5.0)
    lower = max(hmax * 0.1, _TINY)
    upper = max(hmax, lower * 1.01)
    tol = max(lower * 0.1, 1e-8)

    def functional(kind, h):
        return _binned_functional(kind, n, d, cnt, h, xp)

    if method in ("ucv", "bcv"):
        bw_abs = _golden_section_minimize(lambda h: functional(method, h), lower, upper, tol)
    else:
        c1 = 1.0 / (2.0 * _SQRT_PI * n)
        a, b = 1.24 * scale * n**(-1.0 / 7.0), 1.23 * scale * n**(-1.0 / 9.0)
        td = -functional("phi6", b)
        if not math.isfinite(td) or td <= 0.0:
            raise ValueError("sample is too sparse to find TD for 'sj' bandwidth")
        if method == "sj-dpi":
            h_phi4 = (2.394 / (n * td)) ** (1.0 / 7.0)
            sd_h = functional("phi4", h_phi4)
            if not math.isfinite(sd_h) or sd_h <= 0.0:
                raise ValueError("sample is too sparse to find SD for 'sj-dpi' bandwidth")
            bw_abs = (c1 / sd_h) ** (1.0 / 5.0)
        else:
            sd_a = functional("phi4", a)
            if not math.isfinite(sd_a) or sd_a <= 0.0:
                raise ValueError("sample is too sparse to find SD for 'sj-ste' bandwidth")
            alph2 = 1.357 * (sd_a / td) ** (1.0 / 7.0)
            if not math.isfinite(alph2) or alph2 <= 0.0:
                raise ValueError("sample is too sparse to find alph2 for 'sj-ste' bandwidth")

            def objective(h):
                sd_term = functional("phi4", alph2 * h**(5.0 / 7.0))
                if not math.isfinite(sd_term) or sd_term <= 0.0:
                    return float("nan")
                return (c1 / sd_term) ** (1.0 / 5.0) - h

            fl, fu = objective(lower), objective(upper)
            itry = 1
            while (not math.isfinite(fl) or not math.isfinite(fu) or fl * fu > 0.0) and itry <= 99:
                if itry % 2 == 1:
                    upper *= 1.2
                else:
                    lower = max(lower / 1.2, _TINY)
                fl, fu = objective(lower), objective(upper)
                itry += 1
            if not math.isfinite(fl) or not math.isfinite(fu) or fl * fu > 0.0:
                raise ValueError("no solution found for 'sj-ste' bandwidth in the search range")
            bw_abs = _bisection_root(objective, lower, upper, tol)
    if not math.isfinite(bw_abs) or bw_abs <= 0.0:
        raise ValueError("automatic bandwidth rule produced a non-positive value")
    data_sd = math.sqrt(max(_scalar(data_variance), 0.0))
    if data_sd <= 0.0 or not math.isfinite(data_sd):
        data_sd = sample_sd
    return bw_abs / data_sd


def _project_to_principal_axis(x, w, xp):
    if x.ndim != 2:
        raise ValueError("samples_2d must be a 2D array")
    if int(x.shape[0]) != int(w.shape[0]):
        raise ValueError("weights shape is incompatible with samples")
    centered = x - xp.sum(x * w[:, None], axis=0)[None, :]
    cov = (centered.T * w[None, :]) @ centered
    cov = 0.5 * (cov + cov.T)
    evals, evecs = xp.linalg.eigh(cov)
    eigengap_details = _projection_eigengap_diagnostics(evals, int(x.shape[1]), xp)
    _record_projection_ambiguity(eigengap_details)
    index = int(_scalar(xp.argmax(evals)))
    vector = evecs[:, index]
    principal = _scalar(evals[index])
    if not math.isfinite(principal) or principal <= 0.0:
        scales = xp.sqrt(xp_maximum(xp.diag(cov), 0.0, xp))
        index = int(_scalar(xp.argmax(scales)))
        vector = xp.zeros_like(vector)
        vector[index] = 1.0
        principal = _scalar(xp.max(scales)) ** 2
    pivot = int(_scalar(xp.argmax(xp.abs(vector))))
    if _scalar(vector[pivot]) < 0.0:
        vector = -vector
    projected = centered @ vector
    total = _scalar(xp.sum(xp_maximum(evals, 0.0, xp)))
    ratio = principal / total if total > 0.0 else 1.0
    mean = xp.sum(w * projected)
    denom = 1.0 - _scalar(xp.sum(w * w))
    variance = _scalar(xp.sum(w * (projected - mean) ** 2) / denom) if denom > 1e-15 else float("nan")
    if not math.isfinite(variance) or variance <= 0.0:
        variance = _std(projected, xp) ** 2 if int(projected.shape[0]) >= 2 else float("nan")
    if not math.isfinite(variance) or variance <= 0.0:
        variance = _TINY
    return projected, variance, {
        "projection_explained_ratio": ratio,
        "projection_vector": vector,
        "projection_variance": variance,
        "projection_axis_policy": "backend_eigh_largest_eigenvalue",
        "projection_sign_policy": "largest_absolute_loading_positive",
        **eigengap_details,
    }


def select_bandwidth_native(bandwidth, *, n_eff, n_features, samples_2d,
                            weights_1d, data_cov, xp, enable_r_selectors,
                            weighted_r_selector_strategy, multivariate_selector_strategy):
    """Select on the concrete sample device without a host numerical stage."""
    # Operations without an explicit allocation device (FFT/eigh/bincount) must
    # also run under the input's CuPy device, not a possibly different ambient one.
    context = samples_2d.device if xp.__name__ == "cupy" else nullcontext()
    with context:
        return _select_bandwidth_native(
            bandwidth, n_eff=n_eff, n_features=n_features, samples_2d=samples_2d,
            weights_1d=weights_1d, data_cov=data_cov, xp=xp,
            enable_r_selectors=enable_r_selectors,
            weighted_r_selector_strategy=weighted_r_selector_strategy,
            multivariate_selector_strategy=multivariate_selector_strategy,
        )


def _select_bandwidth_native(bandwidth, *, n_eff, n_features, samples_2d,
                             weights_1d, data_cov, xp, enable_r_selectors,
                             weighted_r_selector_strategy, multivariate_selector_strategy):
    from ._bandwidth_selection import (
        BandwidthSelectionResult, _normalize_multivariate_strategy, _normalize_weighted_strategy,
    )
    n_eff, n_features = float(n_eff), int(n_features)
    weighted_strategy = _normalize_weighted_strategy(weighted_r_selector_strategy)
    multivariate_strategy = _normalize_multivariate_strategy(multivariate_selector_strategy)
    w = _normalized_weights(weights_1d, xp, samples_2d)
    weighted = _scalar(xp.max(w) - xp.min(w)) > 1e-12
    if not math.isfinite(n_eff) or n_eff <= 0.0:
        raise ValueError("n_eff must be a finite positive scalar")
    if n_features <= 0:
        raise ValueError("n_features must be a positive integer")
    method = bandwidth.strip().lower() if isinstance(bandwidth, str) else "scalar"
    details = {"input_bandwidth": bandwidth, "n_samples": int(w.shape[0]), "estimator": "kde"}
    r_selector = method in ("ucv", "bcv", "sj", "sj-ste", "sj-dpi")
    multi_used, selector_dim = "none", n_features
    if r_selector or method in ("nrd", "nrd0"):
        if r_selector and not enable_r_selectors:
            raise ValueError("R-style bandwidth selectors are disabled for this estimator")
        x = xp_asarray(samples_2d, dtype=xp.float64, xp=xp, ref_arr=samples_2d)
        if n_features == 1:
            x_work, variance = x[:, 0], data_cov[0, 0]
        else:
            x_work, variance, projection_details = _project_to_principal_axis(x, w, xp)
            details.update(projection_details)
            multi_used = multivariate_strategy
        selector_dim = 1
        details["rule"] = "r_selector" if r_selector else "nrd"
        if r_selector and weighted:
            details["weighted_tie_policy"] = "stable_input_order"
        # The established projected selector normalizes its already normalized
        # weights at the one-dimensional selector boundary once more.
        w_work = _normalized_weights(w, xp, x) if n_features > 1 and r_selector else w
        factor = (_r_factor(method, x_work, w_work, variance, weighted, xp) if r_selector
                  else _nrd_factor(method, x_work, variance, n_eff, xp))
    else:
        factor = _bandwidth_factor(bandwidth, n_eff=n_eff, n_features=n_features)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("bandwidth factor must be a finite positive scalar")
    return BandwidthSelectionResult(
        factor=float(factor), method=method, n_features=n_features, n_eff=n_eff,
        used_r_selector=r_selector, weighted=weighted,
        weighted_strategy=weighted_strategy if weighted else "uniform",
        multivariate_strategy=multi_used, selector_dimension=selector_dim, details=details,
    )
