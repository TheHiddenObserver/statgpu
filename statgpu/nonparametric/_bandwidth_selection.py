"""Bandwidth selection helpers for kernel-based estimators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Union

import numpy as np

from statgpu.nonparametric._kernel_common import (
    _bandwidth_factor,
    _bandwidth_factor_1d_nrd,
    _kernel_values_from_quad,
    _normalize_regression_name,
    _to_float_scalar,
    _to_numpy,
)

_BW_DELMAX = 1000.0
_SQRT_PI = math.sqrt(math.pi)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class BandwidthSelectionResult:
    """Diagnostic result for automatic bandwidth selection."""

    factor: float
    method: str
    n_features: int
    n_eff: float
    used_r_selector: bool
    weighted: bool
    weighted_strategy: str
    multivariate_strategy: str
    selector_dimension: int
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": float(self.factor),
            "method": str(self.method),
            "n_features": int(self.n_features),
            "n_eff": float(self.n_eff),
            "used_r_selector": bool(self.used_r_selector),
            "weighted": bool(self.weighted),
            "weighted_strategy": str(self.weighted_strategy),
            "multivariate_strategy": str(self.multivariate_strategy),
            "selector_dimension": int(self.selector_dimension),
            "details": dict(self.details),
        }


def _normalize_weighted_strategy(strategy: str) -> str:
    name = str(strategy).strip().lower()
    aliases = {
        "quantile_resample": "quantile_resample",
        "quantile": "quantile_resample",
        "resample": "quantile_resample",
    }
    out = aliases.get(name)
    if out is None:
        raise ValueError("weighted_r_selector_strategy must be 'quantile_resample'")
    return out


def _normalize_multivariate_strategy(strategy: str) -> str:
    name = str(strategy).strip().lower()
    aliases = {
        "projection_pca_1d": "projection_pca_1d",
        "projection": "projection_pca_1d",
        "pca": "projection_pca_1d",
    }
    out = aliases.get(name)
    if out is None:
        raise ValueError("multivariate_selector_strategy must be 'projection_pca_1d'")
    return out


def _normalize_estimator_name(estimator: str) -> str:
    name = str(estimator).strip().lower()
    aliases = {
        "kde": "kde",
        "kernel_density": "kde",
        "gaussian_kde": "kde",
        "kernel_regression": "kernel_regression",
        "kreg": "kernel_regression",
        "regression": "kernel_regression",
    }
    out = aliases.get(name)
    if out is None:
        raise ValueError("estimator must be one of: 'kde', 'kernel_regression'")
    return out


# Alias for backward compatibility - delegates to _kernel_common
_normalize_regression_mode = _normalize_regression_name


def _normalized_weights_numpy(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == 0:
        raise ValueError("weights must not be empty")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite")
    if np.min(w) < 0.0:
        raise ValueError("weights must be non-negative")

    w_sum = float(np.sum(w))
    if (not np.isfinite(w_sum)) or w_sum <= 0.0:
        raise ValueError("weights must sum to a positive value")

    return w / w_sum


def _weighted_var_unbiased_1d(x: np.ndarray, w_norm: np.ndarray) -> float:
    x_np = np.asarray(x, dtype=np.float64).reshape(-1)
    w_np = np.asarray(w_norm, dtype=np.float64).reshape(-1)
    if x_np.size != w_np.size:
        raise ValueError("x and weights must have the same length")

    mean = float(np.sum(w_np * x_np))
    denom = 1.0 - float(np.sum(w_np * w_np))
    if (not np.isfinite(denom)) or denom <= 1e-15:
        return float("nan")

    var = float(np.sum(w_np * (x_np - mean) ** 2) / denom)
    if (not np.isfinite(var)) or var < 0.0:
        return float("nan")
    return var


def _weighted_quantile_resample_1d(x: np.ndarray, w_norm: np.ndarray, n_target: int) -> np.ndarray:
    x_np = np.asarray(x, dtype=np.float64).reshape(-1)
    w_np = np.asarray(w_norm, dtype=np.float64).reshape(-1)

    n_t = int(n_target)
    if n_t < 2:
        raise ValueError("n_target must be at least 2 for weighted resampling")

    idx = np.argsort(x_np)
    x_s = x_np[idx]
    w_s = w_np[idx]

    cdf = np.cumsum(w_s)
    cdf[-1] = 1.0

    u = (np.arange(n_t, dtype=np.float64) + 0.5) / float(n_t)
    x_rep = np.interp(u, cdf, x_s)
    return np.asarray(x_rep, dtype=np.float64)


def _project_to_principal_axis(samples_2d, weights_1d) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(_to_numpy(samples_2d), dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("samples_2d must be a 2D array")

    w_norm = _normalized_weights_numpy(np.asarray(_to_numpy(weights_1d), dtype=np.float64))
    if x.shape[0] != w_norm.size:
        raise ValueError("weights shape is incompatible with samples")

    mean = np.sum(x * w_norm[:, None], axis=0)
    xc = x - mean[None, :]

    cov = (xc.T * w_norm[None, :]) @ xc
    cov = 0.5 * (cov + cov.T)

    evals, evecs = np.linalg.eigh(cov)
    idx_max = int(np.argmax(evals))
    principal_vec = np.asarray(evecs[:, idx_max], dtype=np.float64).reshape(-1)
    principal_eval = float(evals[idx_max])

    if (not np.isfinite(principal_eval)) or principal_eval <= 0.0:
        scales = np.sqrt(np.maximum(np.diag(cov), 0.0))
        idx_fallback = int(np.argmax(scales))
        principal_vec = np.zeros(x.shape[1], dtype=np.float64)
        principal_vec[idx_fallback] = 1.0
        principal_eval = float(np.max(scales) ** 2)

    proj = xc @ principal_vec
    total_var = float(np.sum(np.maximum(evals, 0.0)))
    explained_ratio = float(principal_eval / total_var) if total_var > 0.0 else 1.0

    return np.asarray(proj, dtype=np.float64), principal_vec, explained_ratio


# _bandwidth_factor, _bandwidth_factor_1d_nrd, _normalize_regression_mode
# are imported from _kernel_common (see top-level imports)


def _golden_section_minimize(func, lower: float, upper: float, tol: float) -> float:
    a = float(lower)
    b = float(upper)
    if (not np.isfinite(a)) or (not np.isfinite(b)) or a <= 0.0 or b <= a:
        raise ValueError("invalid optimization bounds for bandwidth selection")

    tol_f = float(tol)
    if (not np.isfinite(tol_f)) or tol_f <= 0.0:
        tol_f = max(1e-8, (b - a) * 1e-4)

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0

    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)

    def _eval(x: float) -> float:
        y = float(func(float(x)))
        if not np.isfinite(y):
            return float("inf")
        return y

    fc = _eval(c)
    fd = _eval(d)

    for _ in range(256):
        if (b - a) <= tol_f:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + invphi2 * (b - a)
            fc = _eval(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + invphi * (b - a)
            fd = _eval(d)

    return float(c if fc <= fd else d)


def _bisection_root(func, lower: float, upper: float, tol: float) -> float:
    a = float(lower)
    b = float(upper)
    if (not np.isfinite(a)) or (not np.isfinite(b)) or b <= a:
        raise ValueError("invalid bisection interval")

    fa = float(func(a))
    fb = float(func(b))
    if (not np.isfinite(fa)) or (not np.isfinite(fb)):
        raise ValueError("non-finite values in bisection objective")
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("bisection interval does not bracket a root")

    tol_f = float(tol)
    if (not np.isfinite(tol_f)) or tol_f <= 0.0:
        tol_f = max(1e-8, (b - a) * 1e-4)

    for _ in range(256):
        mid = 0.5 * (a + b)
        if abs(b - a) <= tol_f:
            return float(mid)
        fm = float(func(mid))
        if not np.isfinite(fm):
            raise ValueError("non-finite values in bisection objective")
        if fm == 0.0:
            return float(mid)
        if fa * fm < 0.0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm

    return float(0.5 * (a + b))


def _bw_pair_distance_counts_1d(x: np.ndarray, nb: int = 1000) -> tuple[float, np.ndarray]:
    x_np = np.asarray(x, dtype=np.float64).reshape(-1)
    if x_np.size < 2:
        raise ValueError("need at least 2 data points for automatic bandwidth")
    if not np.all(np.isfinite(x_np)):
        raise ValueError("samples must be finite for automatic bandwidth")

    nb_i = int(nb)
    if nb_i <= 1:
        raise ValueError("nb must be greater than 1")

    xmin = float(np.min(x_np))
    xmax = float(np.max(x_np))
    data_range = xmax - xmin
    if (not np.isfinite(data_range)) or data_range <= 0.0:
        raise ValueError("data are constant in automatic bandwidth calculation")

    d = float(1.01 * data_range / float(nb_i))
    if (not np.isfinite(d)) or d <= 0.0:
        raise ValueError("invalid bin width in automatic bandwidth calculation")

    idx = np.floor((x_np - xmin) / d).astype(np.int64)
    idx = np.clip(idx, 0, nb_i - 1)
    hist = np.bincount(idx, minlength=nb_i).astype(np.float64)

    cnt = np.correlate(hist, hist, mode="full")[nb_i - 1 :].astype(np.float64)
    cnt[0] = np.sum(hist * (hist - 1.0) * 0.5)
    return d, cnt


def _bw_ucv_objective(n: int, d: float, cnt: np.ndarray, h: float) -> float:
    h_f = float(h)
    if (not np.isfinite(h_f)) or h_f <= 0.0:
        return float("inf")

    idx = np.arange(cnt.size, dtype=np.float64)
    delta = (idx * float(d) / h_f) ** 2
    mask = delta < _BW_DELMAX
    delta_m = delta[mask]
    cnt_m = cnt[mask]

    term = np.exp(-delta_m / 4.0) - math.sqrt(8.0) * np.exp(-delta_m / 2.0)
    sum_term = float(np.sum(term * cnt_m))
    return float((0.5 + sum_term / float(n)) / (float(n) * h_f * _SQRT_PI))


def _bw_bcv_objective(n: int, d: float, cnt: np.ndarray, h: float) -> float:
    h_f = float(h)
    if (not np.isfinite(h_f)) or h_f <= 0.0:
        return float("inf")

    idx = np.arange(cnt.size, dtype=np.float64)
    delta = (idx * float(d) / h_f) ** 2
    mask = delta < _BW_DELMAX
    delta_m = delta[mask]
    cnt_m = cnt[mask]

    term = np.exp(-delta_m / 4.0) * (delta_m * delta_m - 12.0 * delta_m + 12.0)
    sum_term = float(np.sum(term * cnt_m))
    return float((1.0 + sum_term / (32.0 * float(n))) / (2.0 * float(n) * h_f * _SQRT_PI))


def _bw_phi4(n: int, d: float, cnt: np.ndarray, h: float) -> float:
    h_f = float(h)
    if (not np.isfinite(h_f)) or h_f <= 0.0:
        return float("nan")

    idx = np.arange(cnt.size, dtype=np.float64)
    delta = (idx * float(d) / h_f) ** 2
    mask = delta < _BW_DELMAX
    delta_m = delta[mask]
    cnt_m = cnt[mask]

    term = np.exp(-delta_m / 2.0) * (delta_m * delta_m - 6.0 * delta_m + 3.0)
    sum_term = float(np.sum(term * cnt_m))
    sum_term = 2.0 * sum_term + 3.0 * float(n)
    denom = float(n * (n - 1)) * (h_f ** 5) * _SQRT_2PI
    if denom <= 0.0:
        return float("nan")
    return float(sum_term / denom)


def _bw_phi6(n: int, d: float, cnt: np.ndarray, h: float) -> float:
    h_f = float(h)
    if (not np.isfinite(h_f)) or h_f <= 0.0:
        return float("nan")

    idx = np.arange(cnt.size, dtype=np.float64)
    delta = (idx * float(d) / h_f) ** 2
    mask = delta < _BW_DELMAX
    delta_m = delta[mask]
    cnt_m = cnt[mask]

    term = np.exp(-delta_m / 2.0) * (
        delta_m * delta_m * delta_m - 15.0 * delta_m * delta_m + 45.0 * delta_m - 15.0
    )
    sum_term = float(np.sum(term * cnt_m))
    sum_term = 2.0 * sum_term - 15.0 * float(n)
    denom = float(n * (n - 1)) * (h_f ** 7) * _SQRT_2PI
    if denom <= 0.0:
        return float("nan")
    return float(sum_term / denom)


def _bandwidth_factor_1d_r_selectors(
    method: str,
    *,
    samples_2d,
    weights_1d,
    data_cov,
    weighted_strategy: str = "quantile_resample",
) -> float:
    method_n = str(method).strip().lower()
    if method_n == "sj":
        method_n = "sj-ste"
    if method_n not in ("ucv", "bcv", "sj-ste", "sj-dpi"):
        raise ValueError("method must be one of: 'ucv', 'bcv', 'sj', 'sj-ste', 'sj-dpi'")

    x = np.asarray(_to_numpy(samples_2d[:, 0]), dtype=np.float64).reshape(-1)
    if x.size < 2:
        raise ValueError("need at least 2 samples for automatic bandwidth selection")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must be finite for automatic bandwidth selection")

    w = _normalized_weights_numpy(np.asarray(_to_numpy(weights_1d), dtype=np.float64).reshape(-1))
    if w.size != x.size:
        raise ValueError("weights shape is incompatible with samples")

    x_work = x
    is_weighted = float(np.max(w) - np.min(w)) > 1e-12
    if is_weighted:
        strategy = _normalize_weighted_strategy(weighted_strategy)
        if strategy == "quantile_resample":
            n_eff = float(1.0 / np.sum(w * w))
            n_rep = int(np.clip(round(max(128.0, min(8192.0, n_eff * 8.0))), 128, 8192))
            x_work = _weighted_quantile_resample_1d(x, w, n_rep)
        else:
            raise ValueError("unsupported weighted strategy")

    n = int(x_work.size)
    d, cnt = _bw_pair_distance_counts_1d(x_work, nb=1000)

    sample_sd = float(np.std(x_work, ddof=1))
    if (not np.isfinite(sample_sd)) or sample_sd <= 0.0:
        raise ValueError("data are constant in automatic bandwidth calculation")

    q75, q25 = np.quantile(x_work, [0.75, 0.25])
    robust = float((q75 - q25) / 1.349)
    scale = min(sample_sd, robust) if np.isfinite(robust) and robust > 0.0 else sample_sd
    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = sample_sd

    if method_n in ("ucv", "bcv"):
        hmax = float(1.144 * sample_sd * (float(n) ** (-1.0 / 5.0)))
        lower = max(hmax * 0.1, float(np.finfo(np.float64).tiny))
        upper = max(hmax, lower * 1.01)
        tol = max(lower * 0.1, 1e-8)

        obj = _bw_ucv_objective if method_n == "ucv" else _bw_bcv_objective
        bw_abs = _golden_section_minimize(lambda h: obj(n, d, cnt, h), lower, upper, tol)
    else:
        hmax = float(1.144 * scale * (float(n) ** (-1.0 / 5.0)))
        lower = max(hmax * 0.1, float(np.finfo(np.float64).tiny))
        upper = max(hmax, lower * 1.01)
        tol = max(lower * 0.1, 1e-8)

        c1 = 1.0 / (2.0 * _SQRT_PI * float(n))
        a = float(1.24 * scale * (float(n) ** (-1.0 / 7.0)))
        b = float(1.23 * scale * (float(n) ** (-1.0 / 9.0)))

        td = -_bw_phi6(n, d, cnt, b)
        if (not np.isfinite(td)) or td <= 0.0:
            raise ValueError("sample is too sparse to find TD for 'sj' bandwidth")

        if method_n == "sj-dpi":
            h_phi4 = float((2.394 / (float(n) * td)) ** (1.0 / 7.0))
            sd_h = _bw_phi4(n, d, cnt, h_phi4)
            if (not np.isfinite(sd_h)) or sd_h <= 0.0:
                raise ValueError("sample is too sparse to find SD for 'sj-dpi' bandwidth")
            bw_abs = float((c1 / sd_h) ** (1.0 / 5.0))
        else:
            sd_a = _bw_phi4(n, d, cnt, a)
            if (not np.isfinite(sd_a)) or sd_a <= 0.0:
                raise ValueError("sample is too sparse to find SD for 'sj-ste' bandwidth")

            alph2 = float(1.357 * ((sd_a / td) ** (1.0 / 7.0)))
            if (not np.isfinite(alph2)) or alph2 <= 0.0:
                raise ValueError("sample is too sparse to find alph2 for 'sj-ste' bandwidth")

            def f_sd(h: float) -> float:
                h_f = float(h)
                sd_term = _bw_phi4(n, d, cnt, alph2 * (h_f ** (5.0 / 7.0)))
                if (not np.isfinite(sd_term)) or sd_term <= 0.0:
                    return float("nan")
                return float((c1 / sd_term) ** (1.0 / 5.0) - h_f)

            fl = float(f_sd(lower))
            fu = float(f_sd(upper))
            itry = 1
            while (not np.isfinite(fl) or not np.isfinite(fu) or (fl * fu > 0.0)) and itry <= 99:
                if itry % 2 == 1:
                    upper *= 1.2
                else:
                    lower /= 1.2
                    lower = max(lower, float(np.finfo(np.float64).tiny))
                fl = float(f_sd(lower))
                fu = float(f_sd(upper))
                itry += 1

            if (not np.isfinite(fl)) or (not np.isfinite(fu)) or (fl * fu > 0.0):
                raise ValueError("no solution found for 'sj-ste' bandwidth in the search range")

            bw_abs = _bisection_root(f_sd, lower, upper, tol)

    if (not np.isfinite(bw_abs)) or bw_abs <= 0.0:
        raise ValueError("automatic bandwidth rule produced a non-positive value")

    data_sd = math.sqrt(max(_to_float_scalar(data_cov[0, 0]), 0.0))
    if data_sd <= 0.0 or (not np.isfinite(data_sd)):
        data_sd = sample_sd

    factor = float(bw_abs / data_sd)
    if (not np.isfinite(factor)) or factor <= 0.0:
        raise ValueError("bandwidth factor must be a finite positive scalar")
    return factor


def _multivariate_factor_from_projected_1d(
    method: str,
    *,
    samples_2d,
    weights_1d,
    data_cov,
    n_eff: float,
    rule_kind: str,
    weighted_r_selector_strategy: str,
) -> tuple[float, Dict[str, Any]]:
    proj, principal_vec, explained_ratio = _project_to_principal_axis(samples_2d, weights_1d)
    w_norm = _normalized_weights_numpy(np.asarray(_to_numpy(weights_1d), dtype=np.float64))

    proj_2d = np.asarray(proj, dtype=np.float64).reshape(-1, 1)

    var_proj = _weighted_var_unbiased_1d(proj, w_norm)
    if (not np.isfinite(var_proj)) or var_proj <= 0.0:
        var_proj = float(np.var(proj, ddof=1)) if proj.size >= 2 else float("nan")
    if (not np.isfinite(var_proj)) or var_proj <= 0.0:
        var_proj = float(np.finfo(np.float64).tiny)

    proj_cov = np.asarray([[var_proj]], dtype=np.float64)

    if rule_kind == "nrd":
        factor = _bandwidth_factor_1d_nrd(
            method,
            n_eff=n_eff,
            samples_2d=proj_2d,
            data_cov=proj_cov,
            xp=np,
        )
    elif rule_kind == "r_selector":
        factor = _bandwidth_factor_1d_r_selectors(
            method,
            samples_2d=proj_2d,
            weights_1d=w_norm,
            data_cov=proj_cov,
            weighted_strategy=weighted_r_selector_strategy,
        )
    else:
        raise ValueError("rule_kind must be one of: 'nrd', 'r_selector'")

    details = {
        "projection_explained_ratio": float(explained_ratio),
        "projection_vector": np.asarray(principal_vec, dtype=np.float64),
        "projection_variance": float(var_proj),
    }
    return float(factor), details


def _as_targets_numpy_2d(targets, n_samples: int) -> np.ndarray:
    y = np.asarray(_to_numpy(targets), dtype=np.float64)
    if y.ndim == 1:
        if y.shape[0] != n_samples:
            raise ValueError("targets length must match samples")
        y = y.reshape(-1, 1)
    elif y.ndim == 2:
        if y.shape[0] != n_samples:
            raise ValueError("targets rows must match samples")
    else:
        raise ValueError("targets must be 1D or 2D")
    return y


def _stable_inverse_cov(cov, xp=np):
    d = int(cov.shape[0])
    cov_work = xp.asarray(cov, dtype=xp.float64)
    cov_work = 0.5 * (cov_work + cov_work.T)

    trace = _to_float_scalar(xp.trace(cov_work))
    base = trace / float(max(1, d)) if np.isfinite(trace) else 1.0
    jitter = max(base * 1e-12, 1e-12)

    for _ in range(8):
        try:
            return xp.linalg.inv(cov_work)
        except Exception:
            cov_work = cov_work + jitter * xp.eye(d, dtype=xp.float64)
            jitter *= 10.0

    return xp.linalg.pinv(cov_work)


def _fill_diagonal_zero(arr, xp=np):
    """Set diagonal entries to zero across NumPy/CuPy/Torch backends."""
    if hasattr(xp, "fill_diagonal"):
        xp.fill_diagonal(arr, 0.0)
        return
    if hasattr(arr, "fill_diagonal_"):
        arr.fill_diagonal_(0.0)
        return
    diag_idx = xp.arange(arr.shape[0])
    arr[diag_idx, diag_idx] = 0.0


def _kernel_regression_cv_score(
    *,
    samples_2d,
    targets_2d,
    weights_norm,
    data_cov,
    kernel_name: str,
    factor: float,
    regression_mode: str,
    xp=np,
) -> float:
    f = float(factor)
    if (not np.isfinite(f)) or f <= 0.0:
        return float("inf")

    n, d = int(samples_2d.shape[0]), int(samples_2d.shape[1])
    if n < 3:
        return float("inf")

    scaled_cov = xp.asarray(data_cov, dtype=xp.float64) * (f ** 2)
    inv_cov = _stable_inverse_cov(scaled_cov, xp=xp)

    if d == 1:
        x = samples_2d[:, 0]
        diff = x[:, None] - x[None, :]
        quad = (diff * diff) * _to_float_scalar(inv_cov[0, 0])
    else:
        s_proj = samples_2d @ inv_cov
        s_quad = xp.sum(s_proj * samples_2d, axis=1)
        cross = s_proj @ samples_2d.T
        quad = s_quad[:, None] + s_quad[None, :] - 2.0 * cross
        quad = xp.maximum(quad, 0.0)

    kernels = _kernel_values_from_quad(quad, kernel_name, xp)
    _fill_diagonal_zero(kernels, xp)

    weighted = kernels * weights_norm[None, :]
    denom = xp.sum(weighted, axis=1)
    tiny = float(np.finfo(np.float64).tiny)

    valid = denom > tiny
    if not _to_float_scalar(xp.any(valid)):
        return float("inf")

    numer_nw = weighted @ targets_2d
    pred_nw = xp.where(
        denom[:, None] > tiny,
        numer_nw / xp.where(denom[:, None] > tiny, denom[:, None], 1.0),
        xp.zeros_like(numer_nw),
    )
    pred = pred_nw

    if regression_mode == "local_linear" and d == 1:
        x = samples_2d[:, 0]
        diff = x[:, None] - x[None, :]

        s0 = denom
        s1 = xp.sum(weighted * diff, axis=1)
        s2 = xp.sum(weighted * diff * diff, axis=1)

        t0 = numer_nw
        t1 = (weighted * diff) @ targets_2d

        det = s0 * s2 - s1 * s1
        det_thresh = tiny * tiny
        use_ll = (s0 > tiny) & (xp.abs(det) > det_thresh)

        safe_det = xp.where(xp.abs(det) > det_thresh, det, 1.0)
        pred_ll = xp.where(
            use_ll[:, None],
            (xp.where(s2 > 0, s2, 0.0)[:, None] * t0 - s1[:, None] * t1) / safe_det[:, None],
            pred_nw,
        )
        pred = xp.where(use_ll[:, None], pred_ll, pred_nw)

    err = targets_2d - pred
    mse_i = xp.mean(err * err, axis=1)

    w_valid = weights_norm * valid.astype(xp.float64)
    wsum = _to_float_scalar(xp.sum(w_valid))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        return float("inf")

    score = _to_float_scalar(xp.sum(w_valid * mse_i) / wsum)
    if not np.isfinite(score):
        return float("inf")
    return score


def _as_targets_2d(targets, n_samples: int, xp=np):
    y = xp.asarray(targets, dtype=xp.float64)
    if y.ndim == 1:
        if y.shape[0] != n_samples:
            raise ValueError("targets length must match samples")
        y = y.reshape(-1, 1)
    elif y.ndim == 2:
        if y.shape[0] != n_samples:
            raise ValueError("targets rows must match samples")
    else:
        raise ValueError("targets must be 1D or 2D")
    return y


def _normalized_weights(w, xp=np):
    w_arr = xp.asarray(w, dtype=xp.float64).reshape(-1)
    w_sum = _to_float_scalar(xp.sum(w_arr))
    if w_sum <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return w_arr / w_sum


def _kernel_regression_cv_factor(
    *,
    samples_2d,
    targets_2d,
    weights_1d,
    data_cov,
    kernel_name: str,
    regression_mode: str,
    n_eff: float,
    n_features: int,
    xp=np,
) -> tuple[float, Dict[str, Any]]:
    x = xp.asarray(samples_2d, dtype=xp.float64)
    y = _as_targets_2d(targets_2d, int(x.shape[0]), xp=xp)
    w = _normalized_weights(weights_1d, xp=xp)
    cov = xp.asarray(data_cov, dtype=xp.float64)

    d = int(n_features)
    if d != int(x.shape[1]):
        raise ValueError("n_features is inconsistent with samples")

    f0 = _bandwidth_factor("scott", n_eff=float(n_eff), n_features=d)
    lower = max(float(f0) * 0.2, 1e-4)
    upper = max(float(f0) * 5.0, lower * 1.01)
    tol = max(1e-4, lower * 0.02)

    best_score_box = {"value": float("inf")}

    def _objective(f: float) -> float:
        score = _kernel_regression_cv_score(
            samples_2d=x,
            targets_2d=y,
            weights_norm=w,
            data_cov=cov,
            kernel_name=kernel_name,
            factor=f,
            regression_mode=regression_mode,
            xp=xp,
        )
        if score < best_score_box["value"]:
            best_score_box["value"] = score
        return score

    factor = _golden_section_minimize(_objective, lower, upper, tol)
    score = _objective(float(factor))

    if (not np.isfinite(factor)) or factor <= 0.0:
        raise ValueError("regression CV bandwidth rule produced a non-positive value")

    details = {
        "cv_objective": "leave_one_out_mse",
        "cv_score": float(score),
        "cv_score_best_seen": float(best_score_box["value"]),
        "cv_search_lower": float(lower),
        "cv_search_upper": float(upper),
        "cv_regression_mode": str(regression_mode),
    }
    return float(factor), details


class _BaseBandwidthSelector:
    def __init__(
        self,
        *,
        estimator: str,
        n_eff: float,
        n_features: int,
        samples_2d,
        weights_1d,
        data_cov,
        xp,
        enable_r_selectors: bool,
        weighted_r_selector_strategy: str,
        multivariate_selector_strategy: str,
    ):
        self.estimator = _normalize_estimator_name(estimator)
        self.n_eff = float(n_eff)
        self.n_features = int(n_features)
        self.samples_2d = samples_2d
        self.weights_1d = weights_1d
        self.data_cov = data_cov
        self.xp = xp
        self.enable_r_selectors = bool(enable_r_selectors)
        self.weighted_r_selector_strategy = _normalize_weighted_strategy(weighted_r_selector_strategy)
        self.multivariate_selector_strategy = _normalize_multivariate_strategy(multivariate_selector_strategy)

        self.weights_np = _normalized_weights_numpy(
            np.asarray(_to_numpy(weights_1d), dtype=np.float64).reshape(-1)
        )
        self.is_weighted = float(np.max(self.weights_np) - np.min(self.weights_np)) > 1e-12

        if (not np.isfinite(self.n_eff)) or self.n_eff <= 0.0:
            raise ValueError("n_eff must be a finite positive scalar")
        if self.n_features <= 0:
            raise ValueError("n_features must be a positive integer")

    def _base_details(self, bandwidth: Union[str, float, int]) -> Dict[str, Any]:
        return {
            "input_bandwidth": bandwidth,
            "n_samples": int(self.weights_np.size),
            "estimator": self.estimator,
        }

    def _select_special(self, bw_name: str, details: Dict[str, Any]) -> Optional[BandwidthSelectionResult]:
        return None

    def select(self, bandwidth: Union[str, float, int]) -> BandwidthSelectionResult:
        used_r_selector = False
        selector_dim = self.n_features
        weighted_used = "uniform" if not self.is_weighted else self.weighted_r_selector_strategy
        multi_used = "none"
        method_label = "scalar"
        details = self._base_details(bandwidth)

        if isinstance(bandwidth, str):
            bw_name = bandwidth.strip().lower()
            method_label = bw_name

            special = self._select_special(bw_name, details)
            if special is not None:
                return special

            if bw_name in ("nrd0", "nrd"):
                if self.n_features == 1:
                    selector_dim = 1
                    factor = _bandwidth_factor_1d_nrd(
                        bw_name,
                        n_eff=self.n_eff,
                        samples_2d=self.samples_2d,
                        data_cov=self.data_cov,
                        xp=self.xp,
                    )
                else:
                    multi_used = self.multivariate_selector_strategy
                    selector_dim = 1
                    factor, proj_details = _multivariate_factor_from_projected_1d(
                        bw_name,
                        samples_2d=self.samples_2d,
                        weights_1d=self.weights_1d,
                        data_cov=self.data_cov,
                        n_eff=self.n_eff,
                        rule_kind="nrd",
                        weighted_r_selector_strategy=self.weighted_r_selector_strategy,
                    )
                    details.update(proj_details)

                details["rule"] = "nrd"
                return BandwidthSelectionResult(
                    factor=float(factor),
                    method=method_label,
                    n_features=self.n_features,
                    n_eff=self.n_eff,
                    used_r_selector=False,
                    weighted=self.is_weighted,
                    weighted_strategy=weighted_used,
                    multivariate_strategy=multi_used,
                    selector_dimension=selector_dim,
                    details=details,
                )

            if bw_name in ("ucv", "bcv", "sj", "sj-ste", "sj-dpi"):
                if not self.enable_r_selectors:
                    raise ValueError("R-style bandwidth selectors are disabled for this estimator")

                used_r_selector = True
                details["rule"] = "r_selector"

                if self.n_features == 1:
                    selector_dim = 1
                    factor = _bandwidth_factor_1d_r_selectors(
                        bw_name,
                        samples_2d=self.samples_2d,
                        weights_1d=self.weights_1d,
                        data_cov=self.data_cov,
                        weighted_strategy=self.weighted_r_selector_strategy,
                    )
                else:
                    multi_used = self.multivariate_selector_strategy
                    selector_dim = 1
                    factor, proj_details = _multivariate_factor_from_projected_1d(
                        bw_name,
                        samples_2d=self.samples_2d,
                        weights_1d=self.weights_1d,
                        data_cov=self.data_cov,
                        n_eff=self.n_eff,
                        rule_kind="r_selector",
                        weighted_r_selector_strategy=self.weighted_r_selector_strategy,
                    )
                    details.update(proj_details)

                return BandwidthSelectionResult(
                    factor=float(factor),
                    method=method_label,
                    n_features=self.n_features,
                    n_eff=self.n_eff,
                    used_r_selector=used_r_selector,
                    weighted=self.is_weighted,
                    weighted_strategy=weighted_used,
                    multivariate_strategy=multi_used,
                    selector_dimension=selector_dim,
                    details=details,
                )

        factor = _bandwidth_factor(
            bandwidth,
            n_eff=self.n_eff,
            n_features=self.n_features,
        )
        return BandwidthSelectionResult(
            factor=float(factor),
            method=method_label,
            n_features=self.n_features,
            n_eff=self.n_eff,
            used_r_selector=used_r_selector,
            weighted=self.is_weighted,
            weighted_strategy=weighted_used,
            multivariate_strategy=multi_used,
            selector_dimension=selector_dim,
            details=details,
        )


class _KDEBandwidthSelector(_BaseBandwidthSelector):
    pass


class _KernelRegressionBandwidthSelector(_BaseBandwidthSelector):
    def __init__(self, *, targets, regression: str, kernel: str, **kwargs):
        super().__init__(**kwargs)
        self.targets = targets
        self.regression = _normalize_regression_mode(regression)
        self.kernel = str(kernel).strip().lower()

    def _select_special(self, bw_name: str, details: Dict[str, Any]) -> Optional[BandwidthSelectionResult]:
        if bw_name not in ("cv", "cv_ls", "cv-nw", "cv-ll"):
            return None

        if bw_name == "cv-ll":
            cv_mode = "local_linear"
        elif bw_name == "cv-nw":
            cv_mode = "nw"
        else:
            cv_mode = self.regression

        factor, cv_details = _kernel_regression_cv_factor(
            samples_2d=self.samples_2d,
            targets_2d=self.targets,
            weights_1d=self.weights_1d,
            data_cov=self.data_cov,
            kernel_name=self.kernel,
            regression_mode=cv_mode,
            n_eff=self.n_eff,
            n_features=self.n_features,
            xp=self.xp,
        )

        details["rule"] = "regression_cv"
        details.update(cv_details)

        return BandwidthSelectionResult(
            factor=float(factor),
            method=bw_name,
            n_features=self.n_features,
            n_eff=self.n_eff,
            used_r_selector=False,
            weighted=self.is_weighted,
            weighted_strategy=(
                "uniform" if not self.is_weighted else self.weighted_r_selector_strategy
            ),
            multivariate_strategy="none",
            selector_dimension=self.n_features,
            details=details,
        )


def select_bandwidth(
    bandwidth: Union[str, float, int],
    *,
    n_eff: float,
    n_features: int,
    samples_2d,
    weights_1d,
    data_cov,
    xp,
    enable_r_selectors: bool = True,
    weighted_r_selector_strategy: str = "quantile_resample",
    multivariate_selector_strategy: str = "projection_pca_1d",
    estimator: str = "kde",
    targets=None,
    regression: str = "nw",
    kernel: str = "gaussian",
) -> BandwidthSelectionResult:
    """Select bandwidth factor and return diagnostic metadata."""

    estimator_name = _normalize_estimator_name(estimator)
    if estimator_name == "kde":
        selector = _KDEBandwidthSelector(
            estimator=estimator_name,
            n_eff=n_eff,
            n_features=n_features,
            samples_2d=samples_2d,
            weights_1d=weights_1d,
            data_cov=data_cov,
            xp=xp,
            enable_r_selectors=enable_r_selectors,
            weighted_r_selector_strategy=weighted_r_selector_strategy,
            multivariate_selector_strategy=multivariate_selector_strategy,
        )
    else:
        selector = _KernelRegressionBandwidthSelector(
            estimator=estimator_name,
            n_eff=n_eff,
            n_features=n_features,
            samples_2d=samples_2d,
            weights_1d=weights_1d,
            data_cov=data_cov,
            xp=xp,
            enable_r_selectors=enable_r_selectors,
            weighted_r_selector_strategy=weighted_r_selector_strategy,
            multivariate_selector_strategy=multivariate_selector_strategy,
            targets=targets,
            regression=regression,
            kernel=kernel,
        )

    return selector.select(bandwidth)


def select_bandwidth_factor(
    bandwidth: Union[str, float, int],
    *,
    n_eff: float,
    n_features: int,
    samples_2d,
    weights_1d,
    data_cov,
    xp,
    enable_r_selectors: bool = True,
    weighted_r_selector_strategy: str = "quantile_resample",
    multivariate_selector_strategy: str = "projection_pca_1d",
    estimator: str = "kde",
    targets=None,
    regression: str = "nw",
    kernel: str = "gaussian",
) -> float:
    """Select bandwidth factor for kernel estimators."""
    result = select_bandwidth(
        bandwidth,
        n_eff=n_eff,
        n_features=n_features,
        samples_2d=samples_2d,
        weights_1d=weights_1d,
        data_cov=data_cov,
        xp=xp,
        enable_r_selectors=enable_r_selectors,
        weighted_r_selector_strategy=weighted_r_selector_strategy,
        multivariate_selector_strategy=multivariate_selector_strategy,
        estimator=estimator,
        targets=targets,
        regression=regression,
        kernel=kernel,
    )
    return float(result.factor)


__all__ = [
    "BandwidthSelectionResult",
    "_bandwidth_factor",
    "_bandwidth_factor_1d_nrd",
    "_bandwidth_factor_1d_r_selectors",
    "select_bandwidth",
    "select_bandwidth_factor",
]
