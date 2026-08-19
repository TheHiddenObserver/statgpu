"""Shared numerical linear algebra policy for panel fit spaces."""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar


# Normal equations are used only as an explicitly certified fast path. A Gram
# eigenvalue ratio of 1e-4 corresponds to kappa_2(X) < 100 in exact arithmetic,
# leaving a very large numerical margin from the SVD rank boundary used below.
_GRAM_CERTIFIED_MIN_EIGEN_RATIO = 1.0e-4


def _axis_max(values, xp, axis):
    if getattr(xp, "__name__", "") == "torch":
        return xp.max(values, dim=axis).values
    return xp.max(values, axis=axis)


def _axis_min(values, xp, axis):
    if getattr(xp, "__name__", "") == "torch":
        return xp.min(values, dim=axis).values
    return xp.min(values, axis=axis)


def _axis_all(values, xp, axis):
    if getattr(xp, "__name__", "") == "torch":
        return xp.all(values, dim=axis)
    return xp.all(values, axis=axis)


def _axis_sum(values, xp, axis):
    if getattr(xp, "__name__", "") == "torch":
        return xp.sum(values, dim=axis)
    return xp.sum(values, axis=axis)


def _rank_mask_backend(X, singular_values, xp):
    """Return the shared SVD retention mask and backend-native rank scalar."""
    if int(singular_values.shape[-1]) == 0:
        raise ValueError("panel design must contain at least one column")
    cutoff_scale = (
        max(int(X.shape[-2]), int(X.shape[-1]))
        * np.finfo(np.float64).eps
    )
    cutoff = xp.max(singular_values) * float(cutoff_scale)
    retained = singular_values > cutoff
    return retained, xp.sum(retained)


def _rank_mask(X, singular_values, xp):
    retained, rank_backend = _rank_mask_backend(X, singular_values, xp)
    # Keep the singular-value cutoff on the active backend. Only the final
    # integer rank must cross the device boundary for fail-closed Python control
    # flow; extracting s_max separately would add an avoidable GPU sync.
    rank = int(_to_float_scalar(rank_backend))
    return retained, rank


def _inverse_values(singular_values, retained, xp):
    safe_values = xp.where(retained, singular_values, xp.ones_like(singular_values))
    return xp.where(retained, 1.0 / safe_values, xp.zeros_like(singular_values))


def _svd_inverse_factors(X, xp):
    """Return SVD factors, inverse singular values, and shared numerical rank."""
    U, singular_values, Vh = xp.linalg.svd(X, full_matrices=False)
    retained, rank = _rank_mask(X, singular_values, xp)
    return U, Vh, _inverse_values(singular_values, retained, xp), rank


def _lstsq_working_design(X, xp, *, batched: bool):
    """Return an SVD working design plus its positive rescaling factor.

    Uniform positive rescaling leaves the relative panel rank cutoff unchanged.
    If an entire independent design lies below ``sqrt(DBL_MIN)``, raise its
    largest element to that threshold before forming inverse retained singular
    values. This prevents ``1 / s`` overflow for a full-rank subnormal design;
    the final least-squares coefficient is multiplied by the same factor to
    restore the original parameterization.
    """
    namespace = getattr(xp, "__name__", "")
    target = float(np.sqrt(np.finfo(np.float64).tiny))

    def _factor(max_abs):
        needs_scale = (max_abs > 0.0) & (max_abs < target)
        bounded_max = xp.where(
            needs_scale, max_abs, xp.full_like(max_abs, float(target))
        )
        relative = bounded_max / float(target)
        safe_relative = xp.where(
            relative > 0.0, relative, xp.ones_like(relative)
        )
        return xp.where(
            needs_scale,
            1.0 / safe_relative,
            xp.ones_like(max_abs),
        )

    if batched:
        view = xp.abs(X).reshape(int(X.shape[0]), -1)
        max_abs = (
            xp.max(view, dim=1).values
            if namespace == "torch"
            else xp.max(view, axis=1)
        )
        factor = _factor(max_abs)
        return X * factor[:, None, None], factor

    max_abs = xp.max(xp.abs(X))
    factor = _factor(max_abs)
    return X * factor, factor


def _first_constant_anchor_2d(X, y, full_rank, xp):
    """Return a safe response anchor for a full-rank exact first-column constant."""
    zero = xp.zeros((), dtype=y.dtype)
    if getattr(y, "ndim", None) != 1 or int(X.shape[1]) == 0:
        return zero, xp.ones((), dtype=y.dtype), xp.asarray(False)
    column = X[:, 0]
    value = column[0]
    exact = (
        xp.all(column == value)
        & xp.isfinite(value)
        & (value != 0.0)
        & full_rank
    )
    anchor_candidate = 0.5 * xp.min(y) + 0.5 * xp.max(y)
    return xp.where(exact, anchor_candidate, zero), value, exact


def _first_constant_anchor_batched(X, y, full_rank, xp):
    """Return per-batch anchors for full-rank exact first-column constants."""
    batch = int(X.shape[0])
    zero = xp.zeros((batch,), dtype=X.dtype)
    if getattr(y, "ndim", None) != 2 or int(X.shape[-1]) == 0:
        return zero, xp.ones((batch,), dtype=X.dtype), xp.zeros(
            (batch,), dtype=xp.bool
        )
    column = X[:, :, 0]
    value = column[:, 0]
    exact = (
        _axis_all(column == value[:, None], xp, 1)
        & xp.isfinite(value)
        & (value != 0.0)
        & full_rank
    )
    anchor_candidate = 0.5 * _axis_min(y, xp, 1) + 0.5 * _axis_max(y, xp, 1)
    return xp.where(exact, anchor_candidate, zero), value, exact


def _svd_project_2d(U, Vh, inverse_values, y, xp, *, stable: bool):
    """Project one response through retained SVD factors."""
    weighted_u_t = inverse_values.reshape(-1, 1) * U.T
    if not stable:
        return Vh.T @ (weighted_u_t @ y)

    # Import lazily so covariance/pseudoinverse users do not pay for the
    # magnitude-tiered reducer unless a response actually needs it.
    from statgpu.panel._reductions import grouped_score_sums

    products = weighted_u_t.T * y.reshape(-1, 1)
    codes = np.zeros(int(products.shape[0]), dtype=np.int64)
    projected = grouped_score_sums(
        products,
        codes,
        n_groups=1,
        xp=xp,
    )[0]
    return Vh.T @ projected


def _resolution_failure_2d(X_work, U, Vh, inverse_values, y, params_work, xp):
    """Return a backend scalar when a tiny coefficient is not numerically certified.

    The trigger is intentionally narrow. A coordinate is considered at risk only
    when its magnitude is at the roundoff scale of the absolute pseudoinverse
    projection. We then require an independently large normal-equation residual
    on column-normalized coordinates before rejecting the fit. Ordinary zeros and
    well-resolved mixed-scale identity problems therefore remain valid, while a
    finite SVD coefficient contaminated by a much larger direction fails closed.
    """
    n = max(1, int(X_work.shape[0]))
    eps = float(np.finfo(np.float64).eps)
    y_scale = xp.max(xp.abs(y))
    safe_y_scale = xp.where(y_scale > 0.0, y_scale, xp.ones_like(y_scale))

    X_pinv_work = (Vh.T * inverse_values) @ U.T
    y_unit_abs = xp.abs(y) / safe_y_scale
    projection_scale = _axis_sum(
        xp.abs(X_pinv_work) * y_unit_abs.reshape(1, -1), xp, 1
    )
    params_unit = xp.abs(params_work) / safe_y_scale
    resolution_risk = (
        (y_scale > 0.0)
        & (projection_scale > 0.0)
        & (
            params_unit
            <= float(64.0 * n * eps) * projection_scale
        )
    )

    residual_unit = y / safe_y_scale - X_work @ (params_work / safe_y_scale)
    column_scale = _axis_max(xp.abs(X_work), xp, 0)
    safe_column_scale = xp.where(
        column_scale > 0.0, column_scale, xp.ones_like(column_scale)
    )
    score_terms = (X_work / safe_column_scale.reshape(1, -1)) * residual_unit.reshape(
        -1, 1
    )
    gradient = _axis_sum(score_terms, xp, 0)
    gradient_scale = _axis_sum(xp.abs(score_terms), xp, 0)
    stationarity_bad = (
        (gradient_scale > 0.0)
        & (
            xp.abs(gradient)
            > float(4096.0 * n * eps) * gradient_scale
        )
    )
    return xp.any(resolution_risk & stationarity_bad)


def _resolution_failure_batched(
    X_work, U, Vh, inverse_values, y, params_work, xp
):
    """Vectorized form of the narrow SVD coefficient-resolution certificate."""
    if getattr(y, "ndim", None) != 2:
        return xp.zeros((int(X_work.shape[0]),), dtype=xp.bool)
    n = max(1, int(X_work.shape[1]))
    eps = float(np.finfo(np.float64).eps)
    y_scale = _axis_max(xp.abs(y), xp, 1)
    safe_y_scale = xp.where(y_scale > 0.0, y_scale, xp.ones_like(y_scale))

    X_pinv_work = xp.matmul(
        xp.swapaxes(Vh, -2, -1) * inverse_values[..., None, :],
        xp.swapaxes(U, -2, -1),
    )
    y_unit_abs = xp.abs(y) / safe_y_scale[:, None]
    projection_scale = _axis_sum(
        xp.abs(X_pinv_work) * y_unit_abs[:, None, :], xp, 2
    )
    params_unit = xp.abs(params_work) / safe_y_scale[:, None]
    resolution_risk = (
        (y_scale[:, None] > 0.0)
        & (projection_scale > 0.0)
        & (
            params_unit
            <= float(64.0 * n * eps) * projection_scale
        )
    )

    residual_unit = y / safe_y_scale[:, None] - xp.matmul(
        X_work, (params_work / safe_y_scale[:, None])[..., None]
    )[..., 0]
    column_scale = _axis_max(xp.abs(X_work), xp, 1)
    safe_column_scale = xp.where(
        column_scale > 0.0, column_scale, xp.ones_like(column_scale)
    )
    score_terms = (
        X_work / safe_column_scale[:, None, :]
    ) * residual_unit[:, :, None]
    gradient = _axis_sum(score_terms, xp, 1)
    gradient_scale = _axis_sum(xp.abs(score_terms), xp, 1)
    stationarity_bad = (
        (gradient_scale > 0.0)
        & (
            xp.abs(gradient)
            > float(4096.0 * n * eps) * gradient_scale
        )
    )
    return _axis_all(~(resolution_risk & stationarity_bad), xp, 1) == False


def _gram_resolution_risk_batched(X, y, params, xp):
    """Conservatively reject Gram candidates below certified float64 resolution."""
    batch = int(X.shape[0])
    if getattr(y, "ndim", None) != 2:
        return xp.zeros((batch,), dtype=xp.bool)
    n = max(1, int(X.shape[1]))
    eps = float(np.finfo(np.float64).eps)
    y_scale = _axis_max(xp.abs(y), xp, 1)
    safe_y_scale = xp.where(y_scale > 0.0, y_scale, xp.ones_like(y_scale))
    column_scale = _axis_max(xp.abs(X), xp, 1)
    safe_column_scale = xp.where(
        column_scale > 0.0, column_scale, xp.ones_like(column_scale)
    )
    X_unit = X / safe_column_scale[:, None, :]
    column_norm_unit = xp.sqrt(_axis_sum(X_unit * X_unit, xp, 1))
    safe_norm = xp.where(
        column_norm_unit > 0.0,
        column_norm_unit,
        xp.ones_like(column_norm_unit),
    )
    signal = (xp.abs(params) / safe_y_scale[:, None]) * safe_column_scale
    # Gram certification already guarantees kappa_2(X) < about 100. The
    # additional factor keeps this a conservative fail-closed certificate rather
    # than a claim that normal equations can resolve machine-epsilon-scale beta.
    bound = float(128.0 * n * eps * 100.0) / safe_norm
    risky = (
        (y_scale[:, None] > 0.0)
        & (xp.abs(params) > 0.0)
        & (signal <= bound)
    )
    return ~_axis_all(~risky, xp, 1)


def panel_svd_working_pseudoinverse(X, xp):
    """Return a safe working-design pseudoinverse and its rescaling factor.

    The returned pseudoinverse belongs to ``X_work = design_scale * X``.
    Consumers that combine the pseudoinverse with residual or covariance scale
    can therefore apply ``design_scale`` only after those smaller quantities
    have reduced the dynamic range, instead of materializing an unrepresentable
    original-scale ``X+`` or ``X+ X+^T`` first. The rank policy is identical to
    :func:`panel_lstsq`.
    """
    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    U, Vh, inverse_values, rank = _svd_inverse_factors(X_work, xp)
    X_pinv_work = (Vh.T * inverse_values) @ U.T
    return X_work, X_pinv_work, design_scale, rank


def panel_working_pseudoinverse(X, xp):
    """Return a working-design pseudoinverse with one-sync Gram certification.

    The Gram candidate is uniformly scaled before ``X'X`` is formed, so the
    certification calculation itself is range-safe without a host-side branch.
    All finite/spectrum/solve checks remain backend-native until one final
    boolean transfer. Certified full-rank designs use the normal-equation
    candidate; every uncertified case falls back to the shared SVD cutoff.
    """
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    if int(X.shape[-1]) == 0:
        raise ValueError("panel design must contain at least one column")

    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    n = max(1, int(X_work.shape[0]))
    k = int(X_work.shape[1])
    namespace = getattr(xp, "__name__", "")

    max_abs = xp.max(xp.abs(X_work))
    gram_limit = float(
        np.sqrt(0.25 * np.finfo(np.float64).max / float(n))
    )
    one = xp.ones_like(max_abs)
    gram_scale = xp.where(
        max_abs > float(gram_limit),
        max_abs / float(gram_limit),
        one,
    )
    X_gram = X_work / gram_scale
    gram = X_gram.T @ X_gram
    rhs = X_gram.T
    gram_finite = xp.all(xp.isfinite(gram))
    rhs_finite = xp.all(xp.isfinite(rhs))

    if namespace == "torch":
        identity = xp.eye(k, dtype=X.dtype, device=X.device)
    else:
        identity = xp.eye(k, dtype=X.dtype)
    spectrum_gram = xp.where(gram_finite, gram, identity)
    eigenvalues = xp.linalg.eigvalsh(spectrum_gram)
    smallest = eigenvalues[0]
    largest = eigenvalues[-1]
    certified = (
        gram_finite
        & rhs_finite
        & xp.isfinite(smallest)
        & xp.isfinite(largest)
        & (largest > 0.0)
        & (
            smallest
            > largest * float(_GRAM_CERTIFIED_MIN_EIGEN_RATIO)
        )
    )

    safe_gram = xp.where(certified, gram, identity)
    safe_rhs = xp.where(certified, rhs, xp.zeros_like(rhs))
    if namespace == "torch" and hasattr(xp.linalg, "solve_ex"):
        candidate, info = xp.linalg.solve_ex(
            safe_gram, safe_rhs, check_errors=False
        )
        certified = certified & (info == 0)
    else:
        candidate = xp.linalg.solve(safe_gram, safe_rhs)
    candidate = candidate / gram_scale
    certified = certified & xp.all(xp.isfinite(candidate))

    # This is the only certification transfer on the accepted Gram path.
    if bool(_to_float_scalar(certified)):
        return X_work, candidate, design_scale, k

    U, Vh, inverse_values, rank = _svd_inverse_factors(X_work, xp)
    X_pinv_work = (Vh.T * inverse_values) @ U.T
    return X_work, X_pinv_work, design_scale, rank


def panel_svd_pseudoinverse(X, xp):
    """Return X+, (X'X)+, and rank from one explicit float64 SVD mask."""
    U, Vh, inverse_values, rank = _svd_inverse_factors(X, xp)
    X_pinv = (Vh.T * inverse_values) @ U.T
    bread = X_pinv @ X_pinv.T
    return X_pinv, bread, rank


def panel_lstsq(X, y, xp):
    """Return a certified minimum-norm least-squares solution.

    Full-rank exact first-column constants remove a safe common response level
    before projection. Cancellation/range-sensitive responses use the shared
    magnitude-tiered reducer for ``diag(1/s) U' y``. For otherwise ordinary
    responses, a narrow coordinate-resolution plus stationarity certificate
    fails closed instead of publishing a finite coefficient contaminated by a
    much larger SVD direction.
    """
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    U, Vh, inverse_values, rank = _svd_inverse_factors(X_work, xp)

    anchor, constant_value, has_constant = _first_constant_anchor_2d(
        X_work, y, rank == int(X_work.shape[1]), xp
    )
    y_work = y - anchor

    from statgpu.panel._reductions import stable_reduction_flags

    stable = bool(stable_reduction_flags(y_work, xp)[0])
    params_work = _svd_project_2d(
        U, Vh, inverse_values, y_work, xp, stable=stable
    )
    if not stable and bool(
        _to_float_scalar(
            _resolution_failure_2d(
                X_work, U, Vh, inverse_values, y_work, params_work, xp
            )
        )
    ):
        raise FloatingPointError(
            "panel least-squares coefficient resolution exceeds float64 precision; "
            "rescale or reformulate the fit"
        )

    restore = xp.zeros_like(params_work)
    restore[0] = xp.where(has_constant, anchor / constant_value, xp.zeros_like(anchor))
    params = (params_work + restore) * design_scale
    if not bool(_to_float_scalar(xp.all(xp.isfinite(params)))):
        raise FloatingPointError(
            "panel least-squares coefficients are not representable in float64"
        )
    return params, rank


def panel_lstsq_deferred_rank(X, y, xp):
    """Return least-squares parameters plus a backend-native rank scalar.

    The final numerical rank remains backend-native so Fama-MacBeth can pack its
    fallback rank decisions. A full-rank exact first-column constant is safely
    centered before the SVD projection. If an otherwise ordinary projection
    fails the narrow coefficient-resolution/stationarity certificate, the
    backend rank is set to zero so the caller fails closed instead of consuming
    the unreliable coefficient vector.
    """
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    U, singular_values, Vh = xp.linalg.svd(X_work, full_matrices=False)
    retained, rank_backend = _rank_mask_backend(X_work, singular_values, xp)
    inverse_values = _inverse_values(singular_values, retained, xp)
    full_rank = rank_backend == int(X_work.shape[1])
    anchor, constant_value, has_constant = _first_constant_anchor_2d(
        X_work, y, full_rank, xp
    )
    y_work = y - anchor
    params_work = _svd_project_2d(
        U, Vh, inverse_values, y_work, xp, stable=False
    )
    failure = _resolution_failure_2d(
        X_work, U, Vh, inverse_values, y_work, params_work, xp
    )
    rank_backend = xp.where(failure, xp.zeros_like(rank_backend), rank_backend)
    restore = xp.zeros_like(params_work)
    restore[0] = xp.where(has_constant, anchor / constant_value, xp.zeros_like(anchor))
    return (params_work + restore) * design_scale, rank_backend


def _validate_batched_lstsq_inputs(X, y):
    if getattr(X, "ndim", None) != 3:
        raise ValueError("batched panel design must have shape (batch, n_obs, n_features)")
    if int(X.shape[-1]) == 0:
        raise ValueError("panel design must contain at least one column")
    if getattr(y, "ndim", None) not in (2, 3):
        raise ValueError("batched panel response must have shape (batch, n_obs[, n_targets])")
    if int(y.shape[0]) != int(X.shape[0]) or int(y.shape[1]) != int(X.shape[1]):
        raise ValueError("batched panel design and response shapes are incompatible")


def panel_lstsq_gram_certified_batched(
    X,
    y,
    xp,
    *,
    min_eigen_ratio: float = _GRAM_CERTIFIED_MIN_EIGEN_RATIO,
):
    """Return a fast Gram-solve candidate and a backend-native safety mask.

    The fast path is deliberately not a replacement for the shared SVD rank
    policy. Clearly well-conditioned matrices are candidates only. A full-rank
    exact first-column constant is response-centered before ``X' y`` so a huge
    common level cannot overflow the Gram RHS or contaminate a smaller slope.
    Candidates whose nonzero coordinates lie below the conservative float64
    resolution bound are also rejected for SVD fallback.
    """
    _validate_batched_lstsq_inputs(X, y)
    ratio = float(min_eigen_ratio)
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("min_eigen_ratio must be finite and strictly between 0 and 1")

    namespace = getattr(xp, "__name__", "")
    if namespace not in {"numpy", "cupy", "torch"}:
        raise NotImplementedError(
            "certified Gram solve requires NumPy, CuPy, or Torch batched linear algebra"
        )

    batch = int(X.shape[0])
    full_rank_candidate = xp.ones((batch,), dtype=xp.bool)
    anchor, constant_value, has_constant = _first_constant_anchor_batched(
        X, y, full_rank_candidate, xp
    )
    y_work = y - anchor[:, None] if getattr(y, "ndim", None) == 2 else y

    transpose = xp.swapaxes(X, -2, -1)
    gram = xp.matmul(transpose, X)
    rhs_input = y_work[..., None] if getattr(y_work, "ndim", None) == 2 else y_work
    rhs = xp.matmul(transpose, rhs_input)

    gram_finite_view = xp.isfinite(gram).reshape(int(gram.shape[0]), -1)
    gram_finite = _axis_all(gram_finite_view, xp, 1)
    rhs_finite_view = xp.isfinite(rhs).reshape(int(rhs.shape[0]), -1)
    rhs_finite = _axis_all(rhs_finite_view, xp, 1)

    k = int(X.shape[-1])
    if namespace == "torch":
        identity = xp.eye(k, dtype=X.dtype, device=X.device)
    else:
        identity = xp.eye(k, dtype=X.dtype)
    spectrum_gram = xp.where(gram_finite[..., None, None], gram, identity)
    eigenvalues = xp.linalg.eigvalsh(spectrum_gram)
    smallest = eigenvalues[..., 0]
    largest = eigenvalues[..., -1]
    certified = (
        gram_finite
        & xp.isfinite(smallest)
        & xp.isfinite(largest)
        & (largest > 0.0)
        & (smallest > largest * ratio)
        & rhs_finite
    )

    safe_gram = xp.where(certified[..., None, None], gram, identity)
    safe_rhs = xp.where(certified[..., None, None], rhs, xp.zeros_like(rhs))

    if namespace == "torch" and hasattr(xp.linalg, "solve_ex"):
        params, info = xp.linalg.solve_ex(safe_gram, safe_rhs, check_errors=False)
        certified = certified & (info == 0)
    else:
        params = xp.linalg.solve(safe_gram, safe_rhs)

    params_finite_view = xp.isfinite(params).reshape(int(params.shape[0]), -1)
    params_finite = _axis_all(params_finite_view, xp, 1)
    certified = certified & params_finite

    if getattr(y_work, "ndim", None) == 2:
        params_centered = params[..., 0]
        resolution_risk = _gram_resolution_risk_batched(
            X, y_work, params_centered, xp
        )
        certified = certified & (~resolution_risk)
        restore = xp.zeros_like(params_centered)
        restore[:, 0] = xp.where(
            has_constant,
            anchor / constant_value,
            xp.zeros_like(anchor),
        )
        params = params_centered + restore
    return params, certified


def panel_lstsq_batched(X, y, xp):
    """Solve equal-shaped panel least-squares problems with one stacked SVD call.

    NumPy and Torch use the same rank cutoff as :func:`panel_lstsq`. Full-rank
    first-column constants remove a safe common response level. A vectorized
    resolution/stationarity certificate maps unresolved SVD solutions to rank
    zero so Fama-MacBeth fails closed after its single packed rank transfer.
    """
    namespace = getattr(xp, "__name__", "")
    if namespace not in {"numpy", "torch"}:
        raise NotImplementedError(
            "panel_lstsq_batched requires a namespace with documented stacked-SVD "
            "support; use panel_lstsq_deferred_rank for two-dimensional CuPy solves"
        )
    _validate_batched_lstsq_inputs(X, y)

    X_work, design_scale = _lstsq_working_design(X, xp, batched=True)
    U, singular_values, Vh = xp.linalg.svd(X_work, full_matrices=False)
    cutoff_scale = (
        max(int(X_work.shape[-2]), int(X_work.shape[-1]))
        * np.finfo(np.float64).eps
    )
    largest = _axis_max(singular_values, xp, -1)
    retained = singular_values > largest[..., None] * float(cutoff_scale)
    ranks = _axis_sum(retained, xp, -1)
    inverse_values = _inverse_values(singular_values, retained, xp)

    full_rank = ranks == int(X_work.shape[-1])
    anchor, constant_value, has_constant = _first_constant_anchor_batched(
        X_work, y, full_rank, xp
    )
    y_work = y - anchor[:, None] if getattr(y, "ndim", None) == 2 else y

    rhs = y_work[..., None] if getattr(y_work, "ndim", None) == 2 else y_work
    weighted_u_t = xp.swapaxes(U, -2, -1) * inverse_values[..., :, None]
    scaled = xp.matmul(weighted_u_t, rhs)
    params_work = xp.matmul(xp.swapaxes(Vh, -2, -1), scaled)

    if getattr(y_work, "ndim", None) == 2:
        params_work = params_work[..., 0]
        failure = _resolution_failure_batched(
            X_work, U, Vh, inverse_values, y_work, params_work, xp
        )
        ranks = xp.where(failure, xp.zeros_like(ranks), ranks)
        restore = xp.zeros_like(params_work)
        restore[:, 0] = xp.where(
            has_constant,
            anchor / constant_value,
            xp.zeros_like(anchor),
        )
        params = (params_work + restore) * design_scale[:, None]
    else:
        params = params_work * design_scale[:, None, None]
    return params, ranks


def panel_matrix_rank(X, xp):
    """Return numerical rank using exactly the panel pseudoinverse cutoff."""
    if getattr(xp, "__name__", "") == "torch":
        singular_values = xp.linalg.svdvals(X)
    else:
        singular_values = xp.linalg.svd(X, compute_uv=False)
    _retained, rank = _rank_mask(X, singular_values, xp)
    return rank
