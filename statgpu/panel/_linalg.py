"""Shared numerical linear algebra policy for panel fit spaces."""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar, xp_eye


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


def _panel_svd(X, xp, *, full_matrices=False):
    """Backend SVD with a numerically exact Torch CUDA driver.

    cuSOLVER's default gesvdj driver leaks ~1e-16 into structurally-zero U
    entries, which large responses amplify into spurious coefficients.  The
    QR-based gesvd driver keeps those entries exact, matching the NumPy/LAPACK
    reference.  Panel code calls ``torch.linalg`` directly (the array module is
    ``torch`` itself), so the driver selection must happen here rather than in
    the backend wrapper.
    """
    if getattr(xp, "__name__", "") == "torch" and getattr(X, "is_cuda", False):
        # Fail closed if the exact driver is unavailable: silently retrying
        # with the default gesvdj driver reintroduces the structurally-zero U
        # leakage this helper exists to prevent.
        return xp.linalg.svd(X, full_matrices=full_matrices, driver="gesvd")
    return xp.linalg.svd(X, full_matrices=full_matrices)


def _rank_mask_backend(X, singular_values, xp):
    """Return the shared SVD retention mask and backend-native rank scalar."""
    if int(singular_values.shape[-1]) == 0:
        raise ValueError("panel design must contain at least one column")
    cutoff_scale = max(int(X.shape[-2]), int(X.shape[-1])) * np.finfo(np.float64).eps
    cutoff = xp.max(singular_values) * float(cutoff_scale)
    retained = singular_values > cutoff
    return retained, xp.sum(retained)


def _rank_mask(X, singular_values, xp):
    retained, rank_backend = _rank_mask_backend(X, singular_values, xp)
    rank = int(_to_float_scalar(rank_backend))
    return retained, rank


def _inverse_values(singular_values, retained, xp):
    safe_values = xp.where(retained, singular_values, xp.ones_like(singular_values))
    return xp.where(retained, 1.0 / safe_values, xp.zeros_like(singular_values))


def _svd_inverse_factors(X, xp):
    """Return SVD factors, inverse singular values, and shared numerical rank."""
    U, singular_values, Vh = _panel_svd(X, xp)
    retained, rank = _rank_mask(X, singular_values, xp)
    return U, Vh, _inverse_values(singular_values, retained, xp), rank


def _lstsq_working_design(X, xp, *, batched: bool):
    """Return an SVD working design plus its positive rescaling factor."""
    target = float(np.sqrt(np.finfo(np.float64).tiny))

    def _factor(max_abs):
        needs_scale = (max_abs > 0.0) & (max_abs < target)
        bounded_max = xp.where(needs_scale, max_abs, xp.full_like(max_abs, target))
        relative = bounded_max / target
        safe_relative = xp.where(relative > 0.0, relative, xp.ones_like(relative))
        return xp.where(needs_scale, 1.0 / safe_relative, xp.ones_like(max_abs))

    if batched:
        view = xp.abs(X).reshape(int(X.shape[0]), -1)
        max_abs = _axis_max(view, xp, 1)
        factor = _factor(max_abs)
        return X * factor[:, None, None], factor

    max_abs = xp.max(xp.abs(X))
    factor = _factor(max_abs)
    return X * factor, factor


def _first_constant_anchor_2d(X, y, full_rank, xp):
    """Return a safe response anchor for a full-rank exact first-column constant."""
    ref = y.reshape(-1)[0]
    zero = xp.zeros_like(ref)
    one = xp.ones_like(ref)
    false = xp.zeros_like(ref, dtype=bool)
    if getattr(y, "ndim", None) != 1 or int(X.shape[1]) == 0:
        return zero, one, false
    column = X[:, 0]
    value = column[0]
    exact = xp.all(column == value) & xp.isfinite(value) & (value != 0.0) & full_rank
    anchor_candidate = 0.5 * xp.min(y) + 0.5 * xp.max(y)
    return xp.where(exact, anchor_candidate, zero), value, exact


def _first_constant_anchor_batched(X, y, full_rank, xp):
    """Return per-batch anchors for full-rank exact first-column constants."""
    ref = X[:, 0, 0]
    zero = xp.zeros_like(ref)
    one = xp.ones_like(ref)
    false = xp.zeros_like(ref, dtype=bool)
    if getattr(y, "ndim", None) != 2 or int(X.shape[-1]) == 0:
        return zero, one, false
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


def _safe_constant_restore(anchor, constant_value, has_constant, xp):
    safe_value = xp.where(has_constant, constant_value, xp.ones_like(constant_value))
    return xp.where(has_constant, anchor / safe_value, xp.zeros_like(anchor))


def _svd_project_2d(U, Vh, inverse_values, y, xp, *, stable: bool):
    """Project one response through retained SVD factors."""
    weighted_u_t = inverse_values.reshape(-1, 1) * U.T
    if not stable:
        return Vh.T @ (weighted_u_t @ y)

    from statgpu.panel._reductions import grouped_score_sums

    products = weighted_u_t.T * y.reshape(-1, 1)
    codes = np.zeros(int(products.shape[0]), dtype=np.int64)
    projected = grouped_score_sums(products, codes, n_groups=1, xp=xp)[0]
    return Vh.T @ projected


def _svd_project_batched_stable(U, Vh, inverse_values, y, xp):
    """Project batched single responses with stable per-period reductions."""
    from statgpu.panel._reductions import grouped_score_sums

    batch = int(U.shape[0])
    n_obs = int(U.shape[1])
    weighted_u_t = xp.swapaxes(U, -2, -1) * inverse_values[..., :, None]
    products = xp.swapaxes(weighted_u_t, -2, -1) * y[..., None]
    flat_products = products.reshape(batch * n_obs, int(products.shape[-1]))
    codes = np.repeat(np.arange(batch, dtype=np.int64), n_obs)
    projected = grouped_score_sums(
        flat_products,
        codes,
        n_groups=batch,
        xp=xp,
    )
    return xp.matmul(xp.swapaxes(Vh, -2, -1), projected[..., None])[..., 0]


def _stable_batched_response_sums(y, xp):
    """Return one magnitude-tiered sum for every row of a response batch.

    The Fama-MacBeth first Gram-RHS coordinate is an exact constant times the
    period response sum. Treat periods as columns of one grouped reduction so a
    tail such as ``[2**55, 1, -2**55]`` is retained without a per-period Python
    loop or host synchronization.
    """
    from statgpu.panel._reductions import grouped_score_sums

    if getattr(y, "ndim", None) != 2:
        raise ValueError("batched response sums require shape (batch, n_obs)")
    n_obs = int(y.shape[1])
    codes = np.zeros(n_obs, dtype=np.int64)
    scores = xp.swapaxes(y, 0, 1)
    return grouped_score_sums(scores, codes, n_groups=1, xp=xp)[0]


def _stable_rhs_2d(X, y, xp):
    """Return a magnitude-tiered normal-equation RHS for one fallback period."""
    from statgpu.panel._reductions import grouped_score_sums

    products = X * y.reshape(-1, 1)
    codes = np.zeros(int(X.shape[0]), dtype=np.int64)
    return grouped_score_sums(products, codes, n_groups=1, xp=xp)[0]


def _stable_rhs_batched(X, y, xp):
    """Return magnitude-tiered normal-equation RHS rows for a fallback batch."""
    from statgpu.panel._reductions import grouped_score_sums

    batch = int(X.shape[0])
    n_obs = int(X.shape[1])
    products = X * y[..., None]
    flat_products = products.reshape(batch * n_obs, int(X.shape[-1]))
    codes = np.repeat(np.arange(batch, dtype=np.int64), n_obs)
    return grouped_score_sums(
        flat_products,
        codes,
        n_groups=batch,
        xp=xp,
    )


def _ambiguous_zero_rhs_2d(X, y, xp, *, ignore_first=False):
    """Detect a raw zero RHS only when stable summation recovers a nonzero tail."""
    raw_rhs = X.T @ y
    stable_rhs = _stable_rhs_2d(X, y, xp)
    ambiguous = (raw_rhs == 0.0) & (stable_rhs != 0.0)
    if int(ambiguous.shape[0]) > 0:
        ambiguous[0] = ambiguous[0] & (~ignore_first)
    return xp.any(ambiguous)


def _ambiguous_zero_rhs_batched(X, y, xp, *, ignore_first=None):
    """Vectorized lost-zero certificate for Fama-MacBeth SVD fallbacks."""
    if getattr(y, "ndim", None) != 2:
        return xp.zeros_like(X[:, 0, 0], dtype=bool)
    rhs = xp.matmul(xp.swapaxes(X, -2, -1), y[..., None])[..., 0]
    stable_rhs = _stable_rhs_batched(X, y, xp)
    ambiguous = (rhs == 0.0) & (stable_rhs != 0.0)
    if ignore_first is not None and int(ambiguous.shape[1]) > 0:
        ambiguous[:, 0] = ambiguous[:, 0] & (~ignore_first)
    return ~_axis_all(~ambiguous, xp, 1)


def _stable_normal_equation_failure_2d(X, y, params, xp, *, ignore_first=False):
    """Reject a well-conditioned SVD fallback that disagrees with stable X' y."""
    n = max(1, int(X.shape[0]))
    k = int(X.shape[1])
    eps = float(np.finfo(np.float64).eps)
    gram = X.T @ X
    gram_finite = xp.all(xp.isfinite(gram))
    identity = xp_eye(k, X.dtype, xp, X)
    spectrum_gram = xp.where(gram_finite, gram, identity)
    eigenvalues = xp.linalg.eigvalsh(spectrum_gram)
    smallest = eigenvalues[0]
    largest = eigenvalues[-1]
    gram_safe = (
        gram_finite
        & xp.isfinite(smallest)
        & xp.isfinite(largest)
        & (largest > 0.0)
        & (smallest > largest * float(_GRAM_CERTIFIED_MIN_EIGEN_RATIO))
    )
    stable_rhs = _stable_rhs_2d(X, y, xp)
    predicted_rhs = gram @ params
    term_scale = xp.abs(gram) @ xp.abs(params)
    normal_scale = xp.maximum(xp.abs(stable_rhs), term_scale)
    finite = (
        xp.isfinite(stable_rhs)
        & xp.isfinite(predicted_rhs)
        & xp.isfinite(normal_scale)
    )
    mismatch = (
        gram_safe
        & finite
        & (
            xp.abs(stable_rhs - predicted_rhs)
            > float(4096.0 * n * eps) * normal_scale
        )
    )
    if int(mismatch.shape[0]) > 0:
        mismatch[0] = mismatch[0] & (~ignore_first)
    return xp.any(mismatch)


def _stable_normal_equation_failure_batched(X, y, params, xp, *, ignore_first=None):
    """Vectorized stable normal-equation certificate for FMB SVD fallbacks."""
    if getattr(y, "ndim", None) != 2:
        return xp.zeros_like(X[:, 0, 0], dtype=bool)
    batch = int(X.shape[0])
    n = max(1, int(X.shape[1]))
    k = int(X.shape[2])
    eps = float(np.finfo(np.float64).eps)
    transpose = xp.swapaxes(X, -2, -1)
    gram = xp.matmul(transpose, X)
    gram_finite = _axis_all(xp.isfinite(gram).reshape(batch, -1), xp, 1)
    identity = xp_eye(k, X.dtype, xp, X)
    spectrum_gram = xp.where(gram_finite[:, None, None], gram, identity)
    eigenvalues = xp.linalg.eigvalsh(spectrum_gram)
    smallest = eigenvalues[:, 0]
    largest = eigenvalues[:, -1]
    gram_safe = (
        gram_finite
        & xp.isfinite(smallest)
        & xp.isfinite(largest)
        & (largest > 0.0)
        & (smallest > largest * float(_GRAM_CERTIFIED_MIN_EIGEN_RATIO))
    )
    stable_rhs = _stable_rhs_batched(X, y, xp)
    predicted_rhs = xp.matmul(gram, params[..., None])[..., 0]
    term_scale = xp.matmul(xp.abs(gram), xp.abs(params)[..., None])[..., 0]
    normal_scale = xp.maximum(xp.abs(stable_rhs), term_scale)
    finite = (
        xp.isfinite(stable_rhs)
        & xp.isfinite(predicted_rhs)
        & xp.isfinite(normal_scale)
    )
    mismatch = (
        gram_safe[:, None]
        & finite
        & (
            xp.abs(stable_rhs - predicted_rhs)
            > float(4096.0 * n * eps) * normal_scale
        )
    )
    if ignore_first is not None and int(mismatch.shape[1]) > 0:
        mismatch[:, 0] = mismatch[:, 0] & (~ignore_first)
    return ~_axis_all(~mismatch, xp, 1)


def _resolution_failure_2d(
    X_work,
    U,
    Vh,
    inverse_values,
    y,
    params_work,
    xp,
    *,
    ignore_first=False,
):
    """Mark a coordinate whose solved value is below the design resolution."""
    n = max(1, int(X_work.shape[0]))
    eps = float(np.finfo(np.float64).eps)
    y_scale = xp.max(xp.abs(y))
    safe_y_scale = xp.where(y_scale > 0.0, y_scale, xp.ones_like(y_scale))

    X_pinv_work = (Vh.T * inverse_values) @ U.T
    y_unit_abs = xp.abs(y) / safe_y_scale
    projection_scale = _axis_sum(
        xp.abs(X_pinv_work) * y_unit_abs.reshape(1, -1), xp, 1
    )
    params_abs = xp.abs(params_work)
    params_unit = params_abs / safe_y_scale
    resolution_risk = (
        (y_scale > 0.0)
        & (params_abs > 0.0)
        & (projection_scale > 0.0)
        & (params_unit <= float(64.0 * n * eps) * projection_scale)
    )
    if int(resolution_risk.shape[0]) > 0:
        resolution_risk[0] = resolution_risk[0] & (~ignore_first)

    # ``resolution_risk`` is itself the deterministic certificate: its
    # threshold (64 n eps * projection_scale) already encodes the per-column
    # pseudoinverse scale (1/s_min through X_pinv_work) and therefore the
    # design's condition, independent of the specific rounding of any SVD
    # driver.  Requiring an additional residual-stationarity confirmation let a
    # numerically stationary candidate for the already-rounded response pass
    # even though the coefficient cannot be resolved, which made the
    # certificate LAPACK-version dependent.
    return xp.any(resolution_risk)


def _resolution_failure_batched(
    X_work,
    U,
    Vh,
    inverse_values,
    y,
    params_work,
    xp,
    *,
    ignore_first=None,
):
    """Vectorized form of the narrow coefficient-resolution certificate."""
    if getattr(y, "ndim", None) != 2:
        return xp.zeros_like(X_work[:, 0, 0], dtype=bool)
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
    params_abs = xp.abs(params_work)
    params_unit = params_abs / safe_y_scale[:, None]
    resolution_risk = (
        (y_scale[:, None] > 0.0)
        & (params_abs > 0.0)
        & (projection_scale > 0.0)
        & (params_unit <= float(64.0 * n * eps) * projection_scale)
    )
    if ignore_first is not None and int(resolution_risk.shape[1]) > 0:
        resolution_risk[:, 0] = resolution_risk[:, 0] & (~ignore_first)

    # ``resolution_risk`` is itself the deterministic certificate (see the 2-D
    # variant): its threshold already encodes the per-column pseudoinverse
    # scale, independent of the SVD driver's rounding.
    return ~_axis_all(~resolution_risk, xp, 1)


def _gram_resolution_risk_batched(
    inverse_gram,
    transpose,
    rhs_input,
    params,
    certified,
    xp,
    *,
    ignore_first=None,
):
    """Bound unresolved Gram coordinates from RHS rounding."""
    if getattr(rhs_input, "ndim", None) != 3 or int(rhs_input.shape[-1]) != 1:
        return xp.zeros_like(certified, dtype=bool)
    n = max(1, int(transpose.shape[-1]))
    eps = float(np.finfo(np.float64).eps)
    rhs_abs = xp.matmul(xp.abs(transpose), xp.abs(rhs_input))[..., 0]
    rhs_error = float(64.0 * n * eps) * rhs_abs
    beta_error = xp.matmul(xp.abs(inverse_gram), rhs_error[..., None])[..., 0]
    risky = (
        certified[:, None]
        & (beta_error > 0.0)
        & (xp.abs(params) <= beta_error)
    )
    if ignore_first is not None and int(risky.shape[1]) > 0:
        risky[:, 0] = risky[:, 0] & (~ignore_first)
    return ~_axis_all(~risky, xp, 1)


def panel_svd_working_pseudoinverse(X, xp):
    """Return a safe working-design pseudoinverse and its rescaling factor."""
    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    U, Vh, inverse_values, rank = _svd_inverse_factors(X_work, xp)
    X_pinv_work = (Vh.T * inverse_values) @ U.T
    return X_work, X_pinv_work, design_scale, rank


def panel_working_pseudoinverse(X, xp):
    """Return a working-design pseudoinverse with one-sync Gram certification."""
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    if int(X.shape[-1]) == 0:
        raise ValueError("panel design must contain at least one column")

    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    n = max(1, int(X_work.shape[0]))
    k = int(X_work.shape[1])
    namespace = getattr(xp, "__name__", "")

    max_abs = xp.max(xp.abs(X_work))
    gram_limit = float(np.sqrt(0.25 * np.finfo(np.float64).max / float(n)))
    one = xp.ones_like(max_abs)
    gram_scale = xp.where(max_abs > gram_limit, max_abs / gram_limit, one)
    X_gram = X_work / gram_scale
    gram = X_gram.T @ X_gram
    rhs = X_gram.T
    gram_finite = xp.all(xp.isfinite(gram))
    rhs_finite = xp.all(xp.isfinite(rhs))

    identity = xp_eye(k, X.dtype, xp, X)
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
        & (smallest > largest * float(_GRAM_CERTIFIED_MIN_EIGEN_RATIO))
    )

    safe_gram = xp.where(certified, gram, identity)
    safe_rhs = xp.where(certified, rhs, xp.zeros_like(rhs))
    if namespace == "torch" and hasattr(xp.linalg, "solve_ex"):
        candidate, info = xp.linalg.solve_ex(safe_gram, safe_rhs, check_errors=False)
        certified = certified & (info == 0)
    else:
        candidate = xp.linalg.solve(safe_gram, safe_rhs)
    candidate = candidate / gram_scale
    certified = certified & xp.all(xp.isfinite(candidate))

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
    """Return a certified minimum-norm least-squares solution."""
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    U, singular_values, Vh = _panel_svd(X_work, xp)
    retained, rank_backend = _rank_mask_backend(X_work, singular_values, xp)
    inverse_values = _inverse_values(singular_values, retained, xp)
    k = int(X_work.shape[1])

    anchor, constant_value, has_constant = _first_constant_anchor_2d(
        X_work, y, rank_backend == k, xp
    )
    y_work = y - anchor

    from statgpu.panel._reductions import stable_reduction_flags

    stable = bool(stable_reduction_flags(y_work, xp)[0])
    params_work = _svd_project_2d(U, Vh, inverse_values, y_work, xp, stable=stable)
    # The coefficient-resolution certificate runs on full-rank designs with
    # more than one column.  ``stable`` only selects how the SVD projection sums
    # cancellation-safe tiers, which is orthogonal to whether a coefficient is
    # below the design's numerical resolution; skipping the check on the stable
    # path let a stationary candidate for the already-rounded response pass even
    # though the coefficient cannot be resolved (the certificate became
    # LAPACK-version dependent).  A single-column design has condition number
    # exactly one (no ill-conditioned direction), so its coefficient is either
    # well resolved or, for a fixed-effect-absorbed column, irrelevant to the
    # fit; a genuinely rank-deficient design skips the certificate too and the
    # caller reports the actual numerical rank.
    failure = _resolution_failure_2d(
        X_work,
        U,
        Vh,
        inverse_values,
        y_work,
        params_work,
        xp,
        ignore_first=has_constant,
    ) & (rank_backend == k) & (k > 1)

    restore = xp.zeros_like(params_work)
    restore[0] = _safe_constant_restore(anchor, constant_value, has_constant, xp)
    params = (params_work + restore) * design_scale
    finite = xp.all(xp.isfinite(params))

    status = xp.where(failure, xp.full_like(rank_backend, -(k + 1)), rank_backend)
    status = xp.where(finite, status, xp.full_like(rank_backend, -2 * (k + 1)))
    status_value = int(_to_float_scalar(status))
    if status_value == -(k + 1):
        raise FloatingPointError(
            "panel least-squares coefficient resolution exceeds float64 precision; "
            "rescale or reformulate the fit"
        )
    if status_value == -2 * (k + 1):
        raise FloatingPointError(
            "panel least-squares coefficients are not representable in float64"
        )
    return params, status_value


def panel_lstsq_deferred_rank(X, y, xp):
    """Return least-squares parameters plus a backend-native rank scalar.

    This function is a Fama-MacBeth fallback after Gram rejection, so its
    response projection always uses the stable grouped reducer. Rank remains
    backend-native for the caller's existing packed control transfer. A
    coefficient-resolution sentinel may replace that rank only after SVD has
    already established full column rank; genuine rank deficiency remains the
    authoritative public failure reason.
    """
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    U, singular_values, Vh = _panel_svd(X_work, xp)
    retained, rank_backend = _rank_mask_backend(X_work, singular_values, xp)
    inverse_values = _inverse_values(singular_values, retained, xp)
    k = int(X_work.shape[1])
    full_rank = rank_backend == k
    anchor, constant_value, has_constant = _first_constant_anchor_2d(
        X_work, y, full_rank, xp
    )
    y_work = y - anchor
    params_work = _svd_project_2d(U, Vh, inverse_values, y_work, xp, stable=True)
    stable_rhs = _stable_rhs_2d(X_work, y_work, xp)
    stable_zero_solution = (
        full_rank & (design_scale == 1.0) & xp.all(stable_rhs == 0.0)
    )
    params_work = xp.where(
        stable_zero_solution, xp.zeros_like(params_work), params_work
    )
    failure = _resolution_failure_2d(
        X_work,
        U,
        Vh,
        inverse_values,
        y_work,
        params_work,
        xp,
        ignore_first=has_constant,
    ) & (k > 1)
    ambiguous_zero = _ambiguous_zero_rhs_2d(
        X_work, y_work, xp, ignore_first=has_constant
    )
    stable_normal_failure = _stable_normal_equation_failure_2d(
        X_work, y_work, params_work, xp, ignore_first=has_constant
    )
    precision_failure = full_rank & (
        failure | ambiguous_zero | stable_normal_failure
    )
    rank_backend = xp.where(
        precision_failure, xp.zeros_like(rank_backend), rank_backend
    )
    restore = xp.zeros_like(params_work)
    restore[0] = _safe_constant_restore(anchor, constant_value, has_constant, xp)
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

    For the single-response Fama-MacBeth path, the exact constant coordinate uses
    a magnitude-tiered response sum before the Gram solve. This preserves a
    cancellation tail in the period intercept without forcing genuine zero
    intercept periods off the fast path. One augmented solve returns both the
    candidate coefficients and ``G^{-1}`` needed to certify the remaining
    non-constant coordinates, so the precision check still uses one factorization.
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

    full_rank_candidate = xp.ones_like(X[:, 0, 0], dtype=bool)
    anchor, constant_value, has_constant = _first_constant_anchor_batched(
        X, y, full_rank_candidate, xp
    )
    y_work = y - anchor[:, None] if getattr(y, "ndim", None) == 2 else y

    transpose = xp.swapaxes(X, -2, -1)
    gram = xp.matmul(transpose, X)
    rhs_input = y_work[..., None] if getattr(y_work, "ndim", None) == 2 else y_work
    rhs = xp.matmul(transpose, rhs_input)
    if getattr(y_work, "ndim", None) == 2:
        stable_response_sum = _stable_batched_response_sums(y_work, xp)
        stable_first_rhs = constant_value * stable_response_sum
        rhs[:, 0, 0] = xp.where(has_constant, stable_first_rhs, rhs[:, 0, 0])

    gram_finite_view = xp.isfinite(gram).reshape(int(gram.shape[0]), -1)
    gram_finite = _axis_all(gram_finite_view, xp, 1)
    rhs_finite_view = xp.isfinite(rhs).reshape(int(rhs.shape[0]), -1)
    rhs_finite = _axis_all(rhs_finite_view, xp, 1)

    k = int(X.shape[-1])
    identity = xp_eye(k, X.dtype, xp, X)
    spectrum_gram = xp.where(gram_finite[..., None, None], gram, identity)
    eigenvalues = xp.linalg.eigvalsh(spectrum_gram)
    smallest = eigenvalues[..., 0]
    largest = eigenvalues[..., -1]
    certified = (
        gram_finite
        & rhs_finite
        & xp.isfinite(smallest)
        & xp.isfinite(largest)
        & (largest > 0.0)
        & (smallest > largest * ratio)
    )

    safe_gram = xp.where(certified[..., None, None], gram, identity)
    safe_rhs = xp.where(certified[..., None, None], rhs, xp.zeros_like(rhs))
    need_inverse = getattr(y_work, "ndim", None) == 2
    if need_inverse:
        identity_batch = xp.broadcast_to(identity, safe_gram.shape)
        solve_rhs = xp.concatenate((safe_rhs, identity_batch), axis=-1)
    else:
        solve_rhs = safe_rhs

    if namespace == "torch" and hasattr(xp.linalg, "solve_ex"):
        solved, info = xp.linalg.solve_ex(safe_gram, solve_rhs, check_errors=False)
        certified = certified & (info == 0)
    else:
        solved = xp.linalg.solve(safe_gram, solve_rhs)

    rhs_width = int(safe_rhs.shape[-1])
    params = solved[..., :rhs_width]
    inverse_gram = solved[..., rhs_width:] if need_inverse else None
    solved_finite_view = xp.isfinite(solved).reshape(int(solved.shape[0]), -1)
    solved_finite = _axis_all(solved_finite_view, xp, 1)
    certified = certified & solved_finite

    if need_inverse:
        params_centered = params[..., 0]
        resolution_risk = _gram_resolution_risk_batched(
            inverse_gram,
            transpose,
            rhs_input,
            params_centered,
            certified,
            xp,
            ignore_first=has_constant,
        )
        certified = certified & (~resolution_risk)
        restore = xp.zeros_like(params_centered)
        restore[:, 0] = _safe_constant_restore(anchor, constant_value, has_constant, xp)
        params = params_centered + restore
    return params, certified


def panel_lstsq_batched(X, y, xp):
    """Solve equal-shaped panel least-squares fallbacks with one stacked SVD call."""
    namespace = getattr(xp, "__name__", "")
    if namespace not in {"numpy", "torch"}:
        raise NotImplementedError(
            "panel_lstsq_batched requires a namespace with documented stacked-SVD "
            "support; use panel_lstsq_deferred_rank for two-dimensional CuPy solves"
        )
    _validate_batched_lstsq_inputs(X, y)

    X_work, design_scale = _lstsq_working_design(X, xp, batched=True)
    U, singular_values, Vh = _panel_svd(X_work, xp)
    cutoff_scale = max(int(X_work.shape[-2]), int(X_work.shape[-1])) * np.finfo(np.float64).eps
    largest = _axis_max(singular_values, xp, -1)
    retained = singular_values > largest[..., None] * float(cutoff_scale)
    ranks = _axis_sum(retained, xp, -1)
    inverse_values = _inverse_values(singular_values, retained, xp)
    k = int(X_work.shape[-1])
    full_rank = ranks == k

    anchor, constant_value, has_constant = _first_constant_anchor_batched(
        X_work, y, full_rank, xp
    )
    y_work = y - anchor[:, None] if getattr(y, "ndim", None) == 2 else y

    if getattr(y_work, "ndim", None) == 2:
        params_work = _svd_project_batched_stable(U, Vh, inverse_values, y_work, xp)
        stable_rhs = _stable_rhs_batched(X_work, y_work, xp)
        stable_zero_solution = (
            full_rank
            & (design_scale == 1.0)
            & _axis_all(stable_rhs == 0.0, xp, 1)
        )
        params_work = xp.where(
            stable_zero_solution[:, None], xp.zeros_like(params_work), params_work
        )
        failure = _resolution_failure_batched(
            X_work,
            U,
            Vh,
            inverse_values,
            y_work,
            params_work,
            xp,
            ignore_first=has_constant,
        )
        ambiguous_zero = _ambiguous_zero_rhs_batched(
            X_work, y_work, xp, ignore_first=has_constant
        )
        stable_normal_failure = _stable_normal_equation_failure_batched(
            X_work, y_work, params_work, xp, ignore_first=has_constant
        )
        precision_failure = full_rank & (
            failure | ambiguous_zero | stable_normal_failure
        )
        ranks = xp.where(precision_failure, xp.zeros_like(ranks), ranks)
        restore = xp.zeros_like(params_work)
        restore[:, 0] = _safe_constant_restore(anchor, constant_value, has_constant, xp)
        params = (params_work + restore) * design_scale[:, None]
    else:
        rhs = y_work
        weighted_u_t = xp.swapaxes(U, -2, -1) * inverse_values[..., :, None]
        scaled = xp.matmul(weighted_u_t, rhs)
        params_work = xp.matmul(xp.swapaxes(Vh, -2, -1), scaled)
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
