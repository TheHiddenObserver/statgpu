"""Shared numerical linear algebra policy for panel fit spaces."""

from __future__ import annotations

import numpy as np

from statgpu.backends import _to_float_scalar


# Normal equations are used only as an explicitly certified fast path.  A Gram
# eigenvalue ratio of 1e-4 corresponds to kappa_2(X) < 100 in exact arithmetic,
# leaving a very large numerical margin from the SVD rank boundary used below.
_GRAM_CERTIFIED_MIN_EIGEN_RATIO = 1.0e-4


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
    # Keep the singular-value cutoff on the active backend.  Only the final
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


def _scaled_lstsq_rhs(y, xp, *, batched: bool):
    """Scale responses only enough to keep ``U.T @ y`` in float64 range.

    For an orthonormal SVD factor ``U``, every projected coordinate obeys
    ``|u.T @ y| <= sqrt(n_obs) * max(abs(y))``.  Use this bound to apply only
    the dimensionless down-scaling needed to keep the projection below half of
    the float64 maximum.  Unlike normalizing every response by its maximum, the
    scale factor is at most ``2 * sqrt(n_obs)`` for finite float64 input, so a
    small but representable component is not needlessly collapsed merely
    because another observation is very large.
    """
    namespace = getattr(xp, "__name__", "")
    ndim = int(getattr(y, "ndim", 0))
    n_obs = int(y.shape[1] if batched else y.shape[0])
    projection_limit = np.finfo(np.float64).max / (
        2.0 * np.sqrt(float(max(n_obs, 1)))
    )

    def _safe_scale(max_abs):
        required = max_abs / float(projection_limit)
        return xp.where(required > 1.0, required, xp.ones_like(required))

    if batched:
        if ndim == 2:
            max_abs = (
                xp.max(xp.abs(y), dim=1).values
                if namespace == "torch"
                else xp.max(xp.abs(y), axis=1)
            )
            safe_scale = _safe_scale(max_abs)
            return y / safe_scale[:, None], safe_scale[:, None]
        if ndim == 3:
            max_abs = (
                xp.max(xp.abs(y), dim=1).values
                if namespace == "torch"
                else xp.max(xp.abs(y), axis=1)
            )
            safe_scale = _safe_scale(max_abs)
            return y / safe_scale[:, None, :], safe_scale[:, None, :]
        raise ValueError("batched panel response must have shape (batch, n_obs[, n_targets])")

    if ndim == 1:
        max_abs = xp.max(xp.abs(y))
        safe_scale = _safe_scale(max_abs)
        return y / safe_scale, safe_scale
    if ndim == 2:
        max_abs = (
            xp.max(xp.abs(y), dim=0).values
            if namespace == "torch"
            else xp.max(xp.abs(y), axis=0)
        )
        safe_scale = _safe_scale(max_abs)
        return y / safe_scale[None, :], safe_scale[None, :]
    raise ValueError("panel response must be one- or two-dimensional")

def panel_svd_pseudoinverse(X, xp):
    """Return X+, (X'X)+, and rank from one explicit float64 SVD mask."""
    U, Vh, inverse_values, rank = _svd_inverse_factors(X, xp)
    X_pinv = (Vh.T * inverse_values) @ U.T
    bread = X_pinv @ X_pinv.T
    return X_pinv, bread, rank


def panel_lstsq(X, y, xp):
    """Return the minimum-norm least-squares solution under the panel SVD policy."""
    U, Vh, inverse_values, rank = _svd_inverse_factors(X, xp)
    y_scaled, response_scale = _scaled_lstsq_rhs(y, xp, batched=False)
    projected = U.T @ y_scaled
    if getattr(projected, "ndim", 1) == 1:
        scaled = inverse_values * projected
        params = Vh.T @ scaled
        return params * response_scale, rank
    scaled = inverse_values.reshape(-1, 1) * projected
    params = Vh.T @ scaled
    return params * response_scale, rank


def panel_lstsq_deferred_rank(X, y, xp):
    """Return least-squares parameters plus a backend-native rank scalar.

    This is the same two-dimensional SVD policy as :func:`panel_lstsq`, but it
    deliberately does not convert the final rank to Python.  GPU callers that
    must solve several independent matrices with a backend lacking a supported
    stacked-SVD API can therefore launch all solves first and transfer one packed
    rank vector after the loop rather than synchronizing once per matrix.
    """
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    U, singular_values, Vh = xp.linalg.svd(X, full_matrices=False)
    retained, rank_backend = _rank_mask_backend(X, singular_values, xp)
    inverse_values = _inverse_values(singular_values, retained, xp)
    y_scaled, response_scale = _scaled_lstsq_rhs(y, xp, batched=False)
    projected = U.T @ y_scaled
    if getattr(projected, "ndim", 1) == 1:
        scaled = inverse_values * projected
        params = Vh.T @ scaled
        return params * response_scale, rank_backend
    scaled = inverse_values.reshape(-1, 1) * projected
    params = Vh.T @ scaled
    return params * response_scale, rank_backend


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

    The fast path is deliberately *not* a replacement for the shared SVD rank
    policy.  It is only valid for matrices whose Gram spectrum is far from the
    numerical rank boundary.  Callers must use the returned ``certified`` mask
    and fall back to :func:`panel_lstsq_batched` or
    :func:`panel_lstsq_deferred_rank` for every uncertified matrix.

    Parameters
    ----------
    X : array-like, shape (batch, n_obs, n_features)
        Equal-shaped design matrices.
    y : array-like, shape (batch, n_obs) or (batch, n_obs, n_targets)
        Matching responses.
    xp : module
        NumPy, CuPy, or Torch numerical namespace.  The operations used here
        (batched Hermitian eigenvalues and batched square solve) are supported by
        the maintained GPU namespaces used by statgpu.
    min_eigen_ratio : float, default=1e-4
        Minimum ``lambda_min(X'X) / lambda_max(X'X)`` accepted for the fast path.
        The default restricts normal-equation solves to clearly well-conditioned
        designs (approximately ``kappa_2(X) < 100``).

    Returns
    -------
    params : array-like
        Candidate least-squares parameters.  Entries corresponding to
        uncertified matrices are placeholders and must not be consumed.
    certified : array-like, shape (batch,)
        Backend-native boolean mask identifying matrices safe for the Gram path.

    Notes
    -----
    Uncertified Gram matrices are replaced by identity matrices before the
    batched solve.  This prevents one singular/ill-conditioned period from
    aborting the whole batch while keeping the unsafe result explicitly masked
    for SVD fallback.
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

    transpose = xp.swapaxes(X, -2, -1)
    gram = xp.matmul(transpose, X)
    rhs_input = y[..., None] if getattr(y, "ndim", None) == 2 else y
    rhs = xp.matmul(transpose, rhs_input)
    rhs_finite_view = xp.isfinite(rhs).reshape(int(rhs.shape[0]), -1)
    rhs_finite = (
        xp.all(rhs_finite_view, dim=1)
        if namespace == "torch"
        else xp.all(rhs_finite_view, axis=1)
    )

    eigenvalues = xp.linalg.eigvalsh(gram)
    smallest = eigenvalues[..., 0]
    largest = eigenvalues[..., -1]
    certified = (
        xp.isfinite(smallest)
        & xp.isfinite(largest)
        & (largest > 0.0)
        & (smallest > largest * ratio)
        & rhs_finite
    )

    k = int(X.shape[-1])
    if namespace == "torch":
        identity = xp.eye(k, dtype=X.dtype, device=X.device)
    else:
        identity = xp.eye(k, dtype=X.dtype)
    safe_gram = xp.where(certified[..., None, None], gram, identity)
    safe_rhs = xp.where(certified[..., None, None], rhs, xp.zeros_like(rhs))

    if namespace == "torch" and hasattr(xp.linalg, "solve_ex"):
        params, info = xp.linalg.solve_ex(safe_gram, safe_rhs, check_errors=False)
        certified = certified & (info == 0)
    else:
        params = xp.linalg.solve(safe_gram, safe_rhs)

    params_finite_view = xp.isfinite(params).reshape(int(params.shape[0]), -1)
    params_finite = (
        xp.all(params_finite_view, dim=1)
        if namespace == "torch"
        else xp.all(params_finite_view, axis=1)
    )
    certified = certified & params_finite

    if getattr(y, "ndim", None) == 2:
        params = params[..., 0]
    return params, certified


def panel_lstsq_batched(X, y, xp):
    """Solve a batch of equal-shaped panel least-squares problems with one SVD call.

    Parameters
    ----------
    X : array-like, shape (batch, n_obs, n_features)
        Equal-shaped design matrices resident on a namespace with documented
        stacked-SVD support.
    y : array-like, shape (batch, n_obs) or (batch, n_obs, n_targets)
        Matching responses.
    xp : module
        NumPy or Torch numerical namespace.  CuPy's maintained public SVD API
        documents two-dimensional input, so CuPy callers use
        :func:`panel_lstsq_deferred_rank` instead of relying on undocumented
        stacked-SVD behavior.

    Returns
    -------
    params : array-like
        Batched minimum-norm solutions.
    ranks : array-like, shape (batch,)
        Backend-native numerical ranks.  Callers can transfer the complete rank
        vector once per batch instead of synchronizing one scalar per period.

    Notes
    -----
    The singular-value cutoff is exactly the same policy as :func:`panel_lstsq`,
    applied independently to each matrix in the batch.
    """
    namespace = getattr(xp, "__name__", "")
    if namespace not in {"numpy", "torch"}:
        raise NotImplementedError(
            "panel_lstsq_batched requires a namespace with documented stacked-SVD "
            "support; use panel_lstsq_deferred_rank for two-dimensional CuPy solves"
        )
    _validate_batched_lstsq_inputs(X, y)

    U, singular_values, Vh = xp.linalg.svd(X, full_matrices=False)
    cutoff_scale = (
        max(int(X.shape[-2]), int(X.shape[-1]))
        * np.finfo(np.float64).eps
    )
    if namespace == "torch":
        largest = xp.max(singular_values, dim=-1, keepdim=True).values
        ranks = xp.sum(singular_values > largest * float(cutoff_scale), dim=-1)
    else:
        largest = xp.max(singular_values, axis=-1, keepdims=True)
        ranks = xp.sum(
            singular_values > largest * float(cutoff_scale),
            axis=-1,
        )
    retained = singular_values > largest * float(cutoff_scale)
    inverse_values = _inverse_values(singular_values, retained, xp)

    y_scaled, response_scale = _scaled_lstsq_rhs(y, xp, batched=True)
    rhs = y_scaled[..., None] if getattr(y_scaled, "ndim", None) == 2 else y_scaled
    projected = xp.matmul(xp.swapaxes(U, -2, -1), rhs)
    scaled = inverse_values[..., None] * projected
    params = xp.matmul(xp.swapaxes(Vh, -2, -1), scaled)
    if getattr(y, "ndim", None) == 2:
        params = params[..., 0] * response_scale
    else:
        params = params * response_scale
    return params, ranks


def panel_matrix_rank(X, xp):
    """Return numerical rank using exactly the panel pseudoinverse cutoff."""
    if getattr(xp, "__name__", "") == "torch":
        singular_values = xp.linalg.svdvals(X)
    else:
        singular_values = xp.linalg.svd(X, compute_uv=False)
    _retained, rank = _rank_mask(X, singular_values, xp)
    return rank
