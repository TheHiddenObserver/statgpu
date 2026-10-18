"""
RidgeCV: Cross-validated Ridge regression with GPU support.
"""

from __future__ import annotations

__all__ = ["RidgeCV"]

from typing import Any, Dict, Optional, Tuple, Union
from collections import OrderedDict
import hashlib
import warnings
import numpy as np

from statgpu._config import Device
from statgpu.cross_validation._base import CVEstimatorBase
from statgpu.backends import get_backend, _torch_dev, xp_maximum, xp_asarray
from statgpu.backends._factory import _cupy_backend, _torch_backend
from statgpu.linear_model.wrappers._ridge import Ridge
from ._device import (
    cv_refit_device,
    resolve_cv_backend,
    validate_cv_sample_weight,
)


# =============================================================================
# CV Cache for Ridge
# =============================================================================

import threading

_RIDGE_CV_ALPHA_CACHE_MAXSIZE = int(64)
_RIDGE_CV_ALPHA_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
_RIDGE_CV_CACHE_LOCK = threading.Lock()


def _ridge_cv_cache_get(key):
    """Get cached Ridge CV results."""
    if key is None:
        return None
    with _RIDGE_CV_CACHE_LOCK:
        val = _RIDGE_CV_ALPHA_CACHE.get(key)
        if val is not None:
            _RIDGE_CV_ALPHA_CACHE.move_to_end(key)
        return val


def _ridge_cv_cache_put(key, value):
    """Put cached Ridge CV results."""
    if key is None:
        return
    with _RIDGE_CV_CACHE_LOCK:
        _RIDGE_CV_ALPHA_CACHE[key] = value
        _RIDGE_CV_ALPHA_CACHE.move_to_end(key)
        while len(_RIDGE_CV_ALPHA_CACHE) > _RIDGE_CV_ALPHA_CACHE_MAXSIZE:
            _RIDGE_CV_ALPHA_CACHE.popitem(last=False)


def _make_ridge_cv_auto_cache_key(X, y, alphas, folds, fit_intercept, use_gpu, sample_weight=None):
    """Generate automatic cache key for Ridge CV.

    Delegates data hashing to shared hash_cv_data() (10M threshold,
    row-index aware), then appends Ridge-specific parameters.
    """
    from statgpu.cross_validation._base import hash_cv_data
    # Shared data hash (10M threshold, row indices for large datasets)
    data_hash = hash_cv_data(X, y, sample_weight)
    # Ridge-specific parameters
    h = hashlib.blake2b(digest_size=32)
    h.update(data_hash)
    h.update(str(X.dtype).encode("utf-8"))
    h.update(np.asarray(alphas, dtype=np.float64).tobytes())
    h.update(str(fit_intercept).encode("utf-8"))
    h.update(str(use_gpu).encode("utf-8"))
    # Hash fold indices (all elements to avoid collisions)
    for train_idx, val_idx in folds:
        h.update(train_idx.tobytes())
        h.update(val_idx.tobytes())
    return h.hexdigest()


# =============================================================================
# K-fold helper
# =============================================================================

from statgpu.cross_validation._base import (
    kfold_indices as _kfold_indices,
    folds_are_complete as _folds_are_complete,
    batch_mse as _batch_mse_cv,
    _coerce_cv_indices,
    _as_backend_array,
    _to_float_scalar,
)


def _validated_ridge_folds(folds, n_samples):
    """Validate and normalize caller-supplied folds without changing their rows."""
    normalized = []
    try:
        fold_iterator = iter(folds)
    except TypeError as exc:
        raise ValueError("cv_splits must contain (train, validation) pairs") from exc
    for fold_idx, pair in enumerate(fold_iterator):
        try:
            train, valid = pair
        except (TypeError, ValueError) as exc:
            raise ValueError("cv_splits must contain (train, validation) pairs") from exc
        train = _coerce_cv_indices(train, fold_idx=fold_idx, name="train")
        valid = _coerce_cv_indices(valid, fold_idx=fold_idx, name="validation")
        for name, indices in (("train", train), ("validation", valid)):
            if indices.size == 0:
                raise ValueError(f"cv_splits fold {fold_idx} {name} indices must be nonempty")
            if np.any(indices < 0) or np.any(indices >= n_samples):
                raise ValueError(f"cv_splits fold {fold_idx} {name} indices are out of bounds")
            if np.unique(indices).size != indices.size:
                raise ValueError(f"cv_splits fold {fold_idx} {name} indices must not contain duplicates")
        if np.intersect1d(train, valid).size:
            raise ValueError(f"cv_splits fold {fold_idx} train and validation indices must be disjoint")
        # Torch rejects negative NumPy strides, even for C-contiguous singletons.
        # Copy only after validation, preserving the caller's row order and arrays.
        if any(stride < 0 for stride in train.strides):
            train = train.copy(order="C")
        if any(stride < 0 for stride in valid.strides):
            valid = valid.copy(order="C")
        normalized.append((train, valid))
    if not normalized:
        raise ValueError("cv_splits must contain at least one fold")
    return normalized


def _ridge_folds_have_complementary_training(folds, n_samples):
    """Whether validated folds permit full-minus-validation statistics."""
    return _folds_are_complete(folds, n_samples=n_samples) and all(
        train.size + valid.size == n_samples for train, valid in folds
    )


# =============================================================================
# Alpha grid generation
# =============================================================================

def _default_ridge_alpha_grid(
    X, y, n_alphas: int = 100, alpha_min_ratio: float = 1e-3,
    sample_weight=None,
):
    """Generate an alpha grid on the package's average-loss scale."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if sample_weight is None:
        normalization = float(X_arr.shape[0])
        X_mean = np.mean(X_arr, axis=0)
        y_mean = float(np.mean(y_arr))
        Xty = (X_arr - X_mean).T @ (y_arr - y_mean)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        normalization = float(np.sum(sw))
        X_mean = np.sum(X_arr * sw[:, None], axis=0) / normalization
        y_mean = float(np.sum(y_arr * sw) / normalization)
        Xty = ((X_arr - X_mean) * sw[:, None]).T @ (y_arr - y_mean)
    alpha_max = float(np.max(np.abs(Xty)) * 2.0 / normalization)
    if alpha_max == 0.0:
        alpha_max = 1.0
    if n_alphas <= 1:
        return np.array([alpha_max])
    return np.logspace(
        np.log10(alpha_max * alpha_min_ratio), np.log10(alpha_max),
        num=n_alphas, dtype=np.float64,
    )


def _default_ridge_alpha_grid_backend(
    X, y, backend, n_alphas: int = 100, alpha_min_ratio: float = 1e-3,
    sample_weight=None,
):
    """Backend-native alpha grid with the same weighted normalization."""
    X_arr = backend.asarray(X)
    y_arr = backend.asarray(y).reshape(-1)
    if sample_weight is None:
        normalization = float(X_arr.shape[0])
        X_mean = backend.mean(X_arr, axis=0)
        y_mean = backend.mean(y_arr)
        Xty = (X_arr - X_mean).T @ (y_arr - y_mean)
    else:
        sw = backend.asarray(sample_weight).reshape(-1)
        normalization = float(backend.sum(sw))
        X_mean = backend.sum(X_arr * sw[:, None], axis=0) / normalization
        y_mean = backend.sum(y_arr * sw) / normalization
        Xty = ((X_arr - X_mean) * sw[:, None]).T @ (y_arr - y_mean)
    alpha_max = float(backend.max(backend.abs(Xty)) * 2.0 / normalization)
    if alpha_max == 0.0:
        alpha_max = 1.0
    if n_alphas <= 1:
        return np.array([alpha_max])
    return np.logspace(
        np.log10(alpha_max * alpha_min_ratio), np.log10(alpha_max),
        num=n_alphas, dtype=np.float64,
    )


# =============================================================================
# Batch MSE computation
# =============================================================================


def _ridge_normalized_root_weights(weights, xp):
    """Normalize square roots without first forming underflow-prone ratios."""
    return xp.sqrt(weights) / xp.sqrt(xp.sum(weights))


def _weighted_ridge_gram(X, y, weights, xp, backend=None, dtype=None):
    """Apply square-root weights before narrowing or forming cross-products."""
    root_weight = _ridge_normalized_root_weights(weights, xp)
    X_weighted = X * root_weight[:, None]
    y_weighted = y * root_weight
    if backend is not None:
        X_weighted = backend.asarray(X_weighted, dtype=dtype)
        y_weighted = backend.asarray(y_weighted, dtype=dtype)
    return X_weighted.T @ X_weighted, X_weighted.T @ y_weighted


def _weighted_ridge_centered_data(X, y, weights, xp):
    """Form centered weighted moments without subtracting raw cross-products."""
    # Center relative to an observed row before averaging. For an exactly
    # constant column, this gives exactly zero centered values even when the
    # constant and the normalized weights are not exactly representable.
    anchor = xp.argmax(weights)
    X_anchor = X[anchor]
    y_anchor = y[anchor]
    active = weights > 0
    X_shifted = xp.where(active[:, None], X, X_anchor) - X_anchor
    y_shifted = xp.where(active, y, y_anchor) - y_anchor
    root_weight = _ridge_normalized_root_weights(weights, xp)
    mass = xp.sum(root_weight ** 2)
    X_mean_shift = xp.sum(
        (X_shifted * root_weight[:, None]) * root_weight[:, None], axis=0,
    ) / mass
    y_mean_shift = xp.sum((y_shifted * root_weight) * root_weight) / mass
    X_centered = X_shifted - X_mean_shift
    y_centered = y_shifted - y_mean_shift
    return X_centered, y_centered, X_anchor + X_mean_shift, y_anchor + y_mean_shift


def _weighted_ridge_centered_gram(X, y, weights, xp, backend=None, dtype=None):
    """Form anchored, square-root-weighted centered cross-products."""
    X_centered, y_centered, X_mean, y_mean = _weighted_ridge_centered_data(X, y, weights, xp)
    gram, cross = _weighted_ridge_gram(
        X_centered, y_centered, weights, xp, backend=backend, dtype=dtype,
    )
    return gram, cross, X_mean, y_mean

def _weighted_ridge_mse(X, y, coefs, intercepts, weights):
    """Subtract finite residuals before square-root weighting and squaring."""
    root_weight = _ridge_normalized_root_weights(weights, np)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = X @ coefs.T + intercepts - y[:, None]
        residuals = np.where(weights[:, None] > 0, raw, 0.) * root_weight[:, None]
    invalid = ~np.isfinite(residuals)
    if np.any(invalid):
        # A raw prediction can overflow although its weighted contribution is
        # finite. Use scaled terms only for those exceptional residuals.
        scaled = (
            (X * root_weight[:, None]) @ coefs.T
            + root_weight[:, None] * intercepts
            - (y * root_weight)[:, None]
        )
        residuals = np.where(invalid, scaled, residuals)
    with np.errstate(over="ignore", invalid="ignore"):
        mse = np.sum(residuals ** 2, axis=0)
    if not np.all(np.isfinite(mse)):
        raise FloatingPointError(
            "RidgeCV weighted validation MSE is not finite at float64 precision; "
            "rescale the response/features or choose different alphas."
        )
    return mse


# =============================================================================
# GPU batch solver for Ridge
# =============================================================================

def _compress_exact_ridge_factors(X, backend):
    """Reduce exact factor duplicates by orthogonal row/column maps.

    Equality is checked on the original factors, before any scaling. Repeated
    rows are replaced by sqrt(m) times their representative; the response is
    kept intact and projected through the expanded left singular vectors.
    Repeated columns are replaced by sqrt(k) times their representative, with
    coefficients expanded as theta/sqrt(k). This preserves predictions and
    the minimum L2 penalty, while row reduction removes only a constant loss.
    """
    xp = backend.xp

    def unique_rows(values):
        if backend.name == "torch":
            return xp.unique(values, dim=0, return_inverse=True, return_counts=True)
        return xp.unique(values, axis=0, return_inverse=True, return_counts=True)

    rows, row_inverse, row_counts = unique_rows(X)
    columns, column_inverse, column_counts = unique_rows(X.T)
    # NumPy 2.0.0 retained the selected axis in inverse-map shapes.
    row_inverse = row_inverse.reshape(-1)
    column_inverse = column_inverse.reshape(-1)
    fewer_rows = rows.shape[0] < X.shape[0]
    fewer_columns = columns.shape[0] < X.shape[1]
    if not fewer_rows and not fewer_columns:
        return None
    reduced = rows if fewer_rows else X
    row_roots = column_roots = None
    if fewer_columns:
        # Every member is exactly equal, so any group representative suffices.
        starts = xp.cumsum(column_counts, axis=0) - column_counts
        representatives = backend.argsort(column_inverse)[starts]
        column_roots = xp.sqrt(backend.asarray(column_counts, dtype=backend.float64))
        reduced = reduced[:, representatives] * column_roots[None, :]
    if fewer_rows:
        row_roots = xp.sqrt(backend.asarray(row_counts, dtype=backend.float64))
        reduced = reduced * row_roots[:, None]
    return (reduced, row_inverse if fewer_rows else None, row_roots,
            column_inverse if fewer_columns else None, column_roots)


def _solve_ridge_path_from_factors(X_factor, y_factor, regularization, backend):
    """Recover an unresolved Gram path with native float64 factor SVD.

    Singular directions indistinguishable from factorization/dot-product
    roundoff are numerical null directions. A direction-specific bound avoids
    discarding independently resolved tiny columns merely because another
    column has a larger scale. Exact duplicate factors can be reduced before
    rejecting a response-bearing ambiguous direction.
    """
    xp = backend.xp
    X_factor = backend.asarray(X_factor, dtype=backend.float64)
    y_factor = backend.asarray(y_factor, dtype=backend.float64)
    row_inverse = row_roots = column_inverse = column_roots = None
    for attempt in range(2):
        if attempt and not bool(_to_float_scalar(xp.all(xp.isfinite(X_factor)))):
            raise FloatingPointError(
                "RidgeCV exact-factor reduction is not finite at float64 precision; "
                "rescale the features."
            )
        U, singular, Vh = xp.linalg.svd(X_factor, full_matrices=False)
        V = Vh.T
        represented = X_factor @ V
        reconstructed = U * singular[None, :]
        eps = float(xp.finfo(X_factor.dtype).eps)
        operations = int(X_factor.shape[1]) + 2
        gamma = operations * eps / (1.0 - operations * eps)
        dot_bound = gamma * xp.linalg.norm(xp.abs(X_factor) @ xp.abs(V), axis=0)
        residual_bound = xp.linalg.norm(represented - reconstructed, axis=0)
        product_bound = eps * xp.linalg.norm(reconstructed, axis=0)
        resolved = singular > (dot_bound + residual_bound + product_bound)
        # Expand the orthogonal row map rather than summing grouped responses.
        # This retains the original response and its dot-product error bound.
        response_basis = U if row_inverse is None else U[row_inverse]
        response_U = response_basis if row_inverse is None else response_basis / row_roots[row_inverse, None]
        projection = response_U.T @ y_factor
        projection_operations = int(y_factor.shape[0]) + 2
        projection_gamma = projection_operations * eps / (1.0 - projection_operations * eps)
        projection_bound = projection_gamma * (xp.abs(response_U).T @ xp.abs(y_factor))
        if attempt:
            # Relative dot-product bounds cannot certify underflowed terms.
            # Inspect operand support: even the absolute dot may round to zero.
            support = xp.any((response_basis != 0) & (y_factor[:, None] != 0), axis=0)
            lost_support = xp.any(
                (response_basis != 0) & (response_U == 0) & (y_factor[:, None] != 0), axis=0,
            )
            invalid_projection = (
                (~xp.isfinite(projection)) | (~xp.isfinite(projection_bound))
                | lost_support
                | (support & (projection_bound < xp.finfo(X_factor.dtype).tiny))
                | ((projection_bound > 0) & (xp.abs(projection) <= projection_bound))
            )
            if bool(_to_float_scalar(xp.any(resolved & invalid_projection))):
                raise FloatingPointError(
                    "RidgeCV factor-SVD response projection is not resolved at "
                    "float64 precision after combining exact duplicate factors; "
                    "response cancellation or underflow prevents a reliable ridge path."
                )
        exact_zero = (singular == 0) & xp.all(represented == 0, axis=0)
        ambiguous_signal = (~resolved) & (~exact_zero) & (xp.abs(projection) > projection_bound)
        if attempt:
            # A duplicate retry must resolve every remaining direction. Even
            # an SVD-reported exact zero can hide a distinct original direction
            # rounded away by sqrt(count) scaling. Do not widen truncation.
            ambiguous_signal = ambiguous_signal | (~resolved)
        if not bool(_to_float_scalar(xp.any(ambiguous_signal))):
            break
        if attempt == 0:
            compressed = _compress_exact_ridge_factors(X_factor, backend)
            if compressed is not None:
                X_factor, row_inverse, row_roots, column_inverse, column_roots = compressed
                continue
        raise FloatingPointError(
            "RidgeCV factor-SVD precision cannot resolve a response-bearing "
            "singular direction for the requested alphas; use larger alphas "
            "or rescale/reparameterize the features. A finite ridge solution "
            "may still exist."
        )
    singular = xp.where(resolved, singular, xp.zeros_like(singular))
    penalty = backend.asarray(regularization, dtype=backend.float64).reshape(1, -1)
    # Normalize before squaring: s**2 can overflow for a finite Gram spectrum,
    # or underflow next to a representable subnormal penalty.
    root_penalty = xp.sqrt(penalty)
    scale = xp.maximum(singular[:, None], root_penalty)
    scaled_singular = singular[:, None] / scale
    scaled_penalty = root_penalty / scale
    denominator = scaled_singular ** 2 + scaled_penalty ** 2
    if not bool(_to_float_scalar(xp.all(xp.isfinite(denominator) & (denominator > 0)))):
        raise FloatingPointError(
            "RidgeCV factor-SVD denominators are not finite and positive at "
            "float64 precision; rescale the data or choose different alphas."
        )
    # Use the original s in the penalty-dominant numerator: s/scale can lose
    # subnormal bits before a later division by a small scale restores them.
    singular_dominant = singular[:, None] >= root_penalty
    safe_penalty = xp.where(singular_dominant, xp.ones_like(penalty), penalty)
    numerator = xp.where(singular_dominant, 1. / scale, singular[:, None] / safe_penalty)
    factors = numerator / denominator
    response_bearing = (singular[:, None] > 0) & (projection[:, None] != 0)
    invalid_gain = (~xp.isfinite(factors)) | (response_bearing & (factors == 0))
    if attempt:
        invalid_gain = invalid_gain | (response_bearing & (factors < xp.finfo(X_factor.dtype).tiny))
    if bool(_to_float_scalar(xp.any(invalid_gain))):
        raise FloatingPointError(
            "RidgeCV factor-SVD gains are not resolved at float64 precision; "
            "rescale the data or choose different alphas."
        )
    signals = factors * projection[:, None]
    if bool(_to_float_scalar(xp.any(
        (~xp.isfinite(signals)) | (response_bearing & (signals == 0))
    ))):
        raise FloatingPointError(
            "RidgeCV factor-SVD coefficient signals are not representable at "
            "float64 precision; rescale the data or choose different alphas."
        )
    coefficients = V @ signals
    if column_inverse is not None:
        expanded = coefficients[column_inverse]
        coefficients = expanded / column_roots[column_inverse, None]
        if bool(_to_float_scalar(xp.any((expanded != 0) & (coefficients == 0)))):
            raise FloatingPointError(
                "RidgeCV exact-factor coefficient expansion underflowed at "
                "float64 precision; rescale the data or choose different alphas."
            )
    if not bool(_to_float_scalar(xp.all(xp.isfinite(coefficients)))):
        raise FloatingPointError(
            "RidgeCV factor-SVD coefficients are not finite; rescale the data "
            "or choose larger alphas."
        )
    return coefficients


def _solve_ridge_path_gpu_from_gram_eig(XtX_batch, Xty_batch, alphas, backend, fit_intercept=True, n_samples_vec=None, factor_factory=None):
    """
    Solve Ridge path for multiple folds using eigendecomposition (vectorized over alphas).

    This function uses eigendecomposition to solve the Ridge regression problem for all
    alphas simultaneously, avoiding repeated Cholesky decompositions.

    Mathematical formulation:
    - Given XtX = Q @ Lambda @ Q.T (eigendecomposition)
    - Ridge solution: coef(alpha) = (XtX + alpha*I)^-1 @ Xty
    - Using eigenbasis: coef(alpha) = Q @ diag(1/(lambda_i + alpha)) @ Q.T @ Xty

    Parameters
    ----------
    XtX_batch : array-like
        Batch of Gram matrices (n_folds, n_features, n_features).
    Xty_batch : array-like
        Batch of cross products (n_folds, n_features).
    alphas : ndarray
        Alpha values (n_alphas,).
    backend : BackendBase
        Backend instance (CuPyBackend or TorchBackend).
    fit_intercept : bool
        Whether intercept is fitted (not used here, kept for API compatibility).

    Returns
    -------
    coefs_desc : ndarray
        Coefficients for each alpha and fold (n_alphas, n_folds, n_features).
    """
    xp = backend.xp

    n_folds = XtX_batch.shape[0]
    n_features = XtX_batch.shape[1]
    n_alphas = alphas.shape[0]

    # Step 1: Eigendecomposition (done once per fold)
    # eigvals: (n_folds, n_features), Q: (n_folds, n_features, n_features)
    eigvals, Q = xp.linalg.eigh(XtX_batch)
    eps = float(xp.finfo(eigvals.dtype).eps)
    gamma = n_features * eps / (1.0 - n_features * eps)
    uncertainty = gamma * backend.max(xp.abs(eigvals), axis=1)
    unresolved_gram = backend.min(eigvals, axis=1) <= uncertainty
    # A Gram matrix is positive semidefinite; clip negative roundoff only.
    # Strictly positive alpha regularizes null directions. A positive absolute
    # floor would change the objective when weights or features are rescaled.
    eigvals = xp_maximum(eigvals, 0.0, xp)

    # Step 2: Project Xty into eigenbasis
    # QTXty = Q.T @ Xty_batch  -> (n_folds, n_features)
    Q_T = backend.transpose(Q, (0, 2, 1))
    QTXty = xp.matmul(Q_T, Xty_batch[:, :, None])[:, :, 0]

    # Step 3: Form the regularized diagonal at the actual solve precision.
    # Scale alpha by n_samples to match Ridge.fit() convention.
    # The effective penalty is n_eff * alpha. Form that product before
    # narrowing: an alpha below float32's range can have a representable
    # positive product for an ordinary training-fold size.
    alphas_arr = backend.asarray(alphas, dtype=backend.float64)
    if n_samples_vec is not None:
        n_arr = backend.asarray(n_samples_vec, dtype=backend.float64).reshape(-1, 1, 1)
        regularization = alphas_arr[None, None, :] * n_arr
    else:
        regularization = alphas_arr[None, None, :]
    regularization64 = regularization
    regularization = backend.asarray(regularization, dtype=eigvals.dtype)
    if not bool(_to_float_scalar(xp.all(
        xp.isfinite(regularization) & (regularization > 0)
    ))):
        raise FloatingPointError(
            "RidgeCV positive regularization is not representable at the CV "
            "precision; use gpu_cv_mixed_precision=False or choose "
            "representable alphas."
        )
    denominator = eigvals[:, :, None] + regularization
    if not bool(_to_float_scalar(xp.all(
        xp.isfinite(denominator) & (denominator > 0)
    ))):
        raise FloatingPointError(
            "RidgeCV normal-equation denominators must be finite and positive "
            "at the CV precision; use gpu_cv_mixed_precision=False or rescale "
            "the data."
        )

    smallest_penalty = backend.min(regularization, axis=2).reshape(-1)
    recover = unresolved_gram & (uncertainty > 0) & (smallest_penalty <= uncertainty)
    # Subnormal Gram entries lose relative precision when factors are squared.
    # Their rounding bound may itself round to zero, so the usual relative
    # test cannot detect the loss. Original factors still resolve a ridge path
    # when both the spectrum and its requested regularization are this small.
    tiny = float(xp.finfo(eigvals.dtype).tiny)
    subnormal_gram = (
        (backend.min(eigvals, axis=1) < tiny) & (smallest_penalty < tiny)
    )
    if factor_factory is None:
        # A caller supplying only an exactly zero Gram and cross-product has
        # the valid zero solution. Public CV supplies original factors, which
        # can distinguish that case from a product that underflowed to zero.
        zero_system = (
            (backend.max(xp.abs(XtX_batch).reshape(n_folds, -1), axis=1) == 0)
            & (backend.max(xp.abs(Xty_batch), axis=1) == 0)
        )
        subnormal_gram = subnormal_gram & (~zero_system)
    recover = recover | subnormal_gram
    recovery_folds = []
    if bool(_to_float_scalar(xp.any(recover))):
        recovery_folds = [
            fold for fold in range(n_folds) if bool(_to_float_scalar(recover[fold]))
        ]
    if recovery_folds and factor_factory is None:
        raise FloatingPointError(
            "RidgeCV Gram precision cannot resolve the requested regularization; "
            "original training factors are required for factor-SVD recovery."
        )
    if recovery_folds:
        # Avoid magnifying numerical-null projection noise before recovery.
        QTXty = xp.where(recover[:, None], xp.zeros_like(QTXty), QTXty)

    # Step 4: Scale projected Xty by inverse diagonal
    # Divide directly: forming a reciprocal can overflow for a representable
    # tiny positive denominator, producing 0*inf in exact null directions.
    scaled = QTXty[:, :, None] / denominator

    # Step 5: Transform back to original basis
    # coefs: (n_folds, n_features, n_alphas)
    coefs = xp.matmul(Q, scaled)
    if recovery_folds:
        coefs = backend.asarray(coefs, dtype=backend.float64)
        for fold in recovery_folds:
            factors_X, factors_y = factor_factory(fold)
            penalty = regularization64[fold if regularization64.shape[0] > 1 else 0]
            coefs[fold] = _solve_ridge_path_from_factors(
                factors_X, factors_y, penalty, backend,
            )
        warnings.warn(
            "RidgeCV used same-device float64 factor-SVD recovery for "
            f"{len(recovery_folds)} fold(s); unresolved singular directions "
            "with only roundoff-sized response projections were treated as numerical zeros.",
            RuntimeWarning, stacklevel=3,
        )

    # Step 6: Reshape to (n_alphas, n_folds, n_features)
    # Current shape: (n_folds, n_features, n_alphas)
    # Need to transpose to: (n_alphas, n_folds, n_features)
    coefs = backend.transpose(coefs, (2, 0, 1))

    # Keep on GPU for further processing (avoid unnecessary H2D transfer)
    return coefs


def _solve_ridge_path_gpu_from_gram(XtX_batch, Xty_batch, n_samples_vec, alphas, backend, fit_intercept=True, factor_factory=None):
    """
    Solve Ridge path for multiple folds using eigendecomposition (optimized).

    This function uses eigendecomposition to solve the Ridge regression problem for all
    alphas simultaneously, avoiding repeated Cholesky decompositions.

    Parameters
    ----------
    XtX_batch : array-like
        Batch of Gram matrices (n_folds, n_features, n_features).
    Xty_batch : array-like
        Batch of cross products (n_folds, n_features).
    n_samples_vec : np.ndarray
        Number of samples for each fold (not used, kept for API compatibility).
    alphas : ndarray
        Alpha values (n_alphas,).
    backend : BackendBase
        Backend instance (CuPyBackend or TorchBackend).
    fit_intercept : bool
        Whether intercept is fitted.

    Returns
    -------
    coefs_desc : ndarray
        Coefficients for each alpha and fold (n_alphas, n_folds, n_features).
    """
    # Use eigendecomposition-based solver (vectorized over alphas)
    return _solve_ridge_path_gpu_from_gram_eig(
        XtX_batch, Xty_batch, alphas, backend, fit_intercept,
        n_samples_vec=n_samples_vec, factor_factory=factor_factory,
    )


# =============================================================================
# Main CV selection function
# =============================================================================

def _select_ridge_alpha_cv(
    X,
    y,
    *,
    alphas=None,
    n_alphas: int = 100,
    alpha_min_ratio: float = 1e-3,
    cv_folds: int = 5,
    cv_splits=None,
    random_state: Optional[int] = None,
    sample_weight=None,
    fit_intercept: bool = True,
    device: Union[str, Device] = Device.CPU,
    return_details: bool = False,
    cache_key: Optional[Tuple[Any, ...]] = None,
    gpu_cv_mixed_precision: bool = True,
):
    """
    Select alpha for Ridge regression via K-fold cross-validation.

    Parameters
    ----------
    X : array-like
        Design matrix (n_samples, n_features).
    y : array-like
        Response vector.
    alphas : array-like or None
        Alpha values to try. If None, generates n_alphas values.
    n_alphas : int
        Number of alpha values (if alphas is None).
    alpha_min_ratio : float
        Minimum alpha ratio.
    cv_folds : int
        Number of CV folds.
    cv_splits : list or None
        Pre-computed CV splits. If None, uses K-fold.
    random_state : int or None
        Random seed for CV splits.
    sample_weight : array-like or None
        Sample weights.
    fit_intercept : bool
        Whether to fit intercept.
    device : str or Device
        Device to use ('cpu' or 'cuda').
    return_details : bool
        Whether to return full CV details.
    cache_key : tuple or None
        Cache key for CV results.
    gpu_cv_mixed_precision : bool
        Whether to use mixed precision on GPU.

    Returns
    -------
    alpha : float
        Best alpha value.
    details : dict (if return_details=True)
        Full CV results including alpha grid, MSE path, etc.
    """
    (
        device_name,
        backend_name,
        backend,
        use_gpu,
        gpu_input_cupy,
        gpu_input_torch,
    ) = resolve_cv_backend(device, X)
    gpu_requested = use_gpu

    X_np = None
    y_np = None
    sample_weight_np = None

    if gpu_input_cupy or gpu_input_torch:
        # backend was selected strictly by resolve_cv_backend above
        if len(tuple(X.shape)) != 2:
            raise ValueError("X must be a 2D array")
        n_samples = int(X.shape[0])
        y_check = backend.asarray(y).reshape(-1)
        if int(y_check.shape[0]) != n_samples:
            raise ValueError("y must have the same number of rows as X")
        if sample_weight is not None:
            sw_check = backend.asarray(sample_weight).reshape(-1)
            if int(sw_check.shape[0]) != n_samples:
                raise ValueError("sample_weight must have the same number of rows as X")
    else:
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64).reshape(-1)
        if sample_weight is not None:
            sample_weight_np = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if X_np.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y_np.shape[0] != X_np.shape[0]:
            raise ValueError("y must have the same number of rows as X")
        if sample_weight_np is not None and sample_weight_np.shape[0] != X_np.shape[0]:
            raise ValueError("sample_weight must have the same number of rows as X")
        n_samples = int(X_np.shape[0])

    validated_weight = validate_cv_sample_weight(sample_weight, n_samples)
    if validated_weight is not None and not use_gpu:
        sample_weight_np = np.asarray(validated_weight, dtype=np.float64).reshape(-1)

    # Generate alpha grid
    if alphas is None:
        if gpu_input_cupy or gpu_input_torch or use_gpu:
            # backend was selected strictly by resolve_cv_backend above
            alpha_grid = _default_ridge_alpha_grid_backend(
                X, y, backend, n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio, sample_weight=sample_weight,
            )
        else:
            alpha_grid = _default_ridge_alpha_grid(
                X_np, y_np, n_alphas=n_alphas,
                alpha_min_ratio=alpha_min_ratio, sample_weight=sample_weight_np,
            )
    else:
        alpha_grid = np.asarray(alphas, dtype=np.float64)
        alpha_grid = alpha_grid[np.isfinite(alpha_grid)]
        alpha_grid = alpha_grid[alpha_grid > 0.0]
        if alpha_grid.size == 0:
            warnings.warn("All provided alphas were filtered; using default grid.", RuntimeWarning)
            if gpu_input_cupy or gpu_input_torch or use_gpu:
                backend = get_backend(
                    backend='torch' if gpu_input_torch else 'cupy', device='cuda'
                )
                alpha_grid = _default_ridge_alpha_grid_backend(
                    X, y, backend, n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio, sample_weight=sample_weight,
                )
            else:
                alpha_grid = _default_ridge_alpha_grid(
                    X_np, y_np, n_alphas=n_alphas,
                    alpha_min_ratio=alpha_min_ratio, sample_weight=sample_weight_np,
                )

    if cv_splits is not None:
        cv_splits = _validated_ridge_folds(cv_splits, int(n_samples))

    # Legacy generated-fold shortcuts must not discard explicit valid folds.
    if cv_splits is None and (
        int(n_samples) < 4 or int(alpha_grid.size) == 1 or int(cv_folds) < 2
    ):
        alpha0 = float(alpha_grid[0])
        if not return_details:
            return alpha0
        return {
            "alpha": alpha0,
            "alphas": alpha_grid.astype(np.float64, copy=False),
            "mse_path": np.full((int(alpha_grid.size), 1), np.nan, dtype=np.float64),
            "mean_mse": np.full(int(alpha_grid.size), np.nan, dtype=np.float64),
        }

    # Generate CV folds
    if cv_splits is not None:
        folds = cv_splits
    else:
        folds = _kfold_indices(n_samples=int(n_samples), n_splits=int(cv_folds), random_state=random_state)

    if validated_weight is not None:
        _, weight_xp, fold_weights = _as_backend_array(validated_weight)
        for fold_idx, (train, valid) in enumerate(folds):
            for name, indices in (("training", train), ("validation", valid)):
                native_indices = xp_asarray(
                    indices, dtype=weight_xp.int64, xp=weight_xp,
                    ref_arr=fold_weights,
                )
                mass = _to_float_scalar(weight_xp.sum(fold_weights[native_indices]))
                if mass <= 0.0:
                    raise ValueError(
                        f"cv_splits fold {fold_idx} {name} sample_weight "
                        "must have a positive sum"
                    )

    complementary_training = _ridge_folds_have_complementary_training(
        folds, n_samples=int(n_samples)
    )

    alpha_grid = alpha_grid.astype(np.float64, copy=False)
    n_alpha = int(alpha_grid.size)
    n_folds = int(len(folds))

    # Cache handling
    # Auto-cache disabled by default to prevent stale results across datasets.
    # Only use explicit cache_key if provided by the caller.
    cache_key_eff = cache_key

    cached_details = _ridge_cv_cache_get(cache_key_eff)
    if cached_details is not None:
        if return_details:
            return cached_details
        return float(cached_details["alpha"])

    # Initialize MSE path
    mse_path = np.full((n_alpha, n_folds), np.nan, dtype=np.float64)

    # GPU path
    if use_gpu:
        try:
            # Get backend based on input data type to avoid cross-backend conversion
            # Torch input -> TorchBackend, CuPy input -> CuPyBackend
            import torch
            try:
                import cupy as cp
                cupy_available = True
            except ImportError:
                cupy_available = False

            # Detect input type and select appropriate backend
            # backend was selected strictly by resolve_cv_backend above

            xp = backend.xp

            cv_dtype = backend.float32 if bool(gpu_cv_mixed_precision) else backend.float64

            # Weighted centering and scoring retain float64; square-root
            # weighted training factors are narrowed before Gram products.
            input_dtype = backend.float64 if sample_weight is not None else cv_dtype
            # Convert inputs to backend arrays
            if gpu_input_cupy or gpu_input_torch:
                # Already on GPU (CuPy or Torch)
                X_full = backend.asarray(X, dtype=input_dtype)
                y_full = backend.asarray(y, dtype=input_dtype).reshape(-1)
                if sample_weight is not None:
                    sw_full = backend.asarray(sample_weight, dtype=backend.float64).reshape(-1)
                else:
                    sw_full = None
            else:
                # Convert from numpy
                X_full = backend.asarray(X_np, dtype=input_dtype)
                y_full = backend.asarray(y_np, dtype=input_dtype)
                if sample_weight_np is not None:
                    sw_full = backend.asarray(sample_weight_np, dtype=backend.float64)
                else:
                    sw_full = None

            # Precompute for fast fold statistics
            XtX_folds = []
            Xty_folds = []
            n_train_folds = []
            X_mean_folds = []
            y_mean_folds = []

            # For batched MSE evaluation (Phase 2 optimization)
            X_val_folds = []
            y_val_folds = []
            sw_val_folds = []
            n_val_folds = []

            fast_fold_stats = (sw_full is None) and bool(complementary_training)
            sw_train = None  # initialized per-fold in slow path; None for fast path
            if fast_fold_stats:
                n_total = int(X_full.shape[0])
                XtX_full = X_full.T @ X_full
                Xty_full = X_full.T @ y_full
                if bool(fit_intercept):
                    X_sum_full = backend.sum(X_full, axis=0)
                    y_sum_full = backend.sum(y_full)
                else:
                    X_sum_full = None
                    y_sum_full = None

            for fold_idx, (train_idx, val_idx) in enumerate(folds):
                train_idx_gpu = backend.asarray(train_idx)
                val_idx_gpu = backend.asarray(val_idx)

                X_val = X_full[val_idx_gpu]
                y_val = y_full[val_idx_gpu]
                sw_val = None if sw_full is None else sw_full[val_idx_gpu]

                # Store validation data for batched MSE
                X_val_folds.append(X_val)
                y_val_folds.append(y_val)
                sw_val_folds.append(sw_val)
                n_val_folds.append(int(val_idx_gpu.shape[0]))

                if fast_fold_stats:
                    n_val = int(val_idx_gpu.shape[0])
                    n_train = int(n_total - n_val)

                    XtX_val = X_val.T @ X_val
                    Xty_val = X_val.T @ y_val
                    XtX_raw = XtX_full - XtX_val
                    Xty_raw = Xty_full - Xty_val

                    if bool(fit_intercept):
                        X_sum_val = backend.sum(X_val, axis=0)
                        y_sum_val = backend.sum(y_val)
                        X_sum_train = X_sum_full - X_sum_val
                        y_sum_train = y_sum_full - y_sum_val

                        inv_n = backend.asarray(1.0 / float(max(1, n_train)), dtype=X_full.dtype)
                        X_mean = X_sum_train * inv_n
                        y_mean = y_sum_train * inv_n
                        XtX = XtX_raw - backend.outer(X_sum_train, X_sum_train) * inv_n
                        Xty = Xty_raw - X_sum_train * y_mean
                    else:
                        X_mean = backend.zeros((X_full.shape[1],), dtype=X_full.dtype)
                        y_mean = backend.array(0.0, dtype=X_full.dtype)
                        XtX = XtX_raw
                        Xty = Xty_raw
                else:
                    X_train = X_full[train_idx_gpu]
                    y_train = y_full[train_idx_gpu]
                    sw_train = None if sw_full is None else sw_full[train_idx_gpu]

                    if sw_train is not None:
                        # Normalize square roots when forming factors, before
                        # mixed-precision casting or cross-products.
                        if bool(fit_intercept):
                            XtX, Xty, X_mean, y_mean = _weighted_ridge_centered_gram(
                                X_train, y_train, sw_train, xp,
                                backend=backend, dtype=cv_dtype,
                            )
                        else:
                            XtX, Xty = _weighted_ridge_gram(
                                X_train, y_train, sw_train, xp,
                                backend=backend, dtype=cv_dtype,
                            )
                            X_mean = backend.zeros((X_train.shape[1],), dtype=X_train.dtype)
                            y_mean = backend.array(0.0, dtype=X_train.dtype)
                        n_train = 1.0  # Factors use the average weighted objective.
                    else:
                        if bool(fit_intercept):
                            X_mean = backend.mean(X_train, axis=0)
                            y_mean = backend.mean(y_train)
                            X_centered = X_train - X_mean
                            y_centered = y_train - y_mean
                        else:
                            X_mean = backend.zeros((X_train.shape[1],), dtype=X_train.dtype)
                            y_mean = backend.array(0.0, dtype=X_train.dtype)
                            X_centered = X_train
                            y_centered = y_train

                        XtX = X_centered.T @ X_centered
                        Xty = X_centered.T @ y_centered
                        n_train = int(X_train.shape[0])

                XtX_folds.append(XtX)
                Xty_folds.append(Xty)
                # For weighted Ridge, n_train is sum(sw) (float); for unweighted, it's the count (int)
                n_train_folds.append(float(n_train) if sw_train is not None else int(n_train))
                X_mean_folds.append(X_mean)
                y_mean_folds.append(y_mean)

            # Batch solve for all alphas (Phase 1 optimization)
            XtX_batch = backend.stack(XtX_folds, axis=0)
            Xty_batch = backend.stack(Xty_folds, axis=0)
            # Use float64 to preserve fractional sum(sw) for weighted Ridge
            n_samples_vec = np.asarray(n_train_folds, dtype=np.float64)

            original_X64 = None
            original_y64 = None

            def factor_factory(fold_idx):
                nonlocal original_X64, original_y64
                if original_X64 is None:
                    original_X64 = backend.asarray(
                        X if (gpu_input_cupy or gpu_input_torch) else X_np,
                        dtype=backend.float64,
                    )
                    original_y64 = backend.asarray(
                        y if (gpu_input_cupy or gpu_input_torch) else y_np,
                        dtype=backend.float64,
                    ).reshape(-1)
                train, valid = folds[fold_idx]
                train = backend.asarray(train)
                valid = backend.asarray(valid)
                X_train = original_X64[train]
                y_train = original_y64[train]
                weights = None if sw_full is None else sw_full[train]
                if fit_intercept:
                    if weights is None:
                        X_shift = X_train - X_train[0]
                        y_shift = y_train - y_train[0]
                        X_shift_mean = backend.mean(X_shift, axis=0)
                        y_shift_mean = backend.mean(y_shift)
                        X_centered = X_shift - X_shift_mean
                        y_centered = y_shift - y_shift_mean
                        X_mean = X_train[0] + X_shift_mean
                        y_mean = y_train[0] + y_shift_mean
                    else:
                        X_centered, y_centered, X_mean, y_mean = _weighted_ridge_centered_data(
                            X_train, y_train, weights, xp,
                        )
                else:
                    X_centered, y_centered = X_train, y_train
                    X_mean = backend.zeros((X_train.shape[1],), dtype=backend.float64)
                    y_mean = backend.array(0., dtype=backend.float64)
                X_mean_folds[fold_idx], y_mean_folds[fold_idx] = X_mean, y_mean
                X_val_folds[fold_idx] = original_X64[valid]
                y_val_folds[fold_idx] = original_y64[valid]
                if weights is not None:
                    roots = _ridge_normalized_root_weights(weights, xp)
                    X_centered = X_centered * roots[:, None]
                    y_centered = y_centered * roots
                return X_centered, y_centered

            coefs_batch = _solve_ridge_path_gpu_from_gram(
                XtX_batch, Xty_batch, n_samples_vec, alpha_grid, backend,
                fit_intercept=bool(fit_intercept), factor_factory=factor_factory,
            )

            # Batch compute intercepts (Phase 2 optimization)
            X_mean_batch = backend.stack(X_mean_folds, axis=0)  # (n_folds, n_features)
            y_mean_batch = backend.stack(y_mean_folds, axis=0)  # (n_folds,)

            intercepts_batch = _compute_intercepts_batch(
                coefs_batch, X_mean_batch, y_mean_batch, backend, fit_intercept=bool(fit_intercept)
            )  # (n_alphas, n_folds)

            # Batch compute MSE for all folds (Phase 2 optimization)
            # Pad validation sets to same size
            n_val_max = max(n_val_folds)
            n_features = int(X_full.shape[1])

            # Pre-allocate padded batches (Phase 3 optimization - memory pre-allocation)
            score_dtype = backend.float64 if original_X64 is not None else input_dtype
            X_val_batch = backend.zeros((n_folds, n_val_max, n_features), dtype=score_dtype)
            y_val_batch = backend.zeros((n_folds, n_val_max), dtype=score_dtype)

            if sw_full is not None:
                sw_val_batch = backend.zeros((n_folds, n_val_max), dtype=backend.float64)
            else:
                sw_val_batch = None

            # Fill padded batches
            for fold_idx in range(n_folds):
                n_val = n_val_folds[fold_idx]
                X_val_batch[fold_idx, :n_val, :] = X_val_folds[fold_idx]
                y_val_batch[fold_idx, :n_val] = y_val_folds[fold_idx]
                if sw_val_batch is not None:
                    sw_val_batch[fold_idx, :n_val] = sw_val_folds[fold_idx]

            # Batched MSE computation (fully vectorized)
            mse_path_gpu = _batch_mse_all_folds(
                X_val_batch, y_val_batch, coefs_batch, intercepts_batch, backend, sw_val_batch,
                n_val_folds=n_val_folds,
            )

            # Convert to numpy
            mse_path = backend.to_numpy(mse_path_gpu)

        except FloatingPointError:
            # Preserve the actionable CV-precision error without substitution.
            raise
        except Exception as exc:
            raise RuntimeError(
                "GPU path failed in _select_ridge_alpha_cv with device='cuda'; "
                "CPU fallback is disabled for strict CUDA execution."
            ) from exc

    # CPU path
    if not use_gpu:
        if gpu_requested:
            raise RuntimeError(
                "device='cuda' requested but GPU path was not executed; "
                "CPU fallback is disabled for strict CUDA execution."
            )

        fast_fold_stats = (sample_weight_np is None) and bool(complementary_training)
        if fast_fold_stats:
            n_total = int(X_np.shape[0])
            XtX_full = X_np.T @ X_np
            Xty_full = X_np.T @ y_np
            if bool(fit_intercept):
                X_sum_full = np.sum(X_np, axis=0)
                y_sum_full = float(np.sum(y_np))
            else:
                X_sum_full = None
                y_sum_full = None

        cpu_recovery_count = 0
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_val = X_np[val_idx]
            y_val = y_np[val_idx]
            sw_val = None if sample_weight_np is None else sample_weight_np[val_idx]

            if fast_fold_stats:
                n_val = int(np.asarray(val_idx, dtype=np.int64).reshape(-1).size)
                n_train = int(n_total - n_val)

                XtX_val = X_val.T @ X_val
                Xty_val = X_val.T @ y_val
                XtX_raw = XtX_full - XtX_val
                Xty_raw = Xty_full - Xty_val

                if bool(fit_intercept):
                    X_sum_val = np.sum(X_val, axis=0)
                    y_sum_val = float(np.sum(y_val))
                    X_sum_train = X_sum_full - X_sum_val
                    y_sum_train = y_sum_full - y_sum_val

                    inv_n = 1.0 / float(max(1, n_train))
                    X_mean = X_sum_train * inv_n
                    y_mean = y_sum_train * inv_n
                    XtX = XtX_raw - np.outer(X_sum_train, X_sum_train) * inv_n
                    Xty = Xty_raw - X_sum_train * y_mean
                else:
                    X_mean = np.zeros((X_np.shape[1],), dtype=np.float64)
                    y_mean = 0.0
                    XtX = XtX_raw
                    Xty = Xty_raw
            else:
                X_train = X_np[train_idx]
                y_train = y_np[train_idx]
                sw_train = None if sample_weight_np is None else sample_weight_np[train_idx]

                if sw_train is not None:
                    # Square-root normalization preserves weighted contributions
                    # even when the corresponding weight ratio underflows.
                    if bool(fit_intercept):
                        XtX, Xty, X_mean, y_mean = _weighted_ridge_centered_gram(
                            X_train, y_train, sw_train, np,
                        )
                    else:
                        XtX, Xty = _weighted_ridge_gram(
                            X_train, y_train, sw_train, np,
                        )
                        X_mean = np.zeros((X_train.shape[1],), dtype=np.float64)
                        y_mean = 0.0
                    n_train = 1.0
                else:
                    if bool(fit_intercept):
                        X_mean = np.mean(X_train, axis=0)
                        y_mean = float(np.mean(y_train))
                        X_centered = X_train - X_mean
                        y_centered = y_train - y_mean
                    else:
                        X_mean = np.zeros((X_train.shape[1],), dtype=np.float64)
                        y_mean = 0.0
                        X_centered = X_train
                        y_centered = y_train

                    XtX = X_centered.T @ X_centered
                    Xty = X_centered.T @ y_centered
                    n_train = int(X_train.shape[0])

            # Solve ordinary weighted systems as before. Only inspect the
            # Gram spectrum when its rounding bound can exceed the smallest
            # requested penalty; default-scale paths avoid an extra eigensolve.
            recover_cpu = False
            if sample_weight_np is not None:
                eps = np.finfo(XtX.dtype).eps
                tiny = np.finfo(XtX.dtype).tiny
                smallest_penalty = float(np.min(alpha_grid))
                gamma = XtX.shape[0] * eps / (1.0 - XtX.shape[0] * eps)
                uncertainty_bound = gamma * np.linalg.norm(XtX, ord=np.inf)
                # Normalized weighted factors can have a subnormal or zero
                # Gram matrix although the raw weighted system was resolved.
                # Compare magnitudes directly, before the error bound loses
                # its own subnormal bits, and recover from original factors.
                recover_cpu = np.max(np.abs(XtX)) < tiny and smallest_penalty < tiny
                if not recover_cpu and 0 < uncertainty_bound and smallest_penalty <= uncertainty_bound:
                    spectrum = np.linalg.eigvalsh(XtX)
                    uncertainty = gamma * np.max(np.abs(spectrum))
                    recover_cpu = (
                        (spectrum[0] <= uncertainty and smallest_penalty <= uncertainty)
                        or (spectrum[0] < tiny and smallest_penalty < tiny)
                    )
            if recover_cpu:
                if fit_intercept:
                    factors_X, factors_y, X_mean, y_mean = _weighted_ridge_centered_data(
                        X_train, y_train, sw_train, np,
                    )
                else:
                    factors_X, factors_y = X_train, y_train
                roots = _ridge_normalized_root_weights(sw_train, np)
                coefs_desc = _solve_ridge_path_from_factors(
                    factors_X * roots[:, None], factors_y * roots, alpha_grid,
                    get_backend(backend="numpy", device="cpu"),
                ).T
                cpu_recovery_count += 1
            else:
                # n_eff scaling matches Ridge.fit() and PGLM exact ridge.
                I = np.eye(XtX.shape[0])
                coefs_desc = []
                for alpha in alpha_grid:
                    XtX_reg = XtX + alpha * float(n_train) * I
                    try:
                        coef = np.linalg.solve(XtX_reg, Xty)
                    except np.linalg.LinAlgError:
                        coef = np.linalg.lstsq(XtX_reg, Xty, rcond=None)[0]
                    coefs_desc.append(coef.flatten())
                coefs_desc = np.stack(coefs_desc, axis=0)

            # Compute intercepts
            if bool(fit_intercept):
                # X_mean: (p,), coefs_desc: (n_alphas, p)
                # X_mean @ coefs_desc.T = coefs_desc @ X_mean = (n_alphas,)
                intercepts_desc = y_mean - coefs_desc @ X_mean
            else:
                intercepts_desc = np.zeros((coefs_desc.shape[0],))

            # Compute MSE
            if sw_val is None:
                mse_desc = _batch_mse_cv(X_val, y_val, coefs_desc, intercepts_desc)
            else:
                mse_desc = _weighted_ridge_mse(
                    X_val, y_val, coefs_desc, intercepts_desc, sw_val,
                )
            mse_path[:, fold_idx] = mse_desc

        if cpu_recovery_count:
            warnings.warn(
                "RidgeCV used same-device float64 factor-SVD recovery for "
                f"{cpu_recovery_count} fold(s); unresolved singular directions "
                "with only roundoff-sized response projections were treated as numerical zeros.",
                RuntimeWarning, stacklevel=2,
            )

    # Compute mean MSE across folds
    mean_mse = np.nanmean(mse_path, axis=1)

    # Find best alpha (minimum MSE)
    best_idx = int(np.nanargmin(mean_mse))
    best_alpha = float(alpha_grid[best_idx])

    details = {
        "alpha": best_alpha,
        "alphas": alpha_grid,
        "mse_path": mse_path,
        "mean_mse": mean_mse,
    }

    _ridge_cv_cache_put(cache_key_eff, details)

    if return_details:
        return details
    return best_alpha


# =============================================================================
# GPU MSE helper — batched across folds
# =============================================================================

def _batch_mse_all_folds(X_val_batch, y_val_batch, coefs_batch, intercepts_batch, backend, sample_weights_batch=None, n_val_folds=None):
    """
    Compute MSE for all folds and all alphas simultaneously (fully vectorized).

    This function batches the MSE computation across all CV folds and all alphas,
    maximizing GPU parallelism and minimizing kernel launch overhead.

    Parameters
    ----------
    X_val_batch : array-like
        Batched validation matrices (n_folds, n_val_max, n_features).
        Padded with zeros if fold sizes differ.
    y_val_batch : array-like
        Batched validation responses (n_folds, n_val_max).
        Padded with zeros if fold sizes differ.
    coefs_batch : array-like
        Coefficient matrix (n_alphas, n_folds, n_features). Same device as X_val_batch.
    intercepts_batch : array-like
        Intercept vector (n_alphas, n_folds). Same device as X_val_batch.
    backend : BackendBase
        Backend instance (CuPyBackend or TorchBackend).
    sample_weights_batch : array-like or None
        Batched sample weights (n_folds, n_val_max), or None.

    Returns
    -------
    mse : array-like
        MSE for each alpha and fold (n_alphas, n_folds). Same device as input.
    """
    xp = backend.xp
    n_folds = X_val_batch.shape[0]

    # coefs_batch and intercepts_batch are already on GPU (no conversion needed)
    # Compute predictions: (n_folds, n_val_max, n_alphas)
    # X_val_batch: (n_folds, n_val_max, n_features)
    # coefs_batch: (n_alphas, n_folds, n_features) -> transpose to (n_folds, n_features, n_alphas)
    coefs_T = backend.transpose(coefs_batch, (1, 2, 0))  # (n_folds, n_features, n_alphas)
    coefs_T = backend.asarray(coefs_T, dtype=X_val_batch.dtype)
    y_pred = xp.matmul(X_val_batch, coefs_T)
    _is_torch = _torch_dev(coefs_batch) is not None
    _expand = lambda a, dim: a.unsqueeze(dim) if _is_torch else xp.expand_dims(a, axis=dim)
    intercepts_expanded = _expand(intercepts_batch.T, 1)
    y_val_expanded = _expand(y_val_batch, 2)
    raw_residuals = y_pred + intercepts_expanded - y_val_expanded
    if sample_weights_batch is not None:
        mass = xp.sum(sample_weights_batch, axis=1, keepdims=True)
        mass = xp.where(mass > 0, mass, xp.ones_like(mass))
        root_weights = xp.sqrt(sample_weights_batch) / xp.sqrt(mass)
        roots = _expand(root_weights, 2)
        active = _expand(sample_weights_batch > 0, 2)
        residuals = xp.where(active, raw_residuals, xp.zeros_like(raw_residuals)) * roots
        invalid = ~xp.isfinite(residuals)
        if bool(_to_float_scalar(xp.any(invalid))):
            scaled_predictions = xp.matmul(X_val_batch * roots, coefs_T)
            scaled = scaled_predictions + roots * intercepts_expanded - roots * y_val_expanded
            residuals = xp.where(invalid, scaled, residuals)
    else:
        residuals = raw_residuals

    # Zero out padded rows to prevent inflated MSE from intercept contribution
    if n_val_folds is not None:
        n_val_max = residuals.shape[1]
        # Create mask: (n_folds, n_val_max) -> (n_folds, n_val_max, 1)
        if _is_torch:
            import torch
            mask = torch.arange(n_val_max, device=residuals.device).unsqueeze(0) < \
                   torch.tensor(n_val_folds, device=residuals.device).unsqueeze(1)
            mask = mask.unsqueeze(2).to(residuals.dtype)
        else:
            mask = xp.arange(n_val_max).reshape(1, -1) < \
                   xp.asarray(n_val_folds).reshape(-1, 1)
            mask = mask[:, :, xp.newaxis].astype(residuals.dtype)
        residuals = residuals * mask

    # Compute MSE — use per-fold n_val to exclude padded zeros
    if sample_weights_batch is not None:
        # Residuals already contain sqrt(w / sum(w)).
        mse = xp.sum(residuals ** 2, axis=1).T
        if not bool(_to_float_scalar(xp.all(xp.isfinite(mse)))):
            raise FloatingPointError(
                "RidgeCV weighted validation MSE is not finite at float64 precision; "
                "rescale the response/features or choose different alphas."
            )
    else:
        ssr = xp.sum(residuals ** 2, axis=1)  # (n_folds, n_alphas)
        if n_val_folds is not None:
            n_val_vec = backend.asarray(n_val_folds, dtype=ssr.dtype).reshape(-1, 1)
            mse = (ssr / n_val_vec).T  # (n_alphas, n_folds)
        else:
            mse = xp.mean(residuals ** 2, axis=1).T

    return mse


def _compute_intercepts_batch(coefs_batch, X_mean_batch, y_mean_batch, backend, fit_intercept=True):
    """
    Compute intercepts for all alphas and all folds simultaneously.

    Parameters
    ----------
    coefs_batch : array-like
        Coefficient matrix (n_alphas, n_folds, n_features). Can be GPU or CPU.
    X_mean_batch : array-like
        Training set means (n_folds, n_features).
    y_mean_batch : array-like
        Training set response means (n_folds,).
    backend : BackendBase
        Backend instance.
    fit_intercept : bool
        Whether to compute intercepts.

    Returns
    -------
    intercepts : array-like
        Intercept matrix (n_alphas, n_folds). Same device as input.
    """
    xp = backend.xp

    if not fit_intercept:
        return backend.zeros((coefs_batch.shape[0], coefs_batch.shape[1]), dtype=coefs_batch.dtype)

    n_alphas = coefs_batch.shape[0]
    n_folds = coefs_batch.shape[1]
    n_features = coefs_batch.shape[2]

    # Compute coefs @ X_mean for each fold
    # Reshape coefs to (n_alphas * n_folds, n_features)
    coefs_reshaped = coefs_batch.reshape((n_alphas * n_folds, n_features))

    # Tile X_mean for each alpha
    X_mean_tiled = xp.tile(X_mean_batch, (n_alphas, 1))

    # Batched dot product: sum over features
    coefs_dot_sum = xp.sum(coefs_reshaped * X_mean_tiled, axis=1)  # (n_alphas * n_folds,)
    coefs_dot_sum = coefs_dot_sum.reshape((n_alphas, n_folds))     # (n_alphas, n_folds)

    # y_mean_batch: (n_folds,) -> (1, n_folds) broadcasts to (n_alphas, n_folds)
    if _torch_dev(coefs_batch) is not None:
        y_mean_expanded = y_mean_batch.unsqueeze(0)
    else:
        y_mean_expanded = xp.expand_dims(y_mean_batch, axis=0)
    intercepts = y_mean_expanded - coefs_dot_sum

    return intercepts


# =============================================================================
# RidgeCV Class
# =============================================================================

class RidgeCV(CVEstimatorBase):
    """Cross-validated Ridge regression for one continuous response.

    Select alpha by mean held-out MSE, then refit all supplied training rows.
    Final inference conditions on selected alpha and does not adjust tuning
    uncertainty. Generated folds are shuffled K-fold, not time/group-aware.

    Parameters
    ----------
    alphas : array-like or None, default=None
        Explicit positive finite candidates; omitted: generate a data-
        dependent grid. Invalid/nonpositive entries are filtered; an empty
        surviving grid falls back to automatic generation.
    n_alphas : int, default=100
        Automatic grid size when alphas is omitted.
    alpha_min_ratio : float, default=0.001
        Minimum/maximum ratio for the automatic grid; choose a positive
        value normally no greater than 1.
    cv : int, default=5
        Generated shuffled K-fold count; must be at least 2, including when
        cv_splits is supplied. Explicit cv_splits determine the actual folds.
    cv_splits : list or None, default=None
        Explicit reusable list of (train_indices, validation_indices).
        Indices must be nonempty, one-dimensional, integer-valued, in bounds,
        duplicate-free within each subset, and disjoint within each pair.
    fit_intercept : bool, default=True
        Fit an intercept in CV and final refit.
    device : str, default='auto'
        cpu, cuda (CuPy), torch (Torch CUDA), auto; explicit unavailable GPU
        requests raise.
    n_jobs : int or None, default=None
        Shared worker configuration; no candidate-parallelism guarantee.
    compute_inference : bool, default=True
        Compute supported inference only on the final full-data refit,
        conditional on selected alpha.
    cov_type : str, default='nonrobust'
        Final Ridge covariance: nonrobust, hc0, hc1, hc2, hc3, hac. There is
        no hac_maxlags constructor control; HAC uses the automatic rule.
    gpu_memory_cleanup : bool, default=False
        Request best-effort GPU cache cleanup for final fitting.
    random_state : int or None, default=None
        Seed for generated folds; not residual-bootstrap randomness.
    gpu_cv_mixed_precision : bool, default=True
        Normally use float32 GPU Gram calculations. Weighted centering and
        validation scoring retain float64. Numerically unresolved Gram systems
        may use same-device float64 factor-SVD recovery with a RuntimeWarning.
        Final-refit inference has its own numerical path.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit finite X (n, p), one-dimensional y (n,), optional analytic weights;
        return self. There is no formula interface.
    predict(X)
        Return NumPy predictions of shape (m,) through the final estimator.
    score(X, y)
        Return unweighted evaluation R-squared; no sample_weight argument.
    summary()
        Print final inference and return None; requires successful inference.
    get_params(deep=True), set_params(**params)
        Read/update constructor configuration; valid updates require refitting.
    adjust_pvalues, combine_pvalues, bootstrap_statistic, permutation_test
        Inherited helpers; they do not automatically repeat CV or model refits.

    Attributes
    ----------
    alpha_ : float
        Selected penalty, minimizing mean validation MSE.
    alphas_, mean_mse_ : numpy.ndarray of shape (n_alphas,)
        Actual candidates and their mean validation losses.
    cv_results_ : dict
        Only mse_path, of shape (n_alphas, n_folds); no mean_mse dictionary key.
    best_score_ : float
        Negative selected mean MSE (larger is better), not final-model R-squared.
    coef_ : numpy.ndarray of shape (n_features,)
        Final full-data prediction slopes.
    intercept_ : float
        Final intercept, or zero when omitted.
    n_iter_ : int
        Final estimator iteration count.
    estimator_ : object
        Fitted direct estimator; read its supported inference outputs here.
    cv_selected_device_ : Device or str
        Resolved final device.

    Notes
    -----
    Custom splits preserve the supplied training and validation rows, with or
    without sample weights. Training subsets need not be validation complements,
    and validation sets need not partition all samples. The final refit uses all
    supplied fit rows. Explicit splits are evaluated even with fewer than four
    rows or one alpha candidate. Each training and validation subset must have
    positive total sample weight; individual zero weights are allowed. The cv
    argument must still be at least 2, but does not generate or replace explicit
    splits. This constructor does not expose solver/max_iter/tol; the final
    Ridge fit uses its exact path. Weighted CPU and GPU CV can recover
    unresolved Gram systems using same-device float64 factor SVD. Only
    unresolved singular directions with roundoff-sized response projections
    are treated as numerical zeros; response-bearing ambiguous directions
    raise FloatingPointError. A finite statistical ridge solution may still
    exist when numerical precision is insufficient. Nonfinite weighted
    validation losses also raise FloatingPointError.
    """

    def __init__(
        self,
        alphas=None,
        n_alphas: int = 100,
        alpha_min_ratio: float = 1e-3,
        cv: int = 5,
        cv_splits=None,
        fit_intercept: bool = True,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
        compute_inference: bool = True,
        cov_type: str = "nonrobust",
        gpu_memory_cleanup: bool = False,
        random_state: Optional[int] = None,
        gpu_cv_mixed_precision: bool = True,
    ):
        super().__init__(
            cv=cv,
            random_state=random_state,
            device=device,
            n_jobs=n_jobs,
        )
        self.alphas = alphas
        self.n_alphas = int(n_alphas)
        self.alpha_min_ratio = float(alpha_min_ratio)
        self.cv = int(cv)
        self.cv_splits = cv_splits
        self.fit_intercept = bool(fit_intercept)
        self.compute_inference = bool(compute_inference)
        self.cov_type = str(cov_type)
        self.gpu_memory_cleanup = bool(gpu_memory_cleanup)
        self.gpu_cv_mixed_precision = bool(gpu_cv_mixed_precision)

        self.alpha_ = None
        self.alphas_ = None
        self.cv_results_ = None
        self.mean_mse_ = None
        self.best_score_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.estimator_ = None
        self.cv_selected_device_ = None

    def _reset_cv_fit_state(self):
        """Clear all fitted outputs before a new CV attempt."""
        self._fitted = False
        self.alpha_ = None
        self.alphas_ = None
        self.cv_results_ = None
        self.mean_mse_ = None
        self.best_score_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.estimator_ = None
        self.cv_selected_device_ = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit Ridge regression with cross-validation to select alpha.

        Parameters
        ----------
        X : array-like
            Training data (n_samples, n_features).
        y : array-like
            Target values.
        sample_weight : array-like or None
            Sample weights.

        Returns
        -------
        self : RidgeCV
            Fitted estimator.
        """
        self._reset_cv_fit_state()
        from statgpu.cross_validation._base import validate_cv_sample_weight
        n_samples = int(X.shape[0]) if hasattr(X, 'shape') else len(X)
        sample_weight = validate_cv_sample_weight(sample_weight, n_samples)

        device_name = self._device
        _, cv_backend_name, _, _, _, _ = resolve_cv_backend(device_name, X)
        refit_device = cv_refit_device(device_name, cv_backend_name)

        # Run CV to select alpha
        details = _select_ridge_alpha_cv(
            X,
            y,
            alphas=self.alphas,
            n_alphas=self._n_alphas,
            alpha_min_ratio=self._alpha_min_ratio,
            cv_folds=self._cv,
            cv_splits=self.cv_splits,
            random_state=self.random_state,
            sample_weight=sample_weight,
            fit_intercept=self._fit_intercept,
            device=device_name,
            gpu_cv_mixed_precision=self._gpu_cv_mixed_precision,
            return_details=True,
        )

        # Keep candidate results local until the final refit succeeds.
        selected_alpha = float(details["alpha"])
        selected_alphas = np.asarray(details["alphas"], dtype=np.float64)
        mse_path = np.asarray(details["mse_path"], dtype=np.float64)
        mean_mse = np.asarray(details["mean_mse"], dtype=np.float64)
        best_score = (
            -float(np.nanmin(mean_mse))
            if np.any(np.isfinite(mean_mse))
            else np.nan
        )

        # Fit final model with selected alpha.
        # Exact solve uses n*alpha on unnormalized X'X, matching the
        # per-sample convention (loss/n + alpha*||w||^2) used by all paths.
        # alpha_ stores the CV-selected value; pass it directly to Ridge.
        estimator = Ridge(
            alpha=selected_alpha,
            fit_intercept=self._fit_intercept,
            device=refit_device,
            n_jobs=self.n_jobs,
            compute_inference=self._compute_inference_enabled,
            cov_type=self._cov_type,
            gpu_memory_cleanup=self._gpu_memory_cleanup,
        )

        estimator.fit(X, y, sample_weight=sample_weight)

        self.alpha_ = selected_alpha
        self.alphas_ = selected_alphas
        self.cv_results_ = {"mse_path": mse_path}
        self.mean_mse_ = mean_mse
        self.best_score_ = best_score
        self.estimator_ = estimator
        self.coef_ = np.asarray(estimator.coef_)
        self.intercept_ = estimator.intercept_
        self.n_iter_ = getattr(estimator, 'n_iter_', None)
        self.cv_selected_device_ = refit_device
        self._fitted = True
        return self

    def predict(self, X):
        """Predict using the fitted Ridge model."""
        self._check_is_fitted()
        return self.estimator_.predict(X)
