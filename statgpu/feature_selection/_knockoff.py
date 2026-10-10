"""Fixed-X and model-X knockoff feature selection across supported backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from statgpu.feature_selection import _knockoff_utils as _kutils
from statgpu.backends._validation import check_finite
from statgpu.feature_selection._knockoff_utils import (
    _build_fixed_x_knockoffs,
    _build_model_x_knockoffs,
    _build_model_x_knockoffs_knockpy_compat,
    _compute_w_statistics,
    _get_xp,
    _knockoff_threshold_and_path,
    _model_x_draw_seed,
    _normalize_compat_mode,
    _normalize_fdr_control,
    _normalize_knockoff_type,
    _normalize_lasso_fast_profile,
    _resolve_backend,
    _standardize_design,
    _standardize_features_unit_variance,
    _to_numpy,
    _validate_q,
)

# Backward compatibility for existing internal profiling scripts.
_random_permutation_inds = _kutils._random_permutation_inds


def _validate_optional_positive_int(value, name: str) -> Optional[int]:
    """Validate an optional strictly-positive integer without truncation."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer or None")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


@dataclass
class KnockoffResult:
    """Structured output for knockoff selection."""

    knockoff_type: str
    selected_features: np.ndarray
    W: np.ndarray
    threshold: float
    q: float
    estimated_fdr: float
    q_trajectory: List[Dict[str, float]]
    method: str
    fdr_control: str
    random_state: Optional[int]
    backend: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "knockoff_type": self.knockoff_type,
            "selected_features": self.selected_features.tolist(),
            "W": self.W.tolist(),
            "threshold": float(self.threshold),
            "q": float(self.q),
            "estimated_fdr": float(self.estimated_fdr),
            "q_trajectory": list(self.q_trajectory),
            "method": self.method,
            "fdr_control": self.fdr_control,
            "random_state": self.random_state,
            "backend": self.backend,
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# Knockpy-compatible interface placeholders
# -----------------------------------------------------------------------------
def knockpy_gaussian_mvr_sampler(
    X,
    *,
    mu=None,
    Sigma=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy GaussianSampler(method='mvr')."""
    raise NotImplementedError("knockpy_gaussian_mvr_sampler is not yet implemented")


def knockpy_gaussian_sdp_sampler(
    X,
    *,
    mu=None,
    Sigma=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy GaussianSampler(method='sdp')."""
    raise NotImplementedError("knockpy_gaussian_sdp_sampler is not yet implemented")


def knockpy_gaussian_maxent_sampler(
    X,
    *,
    mu=None,
    Sigma=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy GaussianSampler(method='maxent')."""
    raise NotImplementedError("knockpy_gaussian_maxent_sampler is not yet implemented")


def knockpy_gaussian_equi_sampler(
    X,
    *,
    mu=None,
    Sigma=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy GaussianSampler(method='equi')."""
    raise NotImplementedError("knockpy_gaussian_equi_sampler is not yet implemented")


def knockpy_gaussian_ci_sampler(
    X,
    *,
    mu=None,
    Sigma=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy GaussianSampler(method='ci')."""
    raise NotImplementedError("knockpy_gaussian_ci_sampler is not yet implemented")


def knockpy_fx_sampler(
    X,
    *,
    y=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy FXSampler."""
    raise NotImplementedError("knockpy_fx_sampler is not yet implemented")


def knockpy_metro_sampler(
    X,
    *,
    y=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy MetroSampler."""
    raise NotImplementedError("knockpy_metro_sampler is not yet implemented")


def knockpy_artk_sampler(
    X,
    *,
    y=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy ARTKSampler."""
    raise NotImplementedError("knockpy_artk_sampler is not yet implemented")


def knockpy_sampler_dispatch(
    sampler: str,
    X,
    *,
    method: Optional[str] = None,
    mu=None,
    Sigma=None,
    y=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Unified dispatcher for knockpy-compatible sampler placeholders."""
    sampler_key = str(sampler).strip().lower().replace("-", "_")
    method_key = None if method is None else str(method).strip().lower().replace("-", "_")

    gaussian_dispatch = {
        "mvr": knockpy_gaussian_mvr_sampler,
        "sdp": knockpy_gaussian_sdp_sampler,
        "maxent": knockpy_gaussian_maxent_sampler,
        "equi": knockpy_gaussian_equi_sampler,
        "ci": knockpy_gaussian_ci_sampler,
    }

    if sampler_key == "gaussian":
        method_key = "mvr" if method_key is None else method_key
        fn = gaussian_dispatch.get(method_key)
        if fn is None:
            raise ValueError("For sampler='gaussian', method must be one of: 'mvr', 'sdp', 'maxent', 'equi', 'ci'")
        return fn(
            X,
            mu=mu,
            Sigma=Sigma,
            groups=groups,
            random_state=random_state,
        )

    gaussian_aliases = {
        "gaussian_mvr": "mvr",
        "gaussian_sdp": "sdp",
        "gaussian_maxent": "maxent",
        "gaussian_equi": "equi",
        "gaussian_ci": "ci",
    }
    if sampler_key in gaussian_aliases:
        fn = gaussian_dispatch[gaussian_aliases[sampler_key]]
        return fn(
            X,
            mu=mu,
            Sigma=Sigma,
            groups=groups,
            random_state=random_state,
        )

    if sampler_key == "fx":
        return knockpy_fx_sampler(
            X,
            y=y,
            groups=groups,
            random_state=random_state,
        )

    if sampler_key == "metro":
        return knockpy_metro_sampler(
            X,
            y=y,
            groups=groups,
            random_state=random_state,
        )

    if sampler_key == "artk":
        return knockpy_artk_sampler(
            X,
            y=y,
            groups=groups,
            random_state=random_state,
        )

    raise ValueError(
        "sampler must be one of: 'gaussian', 'gaussian_mvr', 'gaussian_sdp', "
        "'gaussian_maxent', 'gaussian_equi', 'gaussian_ci', 'fx', 'metro', 'artk'"
    )


def fixed_x_knockoff_filter(
    X,
    y,
    q: float = 0.1,
    method: str = "corr_diff",
    fdr_control: str = "knockoff_plus",
    random_state: Optional[int] = None,
    backend: str = "auto",
    Xk=None,
    compat_mode: str = "statgpu",
    lasso_cv_impl: str = "auto",
    lasso_fast_profile: str = "off",
) -> KnockoffResult:
    """
    Fixed-X knockoff feature selection.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix.
    y : array-like of shape (n_samples,)
        Response vector.
    q : float, default=0.1
        Finite target FDR level in (0, 1). Check
        ``np.isfinite(q) and 0 < q < 1`` before calling. Current validation
        misses NaN and can return an invalid empty selection with
        threshold=inf and estimated_fdr=0.0 under either threshold rule.
    method : {'corr_diff', 'ols_coef_diff', 'lasso_coef_diff'}, default='corr_diff'
        Feature-importance statistic for W construction.
    fdr_control : {'knockoff_plus', 'knockoff'}, default='knockoff_plus'
        Knockoff threshold variant.
    random_state : int, optional
        Random seed for knockoff construction.
    backend : {'auto', 'numpy', 'cupy', 'torch'}, default='auto'
        Compute backend. ``'auto'`` infers from input arrays.
        ``'torch'`` selects the Torch library, not CUDA placement. Fixed-X
        construction follows X's device, including CPU. Keep X, y and any
        supplied Xk on the same device for native statistics. Native model-X
        construction also follows X's device; see ``model_x_knockoff_filter``
        for its construction and seed behavior. This differs from estimator
        ``device='torch'``, which requests CUDA.
    Xk : array-like of shape (n_samples, n_features), optional
        Supplied knockoff matrix; bypasses construction. The caller must ensure
        matched-design validity after any intercept/nuisance projection, not
        merely the same shape or raw Gram matrix as X. Centered valid pairs
        avoid the automatic construction centering issue below.
    compat_mode : {'statgpu', 'knockpy'}, default='statgpu'
        Construction/statistic conventions; does not certify package parity.
    lasso_cv_impl : {'auto', 'statgpu', 'sklearn'}, default='auto'
        Lasso-statistic implementation. Auto requests sklearn in knockpy mode
        and statgpu otherwise. A sklearn import failure or Torch statistic
        computation can switch to statgpu without updating the metadata label.
    lasso_fast_profile : {'off', 'auto', 'moderate', 'aggressive'}, default='off'
        Lasso profile that can change CV folds, candidate penalties, iteration
        budget and tolerance, hence W and selected features. It is not an
        output-preserving speed switch.

    Returns
    -------
    KnockoffResult
        Selected feature indices and full knockoff diagnostics.

    Notes
    -----
    Fixed-X finite-sample interpretation assumes a Gaussian linear response
    with independent homoskedastic normal errors, plus valid matched-design
    geometry and statistics. It is not guaranteed for arbitrary finite y.

    Automatic construction centers X but can generate noncentered Xk. The
    correlation, OLS and native non-knockpy Lasso statistics center y, so equal
    raw Grams can become unequal after the intercept projection. The usual
    null score-pair exchangeability can fail even without threshold ties and
    with n>2p. Increasing n alone does not fix this. A valid centered supplied
    pair avoids this geometry defect; the documented orthogonal QR example
    needs n>=2p+1 and is not a general repair for arbitrary X. Separately, tied
    absolute statistics can give incorrect threshold counts. Do not infer
    nominal FDR control merely from valid dimensions or absence of ties.

    Model-X instead relies on feature-pair exchangeability and conditional
    independence from y given X, allowing arbitrary response relationships;
    the estimated feature-model and multi-draw limitations still apply.

    Construction's input-device behavior does not extend to Torch
    ``method='lasso_coef_diff'``: native Lasso tuning/fitting requests CUDA,
    including for CPU inputs or supplied Xk, and does not guarantee the input
    GPU index. For Torch CPU statistics, use ``corr_diff`` or ``ols_coef_diff``
    when appropriate; for CPU Lasso, use NumPy inputs with ``backend='numpy'``.

    Seeded ``lasso_coef_diff`` calls can reuse stale statistics if X, y, or Xk
    changes in place or previous array memory is reused. A new selector does not
    isolate this cache. Use a fresh Python process for each changed-data call,
    or, with supplied float64 NumPy X/y/Xk, fresh copies while retaining every old
    input unchanged. ``random_state=None`` disables seeded reuse but does not
    provide seed-based repeatability. The feature-selection API reference gives
    a complete safe-copy example and statistical limitations.
    """
    check_finite(X, name="X")
    check_finite(y, name="y")
    if Xk is not None:
        check_finite(Xk, name="Xk")
    q_f = _validate_q(q)
    compat = _normalize_compat_mode(compat_mode)
    lasso_impl = str(lasso_cv_impl).strip().lower()
    if lasso_impl == "auto":
        lasso_impl = "sklearn" if compat == "knockpy" else "statgpu"
    lasso_profile = _normalize_lasso_fast_profile(lasso_fast_profile)

    offset = _normalize_fdr_control(fdr_control)
    backend_name = _resolve_backend(backend, X, y, Xk)
    xp = _get_xp(backend_name)

    X_arr = xp.asarray(X, dtype=xp.float64)
    y_arr = xp.asarray(y, dtype=xp.float64).reshape(-1)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("y must have the same number of rows as X")

    if Xk is None:
        X_work = _standardize_design(X_arr, xp)
        X_knock = _build_fixed_x_knockoffs(X_work, random_state=random_state, xp=xp)
        xk_source = "generated_fixed_x"
    else:
        X_work = X_arr
        X_knock = xp.asarray(Xk, dtype=xp.float64)
        if X_knock.shape != X_work.shape:
            raise ValueError("Xk must have the same shape as X")
        xk_source = "provided"

    W_xp, method_n = _compute_w_statistics(
        X_work,
        X_knock,
        y_arr,
        method=method,
        xp=xp,
        random_state=random_state,
        backend_name=backend_name,
        lasso_cv_impl=lasso_impl,
        lasso_fast_profile=lasso_profile,
        lasso_knockpy_style=(compat == "knockpy"),
    )
    threshold, fdr_hat, trajectory = _knockoff_threshold_and_path(W_xp, q=q_f, offset=offset)

    if np.isfinite(threshold):
        selected_xp = xp.where(W_xp >= threshold)[0]
    else:
        selected_xp = xp.asarray([], dtype=xp.int64)

    W = _to_numpy(W_xp).astype(np.float64, copy=False)
    selected = _to_numpy(selected_xp).astype(np.int64, copy=False)

    return KnockoffResult(
        knockoff_type="fixed_x",
        selected_features=selected,
        W=W,
        threshold=float(threshold),
        q=float(q_f),
        estimated_fdr=float(fdr_hat),
        q_trajectory=trajectory,
        method=method_n,
        fdr_control="knockoff_plus" if offset == 1 else "knockoff",
        random_state=random_state,
        backend=backend_name,
        metadata={
            "n_samples": int(X_arr.shape[0]),
            "n_features": int(X_arr.shape[1]),
            "offset": int(offset),
            "compat_mode": compat,
            "lasso_cv_impl": lasso_impl,
            "lasso_fast_profile": lasso_profile,
            "xk_source": xk_source,
        },
    )

def model_x_knockoff_filter(
    X,
    y,
    q: float = 0.1,
    method: str = "corr_diff",
    fdr_control: str = "knockoff_plus",
    random_state: Optional[int] = None,
    backend: str = "auto",
    Xk=None,
    compat_mode: str = "statgpu",
    lasso_cv_impl: str = "auto",
    lasso_fast_profile: str = "off",
    modelx_covariance_shrinkage: float = 0.20,
    modelx_s_scale: float = 0.999,
    modelx_draws: Optional[int] = None,
    modelx_shrinkage: str = "ledoitwolf",
    modelx_smatrix_method: str = "mvr",
    knockpy_sampler: Optional[str] = None,
    knockpy_sampler_method: Optional[str] = None,
) -> KnockoffResult:
    """
    Model-X knockoff selection (Gaussian second-order approximation).

    Check ``np.isfinite(q) and 0 < q < 1`` before calling or fitting. Current
    validation misses NaN under both threshold rules and can return an invalid
    empty selection with threshold=inf and estimated_fdr=0.0 (an all-false
    selector support mask). This is not a valid no-discoveries result.

    The default native path estimates a Gaussian feature model and builds
    equi-correlated knockoffs from the estimated covariance.

    With compat_mode='statgpu' and Xk=None, native Torch construction uses
    X.device for both its local random generator and random matrix. Torch CPU
    input stays on CPU even when CUDA is available; CUDA input retains its
    device, including its GPU index. As in fixed-X construction, backend='torch'
    selects the array library rather than requesting CUDA. Keep X/y/Xk on the
    same device for native statistics. Supplied, externally validated Xk
    bypasses construction; shape/device agreement alone does not establish
    feature-pair exchangeability or conditional independence from y given X.

    An integer random_state seeds the local generator for each construction
    draw without advancing the global Torch RNG. Repeated construction with
    the same inputs and settings is repeatable within the same backend, dtype,
    device and software environment; this is not a cross-backend or cross-GPU
    equality guarantee, or a guarantee of empirical FDR control. In native
    Torch/CuPy construction, random_state=None uses seed 0 for each draw,
    repeating the construction noise. Use an integer for separately seeded
    draws. NumPy construction with None remains unseeded. Check statistic repeatability
    separately, especially for the Lasso cache limitation below.

    Sampler dispatch is used only with compat_mode='knockpy' and no Xk;
    elsewhere the sampler controls are ignored, not executed.

    Construction's input-device behavior does not extend to Torch
    method='lasso_coef_diff': native Lasso tuning/fitting requests CUDA,
    including for CPU inputs or supplied Xk, and does not guarantee the input
    GPU index. For Torch CPU statistics, use corr_diff or ols_coef_diff when
    appropriate; for CPU Lasso, use NumPy inputs with backend='numpy'.

    Seeded ``lasso_coef_diff`` has the same input-mutation/cache limitation as
    ``fixed_x_knockoff_filter``. Isolate changed-data calls in fresh Python
    processes; a new selector alone does not isolate cached statistics.
    """
    check_finite(X, name="X")
    check_finite(y, name="y")
    if Xk is not None:
        check_finite(Xk, name="Xk")
    q_f = _validate_q(q)
    compat = _normalize_compat_mode(compat_mode)
    lasso_impl = str(lasso_cv_impl).strip().lower()
    if lasso_impl == "auto":
        lasso_impl = "sklearn" if compat == "knockpy" else "statgpu"
    lasso_profile = _normalize_lasso_fast_profile(lasso_fast_profile)

    offset = _normalize_fdr_control(fdr_control)
    backend_name = _resolve_backend(backend, X, y, Xk)
    xp = _get_xp(backend_name)

    X_arr = xp.asarray(X, dtype=xp.float64)
    y_arr = xp.asarray(y, dtype=xp.float64).reshape(-1)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("y must have the same number of rows as X")

    method_key = str(method).strip().lower()
    default_draws = (
        5
        if method_key in ("ols_coef_diff", "ols", "coef_diff", "lasso_coef_diff", "lasso", "lasso_diff")
        else 3
    )
    validated_draws = _validate_optional_positive_int(modelx_draws, "modelx_draws")

    if compat == "knockpy":
        # Preserve backend-native execution when caller supplies Xk and explicitly
        # asks for statgpu CV implementation under knockpy-compatible lasso settings.
        if Xk is not None and lasso_impl == "statgpu":
            X_knock = xp.asarray(Xk, dtype=xp.float64)
            if X_knock.shape != X_arr.shape:
                raise ValueError("Xk must have the same shape as X")

            W_xp, method_n = _compute_w_statistics(
                X_arr,
                X_knock,
                y_arr,
                method=method,
                xp=xp,
                random_state=random_state,
                backend_name=backend_name,
                lasso_cv_impl=lasso_impl,
                lasso_fast_profile=lasso_profile,
                lasso_knockpy_style=True,
            )
            threshold, fdr_hat, trajectory = _knockoff_threshold_and_path(W_xp, q=q_f, offset=offset)

            if np.isfinite(threshold):
                selected_xp = xp.where(W_xp >= threshold)[0]
            else:
                selected_xp = xp.asarray([], dtype=xp.int64)

            W_np = _to_numpy(W_xp).astype(np.float64, copy=False)
            selected = _to_numpy(selected_xp).astype(np.int64, copy=False)

            return KnockoffResult(
                knockoff_type="model_x",
                selected_features=selected,
                W=W_np,
                threshold=float(threshold),
                q=float(q_f),
                estimated_fdr=float(fdr_hat),
                q_trajectory=trajectory,
                method=method_n,
                fdr_control="knockoff_plus" if offset == 1 else "knockoff",
                random_state=random_state,
                backend=backend_name,
                metadata={
                    "n_samples": int(X_arr.shape[0]),
                    "n_features": int(X_arr.shape[1]),
                    "offset": int(offset),
                    "n_modelx_draws": 1,
                    "compat_mode": compat,
                    "lasso_cv_impl": lasso_impl,
                    "lasso_fast_profile": lasso_profile,
                    "modelx_shrinkage": None,
                    "modelx_smatrix_method": None,
                    "knockpy_sampler": knockpy_sampler,
                    "knockpy_sampler_method": knockpy_sampler_method,
                    "xk_source": "provided",
                },
            )

        X_np = np.asarray(_to_numpy(X_arr), dtype=np.float64)
        y_np = np.asarray(_to_numpy(y_arr), dtype=np.float64).reshape(-1)

        draw_specs = []
        if Xk is not None:
            Xk_np = np.asarray(_to_numpy(Xk), dtype=np.float64)
            if Xk_np.shape != X_np.shape:
                raise ValueError("Xk must have the same shape as X")
            draw_specs.append((Xk_np, random_state, {"xk_source": "provided"}))
        else:
            if knockpy_sampler is not None:
                Xk_draw = knockpy_sampler_dispatch(
                    knockpy_sampler,
                    X_np,
                    method=knockpy_sampler_method,
                    y=y_np,
                    random_state=random_state,
                )
                if Xk_draw is None:
                    raise NotImplementedError(
                        "Selected knockpy sampler interface is a placeholder (pass). "
                        "Implement the sampler before enabling this route."
                    )
                Xk_draw = np.asarray(Xk_draw, dtype=np.float64)
                if Xk_draw.shape != X_np.shape:
                    raise ValueError("Dispatched knockoff matrix must have the same shape as X")
                draw_specs.append(
                    (
                        Xk_draw,
                        random_state,
                        {
                            "xk_source": "generated_model_x_dispatch",
                            "knockpy_sampler": str(knockpy_sampler),
                            "knockpy_sampler_method": None
                            if knockpy_sampler_method is None
                            else str(knockpy_sampler_method),
                        },
                    )
                )
            else:
                n_modelx_draws = default_draws if validated_draws is None else validated_draws
                for draw_idx in range(n_modelx_draws):
                    draw_seed = _model_x_draw_seed(random_state, draw_idx)
                    Xk_draw, draw_meta = _build_model_x_knockoffs_knockpy_compat(
                        X_np,
                        random_state=draw_seed,
                        modelx_shrinkage=modelx_shrinkage,
                        modelx_smatrix_method=modelx_smatrix_method,
                    )
                    draw_specs.append((Xk_draw, draw_seed, draw_meta))

        W_acc = None
        method_n = "corr_diff"
        model_meta: Dict[str, Any] = {"xk_source": "generated_model_x"}
        for Xk_draw, draw_seed, draw_meta in draw_specs:
            W_draw, method_n = _compute_w_statistics(
                X_np,
                Xk_draw,
                y_np,
                method=method,
                xp=np,
                random_state=draw_seed,
                backend_name="numpy",
                lasso_cv_impl=lasso_impl,
                lasso_fast_profile=lasso_profile,
                lasso_knockpy_style=True,
            )
            W_acc = W_draw if W_acc is None else (W_acc + W_draw)
            model_meta.update(draw_meta)

        n_modelx_draws = int(len(draw_specs))
        W_np = np.asarray(W_acc / float(n_modelx_draws), dtype=np.float64)
        threshold, fdr_hat, trajectory = _knockoff_threshold_and_path(W_np, q=q_f, offset=offset)

        if np.isfinite(threshold):
            selected = np.where(W_np >= threshold)[0].astype(np.int64, copy=False)
        else:
            selected = np.asarray([], dtype=np.int64)

        return KnockoffResult(
            knockoff_type="model_x",
            selected_features=selected,
            W=W_np,
            threshold=float(threshold),
            q=float(q_f),
            estimated_fdr=float(fdr_hat),
            q_trajectory=trajectory,
            method=method_n,
            fdr_control="knockoff_plus" if offset == 1 else "knockoff",
            random_state=random_state,
            backend="numpy",
            metadata={
                "n_samples": int(X_np.shape[0]),
                "n_features": int(X_np.shape[1]),
                "offset": int(offset),
                "n_modelx_draws": int(n_modelx_draws),
                "compat_mode": compat,
                "lasso_cv_impl": lasso_impl,
                "lasso_fast_profile": lasso_profile,
                "modelx_shrinkage": str(modelx_shrinkage),
                "modelx_smatrix_method": str(modelx_smatrix_method),
                "knockpy_sampler": knockpy_sampler,
                "knockpy_sampler_method": knockpy_sampler_method,
                **model_meta,
            },
        )

    # statgpu default path
    if Xk is not None:
        X_work = X_arr
        X_knock = xp.asarray(Xk, dtype=xp.float64)
        if X_knock.shape != X_work.shape:
            raise ValueError("Xk must have the same shape as X")

        W_xp, method_n = _compute_w_statistics(
            X_work,
            X_knock,
            y_arr,
            method=method,
            xp=xp,
            random_state=random_state,
            backend_name=backend_name,
            lasso_cv_impl=lasso_impl,
            lasso_fast_profile=lasso_profile,
            lasso_knockpy_style=False,
        )
        n_modelx_draws = 1
        model_meta = {
            "xk_source": "provided",
            "covariance_shrinkage": None,
            "s_scale": None,
            "modelx_shrinkage": None,
            "modelx_smatrix_method": None,
        }
    else:
        X_std = _standardize_features_unit_variance(X_arr, xp)
        n_modelx_draws = default_draws if validated_draws is None else validated_draws

        W_acc = None
        method_n = "corr_diff"
        model_meta: Dict[str, Any] = {"xk_source": "generated_model_x"}
        for draw_idx in range(n_modelx_draws):
            draw_seed = _model_x_draw_seed(random_state, draw_idx)
            X_knock, _draw_meta = _build_model_x_knockoffs(
                X_std,
                random_state=draw_seed,
                xp=xp,
                covariance_shrinkage=float(modelx_covariance_shrinkage),
                s_scale=float(modelx_s_scale),
            )
            model_meta.update(_draw_meta)
            W_draw, method_n = _compute_w_statistics(
                X_std,
                X_knock,
                y_arr,
                method=method,
                xp=xp,
                random_state=draw_seed,
                backend_name=backend_name,
                lasso_cv_impl=lasso_impl,
                lasso_fast_profile=lasso_profile,
                lasso_knockpy_style=False,
            )
            W_acc = W_draw if W_acc is None else (W_acc + W_draw)

        W_xp = W_acc / float(n_modelx_draws)

    threshold, fdr_hat, trajectory = _knockoff_threshold_and_path(W_xp, q=q_f, offset=offset)

    if np.isfinite(threshold):
        selected_xp = xp.where(W_xp >= threshold)[0]
    else:
        selected_xp = xp.asarray([], dtype=xp.int64)

    W = _to_numpy(W_xp).astype(np.float64, copy=False)
    selected = _to_numpy(selected_xp).astype(np.int64, copy=False)

    return KnockoffResult(
        knockoff_type="model_x",
        selected_features=selected,
        W=W,
        threshold=float(threshold),
        q=float(q_f),
        estimated_fdr=float(fdr_hat),
        q_trajectory=trajectory,
        method=method_n,
        fdr_control="knockoff_plus" if offset == 1 else "knockoff",
        random_state=random_state,
        backend=backend_name,
        metadata={
            "n_samples": int(X_arr.shape[0]),
            "n_features": int(X_arr.shape[1]),
            "offset": int(offset),
            "n_modelx_draws": int(n_modelx_draws),
            "compat_mode": compat,
            "lasso_cv_impl": lasso_impl,
            "lasso_fast_profile": lasso_profile,
            "modelx_covariance_shrinkage": float(modelx_covariance_shrinkage),
            "modelx_s_scale": float(modelx_s_scale),
            **model_meta,
        },
    )


def knockoff_filter(
    X,
    y,
    knockoff_type: str = "fixed_x",
    q: float = 0.1,
    method: str = "corr_diff",
    fdr_control: str = "knockoff_plus",
    random_state: Optional[int] = None,
    backend: str = "auto",
    Xk=None,
    compat_mode: str = "statgpu",
    lasso_cv_impl: str = "auto",
    lasso_fast_profile: str = "off",
    modelx_covariance_shrinkage: float = 0.20,
    modelx_s_scale: float = 0.999,
    modelx_draws: Optional[int] = None,
    modelx_shrinkage: str = "ledoitwolf",
    modelx_smatrix_method: str = "mvr",
    knockpy_sampler: Optional[str] = None,
    knockpy_sampler_method: Optional[str] = None,
) -> KnockoffResult:
    """Unified knockoff entrypoint for fixed-X and model-X variants.

    Check ``np.isfinite(q) and 0 < q < 1`` before calling or fitting. Current
    validation misses NaN under both threshold rules and can return an invalid
    empty selection with threshold=inf and estimated_fdr=0.0 (an all-false
    selector support mask). This is not a valid no-discoveries result.

    Seeded ``lasso_coef_diff`` can reuse stale statistics after input mutation
    or memory reuse. See ``fixed_x_knockoff_filter`` and the feature-selection
    API reference for process-isolation and retained-input-copy workarounds.

    With compat_mode='statgpu' and Xk=None, native model-X construction uses
    X.device for its local random generator and random matrix, as fixed-X
    construction does. Torch CPU inputs remain on CPU even when CUDA is
    available; CUDA inputs retain their GPU index. Keep X/y/Xk on the same
    device; a supplied, externally validated Xk bypasses construction.
    An integer random_state seeds construction without advancing the global
    Torch RNG. Construction repeatability is scoped to the same backend,
    dtype, device and software environment, not cross-backend or cross-GPU
    equality or empirical FDR control. See model_x_knockoff_filter for
    random_state=None behavior, statistical requirements and the separate
    Lasso cache limitation.

    This device preservation is for construction: Torch method='lasso_coef_diff'
    tuning/fitting still requests CUDA, even for CPU inputs or supplied Xk,
    and does not guarantee the input GPU index. For Torch CPU statistics use
    corr_diff or ols_coef_diff when appropriate; for CPU Lasso use NumPy inputs
    with backend='numpy'.

    """
    kind = _normalize_knockoff_type(knockoff_type)
    if kind == "fixed_x":
        return fixed_x_knockoff_filter(
            X,
            y,
            q=q,
            method=method,
            fdr_control=fdr_control,
            random_state=random_state,
            backend=backend,
            Xk=Xk,
            compat_mode=compat_mode,
            lasso_cv_impl=lasso_cv_impl,
            lasso_fast_profile=lasso_fast_profile,
        )

    return model_x_knockoff_filter(
        X,
        y,
        q=q,
        method=method,
        fdr_control=fdr_control,
        random_state=random_state,
        backend=backend,
        Xk=Xk,
        compat_mode=compat_mode,
        lasso_cv_impl=lasso_cv_impl,
        lasso_fast_profile=lasso_fast_profile,
        modelx_covariance_shrinkage=modelx_covariance_shrinkage,
        modelx_s_scale=modelx_s_scale,
        modelx_draws=modelx_draws,
        modelx_shrinkage=modelx_shrinkage,
        modelx_smatrix_method=modelx_smatrix_method,
        knockpy_sampler=knockpy_sampler,
        knockpy_sampler_method=knockpy_sampler_method,
    )


class _KnockoffSelectorContract:
    """Shared sklearn and finite-input contract for knockoff selectors."""

    def _more_tags(self):
        return {"requires_y": True}

    def __sklearn_tags__(self):
        try:
            from sklearn.utils import Tags, TargetTags, TransformerTags
        except ImportError:
            return self._more_tags()
        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=True),
            transformer_tags=TransformerTags(),
            requires_fit=True,
        )

    def __sklearn_is_fitted__(self):
        return getattr(self, "selected_features_", None) is not None

    @staticmethod
    def _validate_fit_inputs(X, y, Xk=None):
        check_finite(X, name="X")
        check_finite(y, name="y")
        if Xk is not None:
            check_finite(Xk, name="Xk")

    @staticmethod
    def _validate_transform_input(X):
        check_finite(X, name="X")


class KnockoffSelector(_KnockoffSelectorContract):
    """Sklearn-like wrapper for unified knockoff feature selection.

    Check ``np.isfinite(q) and 0 < q < 1`` before calling or fitting. Current
    validation misses NaN under both threshold rules and can return an invalid
    empty selection with threshold=inf and estimated_fdr=0.0 (an all-false
    selector support mask). This is not a valid no-discoveries result.

    ``fit`` returns self; inspect ``result_`` and ``selected_features_``.
    Seeded ``lasso_coef_diff`` can reuse stale statistics across new instances
    after input mutation or memory reuse. See ``fixed_x_knockoff_filter`` and
    the feature-selection API reference for safe repeated-call workflows.

    With compat_mode='statgpu' and Xk=None, native model-X construction uses
    X.device for its local random generator and random matrix, as fixed-X
    construction does. Torch CPU inputs remain on CPU even when CUDA is
    available; CUDA inputs retain their GPU index. Keep X/y/Xk on the same
    device; a supplied, externally validated Xk bypasses construction.
    An integer random_state seeds construction without advancing the global
    Torch RNG. Construction repeatability is scoped to the same backend,
    dtype, device and software environment, not cross-backend or cross-GPU
    equality or empirical FDR control. See model_x_knockoff_filter for
    random_state=None behavior, statistical requirements and the separate
    Lasso cache limitation.

    This device preservation is for construction: Torch method='lasso_coef_diff'
    tuning/fitting still requests CUDA, even for CPU inputs or supplied Xk,
    and does not guarantee the input GPU index. For Torch CPU statistics use
    corr_diff or ols_coef_diff when appropriate; for CPU Lasso use NumPy inputs
    with backend='numpy'.

    """

    def __init__(
        self,
        knockoff_type: str = "fixed_x",
        q: float = 0.1,
        method: str = "corr_diff",
        fdr_control: str = "knockoff_plus",
        random_state: Optional[int] = None,
        backend: str = "auto",
        compat_mode: str = "statgpu",
        lasso_cv_impl: str = "auto",
        lasso_fast_profile: str = "off",
        modelx_covariance_shrinkage: float = 0.20,
        modelx_s_scale: float = 0.999,
        modelx_draws: Optional[int] = None,
        modelx_shrinkage: str = "ledoitwolf",
        modelx_smatrix_method: str = "mvr",
        knockpy_sampler: Optional[str] = None,
        knockpy_sampler_method: Optional[str] = None,
    ):
        self.knockoff_type = knockoff_type
        self.q = q
        self.method = method
        self.fdr_control = fdr_control
        self.random_state = random_state
        self.backend = backend
        self.compat_mode = compat_mode
        self.lasso_cv_impl = lasso_cv_impl
        self.lasso_fast_profile = lasso_fast_profile
        self.modelx_covariance_shrinkage = modelx_covariance_shrinkage
        self.modelx_s_scale = modelx_s_scale
        self.modelx_draws = modelx_draws
        self.modelx_shrinkage = modelx_shrinkage
        self.modelx_smatrix_method = modelx_smatrix_method
        self.knockpy_sampler = knockpy_sampler
        self.knockpy_sampler_method = knockpy_sampler_method

        self.result_: Optional[KnockoffResult] = None
        self.selected_features_: Optional[np.ndarray] = None

    def fit(self, X, y, Xk=None):
        self._validate_fit_inputs(X, y, Xk)
        self.result_ = knockoff_filter(
            X,
            y,
            knockoff_type=self.knockoff_type,
            q=self.q,
            method=self.method,
            fdr_control=self.fdr_control,
            random_state=self.random_state,
            backend=self.backend,
            Xk=Xk,
            compat_mode=self.compat_mode,
            lasso_cv_impl=self.lasso_cv_impl,
            lasso_fast_profile=self.lasso_fast_profile,
            modelx_covariance_shrinkage=self.modelx_covariance_shrinkage,
            modelx_s_scale=self.modelx_s_scale,
            modelx_draws=self.modelx_draws,
            modelx_shrinkage=self.modelx_shrinkage,
            modelx_smatrix_method=self.modelx_smatrix_method,
            knockpy_sampler=self.knockpy_sampler,
            knockpy_sampler_method=self.knockpy_sampler_method,
        )
        self.selected_features_ = self.result_.selected_features
        return self

    def get_support(self) -> np.ndarray:
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet")
        n_features = int(self.result_.W.shape[0])
        mask = np.zeros(n_features, dtype=bool)
        mask[self.selected_features_] = True
        return mask

    def transform(self, X):
        self._validate_transform_input(X)
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet")
        module = type(X).__module__
        if module.startswith("torch"):
            import torch

            X_arr = X
            indices = torch.as_tensor(
                self.selected_features_, dtype=torch.long, device=X.device
            )
        elif module.startswith("cupy"):
            import cupy as cp

            X_arr = X
            indices = cp.asarray(self.selected_features_, dtype=cp.int64)
        else:
            X_arr = np.asarray(X)
            indices = self.selected_features_
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        expected = int(self.result_.W.shape[0])
        if int(X_arr.shape[1]) != expected:
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but selector was fitted with {expected}"
            )
        return X_arr[:, indices]

    def fit_transform(self, X, y, Xk=None):
        return self.fit(X, y, Xk=Xk).transform(X)

    def get_params(self, deep=True):
        return {
            "knockoff_type": self.knockoff_type,
            "q": self.q,
            "method": self.method,
            "fdr_control": self.fdr_control,
            "random_state": self.random_state,
            "backend": self.backend,
            "compat_mode": self.compat_mode,
            "lasso_cv_impl": self.lasso_cv_impl,
            "lasso_fast_profile": self.lasso_fast_profile,
            "modelx_covariance_shrinkage": self.modelx_covariance_shrinkage,
            "modelx_s_scale": self.modelx_s_scale,
            "modelx_draws": self.modelx_draws,
            "modelx_shrinkage": self.modelx_shrinkage,
            "modelx_smatrix_method": self.modelx_smatrix_method,
            "knockpy_sampler": self.knockpy_sampler,
            "knockpy_sampler_method": self.knockpy_sampler_method,
        }

    def set_params(self, **params):
        if not params:
            return self
        valid = self.get_params(deep=False)
        unknown = [name for name in params if name not in valid]
        if unknown:
            name = unknown[0]
            raise ValueError(
                f"Invalid parameter {name!r} for KnockoffSelector. "
                f"Valid parameters are: {', '.join(sorted(valid))}."
            )
        updated = dict(valid)
        updated.update(params)
        fresh = type(self)(**updated)
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        return self


class FixedXKnockoffSelector(_KnockoffSelectorContract):
    """Sklearn-like wrapper for fixed-X knockoff feature selection.

    Check ``np.isfinite(q) and 0 < q < 1`` before calling or fitting. Current
    validation misses NaN under both threshold rules and can return an invalid
    empty selection with threshold=inf and estimated_fdr=0.0 (an all-false
    selector support mask). This is not a valid no-discoveries result.

    ``fit`` returns self; inspect ``result_`` and ``selected_features_``.
    A fresh selector does not prevent seeded ``lasso_coef_diff`` cache reuse
    after input mutation. See ``fixed_x_knockoff_filter`` for safe repeated calls.
    Torch ``method='lasso_coef_diff'`` tuning/fitting requests CUDA even for CPU
    inputs or supplied Xk; construction's input-device behavior does not extend
    to that statistic. For Torch CPU statistics use ``corr_diff`` or
    ``ols_coef_diff`` when appropriate; for CPU Lasso use NumPy inputs with
    ``backend='numpy'``.
    """

    def __init__(
        self,
        q: float = 0.1,
        method: str = "corr_diff",
        fdr_control: str = "knockoff_plus",
        random_state: Optional[int] = None,
        backend: str = "auto",
        compat_mode: str = "statgpu",
        lasso_cv_impl: str = "auto",
        lasso_fast_profile: str = "off",
    ):
        self._selector = KnockoffSelector(
            knockoff_type="fixed_x",
            q=q,
            method=method,
            fdr_control=fdr_control,
            random_state=random_state,
            backend=backend,
            compat_mode=compat_mode,
            lasso_cv_impl=lasso_cv_impl,
            lasso_fast_profile=lasso_fast_profile,
        )

        self.q = q
        self.method = method
        self.fdr_control = fdr_control
        self.random_state = random_state
        self.backend = backend
        self.compat_mode = compat_mode
        self.lasso_cv_impl = lasso_cv_impl
        self.lasso_fast_profile = lasso_fast_profile
        self.result_: Optional[KnockoffResult] = None
        self.selected_features_: Optional[np.ndarray] = None

    def fit(self, X, y, Xk=None):
        self._validate_fit_inputs(X, y, Xk)
        self._selector.fit(X, y, Xk=Xk)
        self.result_ = self._selector.result_
        self.selected_features_ = self._selector.selected_features_
        return self

    def get_support(self) -> np.ndarray:
        return self._selector.get_support()

    def transform(self, X):
        self._validate_transform_input(X)
        return self._selector.transform(X)

    def fit_transform(self, X, y, Xk=None):
        return self.fit(X, y, Xk=Xk).transform(X)

    def get_params(self, deep=True):
        return {
            "q": self.q,
            "method": self.method,
            "fdr_control": self.fdr_control,
            "random_state": self.random_state,
            "backend": self.backend,
            "compat_mode": self.compat_mode,
            "lasso_cv_impl": self.lasso_cv_impl,
            "lasso_fast_profile": self.lasso_fast_profile,
        }

    def set_params(self, **params):
        if not params:
            return self
        valid = self.get_params(deep=False)
        unknown = [name for name in params if name not in valid]
        if unknown:
            name = unknown[0]
            raise ValueError(
                f"Invalid parameter {name!r} for FixedXKnockoffSelector. "
                f"Valid parameters are: {', '.join(sorted(valid))}."
            )
        updated = dict(valid)
        updated.update(params)
        fresh = type(self)(**updated)
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        return self




for _selector_cls in (KnockoffSelector, FixedXKnockoffSelector):
    for _method_name in ("fit", "fit_transform", "transform"):
        _method = getattr(_selector_cls, _method_name, None)
        if callable(_method):
            _method.__statgpu_finite_validation__ = True
del _selector_cls, _method_name, _method
