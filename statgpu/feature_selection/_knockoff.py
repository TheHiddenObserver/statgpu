"""Fixed-X knockoff feature selection skeleton (CPU/GPU)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import _knockoff_utils as _kutils
from ._knockoff_utils import (
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
    pass


def knockpy_gaussian_sdp_sampler(
    X,
    *,
    mu=None,
    Sigma=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy GaussianSampler(method='sdp')."""
    pass


def knockpy_gaussian_maxent_sampler(
    X,
    *,
    mu=None,
    Sigma=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy GaussianSampler(method='maxent')."""
    pass


def knockpy_gaussian_equi_sampler(
    X,
    *,
    mu=None,
    Sigma=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy GaussianSampler(method='equi')."""
    pass


def knockpy_gaussian_ci_sampler(
    X,
    *,
    mu=None,
    Sigma=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy GaussianSampler(method='ci')."""
    pass


def knockpy_fx_sampler(
    X,
    *,
    y=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy FXSampler."""
    pass


def knockpy_metro_sampler(
    X,
    *,
    y=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy MetroSampler."""
    pass


def knockpy_artk_sampler(
    X,
    *,
    y=None,
    groups=None,
    random_state: Optional[int] = None,
):
    """Interface placeholder for knockpy ARTKSampler."""
    pass


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
    Fixed-X knockoff selection skeleton.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix.
    y : array-like of shape (n_samples,)
        Response vector.
    q : float, default=0.1
        Target FDR level in (0, 1).
    method : {'corr_diff', 'ols_coef_diff', 'lasso_coef_diff'}, default='corr_diff'
        Feature-importance statistic for W construction.
    fdr_control : {'knockoff_plus', 'knockoff'}, default='knockoff_plus'
        Knockoff threshold variant.
    random_state : int, optional
        Random seed for knockoff construction.
    backend : {'auto', 'numpy', 'cupy'}, default='auto'
        Compute backend. ``'auto'`` infers from input arrays.

    Returns
    -------
    KnockoffResult
        Selected feature indices and full knockoff diagnostics.
    """
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

    This implementation estimates a Gaussian feature model and builds
    equi-correlated knockoffs from the estimated covariance.
    """
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
                n_modelx_draws = max(1, int(default_draws if modelx_draws is None else modelx_draws))
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
        W_np = np.asarray(W_acc / float(max(1, n_modelx_draws)), dtype=np.float64)
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
        n_modelx_draws = max(1, int(default_draws if modelx_draws is None else modelx_draws))

        W_acc = None
        method_n = "corr_diff"
        model_meta: Dict[str, Any] = {"xk_source": "generated_model_x"}
        for draw_idx in range(n_modelx_draws):
            draw_seed = _model_x_draw_seed(random_state, draw_idx)
            X_knock, model_meta = _build_model_x_knockoffs(
                X_std,
                random_state=draw_seed,
                xp=xp,
                covariance_shrinkage=float(modelx_covariance_shrinkage),
                s_scale=float(modelx_s_scale),
            )
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
    """Unified knockoff entrypoint for fixed-X and model-X variants."""
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


class KnockoffSelector:
    """Sklearn-like wrapper for unified knockoff feature selection."""

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
        if self.selected_features_ is None:
            raise RuntimeError("Selector has not been fitted yet")
        X_arr = np.asarray(X)
        return X_arr[:, self.selected_features_]

    def fit_transform(self, X, y, Xk=None):
        return self.fit(X, y, Xk=Xk).transform(X)


class FixedXKnockoffSelector:
    """Sklearn-like wrapper for fixed-X knockoff feature selection."""

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
        self._selector.fit(X, y, Xk=Xk)
        self.result_ = self._selector.result_
        self.selected_features_ = self._selector.selected_features_
        return self

    def get_support(self) -> np.ndarray:
        return self._selector.get_support()

    def transform(self, X):
        return self._selector.transform(X)

    def fit_transform(self, X, y, Xk=None):
        return self.fit(X, y, Xk=Xk).transform(X)


