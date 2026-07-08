"""
M-estimation sandwich covariance inference engine.

Provides backend-agnostic (NumPy/CuPy/Torch) functions for computing
model-based and sandwich (HC0/HC1) covariance matrices, standard errors,
z-statistics, p-values, and confidence intervals from any loss function
with ``has_hessian=True``.

All math uses **average-scale** convention matching ``loss.hessian()``
which returns ``X'WX / n``:

    H_avg = X'WX / n
    J_avg = (1/n) * sum_i psi_i psi_i'
    cov   = H_avg^{-1} @ J_avg @ H_avg^{-1} / n

Where ``psi_i = per_sample_gradient(eta_i, y_i) * x_i``.

References
----------
- White, H. (1980). "A Heteroskedasticity-Consistent Covariance Matrix
  Estimator and a Direct Test for Heteroskedasticity." Econometrica.
- MacKinnon, J.G. & White, H. (1985). "Some Heteroskedasticity-Consistent
  Covariance Matrix Estimators with Improved Finite Sample Properties."
  Journal of Econometrics.
"""

__all__ = [
    "compute_bread_avg",
    "compute_meat_avg",
    "assemble_cov_avg",
    "m_estimation_inference",
]

from typing import Optional, Dict, Any
import numpy as np


def _resolve_backend_and_xp(X):
    """Resolve backend (numpy/cupy/torch) and return xp module."""
    from statgpu.backends import _resolve_backend
    from statgpu.backends._utils import _get_xp

    backend = _resolve_backend("auto", X)
    xp = _get_xp(backend)
    return backend, xp


def _infer_covariance_convention(cov_type: str, has_curvature: bool) -> str:
    """Map (cov_type, has_curvature) to a covariance convention label."""
    if cov_type == "nonrobust":
        return "penalized_information" if has_curvature else "model_based_nonrobust"
    else:
        return "penalized_sandwich" if has_curvature else "robust_sandwich"


# ---------------------------------------------------------------------------
# Bread: inverse of (scaled) Hessian
# ---------------------------------------------------------------------------

def compute_bread_avg(
    loss,
    X,
    y,
    coef,
    *,
    penalty_curvature_diag=None,
    sample_weight=None,
    use_fisher=False,
) -> np.ndarray:
    """Compute bread = (H_avg + diag(penalty_curvature_diag))^{-1}.

    Parameters
    ----------
    loss : LossBase
        Loss function with ``has_hessian=True``.
    X : ndarray (n, p)
        Design matrix (aligned with coef, including intercept if applicable).
    y : ndarray (n,)
        Response vector.  Ignored when ``use_fisher=True``.
    coef : ndarray (p,)
        Coefficient vector (aligned with X columns).
    penalty_curvature_diag : ndarray (p,) or None
        Diagonal of the penalty Hessian P''(coef) at average scale.
    sample_weight : ndarray (n,) or None
    use_fisher : bool
        If True, use ``loss.fisher_information()`` (expected Fisher, no y-dependence).
        Falls back to ``loss.hessian()`` if Fisher is not implemented.

    Returns
    -------
    bread_avg : ndarray (p, p)
        (H_avg + diag(penalty_curvature_diag))^{-1}
    """
    _, xp = _resolve_backend_and_xp(X)

    if use_fisher and hasattr(loss, 'fisher_information'):
        try:
            H_avg = loss.fisher_information(X, coef, sample_weight=sample_weight)
        except NotImplementedError:
            H_avg = loss.hessian(X, y, coef, sample_weight=sample_weight)
    else:
        H_avg = loss.hessian(X, y, coef, sample_weight=sample_weight)

    if penalty_curvature_diag is not None:
        curv = xp.asarray(penalty_curvature_diag, dtype=H_avg.dtype)
        H_avg = H_avg + xp.diag(curv)

    # Solve H_avg @ bread = I via backend-native solver
    p = H_avg.shape[0]
    from statgpu.backends._utils import xp_eye
    eye = xp_eye(p, H_avg.dtype, xp, ref_arr=H_avg)
    try:
        bread_avg = xp.linalg.solve(H_avg, eye)
    except (np.linalg.LinAlgError, RuntimeError) as e:
        # numpy/cupy raise LinAlgError; torch raises RuntimeError
        raise np.linalg.LinAlgError(
            "Singular Hessian in compute_bread_avg. "
            "The design matrix may be rank-deficient or the penalty is too weak. "
            "Consider adding ridge regularization or checking for collinear features."
        ) from e
    except Exception as e:
        # CuPy may raise bare Exception for cuSOLVER failures
        raise np.linalg.LinAlgError(
            "Singular Hessian in compute_bread_avg. "
            "The design matrix may be rank-deficient or the penalty is too weak. "
            "Consider adding ridge regularization or checking for collinear features."
        ) from e
    return bread_avg


# ---------------------------------------------------------------------------
# Meat: average outer product of scores
# ---------------------------------------------------------------------------

def compute_meat_avg(
    loss,
    X,
    y,
    coef,
    *,
    cov_type="hc0",
    bread_avg=None,
    sample_weight=None,
) -> np.ndarray:
    """Return average score outer product J_avg.

    Unweighted:
        J_avg = sum_i psi_i psi_i' / n

    Analytic weights (matches ``LossBase.gradient`` convention):
        J_avg = sum_i w_i^2 psi_i psi_i' / sum_i w_i

    HC0/HC1 use memory-efficient ``score_outer()``; HC2/HC3/HAC may
    materialize the full (n, p) score matrix.

    Parameters
    ----------
    loss : LossBase
    X : ndarray (n, p)
    y : ndarray (n,)
    coef : ndarray (p,)
    cov_type : str
        One of ``"hc0"``, ``"hc1"``, ``"hc2"``, ``"hc3"``, ``"hac"``.
        HC2/HC3/HAC require ``bread_avg`` for leverage/lag computation.
    bread_avg : ndarray (p, p) or None
        Required for HC2/HC3/HAC.
    sample_weight : ndarray (n,) or None

    Returns
    -------
    J_avg : ndarray (p, p)
    """
    _, xp = _resolve_backend_and_xp(X)

    if cov_type in ("hc2", "hc3"):
        raise NotImplementedError(
            f"HC2/HC3 requires leverage computation; not yet implemented for {loss.name}"
        )
    if cov_type == "hac":
        raise NotImplementedError(
            f"HAC requires lag selection; not yet implemented for {loss.name}"
        )

    # HC0/HC1: memory-efficient via score_outer
    J_sum = loss.score_outer(X, y, coef, sample_weight=sample_weight)
    n_eff = (
        float(xp.sum(sample_weight))
        if sample_weight is not None
        else X.shape[0]
    )
    return J_sum / n_eff


# ---------------------------------------------------------------------------
# Assemble covariance
# ---------------------------------------------------------------------------

def assemble_cov_avg(
    bread_avg,
    meat_avg,
    n_eff,
    k,
    cov_type,
) -> np.ndarray:
    """Assemble covariance: cov = bread_avg @ meat_avg @ bread_avg / n_eff.

    HC1: multiply by n_eff / (n_eff - k).

    Parameters
    ----------
    bread_avg : ndarray (p, p)
    meat_avg : ndarray (p, p)
    n_eff : int
        Effective sample size (n or sum(sample_weight)).
    k : int
        Number of parameters (including intercept if applicable).
    cov_type : str

    Returns
    -------
    cov : ndarray (p, p)
    """
    _, xp = _resolve_backend_and_xp(bread_avg)

    cov = bread_avg @ meat_avg @ bread_avg / n_eff

    if cov_type == "hc1" and n_eff > k:
        cov = cov * (n_eff / (n_eff - k))

    return cov


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------

def m_estimation_inference(
    loss,
    X,
    y,
    coef,
    *,
    cov_type="nonrobust",
    penalty_curvature_diag=None,
    dispersion=None,
    sample_weight=None,
    hac_maxlags=None,
) -> Dict[str, Any]:
    """Full M-estimation inference pipeline.

    Supports BOTH model-based and sandwich covariance:

    - ``cov_type="nonrobust"``: model-based covariance φ·H⁻¹/n
      (skips meat computation entirely)
    - ``cov_type="hc0"``, ``"hc1"``: robust sandwich H⁻¹·J·H⁻¹/n

    Wald test::

        wald = coef' @ cov^{-1} @ coef ~ chi2(k)

    Parameters
    ----------
    loss : LossBase
        Loss function with ``has_hessian=True``.
    X : ndarray (n, p)
        Design matrix aligned with coef.
    y : ndarray (n,)
        Response vector.
    coef : ndarray (p,)
        Coefficient vector.
    cov_type : str
        ``"nonrobust"``, ``"hc0"``, or ``"hc1"``.
    penalty_curvature_diag : ndarray (p,) or None
        Diagonal of penalty Hessian at average scale.
    dispersion : float or None
        Dispersion parameter φ. If None, computed from the loss.
    sample_weight : ndarray (n,) or None
    hac_maxlags : int or None
        Maximum lags for HAC (not yet implemented for non-Gaussian).

    Returns
    -------
    result : dict
        Keys: bse, statistic, pvalues, conf_int, cov_params,
        dispersion, wald_stat, wald_pval, distribution.
    """
    cov_type = cov_type.lower()
    _VALID_COV = {"nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"}
    if cov_type not in _VALID_COV:
        raise ValueError(
            f"Unknown cov_type='{cov_type}'. "
            f"Valid options: {sorted(_VALID_COV)}."
        )
    if cov_type in {"hc2", "hc3"}:
        raise NotImplementedError(
            "HC2/HC3 requires leverage computation; not yet implemented."
        )
    if cov_type == "hac":
        raise NotImplementedError(
            "HAC requires lag selection; not yet implemented."
        )
    _, xp = _resolve_backend_and_xp(X)

    n_eff = float(xp.sum(sample_weight)) if sample_weight is not None else X.shape[0]
    k = int(coef.shape[0])

    # ---- bread ----
    # nonrobust: prefer expected Fisher (matches statsmodels/R summary.glm)
    # sandwich (hc0/hc1): use observed Hessian (standard M-estimation)
    use_fisher = (cov_type == "nonrobust")
    bread_avg = compute_bread_avg(
        loss, X, y, coef,
        penalty_curvature_diag=penalty_curvature_diag,
        sample_weight=sample_weight,
        use_fisher=use_fisher,
    )

    # ---- dispersion (for nonrobust) ----
    if dispersion is None and cov_type == "nonrobust":
        dispersion = _default_dispersion(loss, X, y, coef, n_eff, k)

    # ---- covariance ----
    if cov_type == "nonrobust":
        # Convention A: penalized-information, skip meat
        cov = dispersion * bread_avg / n_eff
    else:
        meat_avg = compute_meat_avg(
            loss, X, y, coef,
            cov_type=cov_type,
            bread_avg=bread_avg,
            sample_weight=sample_weight,
        )
        cov = assemble_cov_avg(bread_avg, meat_avg, n_eff, k, cov_type)

    # ---- standard errors ----
    cov_diag = xp.diag(cov)
    from statgpu.backends._array_ops import _clip
    cov_diag = _clip(cov_diag, 0.0, None)
    bse = xp.sqrt(cov_diag)

    # ---- z-statistics ----
    z_values = coef / (bse + 1e-30)

    # ---- p-values (two-sided normal) ----
    pvalues = _two_sided_pvalue(xp, z_values)

    # ---- confidence intervals (95%) ----
    z_crit = _normal_critical_value(xp, 0.05)
    conf_int = xp.stack([
        coef - z_crit * bse,
        coef + z_crit * bse,
    ], axis=1)

    # ---- Wald test ----
    try:
        # wald = coef' @ cov^{-1} @ coef via solve
        wald_vec = xp.linalg.solve(cov, coef)
        wald_stat = float(xp.dot(coef, wald_vec))
    except (np.linalg.LinAlgError, RuntimeError):
        wald_stat = float("nan")
    from math import isnan as _math_isnan
    wald_pval = _chi2_sf(xp, wald_stat, k) if not _math_isnan(wald_stat) else float("nan")

    # ---- distribution label ----
    distribution = "normal"

    return {
        "bse": _to_numpy_safe(bse),
        "statistic": _to_numpy_safe(z_values),
        "pvalues": _to_numpy_safe(pvalues),
        "conf_int": _to_numpy_safe(conf_int),
        "cov_params": _to_numpy_safe(cov),
        "dispersion": float(dispersion) if dispersion is not None else None,
        "wald_stat": wald_stat,
        "wald_pval": wald_pval,
        "distribution": distribution,
        "hessian_type": "expected_fisher" if use_fisher else "observed",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_dispersion(loss, X, y, coef, n_eff, k):
    """Default dispersion for nonrobust covariance.  Backend-agnostic.

    Canonical-link GLMs (Poisson, logistic, NegBinom): = 1.0.
    Gaussian: = RSS / (n - k).
    Non-canonical GLMs (Gamma, InvGaussian, Tweedie): Pearson chi2 / df_resid.
    Other: = 1.0.
    """
    name = getattr(loss, 'name', '')
    if name in ("squared_error",):
        _, xp = _resolve_backend_and_xp(X)
        eta = X @ coef; mu = eta
        resid = y - mu; rss = float(xp.sum(resid ** 2))
        return rss / max(n_eff - k, 1)

    # Pearson dispersion for non-canonical GLMs (backend-agnostic)
    if name in ("gamma", "inverse_gaussian", "tweedie"):
        _, xp = _resolve_backend_and_xp(X)
        df = max(n_eff - k, 1)
        if hasattr(loss, '_mu_from_eta'):
            eta = X @ coef; mu = loss._mu_from_eta(eta)
        else:
            mu = xp.exp(X @ coef)  # log link default
        # Variance function V(mu) per family
        if name == "gamma":
            V = mu ** 2
        elif name == "inverse_gaussian":
            V = mu ** 3
        elif name == "tweedie":
            p = getattr(loss, 'power', 1.5)
            V = mu ** p
        else:
            return 1.0
        resid_sq = (y - mu) ** 2
        from statgpu.backends._utils import xp_maximum
        pearson = float(xp.sum(resid_sq / xp_maximum(V, 1e-10, xp)))
        return pearson / df

    return 1.0


def _two_sided_pvalue(xp, z_values):
    """Two-sided p-value from standard normal. Backend-agnostic."""
    if xp.__name__ == "torch":
        import torch
        abs_z = xp.abs(z_values)
        from statgpu.inference._distributions_backend import norm
        p = 2.0 * norm.sf(abs_z)
        return p if isinstance(p, torch.Tensor) else torch.as_tensor(p, dtype=z_values.dtype, device=z_values.device)
    elif xp.__name__ == "cupy":
        from statgpu.inference._distributions_backend import norm
        return 2.0 * norm.sf(xp.abs(z_values))
    else:
        from statgpu.inference._distributions_backend import get_distribution
        _norm = get_distribution("norm", backend="numpy")
        return 2.0 * _norm.sf(np.abs(np.asarray(z_values)))


def _normal_critical_value(xp, alpha):
    """Two-sided critical value for (1-alpha) CI."""
    if xp.__name__ == "torch":
        from statgpu.inference._distributions_backend import norm
        import torch
        z = norm.ppf(1.0 - alpha / 2.0)
        return z if isinstance(z, torch.Tensor) else torch.as_tensor(z, dtype=torch.float64)
    elif xp.__name__ == "cupy":
        from statgpu.inference._distributions_backend import norm
        return xp.asarray(norm.ppf(1.0 - alpha / 2.0))
    else:
        from statgpu.inference._distributions_backend import get_distribution
        _norm = get_distribution("norm", backend="numpy")
        return _norm.ppf(1.0 - alpha / 2.0)


def _chi2_sf(xp, x, df):
    """Survival function of chi2(df) at x."""
    if xp.__name__ == "torch":
        from statgpu.inference._distributions_backend import chi2
        import torch
        x_t = torch.as_tensor(float(x), dtype=torch.float64)
        return float(chi2.sf(x_t, df=df))
    elif xp.__name__ == "cupy":
        from statgpu.inference._distributions_backend import chi2
        return float(chi2.sf(float(x), df=df))
    else:
        from statgpu.inference._distributions_backend import get_distribution
        _chi2 = get_distribution("chi2", backend="numpy")
        return float(_chi2.sf(float(x), df=df))


def _to_numpy_safe(arr):
    """Convert array to NumPy, handling cupy/torch."""
    if arr is None:
        return None
    xp_module = type(arr).__module__
    if "cupy" in xp_module:
        return arr.get()
    if "torch" in xp_module:
        return arr.detach().cpu().numpy()
    return np.asarray(arr)
