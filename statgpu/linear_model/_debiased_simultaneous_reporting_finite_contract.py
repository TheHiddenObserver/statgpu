"""Fail-closed feature-only debiased simultaneous inference.

The centered/intercept-inclusive CPU path and the centered CuPy/Torch-native
path already reject non-finite bootstrap max-|Z| draws before quantile
calibration. The historical feature-only reporting-stage helper predates that
contract and can let non-finite draws reach ``np.quantile``. A post-hoc check is
not sufficient because a small number of infinite draws can lie above the
requested quantile while the published critical value remains finite.

Preserve the historical feature-only RNG, multiplier score, target-family, and
quantile formulas, but execute that small reporting-stage helper through an
otherwise equivalent fail-closed implementation that validates every max-|Z|
draw before calibration. Intercept-inclusive centered inference continues to
delegate to its existing corrected implementation, and centered CuPy/Torch
simultaneous numerics continue to bypass this CPU reporting helper entirely.
"""

from __future__ import annotations

import functools

import numpy as np

from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel


_SIMULTANEOUS_REPORTING_FINITE_MARKER = (
    "__statgpu_pr138_simultaneous_reporting_finite__"
)
_ORIGINAL_SIMULTANEOUS = (
    PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap
)


def _clear_failed_simultaneous_publication(model) -> None:
    """Remove result-bearing joint state after a rejected publication."""
    model._conf_int_simultaneous = None
    model._simultaneous_enabled = False
    for name in (
        "_simultaneous_critical_value",
        "_simultaneous_target_mask",
    ):
        model.__dict__.pop(name, None)


def _validate_simultaneous_publication(model) -> None:
    """Reject a non-representable joint result after numerical calibration."""
    if not bool(getattr(model, "_simultaneous_enabled", False)):
        return

    critical = getattr(model, "_simultaneous_critical_value", None)
    conf_int = getattr(model, "_conf_int_simultaneous", None)
    valid_critical = False
    try:
        critical_value = float(critical)
        valid_critical = np.isfinite(critical_value) and critical_value >= 0.0
    except (TypeError, ValueError, OverflowError):
        valid_critical = False

    valid_conf_int = False
    if conf_int is not None:
        try:
            conf_arr = np.asarray(conf_int, dtype=np.float64)
            valid_conf_int = (
                conf_arr.ndim == 2
                and conf_arr.shape[1] == 2
                and bool(np.all(np.isfinite(conf_arr)))
                and bool(np.all(conf_arr[:, 0] <= conf_arr[:, 1]))
            )
        except (TypeError, ValueError, OverflowError):
            valid_conf_int = False

    if valid_critical and valid_conf_int:
        return

    _clear_failed_simultaneous_publication(model)
    if not valid_critical:
        raise FloatingPointError(
            "simultaneous debiased inference produced a non-finite or negative "
            "critical value"
        )
    raise FloatingPointError(
        "simultaneous debiased inference produced non-finite, malformed, or "
        "reversed confidence intervals"
    )


def _compute_feature_only_simultaneous_fail_closed(self):
    """Historical feature-only max-|Z| calculation plus a pre-quantile finite gate."""
    if self._debiased_M_cpu is None:
        return
    if self._y is None or self._resid is None or self._bse is None:
        return

    X = self._X_design
    if X is None:
        return
    X = np.asarray(X, dtype=np.float64)
    if bool(self._effective_intercept):
        X_feat = X[:, 1:]
    else:
        X_feat = X
    n, p = X_feat.shape

    M = np.asarray(self._debiased_M_cpu, dtype=np.float64)
    if M.shape != (p, p):
        raise RuntimeError(
            "debiased precision matrix does not match the simultaneous feature design"
        )
    resid = np.asarray(self._resid, dtype=np.float64).reshape(-1)
    if resid.shape[0] != n:
        raise RuntimeError(
            "simultaneous residual state does not match the feature design"
        )

    params = np.asarray(self._params, dtype=np.float64).reshape(-1)
    bse = np.asarray(self._bse, dtype=np.float64).reshape(-1)
    offset = 1 if bool(self._effective_intercept) else 0
    if params.shape[0] != p + offset or bse.shape[0] != p + offset:
        raise RuntimeError(
            "simultaneous feature target layout does not match debiased parameters"
        )

    # Preserve the maintained target semantics. This helper is entered when the
    # reported intercept is not part of the calibrated family (or no intercept
    # exists), so every feature row remains a simultaneous target.
    param_target_idx = np.arange(offset, len(params), dtype=int)
    feature_target_idx = param_target_idx - offset
    if feature_target_idx.size == 0:
        return
    se_feat = bse[offset:]

    alpha_sim = float(
        getattr(
            self,
            "simultaneous_alpha",
            getattr(self, "_simultaneous_alpha", 0.05),
        )
    )
    if not np.isfinite(alpha_sim) or not (0.0 < alpha_sim < 1.0):
        raise ValueError("simultaneous_alpha must lie strictly between 0 and 1")
    B = int(
        getattr(
            self,
            "simultaneous_n_bootstrap",
            getattr(self, "_simultaneous_n_bootstrap", 1000),
        )
    )
    if B <= 0:
        raise ValueError("simultaneous_n_bootstrap must be positive")
    rng = np.random.default_rng(
        getattr(
            self,
            "simultaneous_random_state",
            getattr(self, "_simultaneous_random_state", None),
        )
    )

    chunk = min(256, B)
    max_stats = np.empty(B, dtype=np.float64)
    filled = 0
    while filled < B:
        bsz = min(chunk, B - filled)
        xi = rng.standard_normal(size=(bsz, n))
        weighted = xi * resid.reshape(1, -1)
        score = (weighted @ X_feat) @ M.T / float(max(n, 1))
        z_star = score / (se_feat.reshape(1, -1) + 1e-30)
        max_stats[filled : filled + bsz] = np.max(
            np.abs(z_star[:, feature_target_idx]),
            axis=1,
        )
        filled += bsz

    if not np.all(np.isfinite(max_stats)):
        _clear_failed_simultaneous_publication(self)
        raise FloatingPointError(
            "simultaneous debiased inference produced non-finite bootstrap "
            "max-|Z| statistics"
        )

    critical = float(np.quantile(max_stats, 1.0 - alpha_sim))
    if not np.isfinite(critical) or critical < 0.0:
        _clear_failed_simultaneous_publication(self)
        raise FloatingPointError(
            "simultaneous debiased inference produced a non-finite or negative "
            "critical value"
        )

    conf_sim = np.array(self._conf_int, copy=True, dtype=np.float64)
    conf_sim[param_target_idx, 0] = (
        params[param_target_idx] - critical * bse[param_target_idx]
    )
    conf_sim[param_target_idx, 1] = (
        params[param_target_idx] + critical * bse[param_target_idx]
    )
    if not np.all(np.isfinite(conf_sim)) or np.any(conf_sim[:, 0] > conf_sim[:, 1]):
        _clear_failed_simultaneous_publication(self)
        raise FloatingPointError(
            "simultaneous debiased inference produced non-finite or reversed "
            "confidence intervals"
        )

    self._conf_int_simultaneous = conf_sim
    self._simultaneous_critical_value = critical
    self._simultaneous_enabled = True


@functools.wraps(_ORIGINAL_SIMULTANEOUS)
def _compute_simultaneous_ci_maxz_bootstrap(self):
    include_intercept = bool(
        getattr(
            self,
            "simultaneous_include_intercept",
            getattr(self, "_simultaneous_include_intercept", False),
        )
    )
    if include_intercept and bool(self._effective_intercept):
        result = _ORIGINAL_SIMULTANEOUS(self)
    else:
        result = _compute_feature_only_simultaneous_fail_closed(self)
    _validate_simultaneous_publication(self)
    return result


def install_debiased_simultaneous_reporting_finite_contract() -> None:
    current = PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap
    if getattr(current, _SIMULTANEOUS_REPORTING_FINITE_MARKER, False):
        return
    setattr(
        _compute_simultaneous_ci_maxz_bootstrap,
        _SIMULTANEOUS_REPORTING_FINITE_MARKER,
        True,
    )
    PenalizedGeneralizedLinearModel._compute_simultaneous_ci_maxz_bootstrap = (
        _compute_simultaneous_ci_maxz_bootstrap
    )


__all__ = [
    "install_debiased_simultaneous_reporting_finite_contract",
    "_validate_simultaneous_publication",
]
