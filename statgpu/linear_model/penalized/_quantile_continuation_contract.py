"""Mark estimator-generated Quantile continuation paths for objective alignment.

The historical path generator remains authoritative for unweighted and exactly
uniform-weight Quantile fits. For genuinely non-uniform analytic weights, the
weighted start is resolved later by the public Quantile solver from backend-
native X/y/weights. This contract therefore avoids first materializing the same
full design and response on NumPy merely to construct path metadata that the
weighted resolver will replace.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import wraps

import numpy as np

from statgpu.backends._array_ops import _xp as _get_xp
from statgpu.solvers._quantile_continuation import (
    mark_auto_quantile_continuation_path,
)
from . import _fit_mixin
from ._fit_mixin import _PenalizedFitMixin


_MARKER = "_statgpu_quantile_continuation_path_contract"
_FIT_MARKER = "_statgpu_quantile_continuation_weight_context_contract"
_QUANTILE_SAMPLE_WEIGHT = ContextVar(
    "statgpu_quantile_continuation_sample_weight", default=None
)


def _is_nonuniform_weight(sample_weight) -> bool:
    """Check exact weight uniformity with only a scalar backend synchronization."""
    if sample_weight is None:
        return False
    module = type(sample_weight).__module__
    if module.startswith("torch") or module.startswith("cupy"):
        values = sample_weight.reshape(-1)
    else:
        values = np.asarray(sample_weight).reshape(-1)
    size = int(values.numel()) if hasattr(values, "numel") else int(values.size)
    if size <= 1:
        return False
    xp = _get_xp(values)
    uniform = xp.all(values == values[0])
    return not bool(uniform.item() if hasattr(uniform, "item") else uniform)


def _quantile_path_metadata(self, n_cont=None):
    """Build only continuation shape/target metadata; start is resolved later."""
    if n_cont is None:
        n_cont = _fit_mixin._N_CONT_STEPS_NONSMOOTH
    n_cont = int(n_cont)
    target_alpha = float(getattr(self._penalty, "alpha", self.alpha))
    if target_alpha > 0.0 and np.isfinite(target_alpha):
        alpha_path = np.geomspace(target_alpha * 1.1, target_alpha, n_cont)
    else:
        alpha_path = np.linspace(max(target_alpha, 0.0), target_alpha, n_cont)
    alpha_path = mark_auto_quantile_continuation_path(alpha_path)
    max_lla_per_step = max(
        _fit_mixin._MAX_LLA_PER_STEP_DEFAULT,
        getattr(self, "_max_lla_iters", 50) // max(n_cont, 1),
    )
    saved_max_iter = self._max_iter
    mi_path = [
        saved_max_iter if i == n_cont - 1 else max(100, saved_max_iter // 10)
        for i in range(n_cont)
    ]
    return alpha_path, max_lla_per_step, mi_path


def _install_fit_weight_context() -> None:
    current = _PenalizedFitMixin._fit_loss_backend
    if getattr(current, _FIT_MARKER, False):
        return

    @wraps(current)
    def _fit_loss_backend_with_quantile_weight_context(
        self, X, y, sample_weight, solver_name, backend_name
    ):
        if str(getattr(getattr(self, "_loss", None), "name", "")).lower() != "quantile":
            return current(self, X, y, sample_weight, solver_name, backend_name)
        token = _QUANTILE_SAMPLE_WEIGHT.set(sample_weight)
        try:
            return current(self, X, y, sample_weight, solver_name, backend_name)
        finally:
            _QUANTILE_SAMPLE_WEIGHT.reset(token)

    setattr(_fit_loss_backend_with_quantile_weight_context, _FIT_MARKER, True)
    _fit_loss_backend_with_quantile_weight_context._statgpu_original = current
    _PenalizedFitMixin._fit_loss_backend = _fit_loss_backend_with_quantile_weight_context


def install_quantile_continuation_contract() -> None:
    current = _PenalizedFitMixin._compute_lla_path
    if not getattr(current, _MARKER, False):

        @wraps(current)
        def _compute_lla_path_with_quantile_provenance(
            self,
            X_work,
            y_arr,
            p,
            loss_name,
            n_cont=None,
        ):
            if (
                str(loss_name or "").lower() == "quantile"
                and _is_nonuniform_weight(_QUANTILE_SAMPLE_WEIGHT.get())
            ):
                # The weighted resolver will compute the actual start from the
                # backend-native objective. Avoid the historical full X/y host
                # snapshot whose numeric start would immediately be discarded.
                return _quantile_path_metadata(self, n_cont=n_cont)

            alpha_path, max_lla_per_step, mi_path = current(
                self,
                X_work,
                y_arr,
                p,
                loss_name,
                n_cont=n_cont,
            )
            if str(loss_name or "").lower() == "quantile":
                alpha_path = mark_auto_quantile_continuation_path(alpha_path)
            return alpha_path, max_lla_per_step, mi_path

        setattr(_compute_lla_path_with_quantile_provenance, _MARKER, True)
        _compute_lla_path_with_quantile_provenance._statgpu_original = current
        _PenalizedFitMixin._compute_lla_path = _compute_lla_path_with_quantile_provenance

    _install_fit_weight_context()


install_quantile_continuation_contract()


__all__ = ["install_quantile_continuation_contract"]
