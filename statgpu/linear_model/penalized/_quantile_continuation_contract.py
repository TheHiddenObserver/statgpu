"""Mark estimator-generated Quantile continuation paths for objective alignment.

The existing path generator remains the single source of continuation length,
target alpha, and unweighted behavior.  This contract adds only provenance so
the public Quantile solver can distinguish an internal automatic path from a
user-supplied low-level ``alpha_path``.
"""

from __future__ import annotations

from functools import wraps

from ._fit_mixin import _PenalizedFitMixin
from statgpu.solvers._quantile_continuation import (
    mark_auto_quantile_continuation_path,
)


_MARKER = "_statgpu_quantile_continuation_path_contract"


def install_quantile_continuation_contract() -> None:
    current = _PenalizedFitMixin._compute_lla_path
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def _compute_lla_path_with_quantile_provenance(
        self,
        X_work,
        y_arr,
        p,
        loss_name,
        n_cont=None,
    ):
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


install_quantile_continuation_contract()


__all__ = ["install_quantile_continuation_contract"]
