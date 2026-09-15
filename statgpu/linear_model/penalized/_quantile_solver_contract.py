"""Truthful solver routing for smooth penalized Quantile models.

Historically the generic auto-dispatch labeled every Quantile fit as FISTA,
while the FISTA branch internally substituted ``QuantileLoss.irls()`` for
L2/no-penalty objectives.  That made ``_selected_solver`` and explicit
``solver='fista'`` requests disagree with the algorithm that actually ran.

Keep this compatibility installer narrow and composable with the existing
penalized-model contract installers:

* auto + Quantile + L2/none resolves to ordinary Quantile IRLS;
* sparse Quantile auto routes keep the existing FISTA-family policy;
* explicit FISTA + smooth Quantile fails before backend numerical dispatch
  instead of silently executing IRLS.

SCAD/MCP continue to use their dedicated Proximal IRLS-CD continuation path;
this installer does not relabel that algorithm as ordinary IRLS.
"""

from __future__ import annotations

from functools import wraps

from . import _fit_mixin as _fit_mixin
from ._base import PenalizedGeneralizedLinearModel


_POLICY_MARKER = "_statgpu_quantile_solver_policy_contract"
_VALIDATE_MARKER = "_statgpu_quantile_solver_validate_contract"
_SMOOTH_PENALTIES = frozenset({"l2", "none", "null", ""})


def _loss_name(value) -> str:
    return str(getattr(value, "name", value) or "").lower().strip()


def _penalty_name(value) -> str:
    return str(getattr(value, "name", value) or "").lower().strip()


def _install_policy_contract() -> None:
    current = _fit_mixin._preferred_penalized_glm_solver
    if getattr(current, _POLICY_MARKER, False):
        return

    @wraps(current)
    def _preferred_with_truthful_quantile_route(
        loss_name,
        penalty_name,
        backend_name=None,
        l1_ratio=0.5,
        cv_mode=False,
        problem_size=None,
    ):
        if (
            _loss_name(loss_name) == "quantile"
            and _penalty_name(penalty_name) in _SMOOTH_PENALTIES
        ):
            return "irls"
        return current(
            loss_name,
            penalty_name,
            backend_name=backend_name,
            l1_ratio=l1_ratio,
            cv_mode=cv_mode,
            problem_size=problem_size,
        )

    setattr(_preferred_with_truthful_quantile_route, _POLICY_MARKER, True)
    _preferred_with_truthful_quantile_route._statgpu_original = current
    _fit_mixin._preferred_penalized_glm_solver = _preferred_with_truthful_quantile_route


def _install_explicit_fista_guard() -> None:
    current = PenalizedGeneralizedLinearModel._validate_solver_penalty
    if getattr(current, _VALIDATE_MARKER, False):
        return

    @wraps(current)
    def _validate_with_quantile_fista_guard(self):
        current(self)
        solver_name = str(getattr(self, "_solver", "") or "").lower()
        loss_name = str(getattr(self, "loss", "") or "").lower()
        penalty_name = _penalty_name(getattr(self, "_penalty", self.penalty))
        if (
            solver_name == "fista"
            and loss_name == "quantile"
            and penalty_name in _SMOOTH_PENALTIES
        ):
            raise ValueError(
                "solver='fista' is not a maintained smooth Quantile route for "
                "L2/no-penalty objectives; use solver='irls' or solver='auto'."
            )

    setattr(_validate_with_quantile_fista_guard, _VALIDATE_MARKER, True)
    _validate_with_quantile_fista_guard._statgpu_original = current
    PenalizedGeneralizedLinearModel._validate_solver_penalty = (
        _validate_with_quantile_fista_guard
    )


def install_quantile_solver_contract() -> None:
    """Install the Quantile solver/provenance reconciliation idempotently."""
    _install_policy_contract()
    _install_explicit_fista_guard()


install_quantile_solver_contract()
