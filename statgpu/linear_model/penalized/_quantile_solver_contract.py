"""Truthful solver routing for penalized Quantile models.

Historically the generic auto-dispatch labeled every Quantile fit as FISTA,
while the fit path internally substituted different algorithms for selected
penalties. That made ``_selected_solver`` and explicit solver requests disagree
with the algorithm that actually ran.

Keep this compatibility installer narrow and composable with the existing
penalized-model contract installers:

* auto + Quantile + L2/none resolves to ordinary Quantile IRLS;
* sparse Quantile auto routes keep the existing FISTA-family policy;
* auto + Quantile + SCAD/MCP resolves to the dedicated Proximal IRLS-CD route;
* incompatible explicit Quantile solver requests fail before backend numerical
  dispatch instead of being silently substituted by another algorithm.
"""

from __future__ import annotations

from functools import wraps

from . import _fit_mixin as _fit_mixin
from ._base import PenalizedGeneralizedLinearModel


_POLICY_MARKER = "_statgpu_quantile_solver_policy_contract"
_VALIDATE_MARKER = "_statgpu_quantile_solver_validate_contract"
_SMOOTH_PENALTIES = frozenset({"l2", "none", "null", ""})
_NONCONVEX_QUANTILE_PENALTIES = frozenset({"scad", "mcp"})
_DEDICATED_NONCONVEX_SOLVER = "proximal_irls_cd"


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
        resolved_loss = _loss_name(loss_name)
        resolved_penalty = _penalty_name(penalty_name)
        if resolved_loss == "quantile":
            if resolved_penalty in _SMOOTH_PENALTIES:
                return "irls"
            if resolved_penalty in _NONCONVEX_QUANTILE_PENALTIES:
                return _DEDICATED_NONCONVEX_SOLVER
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


def _install_explicit_route_guard() -> None:
    current = PenalizedGeneralizedLinearModel._validate_solver_penalty
    if getattr(current, _VALIDATE_MARKER, False):
        return

    @wraps(current)
    def _validate_with_quantile_route_guard(self):
        current(self)
        solver_name = str(getattr(self, "_solver", "") or "").lower()
        loss_name = str(getattr(self, "loss", "") or "").lower()
        penalty_name = _penalty_name(getattr(self, "_penalty", self.penalty))

        if loss_name != "quantile":
            return

        if (
            solver_name in ("fista", "fista_bb")
            and penalty_name in _SMOOTH_PENALTIES
        ):
            raise ValueError(
                f"solver='{solver_name}' is not a maintained smooth Quantile "
                "route for L2/no-penalty objectives; use solver='irls' or "
                "solver='auto'."
            )

        if (
            penalty_name in _NONCONVEX_QUANTILE_PENALTIES
            and solver_name not in ("auto", _DEDICATED_NONCONVEX_SOLVER)
        ):
            raise ValueError(
                f"solver='{solver_name}' is not a maintained Quantile "
                f"{penalty_name.upper()} route; use solver='auto' so the "
                "dedicated Proximal IRLS-CD algorithm is selected."
            )

    setattr(_validate_with_quantile_route_guard, _VALIDATE_MARKER, True)
    _validate_with_quantile_route_guard._statgpu_original = current
    PenalizedGeneralizedLinearModel._validate_solver_penalty = (
        _validate_with_quantile_route_guard
    )


def install_quantile_solver_contract() -> None:
    """Install the Quantile solver/provenance reconciliation idempotently."""
    _install_policy_contract()
    _install_explicit_route_guard()


install_quantile_solver_contract()
