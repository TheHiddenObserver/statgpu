"""Issue #163 internal-CV solver-context regression coverage."""

from __future__ import annotations

import inspect

from statgpu.linear_model.penalized import PenalizedGLM_CV
from statgpu.linear_model.penalized import _quantile_solver_contract as contract


def test_quantile_cv_context_installer_is_idempotent_signature_safe_and_unset():
    before_solver = PenalizedGLM_CV._solver_for_cv
    before_fold = PenalizedGLM_CV._cv_fold_general
    before_refit = PenalizedGLM_CV._refit_best
    solver_signature = inspect.signature(before_solver)
    fold_signature = inspect.signature(before_fold)
    refit_signature = inspect.signature(before_refit)

    assert contract._INTERNAL_CV_RESOLVED_SOLVER.get() is False
    contract.install_quantile_solver_contract()

    assert PenalizedGLM_CV._solver_for_cv is before_solver
    assert PenalizedGLM_CV._cv_fold_general is before_fold
    assert PenalizedGLM_CV._refit_best is before_refit
    assert inspect.signature(before_solver) == solver_signature
    assert inspect.signature(before_fold) == fold_signature
    assert inspect.signature(before_refit) == refit_signature
    assert contract._INTERNAL_CV_RESOLVED_SOLVER.get() is False
    assert getattr(before_solver, contract._CV_PUBLIC_SOLVER_MARKER, False) is True
    assert getattr(PenalizedGLM_CV, contract._CV_CONTEXT_MARKER, False) is True
