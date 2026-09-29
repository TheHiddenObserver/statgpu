"""Installer idempotence coverage for the PR #166 Quantile group route."""

from __future__ import annotations

import importlib

from statgpu.linear_model.penalized import _fit_mixin
from statgpu.linear_model.penalized import _quantile_group_lla_contract as contract
from statgpu.linear_model.penalized import _quantile_solver_contract as quantile_contract


def test_quantile_group_installer_reload_preserves_fit_and_cv_wrappers():
    CV = quantile_contract.PenalizedGLM_CV
    before_fit = _fit_mixin._PenalizedFitMixin._fit_loss_backend
    before_fold = CV._cv_fold_general
    before_refit = CV._refit_best
    before_scores = CV._compute_cv_scores

    importlib.reload(contract)

    assert _fit_mixin._PenalizedFitMixin._fit_loss_backend is before_fit
    assert CV._cv_fold_general is before_fold
    assert CV._refit_best is before_refit
    assert CV._compute_cv_scores is before_scores
    assert getattr(before_fit, contract._MARKER, False)
    assert getattr(CV, contract._CV_MARKER, False)