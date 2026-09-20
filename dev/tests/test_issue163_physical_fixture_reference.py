"""Hosted reproduction of the PR164 physical validator's CPU reference contract."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from dev.benchmarks import run_quantile_smooth_fista_gpu_gate as smooth_wrapper
from dev.benchmarks import validate_quantile_smooth_fista_gpu as smooth_gate
from dev.benchmarks import validate_quantile_solver_provenance_gpu as gate
from statgpu._config import Device
from statgpu.linear_model import PenalizedGLM_CV, QuantileRegression
from statgpu.losses import QuantileLoss
from statgpu.penalties import L1Penalty
from statgpu.solvers import fista_solver
from statgpu.solvers._convergence import ConvergenceWarning
from statgpu.linear_model.penalized import (
    PenalizedGeneralizedLinearModel,
    PenalizedQuantileRegression,
)


def _pinball(y, eta, q, sample_weight):
    y = np.asarray(y, dtype=np.float64).ravel()
    eta = np.asarray(eta, dtype=np.float64).ravel()
    u = y - eta
    values = np.where(u >= 0.0, q * u, (q - 1.0) * u)
    return float(np.average(values, weights=np.asarray(sample_weight, dtype=np.float64)))


def test_pr166_smooth_bootstrap_physical_gate_schema_is_locked():
    assert smooth_gate.SCHEMA_VERSION == 7
    assert smooth_wrapper.EXPECTED_SCHEMA_VERSION == smooth_gate.SCHEMA_VERSION
    assert smooth_gate.BOOTSTRAP_Q != pytest.approx(0.5)
    assert 0.0 < smooth_gate.BOOTSTRAP_Q < 1.0
    assert smooth_gate.BOOTSTRAP_B >= 2
    assert callable(smooth_gate._standalone_bootstrap_public_case)
    assert callable(smooth_gate._async_weighted_l1_case)
    assert smooth_gate.ATOL_ASYNC_L1_OBJECTIVE > 0.0


def test_pr166_async_weighted_l1_fixture_has_spectral_gap_and_cpu_reference():
    X, y, weights, ratio = smooth_gate._async_weighted_data()
    assert ratio > 1.5

    coef, n_iter = fista_solver(
        QuantileLoss(quantile=smooth_gate.Q),
        L1Penalty(alpha=smooth_gate.ASYNC_L1_ALPHA),
        X,
        y,
        max_iter=6000,
        tol=1e-7,
        sample_weight=weights,
        cv_mode=False,
    )
    objective = smooth_gate._l1_objective(
        X,
        y,
        weights,
        coef,
        smooth_gate.ASYNC_L1_ALPHA,
    )

    assert 1 <= n_iter <= 6000
    assert np.all(np.isfinite(np.asarray(coef)))
    assert np.isfinite(objective)


def test_pr166_weighted_l1_cv_physical_fixture_converges_on_cpu():
    X, y, weights, folds = smooth_gate._data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = smooth_gate._cv(
            X,
            y,
            weights,
            folds,
            device="cpu",
            penalty="l1",
        )

    assert model.alpha_ in set(smooth_gate.CV_ALPHA_GRID.tolist())
    scores = np.asarray(
        model.cv_results_["all_scores"],
        dtype=np.float64,
    )
    assert scores.shape == (2, smooth_gate.CV_ALPHA_GRID.size)
    assert np.all(np.isfinite(scores))
    assert model.estimator_._selected_solver == "fista"


def test_pr166_public_bootstrap_physical_fixture_converges_on_cpu():
    X = np.ones((smooth_gate.BOOTSTRAP_N, 1), dtype=np.float64)
    y = np.linspace(
        -4.0,
        4.0,
        smooth_gate.BOOTSTRAP_N,
        dtype=np.float64,
    )
    model = QuantileRegression(
        quantile=smooth_gate.BOOTSTRAP_Q,
        fit_intercept=False,
        max_iter=1600,
        tol=1e-7,
        compute_inference=True,
        inference_method="bootstrap",
        n_bootstrap=smooth_gate.BOOTSTRAP_B,
        random_state=smooth_gate.BOOTSTRAP_SEED,
        device="cpu",
    ).fit(X, y)

    assert model._fitted is True
    assert model._selected_backend_name == "numpy"
    assert model._selected_backend_device == "cpu"
    assert 1 <= model.n_iter_ <= model.max_iter

    result = model._inference_result
    assert result is not None
    assert result.method == "bootstrap"
    metadata = result.metadata
    assert metadata["solver"] == "batched_pinball_fista"
    assert metadata["numerical_backend"] == "numpy"
    assert metadata["numerical_device"] == "cpu"
    assert metadata["reporting_backend"] == "numpy"
    assert metadata["response_construction"] == "backend_native"
    assert metadata["residual_centering"] == "empirical_tau_quantile"
    assert metadata["resampling_schedule"] == "numpy_generator_control_plane"
    schedule_hash = metadata["resampling_schedule_sha256"]
    assert isinstance(schedule_hash, str) and len(schedule_hash) == 64
    int(schedule_hash, 16)
    assert 1 <= int(metadata["solver_n_iter"]) <= model.max_iter
    assert np.all(np.isfinite(np.asarray(result.bse)))
    assert np.all(np.isfinite(np.asarray(result.pvalues)))
    assert np.all(np.isfinite(np.asarray(result.conf_int)))


def test_canonical_physical_artifact_preserves_exact_source_provenance():
    """Canonical GPU evidence must retain auditable exact-source provenance."""
    artifact = (
        Path(__file__).resolve().parents[1]
        / "reviews"
        / "pr164_quantile_solver_provenance_gpu.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))

    assert payload["status"] == "success"
    assert payload["source_clean"] is True
    source_sha = payload["source_sha"]
    assert isinstance(source_sha, str) and len(source_sha) == 40
    int(source_sha, 16)

    assert payload["source_sha_before"] == source_sha
    assert payload["source_sha_after_execution"] == source_sha
    assert payload["source_clean_before"] is True
    assert payload["source_clean_after_execution"] is True

    wrapper = "dev/benchmarks/run_quantile_solver_provenance_gpu_gate.py"
    if "evidence_wrapper" in payload:
        assert payload["evidence_wrapper"] == wrapper
    else:
        assert payload["evidence_wrapper_contract"] == wrapper
        assert payload["evidence_wrapper_executed"] is False
        metadata = payload["evidence_metadata"]
        assert metadata["kind"] == "canonical_provenance_repair"
        assert metadata["physical_rerun_required_for_metadata_repair"] is False


def test_physical_fixture_public_fit_matches_direct_cv_scores():
    """Public fit must not reinterpret fixed folds or requested Quantile scoring."""
    X, y, weights, folds = gate._data()
    alphas = np.asarray([0.045, 0.02], dtype=np.float64)

    direct_owner = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": gate.Q},
        penalty="l2",
        alpha_grid=alphas,
        cv=2,
        cv_splits=folds,
        random_state=164,
        solver="auto",
        device="cpu",
        cv_strategy="strict",
        max_iter=600,
        tol=1e-9,
    )
    direct = direct_owner._compute_cv_scores(
        X,
        y,
        alphas,
        Device.CPU,
        folds,
        sample_weight=weights,
        max_iter=600,
        tol=1e-9,
        strict=True,
    )

    public = gate._cv_fit(
        X,
        y,
        weights,
        folds,
        penalty="l2",
        alpha_grid=alphas,
        device="cpu",
        cv_strategy="strict",
        max_iter=600,
        tol=1e-9,
    )
    public_scores = np.asarray(public.cv_results_["all_scores"], dtype=np.float64)
    np.testing.assert_allclose(public_scores, direct, rtol=0.0, atol=0.0)


def test_physical_fixture_generic_and_typed_irls_children_are_equivalent():
    """Typed Quantile and generic Quantile children must solve the same fold problem."""
    X, y, weights, folds = gate._data()
    alpha = 0.045

    for train_idx, val_idx in folds:
        generic = PenalizedGeneralizedLinearModel(
            loss="quantile",
            loss_kwargs={"quantile": gate.Q},
            penalty="l2",
            alpha=alpha,
            solver="irls",
            device="cpu",
            max_iter=600,
            tol=1e-9,
        ).fit(
            X[train_idx],
            y[train_idx],
            sample_weight=weights[train_idx],
        )
        typed = PenalizedQuantileRegression(
            quantile=gate.Q,
            penalty="l2",
            alpha=alpha,
            solver="irls",
            device="cpu",
            max_iter=600,
            tol=1e-9,
        ).fit(
            X[train_idx],
            y[train_idx],
            sample_weight=weights[train_idx],
        )

        np.testing.assert_allclose(typed.coef_, generic.coef_, rtol=0.0, atol=0.0)
        assert float(typed.intercept_) == float(generic.intercept_)

        generic_score = _pinball(
            y[val_idx],
            X[val_idx] @ np.asarray(generic.coef_) + float(generic.intercept_),
            gate.Q,
            weights[val_idx],
        )
        typed_score = _pinball(
            y[val_idx],
            X[val_idx] @ np.asarray(typed.coef_) + float(typed.intercept_),
            gate.Q,
            weights[val_idx],
        )
        assert typed_score == generic_score


def test_physical_fixture_direct_cv_scores_match_generic_manual_oracle():
    """The existing hosted q-scoring oracle is exercised on the exact P100 fixture."""
    X, y, weights, folds = gate._data()
    alphas = np.asarray([0.045, 0.02], dtype=np.float64)
    owner = PenalizedGLM_CV(
        loss="quantile",
        loss_kwargs={"quantile": gate.Q},
        penalty="l2",
        alpha_grid=alphas,
        cv=2,
        solver="auto",
        device="cpu",
        max_iter=600,
        tol=1e-9,
    )
    observed = owner._compute_cv_scores(
        X,
        y,
        alphas,
        Device.CPU,
        folds,
        sample_weight=weights,
        max_iter=600,
        tol=1e-9,
        strict=True,
    )

    expected = np.empty_like(observed)
    median = np.empty_like(observed)
    for fold_index, (train_idx, val_idx) in enumerate(folds):
        for alpha_index, alpha in enumerate(alphas):
            child = PenalizedGeneralizedLinearModel(
                loss="quantile",
                loss_kwargs={"quantile": gate.Q},
                penalty="l2",
                alpha=float(alpha),
                solver="irls",
                device="cpu",
                max_iter=600,
                tol=1e-9,
            ).fit(
                X[train_idx],
                y[train_idx],
                sample_weight=weights[train_idx],
            )
            eta = X[val_idx] @ np.asarray(child.coef_) + float(child.intercept_)
            expected[fold_index, alpha_index] = _pinball(
                y[val_idx], eta, gate.Q, weights[val_idx]
            )
            median[fold_index, alpha_index] = _pinball(
                y[val_idx], eta, 0.5, weights[val_idx]
            )

    np.testing.assert_allclose(observed, expected, rtol=2e-8, atol=2e-10)
    assert float(np.max(np.abs(observed - median))) > 1e-4


def test_physical_fixture_scad_irls_stopping_is_backend_invariant(monkeypatch):
    """GPU synchronization policy must not change the non-convex SCAD path."""
    from statgpu.losses import QuantileLoss
    from statgpu.penalties import SCADPenalty
    from statgpu.solvers import _proximal_irls_quantile as solver_mod

    X, y, weights, _ = gate._data()
    n = X.shape[0]
    residual = y - float(np.quantile(y, gate.Q))
    psi = np.where(residual >= 0.0, gate.Q, -(1.0 - gate.Q))
    lambda_max = float(np.max(np.abs(X.T @ psi / n)))
    alpha_path = np.geomspace(
        max(lambda_max, gate.SCAD_ALPHA * 1.1), gate.SCAD_ALPHA, 3
    )
    max_iter = [100, 100, 220]
    max_lla_per_step = 16

    loss = QuantileLoss(quantile=gate.Q)
    penalty = SCADPenalty(alpha=gate.SCAD_ALPHA)
    coef_np, intercept_np, _ = solver_mod.proximal_irls_quantile_solver(
        loss,
        penalty,
        X,
        y,
        alpha_path=alpha_path,
        max_lla_per_step=max_lla_per_step,
        lla_tol=1e-6,
        max_iter=max_iter,
        tol=1e-6,
        fit_intercept=True,
        sample_weight=weights,
    )

    # Exercise the GPU control-flow branch without requiring CUDA. Historically
    # this branch checked convergence only every fifth IRLS iteration, which is
    # enough on this exact fixture to move SCAD to a different local basin.
    monkeypatch.setattr(solver_mod, "_resolve_backend", lambda _requested, _X: "cupy")
    monkeypatch.setitem(sys.modules, "cupy", np)
    coef_gpu_policy, intercept_gpu_policy, _ = (
        solver_mod.proximal_irls_quantile_solver(
            loss,
            penalty,
            X,
            y,
            alpha_path=alpha_path,
            max_lla_per_step=max_lla_per_step,
            lla_tol=1e-6,
            max_iter=max_iter,
            tol=1e-6,
            fit_intercept=True,
            sample_weight=weights,
        )
    )

    obj_np = gate._scad_objective(
        X, y, weights, coef_np, intercept_np, alpha=gate.SCAD_ALPHA
    )
    obj_gpu_policy = gate._scad_objective(
        X,
        y,
        weights,
        coef_gpu_policy,
        intercept_gpu_policy,
        alpha=gate.SCAD_ALPHA,
    )
    assert abs(obj_gpu_policy - obj_np) <= gate.ATOL_SCAD_OBJECTIVE
    np.testing.assert_allclose(coef_gpu_policy, coef_np, rtol=0.0, atol=1e-12)
    assert abs(float(intercept_gpu_policy) - float(intercept_np)) <= 1e-12
