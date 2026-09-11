"""Focused review-closure regressions for issue #150."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings

import numpy as np
import pytest

from statgpu.linear_model import GammaRegression, GeneralizedLinearModel
from statgpu.losses import CoxPartialLikelihoodLoss
from statgpu.solvers import lbfgs_solver
from statgpu.solvers._convergence import ConvergenceWarning


def _logistic_data(seed=15401, n=100, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.7, size=(n, p)).astype(np.float64)
    beta = np.array([0.45, -0.28, 0.16])[:p]
    eta = X @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    return X, y


def test_weighted_explicit_solvers_support_no_intercept_representative_row():
    X, y = _logistic_data()
    weights = np.linspace(0.5, 1.6, X.shape[0], dtype=np.float64)

    for solver in ("newton", "lbfgs"):
        model = GeneralizedLinearModel(
            family="binomial",
            solver=solver,
            fit_intercept=False,
            device="cpu",
            max_iter=500,
            tol=1e-8,
        ).fit(X, y, sample_weight=weights)
        assert model._selected_solver == solver
        assert model.intercept_ == 0.0
        assert np.all(np.isfinite(model.coef_))


def test_inverse_link_gamma_weighted_smooth_solvers_use_stable_family_start():
    rng = np.random.default_rng(15402)
    X = rng.normal(scale=0.22, size=(100, 3)).astype(np.float64)
    eta = np.clip(1.0 + X @ np.array([0.08, -0.05, 0.04]), 0.65, 1.35)
    mu = 1.0 / eta
    y = (mu * rng.lognormal(0.0, 0.035, size=X.shape[0])).astype(np.float64)
    weights = np.linspace(0.55, 1.65, X.shape[0], dtype=np.float64)

    for solver in ("newton", "lbfgs"):
        model = GammaRegression(
            link="inverse_power",
            solver=solver,
            device="cpu",
            max_iter=500,
            tol=1e-8,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            warnings.filterwarnings(
                "error",
                message="lbfgs_solver: line search failed.*",
                category=RuntimeWarning,
            )
            model.fit(X, y, sample_weight=weights)
        assert model._selected_solver == solver
        assert np.all(np.isfinite(model.coef_))
        assert np.isfinite(model.intercept_)


def test_direct_cox_lbfgs_nonuniform_weights_remain_fail_closed():
    rng = np.random.default_rng(15403)
    n = 48
    X = rng.normal(size=(n, 3)).astype(np.float64)
    time = rng.exponential(scale=2.0, size=n) + 0.1
    event = np.ones(n, dtype=np.float64)
    y = {"time": time, "event": event}
    weights = np.linspace(0.5, 1.5, n, dtype=np.float64)

    with pytest.raises(ValueError, match="does not support non-uniform sample_weight"):
        lbfgs_solver(
            CoxPartialLikelihoodLoss(ties="breslow"),
            None,
            X,
            y,
            max_iter=20,
            tol=1e-8,
            sample_weight=weights,
        )


def test_installer_import_order_is_stable_in_fresh_interpreter():
    code = r'''
import inspect, json
from statgpu.linear_model._glm_base import GeneralizedLinearModel as RawGLM
raw_fit = RawGLM.fit
raw_smooth = RawGLM._fit_smooth_solver
import statgpu.linear_model as lm
Installed = lm.GeneralizedLinearModel
payload = {
    "same_class": RawGLM is Installed,
    "fit_wrapped": hasattr(Installed.fit, "__wrapped__"),
    "smooth_wrapped": hasattr(Installed._fit_smooth_solver, "__wrapped__"),
    "fit_signature": str(inspect.signature(Installed.fit)),
    "smooth_signature": str(inspect.signature(Installed._fit_smooth_solver)),
    "raw_fit_changed": Installed.fit is not raw_fit,
    "raw_smooth_changed": Installed._fit_smooth_solver is not raw_smooth,
}
print(json.dumps(payload, sort_keys=True))
'''
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["same_class"] is True
    assert payload["fit_wrapped"] is True
    assert payload["smooth_wrapped"] is True
    assert payload["raw_fit_changed"] is True
    assert payload["raw_smooth_changed"] is True
    assert "sample_weight=None" in payload["fit_signature"]
    assert "solver_name" in payload["smooth_signature"]
