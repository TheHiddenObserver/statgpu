"""Fresh-review closure tests for penalized-GLM inference boundaries."""

from __future__ import annotations

import inspect

import numpy as np

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model import _penalized_glm_inference_fit_transaction as _tx
from statgpu.linear_model.penalized._inference_mixin import _PenalizedInferenceMixin


def _poisson_data(seed=14251, n=84, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.45, size=(n, p))
    beta = np.array([0.24, -0.16, 0.11])[:p]
    mu = np.exp(0.12 + X @ beta)
    y = rng.poisson(mu).astype(float)
    return X, y


def test_weighted_penalized_glm_cv_auto_uses_weight_capable_selection_and_refit():
    X, y = _poisson_data()
    weights = np.linspace(0.45, 1.75, X.shape[0])

    cv = PenalizedGLM_CV(
        loss="poisson",
        penalty="l2",
        alpha_grid=np.array([0.08, 0.03]),
        cv=2,
        random_state=7,
        device="cpu",
        solver="auto",
        max_iter=1200,
        tol=1e-7,
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
    ).fit(X, y, sample_weight=weights)

    assert cv.get_params(deep=False)["solver"] == "auto"
    assert getattr(cv, "_solver", None) == "auto"
    assert cv.estimator_._selected_solver == "fista"
    assert cv.inference_method_ == "m_estimation"
    assert cv.penalty_conditioning_ == "cv_selected_penalty"
    assert cv.penalty_selection_adjusted_ is False
    assert np.all(np.isfinite(np.asarray(cv._bse)))
    assert np.all(np.isfinite(np.asarray(cv._pvalues)))


def test_sandwich_alignment_reuses_cross_backend_and_concrete_device_helpers():
    source = inspect.getsource(_tx._align_sandwich_inputs_to_fit_backend)

    # CuPy execution must reuse the BaseEstimator DLPack-aware conversion path,
    # then pin the result to the fit-recorded concrete CUDA ordinal.
    assert "self._to_array" in source
    assert "backend=\"cupy\"" in source
    assert "_cupy_asarray_on_device" in source
    assert "device_id" in source

    # Torch execution must likewise use the DLPack-aware helper with the exact
    # recorded device instead of torch.as_tensor() on an arbitrary container.
    assert "self._to_torch" in source
    assert "device=device" in source
    assert "dtype=torch.float64" in source


def test_final_execution_boundary_installer_is_idempotent():
    before = _PenalizedInferenceMixin._compute_penalized_sandwich_inference
    _tx.install_penalized_glm_inference_fit_transaction()
    after = _PenalizedInferenceMixin._compute_penalized_sandwich_inference
    assert after is before
