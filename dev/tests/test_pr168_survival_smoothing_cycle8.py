"""Mutation checks for the older kernel desired-behavior expected failures.

All real numerical work here is NumPy CPU or explicitly Torch CPU. These tests
do not repair the separately tracked numerical issues or prove CUDA behavior.
"""

from pathlib import Path

import numpy as np
import pytest

from dev.tests import test_pr168_independent_cycle5 as gam_contracts
from dev.tests import test_pr168_independent_kernel_api as kernel_contracts
from dev.tests import test_pr168_survival_smoothing_cycle3 as smoothing_contracts


def _corrupt(array, kind):
    if kind == "exception":
        raise ValueError("unrelated injected kernel failure")
    if kind == "shape":
        return array[:0]
    return np.full_like(array, np.nan if kind == "nonfinite" else .42)


@pytest.mark.parametrize("kind", ["exception", "shape", "nonfinite", "wrong_finite"])
def test_rbf_translation_marker_rejects_unrelated_results(monkeypatch, kind):
    original = kernel_contracts.kernels.rbf_kernel

    def mutated(X, *args, **kwargs):
        result = original(X, *args, **kwargs)
        return _corrupt(result, kind) if np.min(X) > 1e8 else result

    monkeypatch.setattr(kernel_contracts.kernels, "rbf_kernel", mutated)
    with pytest.raises((AssertionError, ValueError)):
        kernel_contracts.test_rbf_kernel_should_be_translation_invariant()


@pytest.mark.parametrize("kind", ["exception", "shape", "nonfinite", "wrong_finite"])
def test_torch_chi2_marker_rejects_unrelated_results(monkeypatch, kind):
    torch = pytest.importorskip("torch")

    def mutated(*args, **kwargs):
        values = _corrupt(np.ones((1, 1)), kind)
        return torch.tensor(values, dtype=torch.float64, device="cpu")

    monkeypatch.setattr(kernel_contracts.kernels, "chi2_kernel", mutated)
    with pytest.raises((AssertionError, ValueError)):
        kernel_contracts.test_torch_cpu_chi2_should_preserve_small_positive_denominators()


@pytest.mark.parametrize("kind", ["exception", "shape", "nonfinite", "wrong_finite"])
@pytest.mark.parametrize("cls", [kernel_contracts.kernels.KernelRidge,
                                 kernel_contracts.kernels.KernelRidgeCV])
def test_chi2_gamma_marker_rejects_unrelated_predictions(monkeypatch, kind, cls):
    original = cls.predict

    def mutated(self, X):
        result = original(self, X)
        return _corrupt(result, kind) if self.gamma == 7.0 else result

    monkeypatch.setattr(cls, "predict", mutated)
    with pytest.raises((AssertionError, ValueError)):
        kernel_contracts.test_chi2_constructor_gamma_should_match_dictionary(cls, "chi2")


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "wrong_finite"])
def test_chi2_gamma_marker_rejects_unrelated_cv_evidence(monkeypatch, kind):
    cls = kernel_contracts.kernels.KernelRidgeCV
    original = cls.fit

    def mutated(self, X, y):
        result = original(self, X, y)
        if self.gamma == 7.0:
            self.cv_results_["mean_mse"] = _corrupt(self.cv_results_["mean_mse"], kind)
        return result

    monkeypatch.setattr(cls, "fit", mutated)
    with pytest.raises(AssertionError):
        kernel_contracts.test_chi2_constructor_gamma_should_match_dictionary(cls, "chi2")


@pytest.mark.parametrize("kind", ["exception", "shape", "nonfinite", "wrong_finite"])
def test_ignored_weight_marker_rejects_unrelated_predictions(monkeypatch, kind):
    cls = smoothing_contracts.KernelRidge
    original = cls.predict

    def mutated(self, X):
        result = original(self, X)
        return _corrupt(result, kind) if len(self.X_fit_) == 4 else result

    monkeypatch.setattr(cls, "predict", mutated)
    with pytest.raises((AssertionError, ValueError)):
        smoothing_contracts.test_kernel_ridge_zero_weight_should_equal_excluding_that_observation()


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "wrong_finite"])
def test_zero_alpha_marker_rejects_unrelated_valid_candidate_damage(monkeypatch, kind):
    cls = smoothing_contracts.KernelRidgeCV
    original = cls.fit

    def mutated(self, X, y):
        result = original(self, X, y)
        table = self.cv_results_["mse_table"].copy()
        if kind == "shape":
            table = table[:, :1]
        else:
            table[1] = _corrupt(table[1], kind)
        self.cv_results_["mse_table"] = table
        return result

    monkeypatch.setattr(cls, "fit", mutated)
    with pytest.raises(AssertionError):
        smoothing_contracts.test_kernel_ridge_cv_should_not_select_nan_candidate()


@pytest.mark.parametrize("kind", ["exception", "shape", "nonfinite", "wrong_finite"])
@pytest.mark.parametrize("lam", [1e10, 1e12])
def test_gam_nullspace_marker_rejects_unrelated_predictions(monkeypatch, kind, lam):
    original = gam_contracts.GAM.predict

    def mutated(self, X):
        return _corrupt(original(self, X), kind)

    monkeypatch.setattr(gam_contracts.GAM, "predict", mutated)
    with pytest.raises((AssertionError, ValueError)):
        gam_contracts.test_large_gam_penalty_should_preserve_constant_response(lam)


@pytest.mark.parametrize("kind", ["exception", "shape", "nonfinite", "wrong_finite", "edf"])
def test_penalized_ls_nullspace_marker_rejects_unrelated_results(monkeypatch, kind):
    original = gam_contracts.penalized_ls

    def mutated(*args, **kwargs):
        beta, edf = original(*args, **kwargs)
        if kind == "edf":
            return beta, 42.
        return _corrupt(beta, kind), edf

    monkeypatch.setattr(gam_contracts, "penalized_ls", mutated)
    with pytest.raises((AssertionError, ValueError)):
        gam_contracts.test_penalized_ls_should_preserve_unpenalized_coordinate()


def test_gam_desired_mean_passes_without_the_known_failure_marker(monkeypatch):
    monkeypatch.setattr(gam_contracts.GAM, "predict", lambda self, X: np.full(len(X), 5.))
    gam_contracts.test_large_gam_penalty_should_preserve_constant_response(1e12)


def test_penalized_ls_desired_solution_passes_without_the_known_failure_marker(monkeypatch):
    desired = np.array([1., 1 / (1 + 1e12)])
    monkeypatch.setattr(gam_contracts, "penalized_ls", lambda *a, **k: (desired, desired.sum()))
    gam_contracts.test_penalized_ls_should_preserve_unpenalized_coordinate()


def test_rbf_desired_result_passes_without_the_known_failure_marker(monkeypatch):
    def exact(X, Y=None, gamma=None, xp=None):
        Y = X if Y is None else Y
        return np.exp(-gamma * np.sum((X[:, None] - Y[None, :]) ** 2, axis=2))

    monkeypatch.setattr(kernel_contracts.kernels, "rbf_kernel", exact)
    kernel_contracts.test_rbf_kernel_should_be_translation_invariant()


def test_torch_chi2_desired_result_passes_without_the_known_failure_marker(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(kernel_contracts.kernels, "chi2_kernel",
                        lambda *a, **k: torch.tensor([[np.exp(-1.)]], dtype=torch.float64))
    kernel_contracts.test_torch_cpu_chi2_should_preserve_small_positive_denominators()


@pytest.mark.parametrize("cls", [kernel_contracts.kernels.KernelRidge,
                                 kernel_contracts.kernels.KernelRidgeCV])
def test_chi2_gamma_desired_result_passes_without_the_known_failure_marker(monkeypatch, cls):
    original = cls.fit

    def corrected_request(self, X, y):
        if self.gamma == 7.0:
            self.kernel_params = {"gamma": 7.0}
        return original(self, X, y)

    monkeypatch.setattr(cls, "fit", corrected_request)
    kernel_contracts.test_chi2_constructor_gamma_should_match_dictionary(cls, "chi2")


@pytest.mark.parametrize("language", ["en", "cn"])
def test_kernel_validation_claim_names_inputs_and_separate_computed_results(language):
    root = Path(__file__).resolve().parents[2]
    page = (root / f"docs/{language}/models/kernel-methods.md").read_text()
    section = page.split("## Backend and Execution Boundaries" if language == "en"
                         else "## 后端与执行边界", 1)[1].split("\n## ", 1)[0]
    if language == "en":
        assert "NaN/Inf in the supplied input arrays" in section
        assert "does not guarantee finite computed" in section
    else:
        assert "输入数组中的 NaN/Inf" in section
        assert "不保证计算得到的核矩阵或学习数组也是有限值" in section


@pytest.mark.parametrize("language", ["en", "cn"])
def test_reference_links_complete_kernel_and_spline_apis(language):
    root = Path(__file__).resolve().parents[2]
    reference = (root / f"docs/{language}/reference/survival-smoothing-api.md").read_text()
    anchors = (
        ("kernel-methods", "complete-estimator-api", "Complete estimator API"),
        ("kernel-methods", "complete-pairwise-function-api", "Complete pairwise-function API"),
        ("splines", "complete-model-specific-calls", "Complete model-specific calls"),
    ) if language == "en" else (
        ("kernel-methods", "完整估计器-api", "完整估计器 API"),
        ("kernel-methods", "完整成对核函数-api", "完整成对核函数 API"),
        ("splines", "完整的模型专属调用", "完整的模型专属调用"),
    )
    for name, anchor, heading in anchors:
        assert f"../models/{name}.md#{anchor}" in reference
        destination = (root / f"docs/{language}/models/{name}.md").read_text()
        assert f"## {heading}" in destination
