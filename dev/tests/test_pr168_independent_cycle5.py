"""Close fifth-cycle independent DOC findings without numerical source edits."""

import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from scipy.interpolate import BSpline

from statgpu import (
    GAM,
    GeneralizedLinearModel,
    LinearRegression,
    PenalizedGeneralizedLinearModel,
)
from statgpu.nonparametric.kernel_methods import _kernels
from statgpu.nonparametric.splines._penalized import penalized_ls
from statgpu.survival import CoxPHCV

ROOT = Path(__file__).resolve().parents[2]


class _NullspaceShrinkage(Exception):
    """A finite solve does not preserve its intended unpenalized direction."""


def test_ordinary_gaussian_glm_uses_normal_while_linear_paths_use_student_t():
    x = np.arange(10.)[:, None]
    y = np.array([.5, 2, 1.5, 4, 5, 3.2, 6.7, 5.9, 9.1, 10.3])
    linear = LinearRegression(device="cpu").fit(x, y)
    glm = GeneralizedLinearModel(family="gaussian", C=0, device="cpu",
                                 compute_inference=True).fit(x, y)
    penalized = PenalizedGeneralizedLinearModel(
        loss="squared_error", penalty="l2", alpha=0, device="cpu", compute_inference=True,
    ).fit(x, y)
    for model in (glm, penalized):
        np.testing.assert_allclose(model.coef_, linear.coef_, atol=1e-12)
        np.testing.assert_allclose(model._bse, linear._bse, atol=1e-12)
    for model in (linear, penalized):
        result = model._inference_result
        assert result.distribution == "t"
        np.testing.assert_allclose(result.pvalues, 2 * stats.t.sf(abs(result.statistic), 8))
    result = glm._inference_result
    assert result.distribution == "normal" and result.method == "m_estimation"
    np.testing.assert_allclose(result.pvalues, 2 * stats.norm.sf(abs(result.statistic)))
    np.testing.assert_allclose(result.conf_int,
                               result.params[:, None] + np.array([-1, 1]) * stats.norm.isf(.025) * result.bse[:, None])
    assert linear._pvalues[1] > 1e-5 and glm._pvalues[1] < 1e-14


@pytest.mark.parametrize("language", ["en", "cn"])
def test_inference_reference_names_the_gaussian_glm_exception(language):
    guide = (ROOT / f"docs/{language}/guides/inference-modes.md").read_text()
    reference = (ROOT / f"docs/{language}/reference/linear-model-api.md").read_text()
    for text in (guide, reference):
        assert "LinearRegression" in text and "Student-t" in text
        assert "_inference_result.distribution" in text
        assert ("normal/z" in text if language == "en" else "正态（z）" in text)
    assert "normal/z-reference" in inspect.getdoc(GeneralizedLinearModel)


def test_current_penalized_ls_edf_help_describes_the_actual_stabilized_operator():
    b = np.eye(2)
    penalty = np.diag([0., 1.])
    lam = 1e12
    beta, edf = penalized_ls(b, np.ones(2), penalty, lam)
    a = b.T @ b + lam * penalty
    delta = 1e-10 * np.trace(a) / b.shape[1]
    used = a + delta * np.eye(b.shape[1])
    np.testing.assert_allclose(beta, np.linalg.solve(used, np.ones(2)), rtol=1e-12)
    assert edf == pytest.approx(np.trace(np.linalg.solve(used, b.T @ b)), rel=1e-12)
    help_text = " ".join(inspect.getdoc(penalized_ls).split())
    for token in ("A_used", "trace(M) clipped to [0, p]", "p is returned", "singular A"):
        assert token in help_text


def test_gam_moderate_penalty_retains_training_mean_control():
    x = np.linspace(-1, 1, 50)[:, None]
    y = np.full(50, 5.)
    model = GAM(n_splines=8, lam=1., device="cpu").fit(x, y)
    np.testing.assert_allclose(model.predict(x), y, atol=1e-8, rtol=1e-8)


@pytest.mark.xfail(strict=True, raises=_NullspaceShrinkage,
                   reason="Issue #238: trace-scaled jitter penalizes the intended GAM intercept")
@pytest.mark.parametrize("lam", [1e10, 1e12])
def test_large_gam_penalty_should_preserve_constant_response(lam):
    x = np.linspace(-1, 1, 50)[:, None]
    y = np.full(50, 5.)
    model = GAM(n_splines=8, lam=lam, device="cpu").fit(x, y)
    prediction = model.predict(x)
    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()
    # Recognize the trace-scaled stabilization specifically, not any finite
    # prediction error. Reconstruct the design independently with SciPy.
    if not np.allclose(prediction, y, atol=1e-8, rtol=1e-8):
        knots = np.r_[np.full(4, -1.), np.linspace(-1., 1., 6)[1:-1], np.full(4, 1.)]
        raw_basis = BSpline(knots, np.eye(8), 3)(x[:, 0])
        basis = np.column_stack([np.ones(len(x)), raw_basis - raw_basis.mean(axis=0)])
        difference = np.diff(np.eye(8), n=2, axis=0)
        penalty = np.zeros((9, 9))
        penalty[1:, 1:] = difference.T @ difference
        unjittered = basis.T @ basis + lam * penalty
        delta = 1e-10 * np.trace(unjittered) / 9
        used = unjittered + delta * np.eye(9)
        expected = np.linalg.solve(used, basis.T @ y)
        assert model.coef_.shape == (9,) and np.isfinite(model.coef_).all()
        np.testing.assert_allclose(model.coef_, expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(prediction, basis @ expected, rtol=1e-10, atol=1e-12)
        assert model.intercept_ == pytest.approx(5 * len(x) / (len(x) + delta), rel=1e-12)
        assert model.edf_ == pytest.approx(np.trace(np.linalg.solve(used, basis.T @ basis)), rel=1e-9)
        assert model.lam_ == lam and model.gcv_score_ is None
        raise _NullspaceShrinkage("constant response should be an exact zero-objective solution")


@pytest.mark.xfail(strict=True, raises=_NullspaceShrinkage,
                   reason="Issue #238: diagonal stabilization shrinks penalty-nullspace coefficients")
def test_penalized_ls_should_preserve_unpenalized_coordinate():
    beta, edf = penalized_ls(np.eye(2), np.ones(2), np.diag([0., 1.]), 1e12)
    assert beta.shape == (2,) and np.isfinite(beta).all() and np.isfinite(edf)
    if not np.allclose(beta, [1., 1 / (1 + 1e12)], atol=1e-10, rtol=1e-8):
        delta = 1e-10 * (2 + 1e12) / 2
        stabilized = 1 / (np.array([1., 1 + 1e12]) + delta)
        np.testing.assert_allclose(beta, stabilized, rtol=1e-12, atol=0)
        assert edf == pytest.approx(stabilized.sum(), rel=1e-12)
        raise _NullspaceShrinkage("the zero-penalty coordinate must remain unshrunk")


@pytest.mark.parametrize("language", ["en", "cn"])
def test_gam_help_and_bilingual_safeguard_disclose_wrong_target_risk(language):
    text = (ROOT / f"docs/{language}/models/semiparametric.md").read_text()
    reference = (ROOT / f"docs/{language}/reference/survival-smoothing-api.md").read_text()
    for token in ("large-smoothing-penalties-and-the-intercept", "lam=1e10", "lam=1e12",
                  "training_prediction.mean()", "rtol=1e-8"):
        assert token in text
    assert "large-smoothing-penalties-and-the-intercept" in reference
    assert "nullspace" in inspect.getdoc(GAM)
    assert "changing lam" in inspect.getdoc(GAM).lower()


@pytest.mark.parametrize("columns", [2, 3])
def test_cox_cv_packed_target_matches_separate_arguments(columns):
    rng = np.random.default_rng(62)
    x = rng.normal(size=(36, 2))
    time = rng.exponential(np.exp(-x @ np.array([.3, -.2]))) + .1
    event = rng.binomial(1, .8, len(x))
    start = np.zeros(len(x))
    options = {"penalties": [.1, 1.], "cv": 2, "random_state": 5,
               "device": "cpu", "max_iter": 100, "compute_inference": False}
    kwargs = {} if columns == 2 else {"start": start}
    separate = CoxPHCV(**options).fit(x, time, event, **kwargs)
    packed = np.column_stack([time, event] if columns == 2 else [start, time, event])
    together = CoxPHCV(**options).fit(x, packed)
    np.testing.assert_allclose(separate.coef_, together.coef_, atol=1e-12)
    assert separate.penalty_ == together.penalty_
    if columns == 3:
        for conflicting in ("entry", "start"):
            with pytest.raises(ValueError, match="entry|start|packed"):
                CoxPHCV(**options).fit(x, packed, **{conflicting: start})
    help_text = " ".join(inspect.getdoc(CoxPHCV.fit).split())
    for token in ("event is None", "[time, event]", "[start, stop, event]", "separate entry/start"):
        assert token in help_text


def test_knockoff_faq_retains_both_gpu_backend_choices():
    page = (ROOT / "docs/en/models/knockoff.md").read_text()
    faq = page.split("## FAQ", 1)[1].split("## ", 1)[0]
    assert 'backend="cupy"' in faq and 'backend="torch"' in faq
    assert "Torch CUDA tensors" in faq and "GPU? Yes" not in faq


@pytest.mark.parametrize("name", ["rbf_kernel", "polynomial_kernel", "linear_kernel",
                                  "laplacian_kernel", "sigmoid_kernel", "cosine_kernel"])
def test_builtin_kernel_help_requires_prepared_backend_arrays(name):
    kernel = getattr(_kernels, name)
    x = np.asarray([[1., 0.], [0., 2.]])
    y = np.asarray([[1., 1.]])
    actual = kernel(x, y, xp=np)
    assert actual.shape == (2, 1) and np.isfinite(actual).all()
    text = inspect.getdoc(kernel)
    assert "X : backend array" in text and "Y : backend array" in text
    assert "convert lists before calling" in text
    if name == "cosine_kernel":
        expected = (x @ y.T) / (np.linalg.norm(x, axis=1)[:, None]
                                * np.linalg.norm(y, axis=1)[None, :] + 1e-10)
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        assert "tiny-norm inputs" in text


def test_kernel_dispatch_help_preserves_chi2_conversion_and_callable_ownership():
    values = [[1., 0.], [0., 2.]]
    np.testing.assert_allclose(_kernels.chi2_kernel(values),
                               _kernels.chi2_kernel(np.asarray(values)))
    seen = {}

    def custom(x, y, xp=None):
        seen.update(x=x, y=y, xp=xp)
        return np.ones((len(x), len(x)))

    result = _kernels.pairwise_kernels(values, metric=custom)
    assert seen == {"x": values, "y": None, "xp": None}
    assert seen["x"] is values and result.shape == (2, 2)
    text = inspect.getdoc(_kernels.pairwise_kernels)
    assert "callable-specific input" in text and "original object unchanged" in text
    assert "chi2 path" in text
    assert "including xp=None" in inspect.getdoc(_kernels)
