"""Fresh cycle-4 docs contracts and independently reproduced smoothing gaps.

CPU Torch evidence is explicit. Strict xfails describe unresolved production
defects rather than endorsing their present behavior as an API contract.
"""
import linecache
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import BSpline

from statgpu.nonparametric import (
    KernelDensityEstimator,
    KernelRegression,
    fit_kde,
    fit_kernel_regression,
    kde_bootstrap_confidence_interval,
    kde_confidence_interval,
)
from statgpu.nonparametric.kernel_methods import KernelPCA, Nystroem
from statgpu.nonparametric.kernel_smoothing import _kernel_common, _kernel_regression
from statgpu.nonparametric.splines import (
    SplineTransformer,
    bspline_basis,
    natural_cubic_spline_basis,
)

ROOT = Path(__file__).resolve().parents[2]
X = np.array([[-1., 0.], [0., 2.], [1., -1.], [2., 1.]])
Y = np.arange(4.)
WEIGHTS = np.array([1., 2., 3., 4.])


class _TorchSizeMethodError(Exception):
    """A known helper tries to convert Tensor.size itself into an integer."""


class _NaturalConstantMismatch(Exception):
    """The rescaled natural basis cannot represent the constant function."""


class _NaturalCurvatureMismatch(Exception):
    """Rescaled endpoint curvature violates the natural boundary condition."""


@contextmanager
def _known_torch_size_failure(helper, size_check):
    try:
        yield
    except TypeError as error:
        traceback = error.__traceback__
        while traceback.tb_next is not None:
            traceback = traceback.tb_next
        # Pin both the message and the precise failing helper/size expression.
        # Other errors in the public call (including signature errors) propagate.
        if (
            str(error) == "int() argument must be a string, a bytes-like object or a real number, "
            "not 'builtin_function_or_method'"
            and traceback.tb_frame.f_code is helper.__code__
            and linecache.getline(traceback.tb_frame.f_code.co_filename, traceback.tb_lineno).strip()
            == size_check
        ):
            raise _TorchSizeMethodError(str(error)) from error
        raise


@pytest.mark.parametrize("language", ["en", "cn"])
def test_torch_restrictions_present_in_learner_and_complete_reference(language):
    for relative in ("models/nonparametric.md", "reference/survival-smoothing-api.md"):
        text = (ROOT / "docs" / language / relative).read_text()
        anchor = "### Current Torch restrictions" if "models" in relative else "### Torch argument restrictions"
        if language == "cn":
            anchor = "### 当前 Torch 限制" if "models" in relative else "### Torch 参数限制"
        section = text.split(anchor, 1)[1].split("\n## ", 1)[0]
        for required in ("weights", "bandwidth_per_feature", "bootstrap", "TypeError", 'backend="numpy"', "X[:1]"):
            assert required.lower() in section.lower()
        assert ("changes the" in section or "not an equivalent" in section) if language == "en" else ("改变" in section)


@pytest.mark.parametrize("language", ["en", "cn"])
def test_natural_boundary_warning_is_scale_specific_and_not_a_repair_claim(language):
    text = (ROOT / f"docs/{language}/models/splines.md").read_text()
    assert "1e6" in text
    assert "TypeError" in text and "(1,p)" in text
    if language == "en":
        assert "does not make the constraints" in text
        assert "independently\nverified natural-spline" in text
        assert "Fit the transformer only on training rows" in text
    else:
        assert "不能\n让约束变得精确" in text
        assert "经过独立验证的自然样条构造" in text
        assert "每个训练折内学习节点" in text
    gam = (ROOT / f"docs/{language}/models/semiparametric.md").read_text()
    assert "TypeError" in gam and "X[:1]" in gam


def test_runtime_help_matches_documented_restrictions_and_result_scope():
    assert "explicit weights" in KernelDensityEstimator.__doc__
    assert "bandwidth_per_feature" in KernelRegression.__doc__
    assert "Torch" in KernelDensityEstimator.pdf.__doc__
    assert "Torch" in KernelRegression.predict.__doc__
    assert "absolute finite-difference step" in natural_cubic_spline_basis.__doc__
    assert "constant functions" in natural_cubic_spline_basis.__doc__
    assert "n_jobs" in KernelPCA.__doc__ and "n_jobs" in Nystroem.__doc__


@pytest.mark.parametrize("kind", ["kde", "nw", "local_linear"])
def test_explicit_matrix_queries_work_on_torch_cpu_and_match_numpy(kind):
    torch = pytest.importorskip("torch")
    def fit(backend):
        if kind == "kde":
            return fit_kde(X, bandwidth=.8, backend=backend)
        return fit_kernel_regression(X, Y, bandwidth=.8, regression=kind, backend=backend)
    actual = fit("torch").predict(X[:1])
    assert isinstance(actual, torch.Tensor) and actual.device.type == "cpu"
    assert actual.shape == (1,)
    assert_allclose(actual.numpy(), fit("numpy").predict(X[:1]), rtol=1e-10, atol=1e-10)


def test_spline_transformer_explicit_matrix_query_on_torch_cpu():
    torch = pytest.importorskip("torch")
    actual = SplineTransformer(device="cpu").fit(torch.tensor(X)).transform(torch.tensor(X[:1]))
    expected = SplineTransformer(device="cpu").fit(X).transform(X[:1])
    assert actual.device.type == "cpu"
    assert_allclose(actual.numpy(), expected, atol=1e-12)


def test_numpy_preserves_requested_weights_and_absolute_widths():
    density = fit_kde(X, weights=WEIGHTS, backend="numpy")
    regression = fit_kernel_regression(X, Y, weights=WEIGHTS, backend="numpy")
    assert_allclose(density.weights_, WEIGHTS / WEIGHTS.sum())
    assert_allclose(regression.weights_, WEIGHTS / WEIGHTS.sum())
    assert_allclose(regression.target_mean_, np.average(Y, weights=WEIGHTS))
    explicit = fit_kernel_regression(X, Y, kernel_metric="diagonal",
                                    bandwidth_per_feature=[.5, 1.], backend="numpy")
    assert_allclose(explicit.covariance_, np.diag([.25, 1.]))
    assert np.isfinite(explicit.predict(X[:1])).all()


def test_unweighted_torch_normal_and_numpy_bootstrap_remain_distinct():
    pytest.importorskip("torch")
    options = {"samples": X[:, 0], "points": [0.], "bandwidth": .5}
    normal = kde_confidence_interval(**options, backend="torch")
    normal_np = kde_confidence_interval(**options, backend="numpy")
    assert normal.n_resamples == 0
    assert_allclose(normal.lower, normal_np.lower)
    assert_allclose(normal.upper, normal_np.upper)
    bootstrap = kde_bootstrap_confidence_interval(
        **options, backend="numpy", n_resamples=5, random_state=11)
    assert bootstrap.n_resamples == 5
    assert np.isfinite(bootstrap.lower).all()


@pytest.mark.xfail(strict=True, raises=_TorchSizeMethodError, reason="Issue #223: Torch Tensor.size is treated as an integer while normalizing weights")
@pytest.mark.parametrize("kind", ["kde", "regression"])
def test_weighted_torch_fit_should_accept_valid_weights(kind):
    pytest.importorskip("torch")
    with _known_torch_size_failure(
        _kernel_common._normalize_weights, "if int(w.size) != int(n_samples):"
    ):
        if kind == "kde":
            model = fit_kde(X, weights=WEIGHTS, backend="torch")
        else:
            model = fit_kernel_regression(X, Y, weights=WEIGHTS, backend="torch")
    assert_allclose(model.weights_.numpy(), WEIGHTS / WEIGHTS.sum())


@pytest.mark.xfail(strict=True, raises=_TorchSizeMethodError, reason="Issue #223: Torch Tensor.size breaks both scalar and vector absolute bandwidths")
@pytest.mark.parametrize("widths", [.5, [.5, .5]])
def test_torch_absolute_widths_should_accept_scalar_or_vector(widths):
    pytest.importorskip("torch")
    with _known_torch_size_failure(
        _kernel_regression._as_bandwidth_per_feature,
        "if int(bw.size) == 1 and int(n_features) > 1:",
    ):
        model = fit_kernel_regression(X, Y, bandwidth_per_feature=widths,
                                      kernel_metric="diagonal", backend="torch")
    assert_allclose(model.covariance_.numpy(), .25 * np.eye(2))


@pytest.mark.xfail(strict=True, raises=_TorchSizeMethodError, reason="Issue #223: Torch multivariate vector shape checks use Tensor.size as an integer")
@pytest.mark.parametrize("kind", ["kde", "regression", "transformer"])
def test_torch_multivariate_vector_should_match_matrix_query(kind):
    torch = pytest.importorskip("torch")
    if kind == "kde":
        model = fit_kde(X, backend="torch")
    elif kind == "regression":
        model = fit_kernel_regression(X, Y, backend="torch")
    else:
        model = SplineTransformer(device="cpu").fit(torch.tensor(X))
    # Establish the working control and prepare inputs outside the known failure.
    expected = model.predict(torch.tensor(X[:1])).numpy()
    query = torch.tensor(X[0])
    helper = (SplineTransformer._prepare_X if kind == "transformer"
              else _kernel_common._as_points_2d)
    size_check = ("elif int(X_arr.size) == expected_features:" if kind == "transformer"
                  else "elif int(arr.size) == n_features:")
    with _known_torch_size_failure(helper, size_check):
        prediction = model.predict(query)
    assert_allclose(prediction.numpy(), expected)


@pytest.mark.xfail(strict=True, raises=_TorchSizeMethodError, reason="Issue #223: Unweighted Torch bootstrap internally supplies weights to a failing size check")
def test_torch_bootstrap_should_accept_equal_weight_samples():
    pytest.importorskip("torch")
    with _known_torch_size_failure(
        _kernel_common._normalize_weights, "if int(w.size) != int(n_samples):"
    ):
        result = kde_bootstrap_confidence_interval(X[:, 0], [0.], backend="torch",
                                                   n_resamples=2, random_state=0)
    assert result.n_resamples == 2
    assert np.isfinite(result.lower).all()


def _natural_projection(scale):
    x = np.linspace(0., scale, 101)
    knots = np.linspace(.1 * scale, .9 * scale, 5)
    ordinary = bspline_basis(x, knots, xp=np)
    natural = natural_cubic_spline_basis(x, knots, xp=np)
    projection = np.linalg.lstsq(ordinary, natural, rcond=None)[0]
    assert_allclose(ordinary @ projection, natural, atol=1e-12)
    augmented = np.r_[np.zeros(4), knots, np.full(4, scale)]
    return x, natural, BSpline(augmented, projection, 3)


def test_unit_range_mitigates_but_does_not_exactly_repair_natural_curvature():
    x, natural, spline = _natural_projection(1.)
    constant = natural @ np.linalg.lstsq(natural, np.ones(len(x)), rcond=None)[0]
    assert_allclose(constant, 1., atol=1e-6)
    # This loose bound verifies the documented mitigation, not exact conditions.
    assert np.max(np.abs(spline.derivative(2)([0., 1.]))) < .01


@pytest.mark.xfail(strict=True, raises=_NaturalConstantMismatch, reason="Issue #224: Absolute-step natural constraints can exclude constant functions after rescaling")
def test_natural_basis_should_represent_constants_independent_of_units():
    x, natural, _ = _natural_projection(1e6)
    constant = natural @ np.linalg.lstsq(natural, np.ones(len(x)), rcond=None)[0]
    assert constant.shape == x.shape
    if not np.allclose(constant, 1., rtol=1e-7, atol=1e-6):
        raise _NaturalConstantMismatch('Rescaled natural basis excludes constant functions')


@pytest.mark.xfail(strict=True, raises=_NaturalCurvatureMismatch, reason="Issue #224: Absolute-step natural constraints differentiate a different basis on very small ranges")
def test_natural_endpoint_curvature_should_remain_small_after_unit_change():
    scale = 1e-8
    _, _, spline = _natural_projection(scale)
    curvature = spline.derivative(2)([0., scale]) * scale**2
    assert curvature.shape == (2, spline.c.shape[1])
    if not np.allclose(curvature, 0., rtol=1e-7, atol=.01):
        raise _NaturalCurvatureMismatch('Rescaled natural endpoint curvature is nonzero')
