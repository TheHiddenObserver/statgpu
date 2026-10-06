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
from scipy.linalg import null_space

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


def _natural_projection(scale, dtype=np.float64):
    x = np.linspace(0., scale, 101, dtype=dtype)
    knots = np.linspace(.1 * scale, .9 * scale, 5, dtype=dtype)
    natural = np.asarray(natural_cubic_spline_basis(x, knots, xp=np))
    # These are ordinary failures, never a reason to forgive issue #224.
    assert natural.shape == (len(x), len(knots) + 2)
    assert np.isfinite(natural).all()
    assert np.linalg.matrix_rank(natural) == natural.shape[1]

    # Use an independent cubic evaluator: a shared production basis regression
    # must not validate itself through a second call to bspline_basis.
    augmented = np.r_[np.full(4, x[0]), knots, np.full(4, x[-1])]
    ordinary = BSpline(augmented, np.eye(len(knots) + 4), 3)(x)
    projection = np.linalg.lstsq(ordinary, natural, rcond=None)[0]
    assert_allclose(ordinary @ projection, natural, rtol=1e-10,
                    atol=1e-12 * np.max(np.abs(natural)))
    # Normalize coefficients so curvature checks cannot be defeated merely by
    # shrinking the columns; column signs/rotations are not part of the API.
    orthogonal, _ = np.linalg.qr(projection)
    return x, natural, BSpline(augmented, orthogonal, 3)


def _assert_known_natural_subspace(spline, defect):
    """Recognize the defect signatures of these fixed regression fixtures."""
    n_basis = len(spline.t) - 4
    if defect == "constant":
        # At range 1e6 the absolute 1e-6 stencil suffers cancellation. Its
        # leftmost coefficient disappears, leaving the -4:1 interior ratio;
        # the right constraint collapses to the last adjacent difference.
        # These limiting rows identify the observed defect without repeating
        # production's floating-point recursion (SciPy rounds it differently).
        constraints = np.zeros((2, n_basis))
        constraints[0, 1:3] = [-4., 1.]
        constraints[1, -2:] = [-1., 1.]
        # The retained right interior term perturbs its projector by 1.3e-6.
        # This dimensionless tolerance admits that cancellation residue, not a
        # different spline space; sign/rotation changes cancel in the projector.
        atol = 5e-6
    else:
        assert defect == "curvature"
        # On these tiny ranges the stencil is wider than the data and erroneously
        # rebuilds the basis on [hi - 2h, lo + 2h]. Independently evaluate that
        # wrong basis with SciPy; do not call either production spline helper.
        lo, hi = spline.t[0], spline.t[-1]
        h = 1e-6
        points = np.array([lo, lo + h, lo + 2*h, hi, hi - h, hi - 2*h])
        expanded = np.r_[np.full(4, points.min()), spline.t[4:-4],
                         np.full(4, points.max())]
        values = BSpline(expanded, np.eye(n_basis), 3)(points)
        constraints = np.vstack([values[2] - 2*values[1] + values[0],
                                 values[5] - 2*values[4] + values[3]])
        atol = 1e-10
    expected = null_space(constraints)
    assert expected.shape == spline.c.shape
    assert_allclose(spline.c @ spline.c.T, expected @ expected.T,
                    rtol=0., atol=atol,
                    err_msg="Unexpected natural-spline constraint subspace")


def _assert_analytic_natural_subspace(spline):
    # Restoring constants alone can leave one boundary broken. A successful
    # return must describe a genuine natural space, not just a partial repair.
    ordinary = BSpline(spline.t, np.eye(len(spline.t) - 4), 3)
    lo, hi = spline.t[0], spline.t[-1]
    constraints = ordinary.derivative(2)([lo, hi]) * (hi - lo)**2
    expected = null_space(constraints)
    assert_allclose(spline.c @ spline.c.T, expected @ expected.T,
                    rtol=0., atol=1e-8,
                    err_msg="Incomplete natural-boundary repair")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unit_range_mitigates_but_does_not_exactly_repair_natural_curvature(dtype):
    x, natural, spline = _natural_projection(1., dtype)
    constant = natural @ np.linalg.lstsq(natural, np.ones(len(x)), rcond=None)[0]
    assert_allclose(constant, 1., atol=1e-6)
    # This loose bound verifies the documented mitigation, not exact conditions.
    assert np.max(np.abs(spline.derivative(2)([0., 1.]))) < .01


@pytest.mark.xfail(strict=True, raises=_NaturalConstantMismatch, reason="Issue #224: Absolute-step natural constraints can exclude constant functions after rescaling")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_natural_basis_should_represent_constants_independent_of_units(dtype):
    x, natural, spline = _natural_projection(1e6, dtype)
    constant = natural @ np.linalg.lstsq(natural, np.ones(len(x)), rcond=None)[0]
    assert constant.shape == x.shape
    if not np.allclose(constant, 1., rtol=1e-7, atol=1e-6):
        _assert_known_natural_subspace(spline, "constant")
        raise _NaturalConstantMismatch('Rescaled natural basis excludes constant functions')
    _assert_analytic_natural_subspace(spline)


@pytest.mark.xfail(strict=True, raises=_NaturalCurvatureMismatch, reason="Issue #224: Absolute-step natural constraints differentiate a different basis on very small ranges")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1e-8, 1e-7])
def test_natural_endpoint_curvature_should_remain_small_after_unit_change(dtype, scale):
    x, _, spline = _natural_projection(scale, dtype)
    curvature = spline.derivative(2)([x[0], x[-1]]) * float(x[-1] - x[0])**2
    assert curvature.shape == (2, spline.c.shape[1])
    assert np.isfinite(curvature).all()
    if not np.allclose(curvature, 0., rtol=1e-7, atol=.01):
        _assert_known_natural_subspace(spline, "curvature")
        raise _NaturalCurvatureMismatch('Rescaled natural endpoint curvature is nonzero')
    _assert_analytic_natural_subspace(spline)
