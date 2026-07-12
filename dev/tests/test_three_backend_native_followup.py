import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.anova import bonferroni, tukey_hsd
from statgpu.covariance import GraphicalLasso, GraphicalLassoCV, MinCovDet
from statgpu.nonparametric import SplineTransformer
from statgpu.panel import FamaMacBeth


def _comparison_matrix(result):
    return np.asarray(
        [
            [c.mean_diff, c.pvalue, c.ci_lower, c.ci_upper, float(c.reject)]
            for c in result.comparisons
        ],
        dtype=float,
    )


def test_graphical_lasso_torch_cpu_matches_numpy_and_preserves_backend():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(123)
    X = rng.normal(size=(80, 5))
    X[:, 1] += 0.4 * X[:, 0]

    cpu = GraphicalLasso(alpha=0.08, max_iter=100, tol=1e-7).fit(X)
    tensor = torch.as_tensor(X, dtype=torch.float64)
    native = GraphicalLasso(alpha=0.08, max_iter=100, tol=1e-7).fit(tensor)

    assert isinstance(native.covariance_, torch.Tensor)
    assert native.covariance_.device == tensor.device
    assert isinstance(native.precision_, torch.Tensor)
    assert_allclose(native.covariance_.numpy(), cpu.covariance_, rtol=2e-6, atol=2e-7)
    assert_allclose(native.precision_.numpy(), cpu.precision_, rtol=2e-5, atol=2e-6)


def test_graphical_lasso_cv_torch_cpu_matches_numpy_selection():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(54, 4))
    kwargs = dict(alphas=[0.03, 0.08], cv=3, random_state=9, max_iter=60, tol=1e-6)
    cpu = GraphicalLassoCV(**kwargs).fit(X)
    native = GraphicalLassoCV(**kwargs).fit(torch.as_tensor(X, dtype=torch.float64))
    assert native.alpha_ == cpu.alpha_
    assert isinstance(native.covariance_, torch.Tensor)
    assert_allclose(native.covariance_.numpy(), cpu.covariance_, rtol=2e-6, atol=2e-7)


def test_min_cov_det_torch_cpu_matches_numpy_and_keeps_support_on_backend():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(70, 3))
    X[:4] += 8.0
    kwargs = dict(support_fraction=0.7, random_state=11)
    cpu = MinCovDet(**kwargs).fit(X)
    native = MinCovDet(**kwargs).fit(torch.as_tensor(X, dtype=torch.float64))

    assert isinstance(native.covariance_, torch.Tensor)
    assert isinstance(native.support_, torch.Tensor)
    assert native.support_.dtype == torch.bool
    assert_allclose(native.location_.numpy(), cpu.location_, rtol=2e-5, atol=2e-6)
    assert_allclose(native.covariance_.numpy(), cpu.covariance_, rtol=3e-5, atol=3e-6)
    assert np.array_equal(native.support_.numpy(), np.asarray(cpu.support_))


@pytest.mark.parametrize("mode", ["constant", "linear", "continue"])
def test_spline_transformer_torch_cpu_matches_numpy_for_all_extrapolations(mode):
    torch = pytest.importorskip("torch")
    train = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    points = np.array([[-0.4], [0.0], [0.25], [1.0], [1.3]])
    kwargs = dict(n_knots=6, degree=3, extrapolation=mode)
    cpu = SplineTransformer(**kwargs).fit(train)
    expected = np.asarray(cpu.transform(points))

    native = SplineTransformer(**kwargs).fit(torch.as_tensor(train, dtype=torch.float64))
    actual = native.transform(torch.as_tensor(points, dtype=torch.float64))
    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == "cpu"
    assert_allclose(actual.numpy(), expected, rtol=2e-12, atol=2e-12)
    assert_allclose(actual.sum(dim=1).numpy(), np.ones(points.shape[0]), atol=2e-12)


def test_fama_macbeth_torch_cpu_matches_numpy_and_predict_stays_native():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    periods = np.repeat(np.arange(8), 18)
    X = rng.normal(size=(periods.size, 2))
    y = 0.7 + X @ np.array([1.2, -0.5]) + rng.normal(scale=0.2, size=periods.size)
    kwargs = dict(cov_type="newey-west", bandwidth=2, min_obs_per_period=8)
    cpu = FamaMacBeth(**kwargs).fit(X, y, periods)
    native = FamaMacBeth(**kwargs).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        periods,
    )

    assert isinstance(native.coef_, torch.Tensor)
    assert isinstance(native.betas_, torch.Tensor)
    assert isinstance(native.cov_params_, torch.Tensor)
    assert_allclose(native.coef_.numpy(), cpu.coef_, rtol=2e-10, atol=2e-10)
    assert_allclose(native.bse_.numpy(), cpu.bse_, rtol=2e-10, atol=2e-10)
    assert_allclose(native.pvalues_.numpy(), cpu.pvalues_, rtol=2e-10, atol=2e-10)
    prediction = native.predict(torch.as_tensor(X[:5], dtype=torch.float64))
    assert isinstance(prediction, torch.Tensor)
    assert_allclose(prediction.numpy(), cpu.predict(X[:5]), rtol=2e-10, atol=2e-10)


def test_posthoc_torch_reductions_match_numpy_without_full_group_conversion():
    torch = pytest.importorskip("torch")
    groups = [
        np.array([1.0, 1.2, 0.8, 1.1]),
        np.array([1.5, 1.6, 1.4, 1.7]),
        np.array([0.5, 0.7, 0.6, 0.4]),
    ]
    tensors = [torch.as_tensor(g, dtype=torch.float64) for g in groups]
    assert_allclose(
        _comparison_matrix(tukey_hsd(*tensors)),
        _comparison_matrix(tukey_hsd(*groups)),
        rtol=1e-12,
        atol=1e-12,
    )
    assert_allclose(
        _comparison_matrix(bonferroni(*tensors)),
        _comparison_matrix(bonferroni(*groups)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_repaired_modules_do_not_convert_full_numeric_designs_to_numpy():
    import statgpu.anova._posthoc as posthoc_module
    import statgpu.covariance._graphical_lasso as glasso_module
    import statgpu.covariance._robust as robust_module
    import statgpu.nonparametric.splines._transformer as spline_module

    assert "_to_numpy" not in inspect.getsource(glasso_module.GraphicalLasso.fit)
    assert "_to_numpy" not in inspect.getsource(robust_module.MinCovDet.fit)
    assert "_to_numpy" not in inspect.getsource(spline_module.SplineTransformer.transform)
    assert "_to_numpy(g)" not in inspect.getsource(posthoc_module.tukey_hsd)
    assert "_to_numpy(g)" not in inspect.getsource(posthoc_module.bonferroni)


def test_optional_cupy_native_paths_match_numpy_when_cuda_is_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("CuPy CUDA runtime unavailable")

    rng = np.random.default_rng(14)
    X = rng.normal(size=(60, 3))
    X_gpu = cp.asarray(X)

    gl_cpu = GraphicalLasso(alpha=0.05, tol=1e-7).fit(X)
    gl_gpu = GraphicalLasso(alpha=0.05, tol=1e-7).fit(X_gpu)
    assert isinstance(gl_gpu.covariance_, cp.ndarray)
    assert_allclose(cp.asnumpy(gl_gpu.covariance_), gl_cpu.covariance_, rtol=3e-6, atol=3e-7)

    spline_cpu = SplineTransformer(n_knots=5, extrapolation="continue").fit(X[:, :1])
    spline_gpu = SplineTransformer(n_knots=5, extrapolation="continue").fit(X_gpu[:, :1])
    points = np.array([[-0.5], [0.25], [1.5]])
    out = spline_gpu.transform(cp.asarray(points))
    assert isinstance(out, cp.ndarray)
    assert_allclose(cp.asnumpy(out), spline_cpu.transform(points), rtol=3e-11, atol=3e-11)
