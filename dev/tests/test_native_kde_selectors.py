"""Native KDE selectors: CPU oracle, device ownership and host-transfer guards."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.nonparametric import select_bandwidth
from statgpu.nonparametric.kernel_smoothing import _bandwidth_selection as legacy
from statgpu.nonparametric.kernel_smoothing import _native_bandwidth as native


METHODS = (0.43, "scott", "silverman", "nrd", "nrd0", "ucv", "bcv", "sj", "sj-ste", "sj-dpi")


@pytest.fixture(params=("torch_cpu", "torch_cuda", "cupy_cuda"))
def backend(request):
    if request.param.startswith("torch"):
        torch = pytest.importorskip("torch")
        if request.param == "torch_cuda" and not torch.cuda.is_available():
            pytest.skip("physical Torch CUDA device required")
        device = "cpu" if request.param == "torch_cpu" else "cuda:0"
        return torch, lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("physical CuPy CUDA device required")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("working CuPy CUDA runtime required")
    return cp, lambda x: cp.asarray(x, dtype=cp.float64)


def _data(dim, weighted, seed=20261022):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(80, dim))
    x[:, 0] *= 1.7
    if dim > 1:
        x[:, 1] += 0.4 * x[:, 0]
    w = rng.uniform(0.1, 2.0, len(x)) if weighted else np.ones(len(x))
    if weighted:
        w[::9] = 0.0
    w /= w.sum()
    centered = x - np.sum(x * w[:, None], axis=0)
    cov = (centered.T * w) @ centered / (1.0 - np.sum(w * w))
    return x, w, cov


def _select(method, x, w, cov, xp):
    # n_eff comes from prepared data in public callers; scalar transfer is part
    # of that existing diagnostic/search-control contract.
    return select_bandwidth(
        method, n_eff=float(1.0 / (w * w).sum()), n_features=x.shape[1],
        samples_2d=x, weights_1d=w, data_cov=cov, xp=xp,
    )


def _host(x):
    if type(x).__module__.startswith("torch"):
        return x.detach().cpu().numpy()
    if type(x).__module__.startswith("cupy"):
        return x.get()
    return x


def _native_policy_factor_oracle(method, x, w, cov, *, projection_vector=None):
    """Independent NumPy formulas conditional on the native PCA axis.

    The chosen axis is a diagnostic input, not a CPU eigensolver-sign oracle.
    Everything after it, including variance, resampling and scalar searches,
    is recomputed with the preserved NumPy reference routines.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = w / w.sum()
    n_eff = 1.0 / np.sum(w * w)
    method_name = method.strip().lower() if isinstance(method, str) else "scalar"
    if method_name not in ("nrd", "nrd0", "ucv", "bcv", "sj", "sj-ste", "sj-dpi"):
        return legacy._bandwidth_factor(method, n_eff=n_eff, n_features=x.shape[1])
    if x.shape[1] > 1:
        assert projection_vector is not None
        centered = x - np.sum(x * w[:, None], axis=0)[None, :]
        x_work = centered @ np.asarray(projection_vector)
        mean = np.sum(w * x_work)
        variance = float(np.sum(w * (x_work - mean)**2) / (1.0 - np.sum(w * w)))
        if not np.isfinite(variance) or variance <= 0:
            variance = float(np.var(x_work, ddof=1))
        if not np.isfinite(variance) or variance <= 0:
            variance = np.finfo(np.float64).tiny
    else:
        x_work = x[:, 0]
        variance = cov[0, 0]
    if method_name in ("nrd", "nrd0"):
        return legacy._bandwidth_factor_1d_nrd(
            method_name, n_eff=n_eff, samples_2d=x_work[:, None],
            data_cov=np.asarray([[variance]]), xp=np,
        )
    if x.shape[1] > 1:
        w = w / w.sum()
    if np.max(w) - np.min(w) > 1e-12:
        n_rep = round(max(128.0, min(8192.0, 8.0 / np.sum(w * w))))
        order = np.argsort(x_work, kind="stable")
        cdf = np.cumsum(w[order])
        cdf[-1] = 1.0
        x_work = np.interp((np.arange(n_rep) + 0.5) / n_rep, cdf, x_work[order])
    return legacy._bandwidth_factor_1d_r_selectors(
        method_name, samples_2d=x_work[:, None],
        weights_1d=np.ones(len(x_work)) / len(x_work),
        data_cov=np.asarray([[variance]]),
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dim", (1, 3))
@pytest.mark.parametrize("weighted", (False, True))
def test_native_selectors_match_numpy_policy_oracle(backend, method, dim, weighted):
    xp, array = backend
    x, w, cov = _data(dim, weighted)
    expected = _select(method, x, w, cov, np)
    actual = _select(method, array(x), array(w), array(cov), xp)
    oracle_factor = _native_policy_factor_oracle(
        method, x, w, cov, projection_vector=_host(actual.details.get("projection_vector")),
    )
    assert actual.factor == pytest.approx(oracle_factor, rel=2e-11, abs=2e-13)
    assert actual.method == expected.method
    assert actual.n_features == dim
    assert actual.n_eff == pytest.approx(expected.n_eff)
    assert actual.used_r_selector == expected.used_r_selector
    assert actual.weighted == expected.weighted
    assert actual.weighted_strategy == expected.weighted_strategy
    assert actual.multivariate_strategy == expected.multivariate_strategy
    assert actual.selector_dimension == expected.selector_dimension
    assert set(expected.details) <= set(actual.details)
    if actual.used_r_selector and weighted:
        assert actual.details["weighted_tie_policy"] == "stable_input_order"
    if "projection_vector" in actual.details:
        vector = actual.details["projection_vector"]
        assert type(vector).__module__.startswith(xp.__name__)
        assert vector.device == array(x).device
        assert actual.to_dict()["details"]["projection_vector"] is vector
        assert actual.details["projection_axis_policy"] == "backend_eigh_largest_eigenvalue"
        assert actual.details["projection_sign_policy"] == "largest_absolute_loading_positive"
        # Eigenvectors describe axes and can differ in orientation by solver.
        np.testing.assert_allclose(
            np.outer(_host(vector), _host(vector)),
            np.outer(expected.details["projection_vector"], expected.details["projection_vector"]),
            rtol=2e-11, atol=2e-13,
        )
        assert actual.details["projection_variance"] == pytest.approx(expected.details["projection_variance"], rel=2e-11)


@pytest.mark.parametrize("method", METHODS)
def test_native_selector_never_converts_arrays_to_host(monkeypatch, backend, method):
    xp, array = backend
    x, w, cov = (array(value) for value in _data(3, True))
    kwargs = dict(n_eff=float(1.0 / (w * w).sum()), n_features=3,
                  samples_2d=x, weights_1d=w, data_cov=cov, xp=xp)

    def forbidden(*args, **kwargs):
        raise AssertionError("native selector attempted host array conversion")

    monkeypatch.setattr(legacy, "_to_numpy", forbidden)
    monkeypatch.setattr(legacy, "_normalized_weights_numpy", forbidden)
    monkeypatch.setattr(legacy, "_bandwidth_factor_1d_nrd", forbidden)
    monkeypatch.setattr(legacy, "_bandwidth_factor_1d_r_selectors", forbidden)
    monkeypatch.setattr(legacy, "_multivariate_factor_from_projected_1d", forbidden)
    if xp.__name__ == "torch":
        monkeypatch.setattr(xp.Tensor, "numpy", forbidden)
        monkeypatch.setattr(xp.Tensor, "cpu", forbidden)
        monkeypatch.setattr(xp.Tensor, "__array__", forbidden)
    else:
        monkeypatch.setattr(xp, "asnumpy", forbidden)
    result = select_bandwidth(method, **kwargs)
    assert np.isfinite(result.factor)


def test_native_histogram_and_functionals_match_cpu(backend):
    xp, array = backend
    x = _data(1, False)[0][:, 0]
    expected_d, expected_counts = legacy._bw_pair_distance_counts_1d(x)
    actual_d, actual_counts = native._pair_distance_counts_1d(array(x), xp)
    assert actual_d == expected_d
    np.testing.assert_array_equal(_host(actual_counts), expected_counts)
    for name, objective in (
        ("ucv", legacy._bw_ucv_objective), ("bcv", legacy._bw_bcv_objective),
        ("phi4", legacy._bw_phi4), ("phi6", legacy._bw_phi6),
    ):
        for h in (0.01, 0.4, 3.0):
            actual = native._binned_functional(name, len(x), actual_d, actual_counts, h, xp)
            expected = objective(len(x), expected_d, expected_counts, h)
            assert actual == pytest.approx(expected, rel=3e-12, abs=2e-12)


@pytest.mark.parametrize("weights", ([0, 1, 0, 1, 0], [1, 0, 0, 1, 0], [0, 0, 0, 0, 1]))
def test_native_linear_quantile_interpolation_preserves_cdf_plateaus(backend, weights):
    xp, array = backend
    x = np.array([-3.0, -1.0, 0.5, 2.0, 4.0])
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    # n=4 gives midpoint queries equal to plateau CDF knots for this case.
    for n in (4, 128):
        expected = legacy._weighted_quantile_resample_1d(x, w, n)
        actual = native._weighted_quantile_resample_1d(array(x), array(w), n, xp)
        np.testing.assert_allclose(_host(actual), expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("method", ("nrd", "nrd0", "ucv", "bcv", "sj", "sj-ste", "sj-dpi"))
def test_native_near_degenerate_weights_match_cpu(backend, method):
    xp, array = backend
    x = np.linspace(-2.0, 2.0, 25)[:, None]
    w = np.full(len(x), 1e-7)
    w[len(x) // 2] = 1.0
    w /= w.sum()
    xc = x - (x * w[:, None]).sum(axis=0)
    cov = (xc.T * w) @ xc / (1.0 - (w * w).sum())
    try:
        expected = _select(method, x, w, cov, np)
    except ValueError as exc:
        with pytest.raises(ValueError, match=str(exc)):
            _select(method, array(x), array(w), array(cov), xp)
    else:
        actual = _select(method, array(x), array(w), array(cov), xp)
        assert actual.factor == pytest.approx(expected.factor, rel=2e-10)


@pytest.mark.parametrize("method", ("nrd", "ucv", "sj-ste"))
def test_native_constant_sample_errors_match_cpu(backend, method):
    xp, array = backend
    with pytest.raises(ValueError, match="constant|positive scale"):
        _select(method, array(np.ones((10, 1))), array(np.ones(10) / 10), array([[1e-12]]), xp)


def test_native_selector_does_not_bypass_r_selector_disable(backend):
    xp, array = backend
    x, w, cov = (array(value) for value in _data(1, False))
    with pytest.raises(ValueError, match="disabled"):
        select_bandwidth("ucv", n_eff=80, n_features=1, samples_2d=x,
                         weights_1d=w, data_cov=cov, xp=xp, enable_r_selectors=False)


def test_kernel_regression_dispatch_is_preserved(monkeypatch):
    torch = pytest.importorskip("torch")
    x, w, cov = (torch.as_tensor(value) for value in _data(1, False))

    def forbidden(*args, **kwargs):
        raise AssertionError("kernel regression entered native KDE selector")

    monkeypatch.setattr(native, "select_bandwidth_native", forbidden)
    result = select_bandwidth("scott", n_eff=80, n_features=1, samples_2d=x,
                              weights_1d=w, data_cov=cov, xp=torch, estimator="kernel_regression")
    assert result.details["estimator"] == "kernel_regression"


def test_native_cupy_selector_restores_ambient_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 2:
            pytest.skip("two physical CuPy CUDA devices required")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("working CuPy CUDA runtime required")
    with cp.cuda.Device(1):
        x, w, cov = (cp.asarray(value) for value in _data(3, True))
    with cp.cuda.Device(0):
        result = _select("sj-dpi", x, w, cov, cp)
        assert cp.cuda.Device().id == 0
    assert result.details["projection_vector"].device.id == 1


def test_weighted_interpolation_stably_orders_equal_samples(backend):
    xp, array = backend
    rng = np.random.default_rng(16)
    x = rng.integers(-2, 3, 100).astype(float)
    w = rng.uniform(0.01, 3.0, len(x))
    w[::7] = 0.0
    w /= w.sum()
    order = np.argsort(x, kind="stable")
    cdf = np.cumsum(w[order])
    cdf[-1] = 1.0
    expected = np.interp((np.arange(128) + 0.5) / 128, cdf, x[order])
    actual = native._weighted_quantile_resample_1d(array(x), array(w), 128, xp)
    np.testing.assert_allclose(_host(actual), expected, atol=2e-13, rtol=2e-13)
    old_order = np.argsort(x)
    old_cdf = np.cumsum(w[old_order])
    old_cdf[-1] = 1.0
    legacy_expected = np.interp((np.arange(128) + 0.5) / 128, old_cdf, x[old_order])
    np.testing.assert_array_equal(legacy._weighted_quantile_resample_1d(x, w, 128), legacy_expected)


@pytest.mark.parametrize("method", ("nrd", "nrd0", "ucv", "bcv", "sj", "sj-ste", "sj-dpi"))
def test_projected_selectors_ignore_eigensolver_sign(monkeypatch, backend, method):
    xp, array = backend
    x, w, cov = _data(3, True)
    native_expected = _select(method, array(x), array(w), array(cov), xp)
    original_native_eigh = xp.linalg.eigh

    def flipped_native(value):
        values, vectors = original_native_eigh(value)
        return values, -vectors

    monkeypatch.setattr(xp.linalg, "eigh", flipped_native)
    native_actual = _select(method, array(x), array(w), array(cov), xp)
    assert native_actual.factor == native_expected.factor
    vector = _host(native_actual.details["projection_vector"])
    expected = _native_policy_factor_oracle(method, x, w, cov, projection_vector=vector)
    assert native_actual.factor == pytest.approx(expected, rel=2e-11)
    assert vector[np.argmax(abs(vector))] > 0.0



@pytest.mark.parametrize("method", ("nrd", "nrd0", "ucv", "bcv", "sj", "sj-ste", "sj-dpi"))
def test_native_selectors_equal_samples_unequal_weights_match_policy(backend, method):
    xp, array = backend
    rng = np.random.default_rng(16)
    x = rng.integers(-3, 4, 100).astype(float)[:, None]
    w = rng.uniform(0.01, 3.0, len(x))
    w[::7] = 0.0
    w /= w.sum()
    xc = x - (x * w[:, None]).sum(axis=0)
    cov = (xc.T * w) @ xc / (1.0 - (w * w).sum())
    expected = _native_policy_factor_oracle(method, x, w, cov)
    actual = _select(method, array(x), array(w), array(cov), xp)
    assert actual.factor == pytest.approx(expected, rel=2e-11)


_FROZEN_BASELINE_SEED7_FACTORS = {
    # Captured from unmodified bf21587e4783801b84e1df7c98ce54dd5c840bb2.
    # The eigensolver principal vector is fixed to its original negative first
    # loading so this regression does not depend on a BLAS sign convention.
    "scott": 0.6784241501679215,
    "silverman": 0.657138650751353,
    "nrd": 0.5045297163030777,
    "nrd0": 0.42837428742714145,
    "ucv": 0.04295947750985378,
    "bcv": 0.41674574737142844,
    "sj": 0.12260194990123138,
    "sj-ste": 0.12260194990123138,
    "sj-dpi": 0.20425690835819504,
}


def _seed7_regression_data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 3))
    w = rng.uniform(size=30)
    w[::4] = 0.0
    w /= w.sum()
    xc = x - (x * w[:, None]).sum(axis=0)
    cov = (xc.T * w) @ xc / (1.0 - (w * w).sum())
    return x, w, cov


@pytest.mark.parametrize("method", tuple(_FROZEN_BASELINE_SEED7_FACTORS))
def test_cpu_selectors_preserve_frozen_original_negative_axis(monkeypatch, method):
    original_eigh = np.linalg.eigh

    def original_axis_sign(cov):
        values, vectors = original_eigh(cov)
        if vectors[0, -1] > 0:
            vectors[:, -1] *= -1
        return values, vectors

    monkeypatch.setattr(np.linalg, "eigh", original_axis_sign)
    x, w, cov = _seed7_regression_data()
    actual = _select(method, x, w, cov, np)
    assert actual.factor == pytest.approx(_FROZEN_BASELINE_SEED7_FACTORS[method], rel=2e-12)
    if "projection_vector" in actual.details:
        assert actual.details["projection_vector"][0] < 0


def test_native_projection_sign_difference_is_explicit_not_claimed_cpu_parity(backend):
    xp, array = backend
    x, w, cov = _seed7_regression_data()
    actual = _select("sj-ste", array(x), array(w), array(cov), xp)
    vector = _host(actual.details["projection_vector"])
    expected = _native_policy_factor_oracle("sj-ste", x, w, cov, projection_vector=vector)
    assert actual.factor == pytest.approx(expected, rel=2e-11)
    # Linear weighted interpolation is not reflection-invariant. Its effect is
    # material here; do not weaken this to a generic CPU/backend tolerance.
    assert actual.factor / _FROZEN_BASELINE_SEED7_FACTORS["sj-ste"] - 1.0 > 0.5
    assert actual.details["projection_sign_policy"] == "largest_absolute_loading_positive"


def test_native_repeated_principal_eigenvalue_reports_chosen_axis(monkeypatch, backend):
    xp, array = backend
    # Exact isotropic covariance with different higher-order marginal shapes.
    x = np.array([[-2., 0.], [2., 0.]] + [[0., -1.], [0., 1.]] * 4 + [[0., 0.]] * 6)
    w = np.ones(16) / 16
    cov = (x.T * w) @ x / (1.0 - (w * w).sum())
    np.testing.assert_array_equal((x.T * w) @ x, np.eye(2) * 0.5)
    with pytest.warns(RuntimeWarning, match="1 ambiguous projection") as observed:
        reference = _select("nrd", array(x), array(w), array(cov), xp)
    assert observed[0].filename == __file__
    original_eigh = xp.linalg.eigh

    def rotated_eigenspace(value):
        values, vectors = original_eigh(value)
        # Rotating a repeated eigenspace gives another equally valid PCA axis.
        rotation = array([[0.6, -0.8], [0.8, 0.6]])
        return values, vectors @ rotation

    monkeypatch.setattr(xp.linalg, "eigh", rotated_eigenspace)
    with pytest.warns(RuntimeWarning, match="1 ambiguous projection"):
        rotated = _select("nrd", array(x), array(w), array(cov), xp)
    axis = _host(rotated.details["projection_vector"])
    expected = _native_policy_factor_oracle("nrd", x, w, cov, projection_vector=axis)
    assert rotated.factor == pytest.approx(expected, rel=2e-11)
    assert not np.allclose(np.outer(axis, axis), np.outer(
        _host(reference.details["projection_vector"]), _host(reference.details["projection_vector"]),
    ))
    assert rotated.details["projection_axis_policy"] == "backend_eigh_largest_eigenvalue"


def _frozen_cpu_selector_policy_oracle(method, x, weights, data_cov):
    """Frozen bf21587e PCA/resampling policy, independent of dispatch/helpers.

    Unmodified scalar one-dimensional searches remain the trusted numerical
    baseline; projection and interpolation are recomputed here rather than
    calling the production helpers whose behavior this regression protects.
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / float(np.sum(w))
    if method in ("scott", "silverman"):
        return legacy._bandwidth_factor(method, n_eff=1.0 / np.sum(weights * weights), n_features=x.shape[1])
    if x.shape[1] == 1:
        projected, variance = np.asarray(x[:, 0]), data_cov[0, 0]
    else:
        mean = np.sum(x * w[:, None], axis=0)
        centered = x - mean[None, :]
        covariance = (centered.T * w[None, :]) @ centered
        covariance = 0.5 * (covariance + covariance.T)
        values, vectors = np.linalg.eigh(covariance)
        index = int(np.argmax(values))
        principal = vectors[:, index]
        if not np.isfinite(values[index]) or values[index] <= 0:
            scales = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            principal = np.zeros(x.shape[1])
            principal[int(np.argmax(scales))] = 1.0
        # Deliberately no sign normalization or degenerate-eigenspace rule.
        projected = centered @ principal
        mean_projected = float(np.sum(w * projected))
        denom = 1.0 - float(np.sum(w * w))
        variance = float(np.sum(w * (projected - mean_projected)**2) / denom) if denom > 1e-15 else float("nan")
        if not np.isfinite(variance) or variance <= 0:
            variance = float(np.var(projected, ddof=1))
        if not np.isfinite(variance) or variance <= 0:
            variance = np.finfo(np.float64).tiny
    if method in ("nrd", "nrd0"):
        return legacy._bandwidth_factor_1d_nrd(
            method, n_eff=1.0 / np.sum(weights * weights),
            samples_2d=projected[:, None], data_cov=np.array([[variance]]), xp=np,
        )
    if x.shape[1] > 1:
        w = w / float(np.sum(w))
    if float(np.max(w) - np.min(w)) > 1e-12:
        effective = float(1.0 / np.sum(w * w))
        n_rep = int(np.clip(round(max(128.0, min(8192.0, effective * 8.0))), 128, 8192))
        # Deliberately use the original NumPy default (unstable) tie policy.
        order = np.argsort(projected)
        cdf = np.cumsum(w[order])
        cdf[-1] = 1.0
        projected = np.interp((np.arange(n_rep, dtype=np.float64) + 0.5) / n_rep,
                              cdf, projected[order])
    return legacy._bandwidth_factor_1d_r_selectors(
        method, samples_2d=projected[:, None],
        weights_1d=np.ones(len(projected)) / len(projected),
        data_cov=np.array([[variance]]),
    )


@pytest.mark.parametrize("method", ("nrd", "nrd0", "ucv", "bcv", "sj", "sj-ste", "sj-dpi"))
@pytest.mark.parametrize("kind", ("seed7", "unequal_tied_weights"))
def test_cpu_preserves_environment_local_frozen_pca_and_sort_policy(method, kind):
    if kind == "seed7":
        x, w, cov = _seed7_regression_data()
    else:
        rng = np.random.default_rng(16)
        x = rng.integers(-3, 4, 100).astype(float)[:, None]
        w = rng.uniform(0.01, 3.0, len(x))
        w[::7] = 0.0
        w /= w.sum()
        centered = x - np.sum(x * w[:, None], axis=0)
        cov = (centered.T * w) @ centered / (1.0 - np.sum(w * w))
    expected = _frozen_cpu_selector_policy_oracle(method, x, w, cov)
    actual = _select(method, x, w, cov, np)
    assert actual.factor == pytest.approx(expected, rel=2e-12)


@pytest.mark.parametrize("eigenvalues,classification", (
    ([1.0, 1.0], "exact"),
    ([1.0, 1.0 + 1e-14], "near"),
    ([1.0, 1.0 + 1e-10], "distinct"),
    ([0.0, 1.0], "distinct"),
    ([0.0, 0.0], "zero_scale"),
))
@pytest.mark.parametrize("scale", (1e-200, 1.0, 1e200))
def test_eigengap_diagnostic_is_scale_relative(backend, eigenvalues, classification, scale):
    xp, array = backend
    values = np.asarray(eigenvalues) * scale
    result = native._projection_eigengap_diagnostics(array(values), 2, xp)
    assert result["projection_eigengap_classification"] == classification
    assert result["projection_eigengap_ambiguous"] == (classification != "distinct")
    expected_scale = np.max(np.abs(values))
    assert result["projection_eigenvalue_scale"] == expected_scale
    assert result["projection_eigengap_relative_tolerance"] == 64 * np.finfo(np.float64).eps * 2
    assert result["projection_eigengap_tolerance"] == result["projection_eigengap_relative_tolerance"] * expected_scale
    if expected_scale:
        assert result["projection_relative_eigengap"] == pytest.approx(
            (values[-1] - values[-2]) / expected_scale, rel=1e-14,
        )


@pytest.mark.parametrize("scale", (1e-80, 1.0, 1e80))
def test_pca_distinct_tiny_and_large_covariances_do_not_warn(backend, scale):
    import warnings

    xp, array = backend
    x = np.array([[-2., 0.], [2., 0.], [0., -1.], [0., 1.]]) * scale
    w = np.ones(4) / 4
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        _, _, details = native._project_to_principal_axis(array(x), array(w), xp)
    assert not observed
    assert details["projection_eigengap_classification"] == "distinct"
    assert details["projection_relative_eigengap"] == pytest.approx(0.75, rel=1e-14)


def test_pca_zero_covariance_warns_and_preserves_existing_fallback(backend):
    xp, array = backend
    with pytest.warns(RuntimeWarning, match="1 ambiguous projection"):
        projected, variance, details = native._project_to_principal_axis(
            array(np.zeros((8, 2))), array(np.ones(8) / 8), xp,
        )
    assert details["projection_eigengap_classification"] == "zero_scale"
    assert details["projection_eigenvalue_scale"] == 0.0
    assert details["projection_eigengap_tolerance"] == 0.0
    assert variance == np.finfo(np.float64).tiny
    np.testing.assert_array_equal(_host(projected), np.zeros(8))
    np.testing.assert_array_equal(_host(details["projection_vector"]), [1.0, 0.0])


def test_projection_warning_scope_aggregates_nested_calls_and_copies_metadata(backend):
    xp, array = backend

    @native.projection_warning_scope
    def nested():
        for eigenvalues in ([1., 1.], [1., 1. + 1e-14], [0., 0.], [0., 1.]):
            native._record_projection_ambiguity(
                native._projection_eigengap_diagnostics(array(eigenvalues), 2, xp),
            )

    @native.projection_warning_scope
    def outer():
        nested()
        nested()
        snapshot = native.projection_warning_diagnostics()
        modified = native.projection_warning_diagnostics()
        modified["count"] = 999
        assert native.projection_warning_diagnostics()["count"] == 6
        return snapshot

    assert outer.__name__ == "outer"
    with pytest.warns(RuntimeWarning, match="6 ambiguous projection") as observed:
        result = outer()
    assert len(observed) == 1
    assert observed[0].filename == __file__
    assert result == {
        "status": "ambiguous_principal_eigenspace", "count": 6,
        "exact_degeneracy_count": 2, "near_degeneracy_count": 2,
        "zero_scale_count": 2, "min_relative_eigengap": 0.0,
    }
    assert native.projection_warning_diagnostics()["count"] == 0


def test_projection_warning_scope_resets_after_error_without_emission(backend):
    import warnings

    xp, array = backend

    @native.projection_warning_scope
    def failed():
        native._record_projection_ambiguity(
            native._projection_eigengap_diagnostics(array([1., 1.]), 2, xp),
        )
        raise ValueError("test failure")

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="test failure"):
            failed()
    assert not observed
    assert native.projection_warning_diagnostics() == {
        "status": "none", "count": 0, "exact_degeneracy_count": 0,
        "near_degeneracy_count": 0, "zero_scale_count": 0,
        "min_relative_eigengap": None,
    }


def test_projection_warning_scope_isolated_between_threads():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import warnings

    barrier = Barrier(2)
    details = {
        "projection_eigengap_ambiguous": True,
        "projection_eigengap_classification": "near",
        "projection_relative_eigengap": 1e-15,
    }

    @native.projection_warning_scope
    def run(count):
        for _ in range(count):
            native._record_projection_ambiguity(details)
        barrier.wait(timeout=5)
        return native.projection_warning_diagnostics()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(run, count) for count in (2, 3)]
            results = [future.result() for future in futures]
    assert [result["count"] for result in results] == [2, 3]
    assert native.projection_warning_diagnostics()["count"] == 0


@pytest.mark.parametrize("method,dimension", ((0.5, 2), ("scott", 2), ("silverman", 2), ("nrd", 1)))
def test_eigengap_warning_only_applies_to_multivariate_pca_selectors(backend, method, dimension):
    import warnings

    xp, array = backend
    x = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])[:, :dimension]
    w = np.ones(4) / 4
    covariance = (x.T * w) @ x / (1.0 - np.sum(w * w))
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        result = _select(method, array(x), array(w), array(covariance), xp)
    assert not observed
    assert "projection_eigengap_classification" not in result.details
