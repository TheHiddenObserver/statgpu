"""Array-like covariance regressions, including optional pandas input.

Pandas is only a test dependency. Real Torch CPU queries exercise native fitted
ownership; physical CUDA cases skip when their actual runtime is unavailable.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from statgpu._config import Device, _device_manager
from statgpu.backends import _to_numpy
from statgpu.covariance import (
    OAS,
    EmpiricalCovariance,
    GraphicalLasso,
    GraphicalLassoCV,
    LedoitWolf,
    MinCovDet,
    ShrunkCovariance,
)

ESTIMATORS = (
    EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance,
    GraphicalLasso, GraphicalLassoCV, MinCovDet,
)
QUERIES = ("score", "predict", "mahalanobis")
ARRAY_ATTRIBUTES = (
    "covariance_", "precision_", "location_", "support_",
    "raw_covariance_", "raw_location_", "dist_",
)


@pytest.fixture(autouse=True)
def automatic_global_device(monkeypatch):
    # Restore the incoming policy on teardown, including an incoming CPU policy
    # from another module in an ordered regression run.
    monkeypatch.setattr(_device_manager, "_current_device", Device.AUTO)


@pytest.fixture
def pd():
    return pytest.importorskip("pandas")


@pytest.fixture
def X():
    data = np.random.default_rng(192).normal(size=(45, 3))
    data[:, 1] += 0.45 * data[:, 0]
    return data


def _model(cls, device="cpu", **kwargs):
    params = {"device": device}
    if cls is GraphicalLasso:
        params.update(alpha=0.08, max_iter=100, tol=1e-7)
    elif cls is GraphicalLassoCV:
        params.update(alphas=[0.03, 0.12], cv=3, random_state=7,
                      max_iter=100, tol=1e-7)
    elif cls is MinCovDet:
        params.update(support_fraction=0.75, random_state=7)
    params.update(kwargs)
    return cls(**params)


def _attributes(model):
    return {name: getattr(model, name) for name in ARRAY_ATTRIBUTES
            if hasattr(model, name)}


def _assert_fit_matches(actual, expected):
    assert actual.n_samples_ == expected.n_samples_
    assert actual.n_features_ == expected.n_features_
    assert _attributes(actual).keys() == _attributes(expected).keys()
    for name, value in _attributes(actual).items():
        assert_allclose(_to_numpy(value), _to_numpy(getattr(expected, name)),
                        rtol=5e-6, atol=5e-7, err_msg=name)
    if hasattr(actual, "shrinkage_"):
        assert_allclose(actual.shrinkage_, expected.shrinkage_, rtol=1e-12)
    if isinstance(actual, GraphicalLassoCV):
        assert actual.alpha_ == expected.alpha_
        assert_allclose([row["scores"] for row in actual.cv_results_],
                        [row["scores"] for row in expected.cv_results_],
                        rtol=5e-6, atol=5e-7)


def _assert_queries_match(actual, expected, data, numpy_data):
    for method in QUERIES:
        result = getattr(actual, method)(data)
        assert_allclose(result, getattr(expected, method)(numpy_data),
                        rtol=5e-6, atol=5e-7, err_msg=method)
        if method == "score":
            assert isinstance(result, float)
        else:
            assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "Float64", "Float32", "Int64"])
@pytest.mark.parametrize("operation", ["fit", "score", "predict", "mahalanobis", "series_fit"])
def test_pandas_original_input_matrix(cls, dtype, operation, X, pd):
    """All original 7 x 5 regressions with NumPy and nullable pandas dtypes."""
    numpy_dtype = np.dtype(dtype.lower())
    data = (X * 100).astype(numpy_dtype) if numpy_dtype.kind == "i" else X.astype(numpy_dtype)
    if operation == "series_fit":
        data = data[:, 0]
        pandas_data = pd.Series(data, index=np.arange(len(data))[::-1], name="x", dtype=dtype)
    else:
        pandas_data = pd.DataFrame(data, columns=["z", "x", "y"],
                                   index=np.arange(len(data))[::-1], dtype=dtype)
    before = pandas_data.copy(deep=True)
    expected = _model(cls).fit(data)
    if operation in ("fit", "series_fit"):
        actual = _model(cls).fit(pandas_data)
        assert actual._backend_name == "numpy"
        assert all(isinstance(value, np.ndarray) for value in _attributes(actual).values())
        _assert_fit_matches(actual, expected)
    else:
        # Fit independently on NumPy to isolate each fitted-query regression.
        result = getattr(expected, operation)(pandas_data)
        assert_allclose(result, getattr(expected, operation)(data), rtol=0, atol=0)
        assert isinstance(result, float if operation == "score" else np.ndarray)
    if operation == "series_fit":
        pd.testing.assert_series_equal(pandas_data, before)
    else:
        pd.testing.assert_frame_equal(pandas_data, before)


class _ArrayLike:
    def __init__(self, values):
        self.values = values

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.values, dtype=dtype)
        return array.copy() if copy else array


class _KeyedGetArray(_ArrayLike):
    def get(self, key):
        raise AssertionError("An ordinary keyed get() is not a device transfer")


class _CpuArray(_ArrayLike):
    def cpu(self):
        raise AssertionError("An ordinary cpu() method is not a device transfer")

    def numpy(self):
        raise AssertionError("An ordinary numpy() method is not a device transfer")


class _GetAndCpuArray(_KeyedGetArray, _CpuArray):
    pass


ARRAY_LIKES = (list, tuple, _ArrayLike, _KeyedGetArray, _CpuArray, _GetAndCpuArray)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("container", ARRAY_LIKES, ids=lambda cls: cls.__name__)
def test_ordinary_array_likes_ignore_unrelated_transfer_methods(cls, container, X):
    data = container(X.tolist())
    expected = _model(cls).fit(X)
    actual = _model(cls).fit(data)
    _assert_fit_matches(actual, expected)
    _assert_queries_match(actual, expected, data, X)


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_mixed_dtype_dataframe_is_positional_and_not_mutated(cls, X, pd):
    frame = pd.DataFrame({"integer": (X[:, 0] * 100).astype(np.int64),
                          "single": X[:, 1].astype(np.float32),
                          "flag": X[:, 2] > 0})
    before = frame.copy(deep=True)
    expected = _model(cls).fit(frame.to_numpy(dtype=np.float64))
    actual = _model(cls).fit(frame)
    _assert_fit_matches(actual, expected)
    # Different query labels do not reorder values in this array-like API.
    query = frame.iloc[::2].rename(columns={"integer": "flag", "flag": "integer"})
    _assert_queries_match(actual, expected, query, query.to_numpy(dtype=np.float64))
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_dataframes_retain_validation(cls, bad_value, X, pd):
    bad = X.copy()
    bad[1, 0] = bad_value
    frame = pd.DataFrame(bad)
    with pytest.raises(ValueError, match="finite|NaN"):
        _model(cls).fit(frame)
    model = _model(cls).fit(X)
    for method in QUERIES:
        with pytest.raises(ValueError, match="finite|NaN"):
            getattr(model, method)(frame)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("dtype", ["Float64", "Float32", "Int64"])
def test_nullable_numeric_missing_values_remain_invalid(cls, dtype, X, pd):
    data = (X * 100).astype(np.int64) if dtype == "Int64" else X
    frame = pd.DataFrame(data, dtype=dtype)
    frame.iloc[1, 0] = pd.NA
    with pytest.raises(ValueError, match="finite|NaN"):
        _model(cls).fit(frame)
    model = _model(cls).fit(X)
    for method in QUERIES:
        with pytest.raises(ValueError, match="finite|NaN"):
            getattr(model, method)(frame)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("shape", [(0, 3), (4, 0), (1, 3)])
def test_dataframe_fit_rejects_invalid_shapes(cls, shape, pd):
    with pytest.raises(ValueError, match="sample|feature|2D"):
        _model(cls).fit(pd.DataFrame(np.zeros(shape)))


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_dataframe_queries_retain_shape_and_feature_validation(cls, X, pd):
    model = _model(cls).fit(X)
    for method in QUERIES:
        for data, message in ((np.empty((0, 3)), "sample"),
                              (np.empty((4, 0)), "feature"),
                              (X[:, :2], "features")):
            with pytest.raises(ValueError, match=message):
                getattr(model, method)(pd.DataFrame(data))
        with pytest.raises(ValueError, match="two-dimensional"):
            getattr(model, method)(_GetAndCpuArray(np.ones((2, 3, 1))))


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_series_queries_keep_existing_one_dimensional_semantics(cls, X, pd):
    multivariate = _model(cls).fit(X)
    row = pd.Series(X[0], index=["z", "x", "y"])
    for method in ("predict", "mahalanobis"):
        assert_allclose(getattr(multivariate, method)(row),
                        getattr(multivariate, method)(X[:1]), rtol=0, atol=0)
    univariate = _model(cls).fit(X[:, :1])
    assert_allclose(univariate.score(pd.Series(X[:, 0])),
                    univariate.score(X[:, :1]), rtol=0, atol=0)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("policy", ["explicit_cpu", "global_cpu", "auto_without_gpu"])
def test_dataframe_cpu_and_automatic_routing(cls, policy, X, pd, monkeypatch):
    if policy == "global_cpu":
        monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
    elif policy == "auto_without_gpu":
        monkeypatch.setattr(_device_manager, "_cupy_available", False)
        monkeypatch.setattr(_device_manager, "_torch_available", False)
        monkeypatch.setattr(_device_manager, "_cuda_available", False)
    else:
        # An explicit estimator policy also overrides a strict global GPU policy.
        monkeypatch.setattr(_device_manager, "_current_device", Device.CUDA)
    device = "cpu" if policy == "explicit_cpu" else "auto"
    actual = _model(cls, device).fit(pd.DataFrame(X))
    assert actual._backend_name == "numpy"
    expected = _model(cls).fit(X)
    _assert_fit_matches(actual, expected)
    _assert_queries_match(actual, expected, pd.DataFrame(X), X)


class _MustNotCoerce:
    def __array__(self, dtype=None, copy=None):
        raise AssertionError("Strict unavailable backends must fail before coercion")


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("device", [Device.CUDA, Device.TORCH])
@pytest.mark.parametrize("global_policy", [False, True], ids=["explicit", "global"])
@pytest.mark.parametrize("input_kind", ["dataframe", "coercion_sentinel"])
def test_unavailable_gpu_precedes_covariance_normalization(
    cls, device, global_policy, input_kind, X, pd, monkeypatch
):
    from statgpu.backends._cupy import CuPyBackend
    from statgpu.backends._torch import TorchBackend

    # Mock availability only. No GPU computation or backend array is simulated.
    monkeypatch.setattr(CuPyBackend, "is_available", lambda self: False)
    monkeypatch.setattr(TorchBackend, "is_available", lambda self: False)
    if global_policy:
        monkeypatch.setattr(_device_manager, "_current_device", device)
    model = _model(cls, "auto" if global_policy else device)
    data = pd.DataFrame(X) if input_kind == "dataframe" else _MustNotCoerce()
    requirement = "PyTorch.*CUDA" if device is Device.TORCH else "CuPy.*CUDA"
    with pytest.raises(RuntimeError, match=requirement):
        if input_kind == "dataframe":
            model.fit(data)
        else:
            # The public finite-input guard may already inspect array-likes.
            # This sentinel checks ordering within the repaired boundary.
            model._prepare_covariance_input(data)
    assert not model._fitted
    assert not hasattr(model, "covariance_")


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("input_kind", ["dataframe", "Float64", "Float32", "Int64",
                                        "mixed_bool", "get", "cpu", "get_and_cpu"])
def test_real_torch_cpu_fitted_queries_keep_backend_and_state(
    cls, input_kind, X, pd, monkeypatch
):
    torch = pytest.importorskip("torch")
    numpy_data = X
    if input_kind in ("Float64", "Float32", "Int64"):
        numpy_dtype = np.dtype(input_kind.lower())
        numpy_data = ((X * 100).astype(numpy_dtype) if numpy_dtype.kind == "i"
                      else X.astype(numpy_dtype))
        data = pd.DataFrame(numpy_data, dtype=input_kind)
    elif input_kind == "mixed_bool":
        data = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "flag": X[:, 2] > 0})
        numpy_data = data.to_numpy(dtype=np.float64)
    else:
        data = {"dataframe": pd.DataFrame, "get": _KeyedGetArray,
                "cpu": _CpuArray, "get_and_cpu": _GetAndCpuArray}[input_kind](X)
    expected = _model(cls).fit(X)
    actual = _model(cls, "auto").fit(torch.as_tensor(X, dtype=torch.float64))
    attributes = _attributes(actual)
    snapshots = {name: value.clone() for name, value in attributes.items()}
    assert actual._backend_name == "torch"
    with monkeypatch.context() as changed:
        changed.setattr(_device_manager, "_current_device", Device.CUDA)
        backend, _, prepared = actual._prepare_covariance_input(data, fitted=True)
        assert backend == "torch"
        assert isinstance(prepared, torch.Tensor)
        assert prepared.device == actual.covariance_.device == torch.device("cpu")
        assert prepared.dtype == torch.float64
        _assert_queries_match(actual, expected, data, numpy_data)
        assert _device_manager._current_device is Device.CUDA
    assert _device_manager._current_device is Device.AUTO
    for name, value in attributes.items():
        assert getattr(actual, name) is value
        assert isinstance(value, torch.Tensor) and value.device.type == "cpu"
        assert torch.equal(value, snapshots[name])
    _assert_fit_matches(actual, expected)


def test_dataframe_cv_folds_scores_and_refit_keep_initial_numpy_ownership(X, pd, monkeypatch):
    expected = _model(GraphicalLassoCV).fit(X)
    fitted_calls, scored_calls = [], []
    fit_prepared = GraphicalLasso._fit_prepared
    score = GraphicalLasso.score

    def observe_fit(self, backend, xp, data):
        fitted_calls.append((backend, type(data), len(data)))
        monkeypatch.setattr(_device_manager, "_current_device", Device.TORCH)
        return fit_prepared(self, backend, xp, data)

    def observe_score(self, data, y=None):
        scored_calls.append((self._backend_name, type(data), len(data)))
        return score(self, data, y)

    monkeypatch.setattr(GraphicalLasso, "_fit_prepared", observe_fit)
    monkeypatch.setattr(GraphicalLasso, "score", observe_score)
    monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
    actual = _model(GraphicalLassoCV, "auto").fit(pd.DataFrame(X))
    assert fitted_calls == [("numpy", np.ndarray, 30)] * 6 + [("numpy", np.ndarray, 45)]
    assert scored_calls == [("numpy", np.ndarray, 15)] * 6
    assert actual._backend_name == "numpy"
    _assert_fit_matches(actual, expected)
    _assert_queries_match(actual, expected, pd.DataFrame(X), X)


@pytest.mark.parametrize("cls", [EmpiricalCovariance, LedoitWolf, OAS,
                                 ShrunkCovariance, GraphicalLasso])
@pytest.mark.parametrize("assume_centered", [False, True])
def test_dataframe_matches_aligned_sklearn_reference(cls, assume_centered, X, pd):
    reference = pytest.importorskip("sklearn.covariance")
    params = {"assume_centered": assume_centered}
    if cls is GraphicalLasso:
        params.update(alpha=0.08, max_iter=100, tol=1e-7)
    actual = _model(cls, **params).fit(pd.DataFrame(X))
    expected = getattr(reference, cls.__name__)(**params).fit(X)
    assert_allclose(actual.covariance_, expected.covariance_, rtol=2e-5, atol=2e-6)
    assert_allclose(actual.precision_, expected.precision_, rtol=2e-5, atol=2e-6)
    assert_allclose(actual.location_, expected.location_, rtol=1e-12, atol=1e-12)
    query = pd.DataFrame(X[::3] + 0.2)
    assert_allclose(actual.score(query), expected.score(query.to_numpy()), rtol=2e-5)
    assert_allclose(actual.mahalanobis(query), expected.mahalanobis(query.to_numpy()),
                    rtol=2e-5, atol=2e-6)


def test_dataframe_empirical_matches_mle_and_gaussian_density(X, pd):
    from scipy.stats import multivariate_normal

    actual = _model(EmpiricalCovariance).fit(pd.DataFrame(X))
    location = X.mean(axis=0)
    covariance = np.cov(X, rowvar=False, bias=True)
    precision = np.linalg.inv(covariance)
    query = X[::3] + 0.2
    centered = query - location
    distances = np.einsum("ij,jk,ik->i", centered, precision, centered)
    assert_allclose(actual.covariance_, covariance, rtol=1e-12, atol=1e-12)
    assert_allclose(actual.precision_, precision, rtol=1e-12, atol=1e-12)
    assert_allclose(actual.score(pd.DataFrame(query)),
                    multivariate_normal.logpdf(query, location, covariance).mean(), rtol=1e-12)
    assert_allclose(actual.predict(pd.DataFrame(query)), distances, rtol=1e-12)
    assert_array_equal(actual.predict(pd.DataFrame(query)), actual.mahalanobis(query))


def _cuda_runtime(device):
    if device == "torch":
        xp = pytest.importorskip("torch")
        if not xp.cuda.is_available():
            pytest.skip("physical Torch CUDA runtime unavailable")
    else:
        xp = pytest.importorskip("cupy")
        try:
            available = xp.cuda.runtime.getDeviceCount() > 0
        except (xp.cuda.runtime.CUDARuntimeError, RuntimeError, OSError) as exc:
            pytest.skip(f"physical CuPy CUDA runtime unavailable: {exc}")
        if not available:
            pytest.skip("physical CuPy CUDA runtime unavailable")
    return xp


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("device", ["torch", "cuda"])
def test_physical_cuda_dataframe_fit_and_fitted_queries(cls, device, X, pd, monkeypatch):
    xp = _cuda_runtime(device)
    mixed = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "flag": X[:, 2] > 0})
    frames = ((pd.DataFrame(X), X), (pd.DataFrame(X, dtype="Float64"), X),
              (mixed, mixed.to_numpy(dtype=np.float64)))
    queries = frames + ((_GetAndCpuArray(X), X),)
    for frame, numpy_fit in frames:
        expected = _model(cls).fit(numpy_fit)
        actual = _model(cls, device).fit(frame)
        assert actual._backend_name == ("torch" if device == "torch" else "cupy")
        for value in _attributes(actual).values():
            if device == "torch":
                assert isinstance(value, xp.Tensor) and value.is_cuda
            else:
                assert isinstance(value, xp.ndarray)
            assert value.device == actual.covariance_.device
        _assert_fit_matches(actual, expected)
        monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
        for data, numpy_data in queries:
            _, _, prepared = actual._prepare_covariance_input(data, fitted=True)
            assert prepared.device == actual.covariance_.device
            _assert_queries_match(actual, expected, data, numpy_data)
