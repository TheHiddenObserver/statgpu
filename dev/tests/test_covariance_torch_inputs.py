"""Torch-to-NumPy covariance dtype/view regressions and ownership controls.

The NumPy oracle uses each tensor's logical, quantized values in float64.
Only float8 cases may skip on old Torch releases such as 2.0.1. Physical CUDA
cases require real hardware; single- and two-GPU selections are independent.
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
REGRESSIONS = ("bfloat16", "float8_e5m2", "lazy_negative")
INPUT_KINDS = REGRESSIONS + (
    "float16", "float32", "float64", "int64", "bool",
    "requires_grad", "noncontiguous",
)
QUERIES = ("score", "predict", "mahalanobis")
ARRAY_ATTRIBUTES = (
    "covariance_", "precision_", "location_", "support_",
    "raw_covariance_", "raw_location_", "dist_",
)


@pytest.fixture(autouse=True)
def automatic_global_device(monkeypatch):
    monkeypatch.setattr(_device_manager, "_current_device", Device.AUTO)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.fixture
def X():
    data = np.random.default_rng(192).normal(size=(45, 3))
    data[:, 1] += 0.45 * data[:, 0]
    return data


def _model(cls, device="cpu"):
    params = {"device": device}
    if cls is GraphicalLasso:
        params.update(alpha=0.08, max_iter=100, tol=1e-7)
    elif cls is GraphicalLassoCV:
        params.update(alphas=[0.03, 0.12], cv=3, random_state=7,
                      max_iter=100, tol=1e-7)
    elif cls is MinCovDet:
        params.update(support_fraction=0.75, random_state=7)
    return cls(**params)


def _tensor(torch, values, kind, device="cpu"):
    if kind == "float8_e5m2" and not hasattr(torch, "float8_e5m2"):
        pytest.skip("Torch float8_e5m2 is unavailable (including Torch 2.0.1)")
    base = torch.tensor(values, dtype=torch.float64, device=device)
    if kind == "lazy_negative":
        # Public operations create a real float64 view with unresolved neg bit.
        result = torch.complex(torch.zeros_like(base), base).conj().imag
        assert result.dtype == torch.float64 and result.is_neg()
        return result
    if kind == "requires_grad":
        return base.requires_grad_(True)
    if kind == "noncontiguous":
        result = base.repeat_interleave(2, dim=1)[:, ::2]
        assert not result.is_contiguous()
        return result.requires_grad_(True)
    if kind == "int64":
        return (base * 10).to(torch.int64)
    if kind == "bool":
        return base > 0
    return base.to(getattr(torch, kind))


def _logical_numpy(tensor):
    # Promote by an actual Torch operation before the NumPy boundary. Copy so
    # the oracle/snapshot cannot alias input storage, including lazy views.
    import torch

    promoted = tensor.detach().cpu().to(dtype=torch.float64).resolve_conj().resolve_neg()
    return promoted.numpy().copy()


def _snapshot(tensor):
    metadata = (tensor.dtype, tensor.device, tensor.shape, tensor.stride(),
                tensor.storage_offset(), tensor.data_ptr(), tensor.requires_grad,
                tensor.is_leaf, tensor.is_neg(), tensor.is_conj(), tensor._version)
    base = tensor._base
    # Preserve the entire backing array as well as the view's logical values.
    backing = None if base is None else base.detach().resolve_conj().resolve_neg().cpu().clone()
    return metadata, _logical_numpy(tensor), backing


def _assert_unchanged(tensor, before):
    metadata, values, backing = before
    after = _snapshot(tensor)
    assert after[0] == metadata
    assert_array_equal(after[1], values)
    if backing is not None:
        import torch

        torch.testing.assert_close(after[2], backing, rtol=0, atol=0, equal_nan=True)
    if tensor.is_leaf:
        assert tensor.grad is None


def _attributes(model):
    return {name: getattr(model, name) for name in ARRAY_ATTRIBUTES
            if hasattr(model, name)}


def _assert_numpy_model(model):
    assert model._backend_name == "numpy"
    for name, value in _attributes(model).items():
        assert isinstance(value, np.ndarray), name
        assert value.dtype == (np.bool_ if name == "support_" else np.float64), name


def _assert_fit_matches(actual, expected, *, native=False):
    # Preserve the existing covariance backend parity thresholds. NumPy-target
    # conversion has a tighter threshold because both fits use the same solver.
    rtol, atol = (5e-6, 5e-7) if native else (1e-12, 1e-12)
    assert actual.n_samples_ == expected.n_samples_
    assert actual.n_features_ == expected.n_features_
    assert _attributes(actual).keys() == _attributes(expected).keys()
    for name, value in _attributes(actual).items():
        want = _to_numpy(getattr(expected, name))
        if name == "support_":
            assert_array_equal(_to_numpy(value), want)
        else:
            assert_allclose(_to_numpy(value), want, rtol=rtol, atol=atol, err_msg=name)
    if hasattr(actual, "shrinkage_"):
        assert_allclose(actual.shrinkage_, expected.shrinkage_, rtol=1e-12, atol=1e-12)
    if isinstance(actual, GraphicalLassoCV):
        assert actual.alpha_ == expected.alpha_
        for field in ("alpha", "scores", "mean_score"):
            assert_allclose([row[field] for row in actual.cv_results_],
                            [row[field] for row in expected.cv_results_],
                            rtol=rtol, atol=atol, err_msg=field)


def _assert_queries_match(actual, expected, tensor, values, *, native=False):
    rtol, atol = (5e-6, 5e-7) if native else (1e-12, 1e-12)
    for method in QUERIES:
        result = getattr(actual, method)(tensor)
        assert isinstance(result, float if method == "score" else np.ndarray)
        assert_allclose(result, getattr(expected, method)(values),
                        rtol=rtol, atol=atol, err_msg=method)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("kind", INPUT_KINDS)
def test_numpy_target_fit_accepts_torch_dtypes_and_views(cls, kind, X, torch):
    tensor = _tensor(torch, X, kind)
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    actual = _model(cls).fit(tensor)
    _assert_numpy_model(actual)
    _assert_fit_matches(actual, expected)
    _assert_queries_match(actual, expected, tensor, values)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("method", QUERIES)
def test_numpy_fitted_queries_accept_torch_dtypes_and_views(cls, kind, method, X, torch, monkeypatch):
    model = _model(cls).fit(X)
    tensor = _tensor(torch, X[::3] + 0.2, kind)
    before = _snapshot(tensor)
    expected = getattr(model, method)(_logical_numpy(tensor))
    fitted = {name: (value, value.copy()) for name, value in _attributes(model).items()}
    # A changed global GPU policy must not make a CPU-fitted query select Torch.
    monkeypatch.setattr(_device_manager, "_current_device", Device.TORCH)
    result = getattr(model, method)(tensor)
    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert isinstance(result, float if method == "score" else np.ndarray)
    _assert_numpy_model(model)
    for name, (original, values) in fitted.items():
        assert getattr(model, name) is original
        assert_array_equal(getattr(model, name), values)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("kind", REGRESSIONS)
def test_global_cpu_converts_torch_then_keeps_fitted_numpy(cls, kind, X, torch, monkeypatch):
    tensor = _tensor(torch, X, kind)
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
    actual = _model(cls, "auto").fit(tensor)
    _assert_numpy_model(actual)
    _assert_fit_matches(actual, expected)
    monkeypatch.setattr(_device_manager, "_current_device", Device.CUDA)
    _assert_queries_match(actual, expected, tensor, values)
    _assert_numpy_model(actual)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("kind", REGRESSIONS)
def test_true_auto_keeps_torch_cpu_native_for_dtypes_and_views(cls, kind, X, torch, monkeypatch):
    tensor = _tensor(torch, X, kind)
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    actual = _model(cls, "auto").fit(tensor)
    assert actual._backend_name == "torch"
    for name, value in _attributes(actual).items():
        assert isinstance(value, torch.Tensor) and value.device.type == "cpu"
        assert value.dtype == (torch.bool if name == "support_" else torch.float64)
    _assert_fit_matches(actual, expected, native=True)
    monkeypatch.setattr(_device_manager, "_current_device", Device.CUDA)
    _assert_queries_match(actual, expected, tensor, values, native=True)
    assert actual._backend_name == "torch"
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("kind", REGRESSIONS)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_torch_inputs_retain_validation(cls, kind, bad_value, X, torch):
    bad = X.copy()
    bad[1, 0] = bad_value
    tensor = _tensor(torch, bad, kind)
    before = _snapshot(tensor)
    with pytest.raises(ValueError, match="finite|NaN"):
        _model(cls).fit(tensor)
    model = _model(cls).fit(X)
    for method in QUERIES:
        with pytest.raises(ValueError, match="finite|NaN"):
            getattr(model, method)(tensor)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("kind", REGRESSIONS)
@pytest.mark.parametrize("shape", [(0, 3), (1, 3), (4, 0), (2, 3, 1)])
def test_invalid_torch_fit_shapes_retain_validation(cls, kind, shape, torch):
    tensor = _tensor(torch, np.zeros(shape), kind)
    with pytest.raises(ValueError, match="sample|feature|2D|two-dimensional"):
        _model(cls).fit(tensor)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("kind", REGRESSIONS)
def test_torch_queries_retain_shape_and_feature_validation(cls, kind, X, torch):
    model = _model(cls).fit(X)
    for method in QUERIES:
        for shape, message in (((0, 3), "sample"), ((4, 0), "feature"),
                               ((4, 2), "features"), ((2, 3, 1), "two-dimensional")):
            tensor = _tensor(torch, np.zeros(shape), kind)
            with pytest.raises(ValueError, match=message):
                getattr(model, method)(tensor)


@pytest.mark.parametrize("kind", REGRESSIONS)
def test_robust_torch_conversion_preserves_contaminated_support(kind, X, torch):
    data = X.copy()
    data[:8] += np.array([25.0, -35.0, 45.0])
    tensor = _tensor(torch, data, kind)
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(MinCovDet).fit(values)
    actual = _model(MinCovDet).fit(tensor)
    _assert_fit_matches(actual, expected)
    assert np.count_nonzero(actual.support_[:8]) == 0
    _assert_queries_match(actual, expected, tensor, values)
    _assert_unchanged(tensor, before)


@pytest.mark.parametrize("kind", REGRESSIONS)
@pytest.mark.parametrize("policy", ["explicit_cpu", "global_cpu", "native_auto"])
def test_cv_torch_conversion_preserves_folds_selection_and_final_refit(kind, policy, X, torch, monkeypatch):
    tensor = _tensor(torch, X, kind)
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(GraphicalLassoCV).fit(values)
    fit_calls, score_calls = [], []
    original_fit = GraphicalLasso._fit_prepared
    original_score = GraphicalLasso.score
    backend = "torch" if policy == "native_auto" else "numpy"
    native = policy == "native_auto"
    if policy == "global_cpu":
        monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)

    def observe_fit(self, backend_name, xp, prepared):
        assert backend_name == backend
        assert prepared.dtype == (torch.float64 if native else np.float64)
        fit_calls.append((self.alpha, _logical_numpy(prepared) if native else prepared.copy()))
        monkeypatch.setattr(_device_manager, "_current_device", Device.CUDA)
        return original_fit(self, backend_name, xp, prepared)

    def observe_score(self, prepared, y=None):
        assert self._backend_name == backend
        score_calls.append(_logical_numpy(prepared) if native else prepared.copy())
        return original_score(self, prepared, y=y)

    monkeypatch.setattr(GraphicalLasso, "_fit_prepared", observe_fit)
    monkeypatch.setattr(GraphicalLasso, "score", observe_score)
    actual = _model(GraphicalLassoCV, "cpu" if policy == "explicit_cpu" else "auto").fit(tensor)
    folds = np.array_split(np.random.RandomState(7).permutation(len(values)), 3)
    expected_train, expected_test = [], []
    for alpha in (0.03, 0.12):
        for index, test in enumerate(folds):
            train = np.concatenate([folds[j] for j in range(3) if j != index])
            expected_train.append((alpha, values[train]))
            expected_test.append(values[test])
    expected_train.append((actual.alpha_, values))
    assert len(fit_calls) == 7 and len(score_calls) == 6
    for (alpha, got), (wanted_alpha, want) in zip(fit_calls, expected_train):
        assert alpha == wanted_alpha
        assert_array_equal(got, want)
    for got, want in zip(score_calls, expected_test):
        assert_array_equal(got, want)
    _assert_fit_matches(actual, expected, native=native)
    assert all(np.isfinite(row["scores"]).all() for row in actual.cv_results_)
    assert actual._backend_name == backend
    _assert_unchanged(tensor, before)


def test_torch_to_numpy_moves_to_cpu_before_float64_cast_routing(torch, monkeypatch):
    """Observe real CPU conversion order; this is not physical MPS evidence."""
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16,
                          requires_grad=True)
    calls = []
    original_cpu = torch.Tensor.cpu
    original_to = torch.Tensor.to
    original_numpy = torch.Tensor.numpy

    def cpu(self, *args, **kwargs):
        calls.append("cpu")
        return original_cpu(self, *args, **kwargs)

    def to(self, *args, **kwargs):
        assert torch.float64 in args or kwargs.get("dtype") == torch.float64
        calls.append("float64")
        return original_to(self, *args, **kwargs)

    def numpy(self, *args, **kwargs):
        calls.append("numpy")
        assert self.device.type == "cpu" and self.dtype == torch.float64
        assert not self.requires_grad and not self.is_neg()
        return original_numpy(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.Tensor, "numpy", numpy)
    backend, _, prepared = _model(EmpiricalCovariance)._prepare_covariance_input(tensor)
    assert backend == "numpy"
    assert calls == ["cpu", "float64", "numpy"]
    assert_array_equal(prepared, [[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_physical_mps_to_numpy_preserves_float32_input_conversion(cls, X, torch):
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        pytest.skip("physical Torch MPS runtime unavailable")
    # MPS cannot represent float64. The existing explicit CPU transfer must
    # happen before covariance's dtype promotion, including detached queries.
    tensor = torch.tensor(X.astype(np.float32), dtype=torch.float32,
                          device="mps", requires_grad=True)
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    actual = _model(cls).fit(tensor)
    _assert_numpy_model(actual)
    _assert_fit_matches(actual, expected)
    _assert_queries_match(actual, expected, tensor, values)
    _assert_queries_match(expected, expected, tensor, values)
    _assert_unchanged(tensor, before)
    torch.mps.synchronize()


def _require_cuda(torch, minimum_devices):
    if not torch.cuda.is_available() or torch.cuda.device_count() < minimum_devices:
        pytest.skip(f"physical Torch CUDA test requires {minimum_devices} CUDA device(s)")


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("kind", REGRESSIONS)
@pytest.mark.parametrize("policy", ["explicit_cpu", "global_cpu"])
def test_physical_single_gpu_torch_to_numpy_fit_and_queries(cls, kind, policy, X, torch, monkeypatch):
    _require_cuda(torch, 1)
    tensor = _tensor(torch, X, kind, device="cuda:0")
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    if policy == "global_cpu":
        monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
    actual = _model(cls, "cpu" if policy == "explicit_cpu" else "auto").fit(tensor)
    _assert_numpy_model(actual)
    _assert_fit_matches(actual, expected)
    monkeypatch.setattr(_device_manager, "_current_device", Device.TORCH)
    _assert_queries_match(actual, expected, tensor, values)
    # Also isolate NumPy-trained queries from direct-fit conversion.
    _assert_queries_match(expected, expected, tensor, values)
    _assert_unchanged(tensor, before)
    torch.cuda.synchronize(0)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("kind", REGRESSIONS)
def test_physical_single_gpu_true_auto_keeps_torch_native(cls, kind, X, torch, monkeypatch):
    _require_cuda(torch, 1)
    tensor = _tensor(torch, X, kind, device="cuda:0")
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    actual = _model(cls, "auto").fit(tensor)
    assert actual._backend_name == "torch"
    for name, value in _attributes(actual).items():
        assert isinstance(value, torch.Tensor) and value.device == tensor.device
        assert value.dtype == (torch.bool if name == "support_" else torch.float64)
    _assert_fit_matches(actual, expected, native=True)
    monkeypatch.setattr(_device_manager, "_current_device", Device.CPU)
    _assert_queries_match(actual, expected, tensor, values, native=True)
    _assert_unchanged(tensor, before)
    torch.cuda.synchronize(0)


@pytest.mark.parametrize("cls", ESTIMATORS)
@pytest.mark.parametrize("kind", REGRESSIONS)
def test_physical_two_gpu_nondefault_torch_to_numpy(cls, kind, X, torch, monkeypatch):
    _require_cuda(torch, 2)
    tensor = _tensor(torch, X, kind, device="cuda:1")
    before = _snapshot(tensor)
    values = _logical_numpy(tensor)
    expected = _model(cls).fit(values)
    with torch.cuda.device(0):
        actual = _model(cls).fit(tensor)
        _assert_numpy_model(actual)
        _assert_fit_matches(actual, expected)
        monkeypatch.setattr(_device_manager, "_current_device", Device.TORCH)
        _assert_queries_match(actual, expected, tensor, values)
        _assert_queries_match(expected, expected, tensor, values)
        assert torch.cuda.current_device() == 0
    assert tensor.device.index == 1
    _assert_unchanged(tensor, before)
    torch.cuda.synchronize(1)
