"""Preserve Student-t low-df dispatch while making the class its source owner."""

import importlib
import inspect
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from statgpu.inference import TDistributionBase, get_distribution
from statgpu.inference import _low_df_reference_contract as legacy
from statgpu.inference._reference_distribution import two_sided_reference_inference


@pytest.mark.parametrize(
    "name, signature",
    [
        ("two_sided_pvalue", "(self, stat_abs, df)"),
        ("two_sided_critical_value", "(self, alpha, df, *, max_bisect_steps=60)"),
    ],
)
def test_t_reference_signature_pickle_and_unwrapped_compatibility(name, signature):
    method = getattr(TDistributionBase, name)
    assert str(inspect.signature(method)) == signature
    assert inspect.signature(method.__wrapped__) == inspect.signature(method)
    assert method.__name__ == name
    assert method.__qualname__ == f"TDistributionBase.{name}"
    assert method.__module__ == "statgpu.inference._distributions_backend"
    assert method._statgpu_original is method.__wrapped__
    assert method._statgpu_low_df_reference is True
    assert pickle.loads(pickle.dumps(method)) is method
    bound = pickle.loads(pickle.dumps(getattr(TDistributionBase(SimpleNamespace()), name)))
    assert bound.__func__ is method
    # Registered special-function providers contain modules and were never picklable.
    with pytest.raises(TypeError, match="cannot pickle 'module' object"):
        pickle.dumps(getattr(get_distribution("t", backend="numpy"), name))


@pytest.mark.parametrize("df", [1, 2, np.float64(1), "2"])
@pytest.mark.parametrize("statistic", [0, -1, [0, -1, 4], np.array([-3, 2], dtype=np.float32), np.array([1e16, 1e154, np.inf, np.nan])])
def test_low_df_pvalues_match_existing_stable_reference_exactly(df, statistic):
    distribution = get_distribution("t", backend="numpy")
    statistic = np.asarray(statistic)
    before = statistic.copy()
    with np.errstate(all="ignore"):
        actual = distribution.two_sided_pvalue(statistic, df)
        expected, _ = two_sided_reference_inference(
            np.abs(np.asarray(statistic, dtype=np.float64)), distribution="t",
            alpha=0.05, backend="numpy", xp=np, df=float(df),
        )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(statistic, before)
    assert np.asarray(actual).dtype == np.float64


@pytest.mark.parametrize("df", [1, 2])
@pytest.mark.parametrize("alpha", [0.05, 0.2, np.float32(0.1), 1e-12])
def test_low_df_critical_value_ignores_bisection_budget_as_before(df, alpha):
    distribution = get_distribution("t", backend="numpy")
    _, expected = two_sided_reference_inference(
        np.asarray(0.), distribution="t", alpha=alpha, backend="numpy", xp=np, df=df,
    )
    actual = distribution.two_sided_critical_value(alpha, df, max_bisect_steps=object())
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("df", [0, -1, 0.5, 3, 25, np.nan])
def test_general_df_matches_original_generic_methods(df):
    distribution = get_distribution("t", backend="numpy")
    values = np.asarray([0., -1., 3.])
    for name, arg in (("two_sided_pvalue", values), ("two_sided_critical_value", 0.05)):
        method = getattr(TDistributionBase, name)
        with np.errstate(all="ignore"):
            expected = method.__wrapped__(distribution, arg, df)
            actual = method(distribution, arg, df)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("alpha", [0, 1, -1, np.nan, np.inf])
@pytest.mark.parametrize("df", [1, 2, 3])
def test_invalid_alpha_retains_low_df_error_and_general_df_nan(df, alpha):
    distribution = get_distribution("t", backend="numpy")
    if df in (1, 2):
        with pytest.raises(ValueError, match="alpha must be finite and strictly between 0 and 1"):
            distribution.two_sided_critical_value(alpha, df)
    else:
        assert np.isnan(distribution.two_sided_critical_value(alpha, df))


def test_df_conversion_and_generic_dispatch_order(monkeypatch):
    distribution = get_distribution("t", backend="numpy")
    events = []

    class DF:
        def __float__(self):
            events.append("df")
            return 3.

    df = DF()
    marker = object()
    def generic(statistic, passed_df):
        events.append("generic")
        assert statistic is marker and passed_df is df
        return "pvalue"
    monkeypatch.setattr(distribution, "_two_sided_pvalue_standard", generic)
    assert distribution.two_sided_pvalue(marker, df) == "pvalue"
    assert events == ["df", "generic"]
    with pytest.raises(TypeError):
        distribution.two_sided_critical_value(0, None)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        distribution.two_sided_pvalue(1, 1, unknown=True)


@pytest.mark.parametrize("backend", ["numpy", "cupy", "torch"])
@pytest.mark.parametrize("has_statistic_device", [False, True])
def test_low_df_backend_context_and_concrete_device_forwarding(monkeypatch, backend, has_statistic_device):
    # Spies prove dispatch/ownership only; this is not accelerator execution.
    conversions = []
    class TorchSpy:
        float64 = np.float64
        abs = staticmethod(np.abs)
        @staticmethod
        def as_tensor(value, **kwargs):
            conversions.append(("tensor", kwargs))
            return np.asarray([-2., 3.])
        @staticmethod
        def zeros(shape, **kwargs):
            conversions.append(("zeros", kwargs))
            return np.zeros(shape)
    xp = TorchSpy if backend == "torch" else np
    sf = SimpleNamespace(_device="cuda:3")
    if backend != "numpy":
        setattr(sf, "_torch" if backend == "torch" else "_cp", xp)
    distribution = TDistributionBase(sf)
    statistic = SimpleNamespace(device="cuda:7") if backend == "torch" and has_statistic_device else [-2., 3.]
    calls = []
    def reference(value, **kwargs):
        calls.append((value, kwargs))
        return "pvalue", "critical"
    monkeypatch.setattr(legacy, "two_sided_reference_inference", reference)
    assert distribution.two_sided_pvalue(statistic, 1) == "pvalue"
    assert distribution.two_sided_critical_value(0.2, 2) == "critical"
    np.testing.assert_array_equal(calls[0][0], [2., 3.])
    assert np.asarray(calls[1][0]).shape == ()
    for index, (_, kwargs) in enumerate(calls):
        expected_device = None
        if backend == "torch":
            expected_device = "cuda:7" if index == 0 and has_statistic_device else "cuda:3"
        assert kwargs == dict(distribution="t", alpha=0.05 if index == 0 else 0.2,
                              backend=backend, xp=xp, df=float(index + 1), device=expected_device)
    if backend == "torch":
        assert [kind for kind, _ in conversions] == ["tensor", "zeros"]
        assert [kwargs["device"] for _, kwargs in conversions] == [calls[0][1]["device"], "cuda:3"]


def test_legacy_installer_is_idempotent_and_helpers_remain_importable():
    methods = (TDistributionBase.two_sided_pvalue, TDistributionBase.two_sided_critical_value)
    legacy._install_low_df_reference_contract()
    importlib.reload(legacy)
    legacy._install_low_df_reference_contract()
    assert methods == (TDistributionBase.two_sided_pvalue, TDistributionBase.two_sided_critical_value)
    for name in ("_backend_context", "_statistic_abs_on_backend", "_reference_scalar"):
        assert callable(getattr(legacy, name))


@pytest.mark.parametrize("first", ["_distributions_backend", "_reference_distribution", "_low_df_reference_contract"])
def test_cold_import_orders_retain_stable_tails(first):
    root = Path(__file__).resolve().parents[2]
    script = f'''import importlib
importlib.import_module("statgpu.inference.{first}")
from statgpu.inference import get_distribution
value = get_distribution("t", backend="numpy").two_sided_pvalue(1e154, 2)
assert value > 0
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_t_reference_methods_are_defined_on_the_class_not_installed():
    owner = "_distributions_backend.py"
    assert Path(TDistributionBase.two_sided_pvalue.__code__.co_filename).name == owner
    assert Path(TDistributionBase.two_sided_critical_value.__code__.co_filename).name == owner
    # A compatibility installer must not replace a caller's own customization.
    sentinel = lambda *args, **kwargs: None
    original = TDistributionBase.two_sided_pvalue
    try:
        TDistributionBase.two_sided_pvalue = sentinel
        legacy._install_low_df_reference_contract()
        assert TDistributionBase.two_sided_pvalue is sentinel
    finally:
        TDistributionBase.two_sided_pvalue = original


def test_legacy_reference_helpers_keep_backend_priority_and_fallback():
    # CuPy context takes precedence if both provider attributes are present.
    cupy, torch = object(), object()
    sf = SimpleNamespace(_cp=cupy, _torch=torch, _device="cuda:3")
    assert legacy._backend_context(TDistributionBase(sf)) == ("cupy", cupy, None)
    sf = SimpleNamespace(_torch=torch, _device="cuda:3")
    distribution = TDistributionBase(sf)
    assert legacy._backend_context(distribution, SimpleNamespace(device="cpu")) == ("torch", torch, "cpu")
    assert legacy._backend_context(distribution, SimpleNamespace(device=None)) == ("torch", torch, "cuda:3")
    assert legacy._backend_context(TDistributionBase(SimpleNamespace(_torch=torch))) == ("torch", torch, None)


@pytest.mark.parametrize("name", ["two_sided_pvalue", "two_sided_critical_value"])
def test_historical_class_method_pickle_lookup_remains_available(name):
    # Explicit historical module/class/name lookup, not only current dumps().
    payload = (
        "cbuiltins\ngetattr\n(cstatgpu.inference._distributions_backend\n"
        f"TDistributionBase\nV{name}\ntR."
    ).encode()
    assert pickle.loads(payload) is getattr(TDistributionBase, name)
