"""Native Torch model-X keeps input placement and a local construction RNG.

CPU tests simulate CUDA availability without redirecting allocation. A separate
ordinal-routing probe is explicitly simulated; CUDA integration tests execute
only when the required physical device exists. These tests do not establish
statistical FDR validity or bitwise numerical parity across devices.
"""
from __future__ import annotations

import inspect
import pydoc
from pathlib import Path

import numpy as np
import pytest

from statgpu.feature_selection import (
    FixedXKnockoffSelector,
    KnockoffSelector,
    fixed_x_knockoff_filter,
    knockoff_filter,
    model_x_knockoff_filter,
)

ROOT = Path(__file__).resolve().parents[2]
MODELX_ENTRYPOINTS = ("function", "unified", "selector")


def _problem():
    rng = np.random.default_rng(824)
    X = rng.normal(size=(80, 5))
    y = 2 * X[:, 0] + rng.normal(size=80)
    # Under this known independent-Gaussian feature model, an independent draw
    # is exchangeable with X and independent of y given X. This is a device
    # control, not a recipe for supplying arbitrary same-shaped knockoffs.
    Xk = rng.normal(size=X.shape)
    return X, y, Xk


def _result(entrypoint, kind, X, y, **kwargs):
    if entrypoint == "function":
        function = fixed_x_knockoff_filter if kind == "fixed_x" else model_x_knockoff_filter
        return function(X, y, **kwargs)
    if entrypoint == "unified":
        return knockoff_filter(X, y, knockoff_type=kind, **kwargs)
    Xk = kwargs.pop("Xk", None)
    if entrypoint == "fixed_selector":
        return FixedXKnockoffSelector(**kwargs).fit(X, y, Xk=Xk).result_
    return KnockoffSelector(knockoff_type=kind, **kwargs).fit(X, y, Xk=Xk).result_


@pytest.fixture
def allocation_probe(monkeypatch):
    torch = pytest.importorskip("torch")
    from statgpu.feature_selection import _knockoff

    original_generator = torch.Generator
    original_randn = torch.randn
    original_matmul = torch.Tensor.__matmul__
    original_statistics = _knockoff._compute_w_statistics
    requests, noise, noise_products, statistic_devices = [], [], [], []
    generators = []

    def generator(*args, **kwargs):
        assert not args, "The probe expects an explicit device keyword."
        requests.append(("Generator", str(kwargs["device"])))
        output = original_generator(*args, **kwargs)
        generators.append(output)
        return output

    def randn(*args, **kwargs):
        requests.append(("randn", str(kwargs["device"])))
        assert kwargs["generator"] is generators[-1]
        assert kwargs["generator"].device == torch.device(kwargs["device"])
        output = original_randn(*args, **kwargs)
        noise.append(output)
        return output

    def matmul(left, right):
        if any(left is draw for draw in noise):
            noise_products.append({
                "left_shape": tuple(left.shape),
                "right_shape": tuple(right.shape),
                "right_device": str(right.device),
                "right_finite": bool(torch.isfinite(right).all()),
            })
        return original_matmul(left, right)

    def statistics(X, Xk, y, **kwargs):
        statistic_devices.append(tuple(str(value.device) for value in (X, Xk, y))
                                 if kwargs["backend_name"] == "torch" else ("numpy",) * 3)
        return original_statistics(X, Xk, y, **kwargs)

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(917)
        monkeypatch.setattr(torch, "Generator", generator)
        monkeypatch.setattr(torch, "randn", randn)
        monkeypatch.setattr(torch.Tensor, "__matmul__", matmul)
        monkeypatch.setattr(_knockoff, "_compute_w_statistics", statistics)
        yield torch, requests, noise_products, statistic_devices


def _inputs(torch, container):
    values = _problem()
    if container == "torch":
        return tuple(torch.as_tensor(value, device="cpu") for value in values)
    return values


@pytest.mark.parametrize("draws", [1, 3, None], ids=["one", "three", "default"])
@pytest.mark.parametrize("cuda_available", [False, True])
@pytest.mark.parametrize("container,backend", [("numpy", "torch"), ("torch", "torch"), ("torch", "auto")])
@pytest.mark.parametrize("entrypoint", MODELX_ENTRYPOINTS)
def test_generated_modelx_keeps_cpu_input_device(
    allocation_probe, monkeypatch, cuda_available, container, backend, entrypoint, draws,
):
    torch, requests, noise_products, statistic_devices = allocation_probe
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    X, y, _ = _inputs(torch, container)
    opts = {"backend": backend, "compat_mode": "statgpu", "random_state": 21,
            "method": "corr_diff"}
    if draws is not None:
        opts["modelx_draws"] = draws
    result = _result(entrypoint, "model_x", X, y, **opts)
    expected_draws = 3 if draws is None else draws

    assert requests == [("Generator", "cpu"), ("randn", "cpu")] * expected_draws
    assert noise_products == [{"left_shape": (80, 5), "right_shape": (5, 5),
                               "right_device": "cpu", "right_finite": True}] * expected_draws
    assert statistic_devices == [("cpu", "cpu", "cpu")] * expected_draws
    assert result.metadata["n_modelx_draws"] == expected_draws
    assert result.backend == "torch"
    assert result.metadata["xk_source"] == "generated_model_x"
    assert result.W.shape == (5,) and np.isfinite(result.W).all()


@pytest.mark.parametrize("container", ["numpy", "torch"])
@pytest.mark.parametrize("entrypoint", (*MODELX_ENTRYPOINTS, "fixed_selector"))
def test_fixedx_construction_keeps_cpu_input_device_when_cuda_is_available(
    allocation_probe, monkeypatch, container, entrypoint,
):
    torch, requests, _, statistic_devices = allocation_probe
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    X, y, _ = _inputs(torch, container)
    result = _result(entrypoint, "fixed_x", X, y, backend="torch",
                     random_state=21, method="corr_diff")
    assert requests == [("Generator", "cpu"), ("randn", "cpu")]
    assert statistic_devices == [("cpu", "cpu", "cpu")]
    assert result.metadata["xk_source"] == "generated_fixed_x"
    assert result.W.shape == (5,) and np.isfinite(result.W).all()


@pytest.mark.parametrize("container", ["numpy", "torch"])
@pytest.mark.parametrize("entrypoint", MODELX_ENTRYPOINTS)
def test_valid_external_modelx_pair_bypasses_random_construction_on_cpu(
    allocation_probe, monkeypatch, container, entrypoint,
):
    from statgpu.feature_selection import _knockoff

    torch, requests, noise_products, statistic_devices = allocation_probe
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def unexpected_construction(*args, **kwargs):
        pytest.fail("Supplied Xk must bypass native model-X construction.")

    monkeypatch.setattr(_knockoff, "_build_model_x_knockoffs", unexpected_construction)
    X, y, Xk = _inputs(torch, container)
    result = _result(entrypoint, "model_x", X, y, Xk=Xk, backend="torch",
                     compat_mode="statgpu", lasso_cv_impl="statgpu", method="corr_diff")
    assert requests == [] and noise_products == []
    assert statistic_devices == [("cpu", "cpu", "cpu")]
    assert result.metadata["xk_source"] == "provided"
    numpy_X, numpy_y, numpy_Xk = _problem()
    centered_y = numpy_y - numpy_y.mean()
    expected = np.abs(numpy_X.T @ centered_y) - np.abs(numpy_Xk.T @ centered_y)
    np.testing.assert_allclose(result.W, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("entrypoint", MODELX_ENTRYPOINTS)
def test_explicit_numpy_modelx_alternative_does_not_request_torch_allocation(
    allocation_probe, monkeypatch, entrypoint,
):
    torch, requests, noise_products, statistic_devices = allocation_probe
    X, y, _ = _problem()
    kwargs = {"backend": "numpy", "compat_mode": "statgpu", "random_state": 21,
              "modelx_draws": 1, "method": "corr_diff"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    available = _result(entrypoint, "model_x", X, y, **kwargs)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    unavailable = _result(entrypoint, "model_x", X, y, **kwargs)
    assert requests == [] and noise_products == []
    assert statistic_devices == [("numpy",) * 3, ("numpy",) * 3]
    assert available.backend == unavailable.backend == "numpy"
    np.testing.assert_array_equal(available.W, unavailable.W)
    assert np.isfinite(available.W).all()


def test_modelx_nondefault_device_routing_with_simulated_tensor_metadata(monkeypatch):
    """Exercise ordinal selection on CPU; this is not CUDA execution evidence."""
    torch = pytest.importorskip("torch")
    from statgpu.feature_selection import _knockoff_utils

    reported_device = torch.device("cuda:1")
    original_eye, original_generator, original_randn = torch.eye, torch.Generator, torch.randn
    requests, generators = [], []

    class ReportedDeviceTensor(torch.Tensor):
        @property
        def device(self):
            return reported_device

        def to(self, device):
            assert device == reported_device
            return self

    def eye(*args, **kwargs):
        assert kwargs.pop("device") == reported_device
        return original_eye(*args, **kwargs, device="cpu").as_subclass(ReportedDeviceTensor)

    def generator(*, device):
        requests.append(("Generator", str(device)))
        value = original_generator(device="cpu")
        generators.append(value)
        return value

    def randn(*args, **kwargs):
        requests.append(("randn", str(kwargs.pop("device"))))
        assert kwargs["generator"] is generators[-1]
        return original_randn(*args, **kwargs, device="cpu")

    X, _, _ = _problem()
    X = torch.as_tensor(X).as_subclass(ReportedDeviceTensor)
    monkeypatch.setattr(_knockoff_utils, "_get_torch_device_str", lambda: "cuda:0")
    monkeypatch.setattr(torch, "eye", eye)
    monkeypatch.setattr(torch, "Generator", generator)
    monkeypatch.setattr(torch, "randn", randn)
    output, _ = _knockoff_utils._build_model_x_knockoffs(X, random_state=21, xp=torch)
    assert requests == [("Generator", "cuda:1"), ("randn", "cuda:1")]
    assert output.shape == X.shape
    assert bool(torch.isfinite(output).all())


@pytest.mark.parametrize("ordinal", [0, 1], ids=["cuda0", "cuda1"])
@pytest.mark.parametrize("backend", ["torch", "auto"])
@pytest.mark.parametrize("entrypoint", MODELX_ENTRYPOINTS)
@pytest.mark.parametrize("draws", [1, 3, None], ids=["one", "three", "default"])
def test_generated_modelx_on_physical_cuda_device(monkeypatch, ordinal, backend, entrypoint, draws):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= ordinal:
        pytest.skip(f"Requires physical CUDA device cuda:{ordinal}")
    from statgpu.feature_selection import _knockoff

    target_device = torch.device(f"cuda:{ordinal}")
    original_statistics = _knockoff._compute_w_statistics
    observed_devices = []

    def statistics(X, Xk, y, **kwargs):
        observed_devices.append((X.device, Xk.device, y.device))
        return original_statistics(X, Xk, y, **kwargs)

    monkeypatch.setattr(_knockoff, "_compute_w_statistics", statistics)
    # Keep default CUDA device at zero to expose accidental default-device
    # allocations when the input actually resides on cuda:1.
    with torch.cuda.device(0), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        X, y, _ = (torch.as_tensor(value, device=target_device) for value in _problem())
        opts = {"backend": backend, "compat_mode": "statgpu", "method": "corr_diff"}
        if draws is not None:
            opts["modelx_draws"] = draws
        cpu_before = torch.get_rng_state().clone()
        cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
        first = _result(entrypoint, "model_x", X, y, random_state=21, **opts)
        repeated = _result(entrypoint, "model_x", X, y, random_state=21, **opts)
        changed = _result(entrypoint, "model_x", X, y, random_state=22, **opts)
        assert torch.equal(cpu_before, torch.get_rng_state())
        assert all(torch.equal(before, after)
                   for before, after in zip(cuda_before, torch.cuda.get_rng_state_all()))
    expected_draws = 3 if draws is None else draws
    assert observed_devices == [(target_device,) * 3] * (3 * expected_draws)
    np.testing.assert_array_equal(first.W, repeated.W)
    assert not np.array_equal(first.W, changed.W)
    assert all(np.isfinite(result.W).all() for result in (first, repeated, changed))
    assert first.metadata["n_modelx_draws"] == expected_draws


@pytest.mark.parametrize("consumer", [model_x_knockoff_filter, knockoff_filter, KnockoffSelector])
def test_installed_modelx_help_describes_input_device_and_local_seed(consumer):
    # Imported help must state the same contract without a source checkout.
    for text in (inspect.getdoc(consumer), pydoc.render_doc(consumer, renderer=pydoc.plaintext)):
        text = " ".join(line.strip().removeprefix("|").strip() for line in text.splitlines())
        text = " ".join(text.split())
        for required in ("compat_mode='statgpu'", "Xk=None", "X.device",
                         "local random generator", "random matrix", "Torch CPU",
                         "CUDA", "GPU index", "X/y/Xk", "same device",
                         "externally validated", "random_state", "global Torch RNG",
                         "backend", "dtype", "software environment", "cross-GPU",
                         "empirical FDR", "Lasso cache limitation", "tuning/fitting",
                         "requests CUDA", "CPU Lasso", "corr_diff", "ols_coef_diff"):
            assert required in text
        for obsolete in ("currently ignores random_state",
                         "repeated seeded calls can differ",
                         "Native generated model-X has a different device limitation"):
            assert obsolete not in text
        assert "fixed-x" in text.lower()


def test_installed_fixedx_help_and_modelx_help_agree_on_input_device():
    text = " ".join(inspect.getdoc(fixed_x_knockoff_filter).split())
    assert "Fixed-X construction follows X's device, including CPU" in text
    assert "Native model-X construction also follows X's device" in text
    assert "Torch library, not CUDA placement" in text
    assert "model-X has a different device limitation" not in text
    assert "native Lasso tuning/fitting requests CUDA" in text
    assert "CPU Lasso" in text


@pytest.mark.parametrize("language", ["en", "cn"])
def test_bilingual_docs_describe_device_seed_scope_and_valid_external_pairs(language):
    reference = (ROOT / f"docs/{language}/reference/feature-selection-api.md").read_text()
    model = (ROOT / f"docs/{language}/models/knockoff.md").read_text()
    for text in (reference, model):
        text = " ".join(text.split())
        for required in ('compat_mode="statgpu"', 'Xk=None', 'backend="numpy"',
                         "X/y/Xk", "Torch CPU", "KnockoffSelector",
                         "torch-device-placement", "random_state=None", "dtype", "CPU Lasso"):
            assert required in text
        if language == "en":
            for required in ("local", "random generator", "random matrix", "GPU index",
                             "externally validated", "exchangeability", "conditional independence",
                             "global Torch RNG", "cross-backend", "cross-GPU", "empirical FDR",
                             "seed 0", "same construction noise", "Lasso cache limitation"):
                assert required in text
            for obsolete in ("currently ignores random_state", "currently use the global",
                             "currently draws from the global", "seed limitation below"):
                assert obsolete not in text
        else:
            for required in ("局部", "随机数生成器", "随机矩阵", "GPU 编号", "外部验证",
                             "交换性", "条件独立性", "全局 Torch", "跨后端", "跨 GPU",
                             "实际 FDR", "种子 0", "相同的构造噪声", "Lasso 缓存限制"):
                assert required in text
            for obsolete in ("种子限制", "当前不会使用它"):
                assert obsolete not in text
    assert "X.device" in reference
    assert "torch-lasso-device-routing" in model
    assert ("### Torch Lasso device routing" in reference if language == "en"
            else "### Torch Lasso 的设备选择" in reference)
