"""Regression tests for core contracts found during iterative review."""

import sys
import types

import numpy as np
import pytest

from statgpu._base import BaseEstimator
from statgpu._config import Device, _DeviceManager
from statgpu.backends import get_backend
from statgpu.inference._resampling import _rng_default
from statgpu.unsupervised import UMAP
import statgpu.unsupervised._umap as umap_module


class DummyEstimator(BaseEstimator):
    def __init__(self, value=1, child=None, device=Device.CPU):
        super().__init__(device=device)
        self.value = value
        self.child = child

    def fit(self, X, y=None, **fit_params):
        self._fitted = True
        return self

    def predict(self, X):
        return X


def test_backend_factory_rejects_invalid_backend_and_device():
    with pytest.raises(ValueError, match="backend must be one of"):
        get_backend("numpyy")
    with pytest.raises(ValueError, match="device must be one of"):
        get_backend(device="gpu0")


def test_device_manager_checks_requested_cupy_backend(monkeypatch):
    manager = _DeviceManager()
    monkeypatch.setattr(manager, "_check_cupy", lambda: False)
    monkeypatch.setattr(manager, "_check_torch", lambda: True)
    with pytest.warns(RuntimeWarning, match="CuPy"):
        manager.set_device("cuda")
    assert manager.get_device() is Device.CUDA


def test_device_manager_checks_requested_torch_backend(monkeypatch):
    manager = _DeviceManager()
    monkeypatch.setattr(manager, "_check_cupy", lambda: True)
    monkeypatch.setattr(manager, "_check_torch", lambda: False)
    with pytest.warns(RuntimeWarning, match="PyTorch"):
        manager.set_device("torch")
    assert manager.get_device() is Device.TORCH


def test_set_params_rejects_unknown_and_supports_nested_estimators():
    child = DummyEstimator(value=1)
    parent = DummyEstimator(child=child)
    assert parent.set_params(child__value=7) is parent
    assert child.value == 7
    assert parent.get_params(deep=True)["child__value"] == 7
    with pytest.raises(ValueError, match="Invalid parameter"):
        parent.set_params(unknown=3)
    parent.set_params(device="auto")
    assert parent.device is Device.AUTO


def test_torch_rng_none_uses_entropy(monkeypatch):
    created = []

    class FakeGenerator:
        def __init__(self, device):
            self.device = device
            self.seed_called = False
            self.manual_seed_value = None

        def seed(self):
            self.seed_called = True
            return 123

        def manual_seed(self, value):
            self.manual_seed_value = value
            return self

    fake_torch = types.ModuleType("torch")

    def generator_factory(device):
        generator = FakeGenerator(device)
        created.append(generator)
        return generator

    fake_torch.Generator = generator_factory
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    _rng_default("torch", None, device="cuda:1")
    assert created[-1].seed_called
    assert created[-1].manual_seed_value is None
    _rng_default("torch", 19, device="cuda")
    assert not created[-1].seed_called
    assert created[-1].manual_seed_value == 19


def test_cupy_rng_none_does_not_force_seed_zero(monkeypatch):
    calls = []
    fake_cupy = types.ModuleType("cupy")
    fake_cupy.random = types.SimpleNamespace(
        RandomState=lambda *args: calls.append(args) or object()
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    _rng_default("cupy", None)
    _rng_default("cupy", 23)
    assert calls == [(), (23,)]


def test_umap_fuzzy_graph_uses_reverse_edge_memberships(monkeypatch):
    model = UMAP(
        n_neighbors=2,
        n_components=2,
        n_epochs=1,
        init="random",
        random_state=0,
        device="cpu",
    )
    neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
    neighbor_distances = np.zeros((3, 2), dtype=np.float64)
    memberships = np.array(
        [[0.2, 0.4], [0.6, 0.8], [0.3, 0.5]], dtype=np.float64
    )
    monkeypatch.setattr(
        umap_module,
        "topk_smallest",
        lambda backend, distances, k: (neighbor_distances, neighbor_indices),
    )
    monkeypatch.setattr(
        model,
        "_smooth_knn_membership",
        lambda backend, distances: memberships,
    )
    src, dst, weights, n_samples = model._fuzzy_graph(
        get_backend("numpy"), np.arange(3.0).reshape(-1, 1)
    )
    graph = np.zeros((n_samples, n_samples), dtype=np.float64)
    graph[src, dst] = weights
    expected = np.array(
        [[0.0, 0.68, 0.58], [0.68, 0.0, 0.90], [0.58, 0.90, 0.0]]
    )
    np.testing.assert_allclose(graph, expected, rtol=0.0, atol=1e-12)
