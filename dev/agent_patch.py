"""Apply the reviewed one-shot patch on the code-review branch."""

from pathlib import Path
import re
from textwrap import dedent, indent


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}")
    target.write_text(text.replace(old, new))


old_rng = indent(
    dedent(
        '''\
if backend_name == "torch":
    import torch
    g = torch.Generator(device=device)
    if random_state is not None:
        g.manual_seed(int(random_state))
    return g
import cupy as cp

seed = 0 if random_state is None else int(random_state)
return cp.random.RandomState(seed)
'''
    ),
    "    ",
)
new_rng = indent(
    dedent(
        '''\
if backend_name == "torch":
    import torch
    g = torch.Generator(device=device)
    if random_state is None:
        g.seed()
    else:
        g.manual_seed(int(random_state))
    return g
import cupy as cp

if random_state is None:
    return cp.random.RandomState()
return cp.random.RandomState(int(random_state))
'''
    ),
    "    ",
)
replace_once("statgpu/inference/_resampling.py", old_rng, new_rng)


umap_path = Path("statgpu/unsupervised/_umap.py")
umap_text = umap_path.read_text()
umap_pattern = re.compile(
    r"        # Build COO sparse edges directly \(O\(n\*k\) memory, not O\(n²\)\).*?"
    r"        return \(all_src, all_dst, rev_w, n_samples\)\n",
    re.S,
)
umap_replacement = indent(
    dedent(
        '''\
# Build the directed membership graph on the host, then apply
# UMAP's fuzzy union W + W.T - W * W.T. The previous code used
# 2W - W^2 without looking up reverse-edge memberships, leaving
# the graph asymmetric and assigning incorrect edge strengths.
from scipy.sparse import coo_matrix

all_src_np = np.repeat(np.arange(n_samples, dtype=np.int64), k)
if hasattr(neighbor_indices, "get"):
    import cupy as cp
    all_dst_np = cp.asnumpy(neighbor_indices).ravel().astype(np.int64)
    all_w_np = cp.asnumpy(membership).ravel().astype(np.float64)
elif hasattr(neighbor_indices, "cpu"):
    all_dst_np = neighbor_indices.detach().cpu().numpy().ravel().astype(np.int64)
    all_w_np = membership.detach().cpu().numpy().ravel().astype(np.float64)
else:
    all_dst_np = np.asarray(neighbor_indices, dtype=np.int64).ravel()
    all_w_np = np.asarray(membership, dtype=np.float64).ravel()

directed = coo_matrix(
    (all_w_np, (all_src_np, all_dst_np)),
    shape=(n_samples, n_samples),
).tocsr()
directed.sum_duplicates()
reverse = directed.T.tocsr()
fuzzy = directed + reverse - directed.multiply(reverse)
fuzzy.setdiag(0.0)
fuzzy.eliminate_zeros()
fuzzy = fuzzy.tocoo()

all_src = backend.asarray(
    fuzzy.row.astype(np.int64, copy=False), dtype=backend.int64
)
all_dst = backend.asarray(
    fuzzy.col.astype(np.int64, copy=False), dtype=backend.int64
)
all_w = backend.asarray(
    np.clip(fuzzy.data, 0.0, 1.0).astype(np.float64, copy=False),
    dtype=backend.float64,
)
return (all_src, all_dst, all_w, n_samples)
'''
    ),
    "        ",
)
umap_text, count = umap_pattern.subn(umap_replacement, umap_text)
if count != 1:
    raise RuntimeError(f"expected one UMAP graph block, found {count}")
umap_path.write_text(umap_text)


base_path = Path("statgpu/_base.py")
base_text = base_path.read_text()
base_pattern = re.compile(r"    def get_params\(self, deep=True\):.*\Z", re.S)
base_replacement = indent(
    dedent(
        '''\
def get_params(self, deep=True):
    """Get constructor parameters for this estimator.

    Nested estimator parameters are exposed as ``name__param`` when
    ``deep=True``, matching the scikit-learn estimator contract.
    """
    import inspect

    params = {}
    try:
        sig = inspect.signature(type(self).__init__)
    except (ValueError, TypeError):
        return params

    for name, parameter in sig.parameters.items():
        if name == "self" or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if hasattr(self, name):
            params[name] = getattr(self, name)
        elif hasattr(self, f"_{name}"):
            params[name] = getattr(self, f"_{name}")

    if deep:
        for name, value in list(params.items()):
            if hasattr(value, "get_params"):
                for sub_name, sub_value in value.get_params(deep=True).items():
                    params[f"{name}__{sub_name}"] = sub_value
    return params


def set_params(self, **params):
    """Set estimator parameters, validating names and nesting."""
    if not params:
        return self

    valid_params = self.get_params(deep=True)
    nested_params = {}

    for key, value in params.items():
        root, delimiter, sub_key = key.partition("__")
        if root not in valid_params:
            valid_names = sorted(name for name in valid_params if "__" not in name)
            raise ValueError(
                f"Invalid parameter {root!r} for estimator "
                f"{self.__class__.__name__}. Valid parameters are: "
                f"{', '.join(valid_names)}."
            )

        if delimiter:
            nested_params.setdefault(root, {})[sub_key] = value
            continue

        if root == "device" and isinstance(value, str):
            value = Device(value)
        if hasattr(self, root):
            setattr(self, root, value)
        else:
            setattr(self, f"_{root}", value)

    for root, sub_params in nested_params.items():
        nested_estimator = getattr(self, root, None)
        if nested_estimator is None:
            nested_estimator = getattr(self, f"_{root}", None)
        if not hasattr(nested_estimator, "set_params"):
            raise ValueError(
                f"Parameter {root!r} of {self.__class__.__name__} "
                "does not support nested parameters."
            )
        nested_estimator.set_params(**sub_params)

    return self
'''
    ),
    "    ",
)
base_text, count = base_pattern.subn(base_replacement, base_text)
if count != 1:
    raise RuntimeError(f"expected one BaseEstimator parameter block, found {count}")
base_path.write_text(base_text)


replace_once("README.md", "- Python >= 3.8", "- Python >= 3.9")


Path("dev/tests/test_core_contracts.py").write_text(
    dedent(
        r'''\
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
'''
    )
)


Path(".github/workflows/test.yml").write_text(
    dedent(
        '''\
name: Tests

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[validation]"

    - name: Run tests
      run: |
        python -m pytest dev/tests/test_refactor_safety_net.py dev/tests/test_penalized_correctness.py dev/tests/test_penalized_robust_cox.py dev/tests/test_multitarget_linear.py dev/tests/test_multitarget_glm.py dev/tests/test_multitarget_penalized.py dev/tests/test_sklearn_contract.py dev/tests/test_cox_cv.py dev/tests/test_quantile_regression.py dev/tests/test_quantile_penalized.py dev/tests/test_backend_utils_parity.py dev/tests/test_glm_torch_backend.py dev/tests/test_model_contracts.py dev/tests/test_unsupervised_minibatch.py dev/tests/test_unsupervised_kmeans.py dev/tests/test_unsupervised_gmm.py dev/tests/test_unsupervised_pca.py dev/tests/test_unsupervised_truncated_svd.py dev/tests/test_unsupervised_nmf.py dev/tests/test_unsupervised_minibatch_nmf.py dev/tests/test_unsupervised_dbscan.py dev/tests/test_unsupervised_agglomerative.py dev/tests/test_unsupervised_tsne.py dev/tests/test_unsupervised_umap.py dev/tests/test_inference_resampling.py dev/tests/test_core_contracts.py -q --tb=short
'''
    )
)
