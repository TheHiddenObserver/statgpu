"""Maintenance regressions for issues #45, #81, #82, and #83."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def test_iterative_compile_policy_defaults_to_non_cudagraph_mode(monkeypatch):
    from statgpu.backends._torch_compile import resolve_torch_compile_mode

    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)
    assert resolve_torch_compile_mode(workload="iterative") == "default"
    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "disable")
    assert resolve_torch_compile_mode(workload="iterative") is None
    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "reduce-overhead")
    assert resolve_torch_compile_mode(workload="iterative") == "reduce-overhead"


def test_compile_runtime_cudagraph_failure_falls_back_once(monkeypatch):
    fake_torch = types.ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    calls = {"compiled": 0, "eager": 0}

    def fake_compile(fn, *, mode, **kwargs):
        assert mode == "default"

        def broken(*args, **call_kwargs):
            calls["compiled"] += 1
            raise RuntimeError(
                "accessing tensor output of CUDAGraphs that has been "
                "overwritten by a subsequent run"
            )

        return broken

    fake_torch.cuda = FakeCuda()
    fake_torch.compile = fake_compile
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)

    from statgpu.backends._torch_compile import compile_torch

    def eager(value):
        calls["eager"] += 1
        return value + 1

    guarded = compile_torch(eager, workload="iterative")
    with pytest.warns(RuntimeWarning, match="CUDA Graph lifecycle"):
        assert guarded(1) == 2
    assert guarded(2) == 3
    assert calls == {"compiled": 1, "eager": 2}


def test_repository_has_no_unscoped_reduce_overhead_calls():
    from pathlib import Path

    offenders = []
    for path in Path("statgpu").rglob("*.py"):
        if path.name == "_torch_compile.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "mode='reduce-overhead'" in text or 'mode="reduce-overhead"' in text:
            offenders.append(path.as_posix())
    assert offenders == []


def test_numpy_finite_validation_handles_nested_inputs():
    from statgpu.backends._validation import check_finite

    value = [np.array([1.0, 2.0]), np.array([3.0])]
    assert check_finite(value, name="X") is value
    with pytest.raises(ValueError, match="X\[1\].*finite"):
        check_finite([np.array([1.0]), np.array([np.inf])], name="X")
    with pytest.raises(ValueError, match="sample_weight.*finite"):
        check_finite(np.array([1.0, np.nan]), name="sample_weight")


def test_torch_finite_validation_stays_on_device():
    torch = pytest.importorskip("torch")
    from statgpu.backends._validation import check_finite

    tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
    result = check_finite(tensor, name="X")
    assert result is tensor
    with pytest.raises(ValueError, match="X.*finite"):
        check_finite(torch.tensor([1.0, float("inf")]), name="X")


def test_public_method_guard_rejects_nonfinite_before_fit_body():
    from statgpu._base import BaseEstimator

    class DummyEstimator(BaseEstimator):
        def __init__(self, options=None, device="cpu"):
            super().__init__(device=device)
            self.options = dict(options or {})
            self.body_called = False

        def fit(self, X, y=None):
            self.body_called = True
            self._fitted = True
            return self

        def predict(self, X):
            self.body_called = True
            return np.zeros(len(X))

    estimator = DummyEstimator()
    with pytest.raises(ValueError, match="X.*finite"):
        estimator.fit(np.array([[1.0], [np.nan]]), np.array([0.0, 1.0]))
    assert estimator.body_called is False


def test_legacy_clone_preserves_raw_constructor_identity():
    from sklearn.base import clone
    from statgpu._base import BaseEstimator

    class CopyingEstimator(BaseEstimator):
        def __init__(self, options=None, solver="AUTO", device="cpu"):
            super().__init__(device=device)
            self.options = dict(options or {})
            self.solver = str(solver).lower()

        def fit(self, X, y=None):
            self._fitted = True
            return self

        def predict(self, X):
            return np.zeros(len(X))

    options = {"threshold": 1.0}
    estimator = CopyingEstimator(options=options, solver="AUTO")
    assert estimator.get_params(deep=False)["options"] is options
    assert estimator.get_params(deep=False)["solver"] == "AUTO"
    cloned = clone(estimator)
    assert type(cloned) is CopyingEstimator
    assert cloned.solver == "auto"
    replacement = {"threshold": 2.0}
    cloned.set_params(options=replacement)
    assert cloned.get_params(deep=False)["options"] is replacement
