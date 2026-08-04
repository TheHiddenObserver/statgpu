"""Maintenance regressions for issues #45, #81, #82, and #83."""

from __future__ import annotations

import inspect
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
    assert guarded.__statgpu_compile_status__ == "runtime-fallback"
    assert "overwritten" in guarded.__statgpu_compile_error__


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


def test_numpy_finite_validation_vectorizes_sequences():
    from statgpu.backends._validation import check_finite

    value = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    assert check_finite(value, name="X") is value
    with pytest.raises(ValueError, match=r"X.*finite"):
        check_finite([np.array([1.0]), np.array([np.inf])], name="X")
    with pytest.raises(ValueError, match=r"sample_weight.*finite"):
        check_finite(np.array([1.0, np.nan]), name="sample_weight")


def test_torch_finite_validation_stays_on_device():
    torch = pytest.importorskip("torch")
    from statgpu.backends._validation import check_finite

    tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
    result = check_finite(tensor, name="X")
    assert result is tensor
    with pytest.raises(ValueError, match=r"X.*finite"):
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
    with pytest.raises(ValueError, match=r"X.*finite"):
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
    assert cloned.solver == "AUTO"
    assert cloned._solver == "auto"
    replacement = {"threshold": 2.0}
    cloned.set_params(options=replacement)
    assert cloned.get_params(deep=False)["options"] is replacement


def test_torch_lasso_py21_iterative_compile_smoke(monkeypatch):
    """Exercise the original Issue #45 path on a physical modern CUDA GPU."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a physical Torch CUDA backend")

    from packaging.version import Version

    torch_version = Version(torch.__version__.split("+", 1)[0])
    if torch_version < Version("2.1"):
        pytest.skip("Issue #45 requires PyTorch 2.1 or newer")
    if torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("torch.compile acceptance requires CUDA capability >= 7")

    from statgpu.backends import _to_numpy
    from statgpu.backends._torch_compile import get_torch_compile_diagnostics
    from statgpu.linear_model import Lasso

    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)
    get_torch_compile_diagnostics(clear=True)
    rng = np.random.default_rng(20260804)
    X = rng.normal(size=(384, 24)).astype(np.float64)
    beta = np.zeros(24, dtype=np.float64)
    beta[:6] = np.array([1.5, -1.0, 0.8, -0.6, 0.4, -0.2])
    y = X @ beta + 0.05 * rng.normal(size=X.shape[0])

    kwargs = {"alpha": 0.01, "device": "torch"}
    signature = inspect.signature(Lasso)
    if "max_iter" in signature.parameters:
        kwargs["max_iter"] = 80
    if "tol" in signature.parameters:
        kwargs["tol"] = 1e-7

    model = Lasso(**kwargs)
    model.fit(X, y)
    first = np.asarray(_to_numpy(model.predict(X)))
    model.fit(X, y)
    second = np.asarray(_to_numpy(model.predict(X)))

    assert first.shape == y.shape
    assert second.shape == y.shape
    assert np.isfinite(first).all()
    assert np.isfinite(second).all()
    np.testing.assert_allclose(first, second, rtol=1e-7, atol=1e-8)
    events = get_torch_compile_diagnostics(clear=True)
    assert any(event["status"] == "compiled" for event in events)
    assert not any("fallback" in event["status"] for event in events)



def test_compile_construction_fallback_is_visible(monkeypatch):
    fake_torch = types.ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    def broken_compile(fn, **kwargs):
        raise RuntimeError("compiler unavailable")

    fake_torch.cuda = FakeCuda()
    fake_torch.compile = broken_compile
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)

    from statgpu.backends._torch_compile import (
        compile_torch,
        get_torch_compile_diagnostics,
    )

    get_torch_compile_diagnostics(clear=True)
    with pytest.warns(RuntimeWarning, match="construction failed"):
        wrapped = compile_torch(lambda x: x + 1, workload="iterative")
    assert wrapped(2) == 3
    assert wrapped.__statgpu_compile_status__ == "construction-fallback"
    assert "compiler unavailable" in wrapped.__statgpu_compile_error__
    assert get_torch_compile_diagnostics(clear=True)[-1]["status"] == "construction-fallback"


def test_set_params_rebuilds_normalized_panel_state():
    from statgpu.panel import PooledOLS

    model = PooledOLS()
    model._fitted = True
    model.set_params(cov_type="HAC")
    assert model.get_params(deep=False)["cov_type"] == "HAC"
    assert model.cov_type == "HAC"
    assert model._cov_type == "hac"
    assert model._fitted is False


def test_current_sklearn_classifier_and_regressor_tags():
    pytest.importorskip("sklearn")
    from sklearn.base import is_classifier, is_regressor
    from statgpu.linear_model import LogisticRegression, Ridge

    from statgpu.covariance import GraphicalLasso

    assert is_classifier(LogisticRegression())
    assert is_regressor(Ridge(compute_inference=False))
    assert not is_regressor(GraphicalLasso())
    assert not is_classifier(GraphicalLasso())

    class ExternalRidge(Ridge):
        pass

    assert is_regressor(ExternalRidge(compute_inference=False))


def test_extended_public_finite_validation_matrix():
    from statgpu.backends._validation import check_finite
    from statgpu.unsupervised import PCA

    with pytest.raises(ValueError, match="finite"):
        check_finite(np.array([1.0, np.nan], dtype=object), name="X")

    X = np.arange(24, dtype=float).reshape(8, 3)
    model = PCA(n_components=2).fit(X)
    with pytest.raises(ValueError, match="finite"):
        model.inverse_transform(np.array([[np.nan, 0.0]]))


def _require_modern_torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a physical Torch CUDA backend")
    from packaging.version import Version
    if Version(torch.__version__.split("+", 1)[0]) < Version("2.1"):
        pytest.skip("requires PyTorch 2.1 or newer")
    if torch.cuda.get_device_capability()[0] < 7:
        pytest.skip("requires CUDA capability >= 7")
    return torch


def test_physical_cuda_compile_path_is_observable(monkeypatch):
    torch = _require_modern_torch_cuda()
    from statgpu.backends._torch_compile import (
        compile_torch,
        get_torch_compile_diagnostics,
    )

    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)
    get_torch_compile_diagnostics(clear=True)

    def add_one(x):
        return x + 1

    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    before_graphs = int(counters["stats"].get("unique_graphs", 0))
    compiled = compile_torch(add_one, workload="iterative")
    x = torch.arange(16, device="cuda", dtype=torch.float64)
    result = compiled(x)
    torch.cuda.synchronize()
    after_graphs = int(counters["stats"].get("unique_graphs", 0))
    assert compiled.__statgpu_compile_status__ == "compiled"
    assert after_graphs > before_graphs
    assert torch.allclose(result, x + 1)
    assert get_torch_compile_diagnostics(clear=True)[-1]["status"] == "compiled"


def test_torch_penalty_compile_matrix_py21(monkeypatch):
    torch = _require_modern_torch_cuda()
    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)

    from statgpu.backends._torch_compile import get_torch_compile_diagnostics
    import statgpu.penalties._adaptive_l1 as adaptive_module
    import statgpu.penalties._group_lasso as group_lasso_module
    import statgpu.penalties._group_mcp as group_mcp_module
    import statgpu.penalties._group_scad as group_scad_module
    import statgpu.penalties._l1 as l1_module
    import statgpu.penalties._mcp as mcp_module
    import statgpu.penalties._scad as scad_module
    from statgpu.penalties import (
        AdaptiveL1Penalty,
        GroupLassoPenalty,
        GroupMCPPenalty,
        GroupSCADPenalty,
        L1Penalty,
        MCPPenalty,
        SCADPenalty,
    )

    l1_module._L1_PROXIMAL_TORCH_COMPILED = None
    adaptive_module._ADAPTIVE_L1_PROXIMAL_TORCH_COMPILED = None
    scad_module._SCAD_PROXIMAL_TORCH_COMPILED = None
    mcp_module._MCP_PROXIMAL_TORCH_COMPILED = None
    group_lasso_module._GROUP_LASSO_PROXIMAL_TORCH_COMPILED_EQUAL = None
    group_scad_module._GROUP_SCAD_PROXIMAL_TORCH_COMPILED = None
    group_mcp_module._GROUP_MCP_PROXIMAL_TORCH_COMPILED = None
    get_torch_compile_diagnostics(clear=True)
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    before_graphs = int(counters["stats"].get("unique_graphs", 0))

    groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    penalties = [
        L1Penalty(alpha=0.2),
        AdaptiveL1Penalty(alpha=0.2, weights=np.ones(8)),
        SCADPenalty(alpha=0.2),
        MCPPenalty(alpha=0.2),
        GroupLassoPenalty(alpha=0.2, groups=groups),
        GroupSCADPenalty(alpha=0.2, groups=groups),
        GroupMCPPenalty(alpha=0.2, groups=groups),
    ]
    w = torch.linspace(-2.0, 2.0, 8, device="cuda", dtype=torch.float64)
    for penalty in penalties:
        result = penalty.proximal(w, step=0.1, backend="torch")
        assert result.is_cuda
        assert torch.isfinite(result).all()
    torch.cuda.synchronize()

    after_graphs = int(counters["stats"].get("unique_graphs", 0))
    events = get_torch_compile_diagnostics(clear=True)
    assert after_graphs > before_graphs
    assert len([event for event in events if event["status"] == "compiled"]) >= len(penalties)
    assert [event for event in events if "fallback" in event["status"]] == []


def test_cupy_finite_validation_stays_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.backends._validation import check_finite

    value = cp.asarray([1.0, 2.0])
    assert check_finite(value, name="X") is value
    with pytest.raises(ValueError, match="finite"):
        check_finite(cp.asarray([1.0, cp.inf]), name="X")



def test_set_params_preserves_estimator_fit_validation_boundary():
    from statgpu.survival import CoxPH

    model = CoxPH()
    model.set_params(compute_inference="False")
    assert model.get_params(deep=False)["compute_inference"] == "False"



def test_compile_call_sites_do_not_swallow_policy_errors():
    import ast
    from pathlib import Path

    offenders = []
    for path in Path("statgpu").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "compile_torch" not in source and "suppress_errors" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "compile_torch"
                    ):
                        offenders.append((path.as_posix(), decorator.lineno, "decorator"))
            if isinstance(node, ast.Try):
                contains_compile = any(
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "compile_torch"
                    for stmt in node.body
                    for child in ast.walk(stmt)
                )
                if contains_compile:
                    offenders.append((path.as_posix(), node.lineno, "caught"))
        if "torch._dynamo.config.suppress_errors" in source:
            offenders.append((path.as_posix(), 0, "suppress_errors"))
    assert offenders == []


def test_invalid_compile_mode_reaches_penalty_callsite(monkeypatch):
    fake_torch = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "invalid-mode")

    import statgpu.penalties._l1 as l1_module
    from statgpu.penalties import L1Penalty

    l1_module._L1_PROXIMAL_TORCH_COMPILED = None
    with pytest.raises(ValueError, match="STATGPU_TORCH_COMPILE_MODE"):
        L1Penalty(alpha=0.1).proximal(np.array([1.0]), 0.1, backend="torch")


def test_set_params_invalid_update_is_transactional():
    from statgpu.panel import PooledOLS

    model = PooledOLS(cov_type="robust", kernel="bartlett")
    model._fitted = True
    model.marker_ = object()
    marker = model.marker_
    before = model.get_params(deep=False).copy()

    with pytest.raises(ValueError, match="cov_type"):
        model.set_params(cov_type="invalid", kernel="PARZEN")

    assert model.get_params(deep=False) == before
    assert model.cov_type == "robust"
    assert model.kernel == "bartlett"
    assert model._fitted is True
    assert model.marker_ is marker


def test_pandas_nullable_boolean_missing_is_rejected():
    pd = pytest.importorskip("pandas")
    from statgpu.backends._validation import check_finite

    with pytest.raises(ValueError, match="finite"):
        check_finite(pd.Series([True, pd.NA], dtype="boolean"), name="X")


def test_public_sklearn_tags_are_available_and_transformers_are_marked():
    import inspect
    import statgpu
    from sklearn.utils import get_tags

    errors = []
    missing_transformer_tags = []
    for name in statgpu.__all__:
        cls = getattr(statgpu, name, None)
        if not inspect.isclass(cls) or not hasattr(cls, "fit") or inspect.isabstract(cls):
            continue
        signature = inspect.signature(cls)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect._empty
            and parameter.kind
            not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        ]
        if required:
            continue
        try:
            estimator = cls()
            tags = get_tags(estimator)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if callable(getattr(estimator, "transform", None)) and tags.transformer_tags is None:
            missing_transformer_tags.append(name)

    assert errors == []
    assert missing_transformer_tags == []


def test_knockoff_selectors_reject_nonfinite_inputs():
    from statgpu.feature_selection import FixedXKnockoffSelector, KnockoffSelector

    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    y = np.array([0.0, 1.0])
    for selector in (KnockoffSelector(), FixedXKnockoffSelector()):
        with pytest.raises(ValueError, match="finite"):
            selector.fit(X, y)


def test_base_inference_helpers_reject_nonfinite_inputs():
    from statgpu.linear_model import LinearRegression

    model = LinearRegression()
    with pytest.raises(ValueError, match="finite"):
        model.combine_pvalues(np.array([0.1, np.nan]))



def _default_public_estimators():
    import inspect
    import statgpu

    for name in statgpu.__all__:
        cls = getattr(statgpu, name, None)
        if not inspect.isclass(cls) or not hasattr(cls, "fit") or inspect.isabstract(cls):
            continue
        signature = inspect.signature(cls)
        required = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.default is inspect._empty
            and parameter.kind
            not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        ]
        if required:
            continue
        try:
            yield name, cls()
        except Exception:
            continue


def test_public_constructor_attributes_preserve_identity():
    mismatches = []
    for estimator_name, estimator in _default_public_estimators():
        for parameter, value in estimator.get_params(deep=False).items():
            if not hasattr(estimator, parameter):
                mismatches.append((estimator_name, parameter, "missing"))
            elif getattr(estimator, parameter) is not value:
                mismatches.append((estimator_name, parameter, "identity"))
    assert mismatches == []


def test_public_raw_private_normalized_choice_contracts():
    from statgpu.linear_model import LassoCV
    from statgpu.panel import PooledOLS

    panel = PooledOLS(cov_type="HAC")
    assert panel.cov_type == "HAC"
    assert panel._cov_type == "hac"

    lasso = LassoCV(method="STANDARD", solver="AUTO")
    assert lasso.method == "STANDARD"
    assert lasso._method == "standard"
    assert lasso.solver == "AUTO"
    assert lasso._solver == "AUTO"


def test_public_raw_private_mutable_kwargs_are_decoupled():
    from statgpu.linear_model import PenalizedLinearRegression

    penalty_kwargs = {"gamma": 3.0}
    loss_kwargs = {"scale": 2.0}
    model = PenalizedLinearRegression(
        penalty_kwargs=penalty_kwargs,
        loss_kwargs=loss_kwargs,
    )
    assert model.penalty_kwargs is penalty_kwargs
    assert model.loss_kwargs is loss_kwargs
    assert model._penalty_kwargs == penalty_kwargs
    assert model._loss_kwargs == loss_kwargs
    assert model._penalty_kwargs is not penalty_kwargs
    assert model._loss_kwargs is not loss_kwargs

    penalty_kwargs["external"] = True
    loss_kwargs["external"] = True
    assert "external" not in model._penalty_kwargs
    assert "external" not in model._loss_kwargs


def test_device_public_value_and_private_runtime_are_separate():
    from statgpu.linear_model import Ridge
    from statgpu._config import Device

    model = Ridge(device="cpu", compute_inference=False)
    assert model.device == "cpu"
    assert model._device is Device.CPU
    assert model._get_compute_device() is Device.CPU


def test_set_params_refreshes_public_and_private_constructor_state():
    from statgpu.panel import PooledOLS

    model = PooledOLS(cov_type="robust")
    model._fitted = True
    model.set_params(cov_type="HAC")
    assert model.cov_type == "HAC"
    assert model._cov_type == "hac"
    assert model._fitted is False


def test_delegated_wrapper_parameters_exist_publicly():
    from statgpu.linear_model import (
        GammaRegression,
        NegativeBinomialRegression,
        TweedieRegression,
    )

    gamma = GammaRegression(link="log")
    negative_binomial = NegativeBinomialRegression(alpha=0.75)
    tweedie = TweedieRegression(power=1.7)
    assert gamma.link == "log"
    assert negative_binomial.alpha == 0.75
    assert tweedie.power == 1.7
