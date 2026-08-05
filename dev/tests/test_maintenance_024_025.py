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
    torch._dynamo.reset()
    before_graphs = _dynamo_unique_graphs(torch)
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
    after_graphs = _dynamo_unique_graphs(torch)
    assert after_graphs > before_graphs
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


def _dynamo_unique_graphs(torch):
    return int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0))


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
        case_before_graphs = _dynamo_unique_graphs(torch)
        result = penalty.proximal(w, step=0.1, backend="torch")
        torch.cuda.synchronize()
        case_after_graphs = _dynamo_unique_graphs(torch)
        assert case_after_graphs > case_before_graphs, penalty.name
        assert result.is_cuda
        assert torch.isfinite(result).all()

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

    try:
        from sklearn.utils import get_tags
    except ImportError:
        get_tags = None
        from sklearn.utils._tags import _safe_tags

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
            if get_tags is None:
                tags = _safe_tags(estimator)
            else:
                tags = get_tags(estimator)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if (
            get_tags is not None
            and callable(getattr(estimator, "transform", None))
            and tags.transformer_tags is None
        ):
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


def test_public_raw_private_mutable_kwargs_preserve_runtime_identity():
    from statgpu.linear_model import PenalizedLinearRegression

    penalty_kwargs = {"gamma": 3.0}
    loss_kwargs = {"scale": 2.0}
    model = PenalizedLinearRegression(
        penalty_kwargs=penalty_kwargs,
        loss_kwargs=loss_kwargs,
    )
    assert model.penalty_kwargs is penalty_kwargs
    assert model.loss_kwargs is loss_kwargs
    assert model._penalty_kwargs is penalty_kwargs
    assert model._loss_kwargs is loss_kwargs

    penalty_kwargs["external"] = True
    loss_kwargs["external"] = True
    assert model._penalty_kwargs["external"] is True
    assert model._loss_kwargs["external"] is True


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



def test_all_public_numeric_methods_expose_finite_contract():
    import inspect
    import statgpu

    candidate_names = {
        "X", "X_new", "x", "y", "sample_weight", "weights", "offset",
        "exposure", "entry", "start", "stop", "time", "event", "times",
        "cluster", "clusters", "strata", "subject", "subjects", "groups",
        "init", "init_coef", "initial_coef", "time_index", "entity_ids",
        "time_ids", "pvalues", "arrays", "scores", "thresholds", "Xk",
        "mu", "Sigma",
    }
    missing = []
    for estimator_name, estimator in _default_public_estimators():
        cls = type(estimator)
        for method_name in dir(cls):
            if method_name.startswith("_"):
                continue
            method = getattr(cls, method_name, None)
            if not callable(method):
                continue
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError):
                continue
            if not (set(signature.parameters) & candidate_names):
                continue
            if not getattr(method, "__statgpu_finite_validation__", False):
                missing.append((estimator_name, method_name))
    assert missing == []


def test_inherited_penalized_fit_rejects_nonfinite_before_solver():
    from statgpu.linear_model import PenalizedLinearRegression

    model = PenalizedLinearRegression(compute_inference=False, device="cpu")
    X = np.array([[1.0, 0.0], [np.nan, 1.0], [2.0, 2.0]])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"X.*finite"):
        model.fit(X, y)
    assert model._fitted is False


def test_inherited_ridge_predict_rejects_nonfinite():
    from statgpu.linear_model import Ridge

    X = np.arange(24, dtype=float).reshape(8, 3)
    y = np.arange(8, dtype=float)
    model = Ridge(compute_inference=False, device="cpu").fit(X, y)
    bad = X.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"X.*finite"):
        model.predict(bad)


def test_inherited_lasso_score_rejects_nonfinite_target():
    from statgpu.linear_model import Lasso

    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10, dtype=float)
    model = Lasso(alpha=0.01, compute_inference=False, device="cpu").fit(X, y)
    bad_y = y.copy()
    bad_y[0] = np.nan
    with pytest.raises(ValueError, match=r"y.*finite"):
        model.score(X, bad_y)


def test_knockoff_manual_validation_is_marked():
    from statgpu.feature_selection import FixedXKnockoffSelector, KnockoffSelector

    for cls in (KnockoffSelector, FixedXKnockoffSelector):
        for method_name in ("fit", "fit_transform", "transform"):
            assert getattr(
                getattr(cls, method_name),
                "__statgpu_finite_validation__",
                False,
            )



def test_custom_get_params_do_not_expose_normalized_private_values():
    import ast
    from pathlib import Path

    normalized_private = {
        "_device", "_cov_type", "_hac_maxlags", "_gpu_memory_cleanup",
        "_solver", "_cpu_solver", "_stopping", "_inference_method",
        "_simultaneous_method", "_n_bootstrap",
        "_enable_simultaneous_inference", "_simultaneous_alpha",
        "_simultaneous_n_bootstrap", "_simultaneous_include_intercept",
        "_method", "_admm_rho", "_alpha_min_ratio", "_cd_kkt_check_every",
        "_compute_inference_enabled", "_cv", "_fit_intercept",
        "_gpu_cv_mixed_precision", "_max_iter", "_n_alphas", "_tol",
        "_n_Cs", "_C_min_ratio", "_penalty_kwargs", "_loss_kwargs",
        "_epsilon", "_ties", "_acknowledge_approx", "_refine_top_k",
        "_batch_size", "_min_effective_weight", "_quantile", "_cv_strategy",
    }
    offenders = []
    for path in Path("statgpu").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Attribute(self, node):
                if (
                    stack
                    and stack[-1] == "get_params"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and node.attr in normalized_private
                ):
                    offenders.append((path.as_posix(), node.lineno, node.attr))
                self.generic_visit(node)

        Visitor().visit(tree)
    assert offenders == []


def test_tsne_nondefault_get_params_preserve_raw_identity():
    from sklearn.base import clone
    from statgpu.unsupervised import TSNE

    max_iter = np.int64(300)
    device = "".join(("c", "pu"))
    model = TSNE(max_iter=max_iter, device=device)
    params = model.get_params(deep=False)
    assert params["max_iter"] is max_iter
    assert params["device"] is device

    cloned = clone(model)
    cloned_params = cloned.get_params(deep=False)
    assert isinstance(cloned_params["max_iter"], np.integer)
    assert cloned_params["device"] == "cpu"



def test_supervised_generic_estimators_have_sklearn_types():
    from sklearn.base import is_classifier, is_regressor
    from statgpu.linear_model import (
        GeneralizedLinearModel,
        OrderedGeneralizedLinearModel,
        PenalizedGLM_CV,
        PenalizedGeneralizedLinearModel,
    )
    from statgpu.nonparametric import KernelRegression

    assert is_regressor(GeneralizedLinearModel())
    assert is_classifier(OrderedGeneralizedLinearModel())
    assert is_regressor(PenalizedGeneralizedLinearModel())
    assert is_regressor(PenalizedGLM_CV())
    assert is_regressor(KernelRegression())


# PR87_REVIEW_FIX_BATCH_TESTS
def test_formula_history_does_not_bypass_direct_pandas_finite_guard():
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import LinearRegression

    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0]}
    )
    model = LinearRegression(device="cpu").fit(formula="y ~ x", data=data)
    X_bad = pd.DataFrame({"x": [0.0, np.nan, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match=r"X.*finite"):
        model.fit(X_bad, y)


def test_stepwise_selector_finite_and_supervised_tag_contract():
    from statgpu.feature_selection import StepwiseSelector
    from statgpu.linear_model import LinearRegression

    selector = StepwiseSelector(LinearRegression)
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match=r"X.*finite"):
        selector.fit(X, y)
    assert selector._more_tags()["requires_y"] is True

    try:
        from sklearn.utils import get_tags
    except ImportError:
        from sklearn.utils._tags import _safe_tags

        assert _safe_tags(selector)["requires_y"] is True
    else:
        assert get_tags(selector).target_tags.required is True
        assert get_tags(selector).transformer_tags is not None


@pytest.mark.parametrize(
    "entrypoint",
    [
        "fixed_x_knockoff_filter",
        "model_x_knockoff_filter",
        "knockoff_filter",
    ],
)
def test_function_style_knockoff_entrypoints_reject_nonfinite(entrypoint):
    import statgpu.feature_selection as feature_selection

    fn = getattr(feature_selection, entrypoint)
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    y = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match=r"X.*finite"):
        fn(X, y, backend="numpy")


def test_nested_set_params_is_atomic_and_does_not_mutate_shared_children():
    from statgpu._base import BaseEstimator

    class Child(BaseEstimator):
        def __init__(self, value=1, device="cpu"):
            super().__init__(device=device)
            self.value = value

        def fit(self, X, y=None):
            self._fitted = True
            return self

        def predict(self, X):
            return np.zeros(len(X))

    class Parent(BaseEstimator):
        def __init__(self, left=None, right=None, device="cpu"):
            super().__init__(device=device)
            self.left = left
            self.right = right

        def fit(self, X, y=None):
            self._fitted = True
            return self

        def predict(self, X):
            return np.zeros(len(X))

    left = Child(value=1)
    right = Child(value=2)
    parent = Parent(left=left, right=right)

    with pytest.raises(ValueError, match="Invalid parameter"):
        parent.set_params(left__value=7, right__unknown=3)
    assert parent.left is left
    assert parent.right is right
    assert left.value == 1
    assert right.value == 2

    parent.set_params(left__value=7)
    assert left.value == 1
    assert parent.left is not left
    assert parent.left.value == 7


def test_torch_public_finite_validation_stays_on_cuda():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model import LinearRegression

    X = torch.tensor(
        [[1.0, 2.0], [3.0, float("nan")]],
        dtype=torch.float64,
        device="cuda",
    )
    y = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match=r"X.*finite"):
        LinearRegression(device="torch").fit(X, y)
    assert X.is_cuda and y.is_cuda


def test_cupy_public_finite_validation_stays_on_cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model import LinearRegression

    X = cp.asarray([[1.0, 2.0], [3.0, cp.nan]], dtype=cp.float64)
    y = cp.asarray([1.0, 2.0], dtype=cp.float64)
    with pytest.raises(ValueError, match=r"X.*finite"):
        LinearRegression(device="cuda").fit(X, y)
    assert isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)


@pytest.mark.parametrize(
    "penalty,penalty_kwargs",
    [
        ("scad", {}),
        ("mcp", {}),
        ("group_scad", {"groups": [[0, 1], [2, 3], [4, 5]]}),
        ("group_mcp", {"groups": [[0, 1], [2, 3], [4, 5]]}),
    ],
)
def test_torch_nonconvex_model_level_compile_matrix_py21(
    monkeypatch, penalty, penalty_kwargs
):
    torch = _require_modern_torch_cuda()
    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)

    from statgpu.backends import _to_numpy
    from statgpu.backends._torch_compile import get_torch_compile_diagnostics
    from statgpu.linear_model import PenalizedLinearRegression
    import statgpu.solvers._fista_lla as fista_lla_module

    fista_lla_module._SQERR_PROXIMAL_TORCH = None
    fista_lla_module._FUSED_PROXIMAL_CLIP_TORCH = None
    if penalty == "group_scad":
        import statgpu.penalties._group_scad as group_scad_module

        group_scad_module._GROUP_SCAD_PROXIMAL_TORCH_COMPILED = None
    elif penalty == "group_mcp":
        import statgpu.penalties._group_mcp as group_mcp_module

        group_mcp_module._GROUP_MCP_PROXIMAL_TORCH_COMPILED = None
    get_torch_compile_diagnostics(clear=True)
    torch._dynamo.reset()
    before_graphs = _dynamo_unique_graphs(torch)

    rng = np.random.default_rng(20260805)
    X = rng.normal(size=(192, 6)).astype(np.float64)
    beta = np.array([1.2, -0.8, 0.6, 0.0, 0.0, 0.0])
    y = X @ beta + 0.05 * rng.normal(size=X.shape[0])
    model = PenalizedLinearRegression(
        penalty=penalty,
        penalty_kwargs=penalty_kwargs,
        alpha=0.03,
        max_iter=40,
        max_lla_iters=3,
        tol=1e-6,
        lla_tol=1e-6,
        device="torch",
        compute_inference=False,
    ).fit(X, y)
    prediction = np.asarray(_to_numpy(model.predict(X)))

    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()
    assert np.isfinite(np.asarray(_to_numpy(model.coef_))).all()
    events = get_torch_compile_diagnostics(clear=True)
    after_graphs = _dynamo_unique_graphs(torch)
    assert after_graphs > before_graphs
    assert any(event["status"] == "compiled" for event in events)
    assert not any("fallback" in event["status"] for event in events)


def test_torch_elasticnet_model_level_compile_path_py21(monkeypatch):
    torch = _require_modern_torch_cuda()
    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)
    from statgpu.backends import _to_numpy
    from statgpu.backends._torch_compile import get_torch_compile_diagnostics
    from statgpu.linear_model import ElasticNet

    get_torch_compile_diagnostics(clear=True)
    torch._dynamo.reset()
    before_graphs = _dynamo_unique_graphs(torch)
    rng = np.random.default_rng(20260806)
    X = rng.normal(size=(192, 10)).astype(np.float64)
    y = X[:, 0] - 0.5 * X[:, 1] + 0.05 * rng.normal(size=X.shape[0])
    model = ElasticNet(
        alpha=0.02,
        l1_ratio=0.6,
        max_iter=60,
        tol=1e-6,
        device="torch",
    ).fit(X, y)
    prediction = np.asarray(_to_numpy(model.predict(X)))
    assert np.isfinite(prediction).all()
    events = get_torch_compile_diagnostics(clear=True)
    after_graphs = _dynamo_unique_graphs(torch)
    assert after_graphs > before_graphs
    assert any(event["status"] == "compiled" for event in events)
    assert not any("fallback" in event["status"] for event in events)


# PR87_FORMULA_PREDICT_OWNERSHIP_TEST
def test_formula_predict_dataframe_keeps_formula_missing_row_semantics():
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import LinearRegression

    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0], "x": [0.0, 1.0, 2.0, 3.0]}
    )
    model = LinearRegression(device="cpu").fit(formula="y ~ x", data=data)
    new_data = pd.DataFrame({"x": [0.0, np.nan, 2.0, 3.0]})
    prediction = np.asarray(model.predict(new_data))
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()


# PR87_STEPWISE_LIFECYCLE_TESTS
def test_stepwise_transform_and_set_params_lifecycle():
    from statgpu.feature_selection import StepwiseSelector
    from statgpu.linear_model import LinearRegression

    X = np.column_stack([np.arange(8.0), np.arange(8.0) ** 2])
    y = 1.0 + 2.0 * X[:, 0]
    selector = StepwiseSelector(
        LinearRegression, max_features=1, device="cpu"
    ).fit(X, y)

    transformed = selector.transform(X)
    assert transformed.shape == (X.shape[0], 1)
    assert selector.__sklearn_is_fitted__() is True

    selector.set_params(criterion="BIC")
    assert selector.criterion == "BIC"
    assert selector._criterion == "bic"
    assert selector.__sklearn_is_fitted__() is False
    with pytest.raises(RuntimeError, match="not been fitted"):
        selector.predict(X)

    before = selector.get_params(deep=False)
    with pytest.raises(ValueError, match="criterion"):
        selector.set_params(criterion="invalid")
    assert selector.get_params(deep=False) == before


# PR87_DATA_ONLY_FORMULA_GUARD_TEST
def test_data_argument_alone_does_not_disable_direct_pandas_finite_guard():
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import LinearRegression

    X_bad = pd.DataFrame({"x": [0.0, np.nan, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    unrelated_data = pd.DataFrame({"z": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match=r"X.*finite"):
        LinearRegression(device="cpu").fit(X_bad, y, data=unrelated_data)


# PR87_KNOCKOFF_SET_PARAMS_TRANSACTION_TESTS
@pytest.mark.parametrize("selector_name", ["KnockoffSelector", "FixedXKnockoffSelector"])
def test_knockoff_selector_set_params_is_transactional(selector_name):
    import statgpu.feature_selection as feature_selection

    selector = getattr(feature_selection, selector_name)(q=0.1)
    sentinel_result = object()
    sentinel_features = np.array([0], dtype=np.int64)
    selector.result_ = sentinel_result
    selector.selected_features_ = sentinel_features

    with pytest.raises(ValueError, match="Invalid parameter"):
        selector.set_params(q=0.2, unknown_parameter=1)
    assert selector.q == 0.1
    assert selector.result_ is sentinel_result
    assert selector.selected_features_ is sentinel_features

    selector.set_params(q=0.2)
    assert selector.q == 0.2
    assert selector.result_ is None
    assert selector.selected_features_ is None


# PR87_SECOND_REVIEW_FORMULA_WEIGHT_TESTS
def test_glm_formula_sample_weight_aligns_patsy_retained_rows():
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 5.0, 8.0],
            "x": [0.0, 1.0, np.nan, 3.0, 4.0],
        }
    )
    original_weights = np.array([1.0, 2.0, 1000.0, 4.0, 5.0])
    retained = np.array([0, 1, 3, 4])

    formula_model = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0, device="cpu"
    ).fit(
        formula="y ~ x", data=data, sample_weight=original_weights
    )
    direct_model = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0, device="cpu"
    ).fit(
        data.loc[retained, ["x"]].to_numpy(),
        data.loc[retained, "y"].to_numpy(),
        sample_weight=original_weights[retained],
    )
    np.testing.assert_allclose(formula_model.coef_, direct_model.coef_)
    np.testing.assert_allclose(formula_model.intercept_, direct_model.intercept_)

    aligned_model = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0, device="cpu"
    ).fit(
        formula="y ~ x", data=data, sample_weight=original_weights[retained]
    )
    np.testing.assert_allclose(aligned_model.coef_, direct_model.coef_)
    np.testing.assert_allclose(aligned_model.intercept_, direct_model.intercept_)

    with pytest.raises(ValueError, match="sample_weight must (?:have length|match)"):
        GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0, device="cpu"
        ).fit(
            formula="y ~ x", data=data, sample_weight=np.ones(3)
        )


def test_compile_benchmark_has_hard_per_case_graph_gate():
    import importlib.util
    from pathlib import Path

    path = Path("dev/benchmarks/benchmark_torch_compile_maintenance.py")
    spec = importlib.util.spec_from_file_location("compile_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module._validate_compile_evidence(
        "default", "lasso", [{"status": "compiled"}], graph_delta=1
    )
    with pytest.raises(RuntimeError, match="Dynamo graph"):
        module._validate_compile_evidence(
            "default", "lasso", [{"status": "compiled"}], graph_delta=0
        )
    with pytest.raises(RuntimeError, match="compiled diagnostic"):
        module._validate_compile_evidence(
            "default", "lasso", [], graph_delta=1
        )


# PR87_FORMULA_WEIGHT_SHARED_ALIGNMENT_TESTS
@pytest.mark.parametrize("kind", ["linear", "glm", "penalized"])
def test_formula_sample_weight_validates_after_row_alignment(kind):
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import (
        GeneralizedLinearModel,
        LinearRegression,
        PenalizedLinearRegression,
    )

    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 5.0], "x": [0.0, 1.0, np.nan, 3.0]}
    )
    if kind == "linear":
        factory = lambda: LinearRegression(device="cpu", compute_inference=False)
    elif kind == "glm":
        factory = lambda: GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0, device="cpu"
        )
    else:
        factory = lambda: PenalizedLinearRegression(
            penalty="l1",
            alpha=0.01,
            max_iter=30,
            device="cpu",
            compute_inference=False,
        )

    # Non-finite value belongs only to the row Patsy drops, so it is removed
    # before the retained side array is validated.
    model = factory().fit(
        formula="y ~ x",
        data=data,
        sample_weight=np.array([1.0, 1.0, np.nan, 1.0]),
    )
    assert model is not None

    with pytest.raises(ValueError, match=r"sample_weight.*finite"):
        factory().fit(
            formula="y ~ x",
            data=data,
            sample_weight=pd.Series([1.0, np.nan, 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        factory().fit(
            formula="y ~ x",
            data=data,
            sample_weight=np.ones((len(data), 1)),
        )
    with pytest.raises(ValueError, match="non-negative"):
        factory().fit(
            formula="y ~ x",
            data=data,
            sample_weight=np.array([1.0, -1.0, 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="positive sum"):
        factory().fit(
            formula="y ~ x",
            data=data,
            sample_weight=np.zeros(len(data)),
        )


def test_glm_direct_sample_weight_semantic_contract():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.arange(8.0).reshape(4, 2)
    y = np.arange(4.0)
    factory = lambda: GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0, device="cpu"
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        factory().fit(X, y, sample_weight=np.ones((4, 1)))
    with pytest.raises(ValueError, match="non-negative"):
        factory().fit(X, y, sample_weight=np.array([1.0, 1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="positive sum"):
        factory().fit(X, y, sample_weight=np.zeros(4))


# PR87_FORMULA_WEIGHT_GPU_DEVICE_TESTS
def test_torch_formula_sample_weight_alignment_stays_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.core.formula import align_formula_sample_weight

    weights = torch.tensor(
        [1.0, float("nan"), 3.0, 4.0],
        dtype=torch.float64,
        device="cuda",
    )
    aligned = align_formula_sample_weight(
        weights,
        data_length=4,
        retained_rows=np.array([0, 2, 3], dtype=np.int64),
        retained_length=3,
    )
    assert aligned.is_cuda
    assert aligned.device == weights.device
    assert torch.isfinite(aligned).all()
    torch.testing.assert_close(
        aligned,
        torch.tensor([1.0, 3.0, 4.0], dtype=torch.float64, device="cuda"),
    )
    assert weights.is_cuda

    with pytest.raises(ValueError, match=r"sample_weight.*finite"):
        align_formula_sample_weight(
            weights,
            data_length=4,
            retained_rows=np.array([0, 1, 3], dtype=np.int64),
            retained_length=3,
        )
    assert weights.is_cuda


def test_cupy_formula_sample_weight_alignment_stays_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.core.formula import align_formula_sample_weight

    weights = cp.asarray([1.0, cp.nan, 3.0, 4.0], dtype=cp.float64)
    aligned = align_formula_sample_weight(
        weights,
        data_length=4,
        retained_rows=np.array([0, 2, 3], dtype=np.int64),
        retained_length=3,
    )
    assert isinstance(aligned, cp.ndarray)
    assert int(aligned.device.id) == int(weights.device.id)
    assert bool(cp.isfinite(aligned).all().item())
    cp.testing.assert_allclose(aligned, cp.asarray([1.0, 3.0, 4.0]))
    assert isinstance(weights, cp.ndarray)

    with pytest.raises(ValueError, match=r"sample_weight.*finite"):
        align_formula_sample_weight(
            weights,
            data_length=4,
            retained_rows=np.array([0, 1, 3], dtype=np.int64),
            retained_length=3,
        )
    assert isinstance(weights, cp.ndarray)
