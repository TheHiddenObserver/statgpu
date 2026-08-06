"""Maintenance regressions for issues #45, #81, #82, and #83."""

from __future__ import annotations

import inspect
import sys
import types

import numpy as np
import pytest


def test_iterative_compile_policy_defaults_to_eager_and_allows_opt_in(monkeypatch):
    from statgpu.backends._torch_compile import resolve_torch_compile_mode

    monkeypatch.delenv("STATGPU_TORCH_COMPILE_MODE", raising=False)
    assert resolve_torch_compile_mode(workload="iterative") is None
    assert resolve_torch_compile_mode(
        workload="general", requested_mode="default"
    ) is None
    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "auto")
    assert resolve_torch_compile_mode(workload="iterative") is None
    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "disable")
    assert resolve_torch_compile_mode(workload="iterative") is None
    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "default")
    assert resolve_torch_compile_mode(workload="iterative") == "default"
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
    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "default")

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

    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "default")
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
    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "default")

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

    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "default")
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
    monkeypatch.setenv("STATGPU_TORCH_COMPILE_MODE", "default")

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


# PR87_GLM_WEIGHT_INFERENCE_DEVICE_TESTS
def test_torch_glm_formula_weight_inference_avoids_cpu_roundtrip(monkeypatch):
    torch = _require_modern_torch_cuda()
    pd = pytest.importorskip("pandas")
    import statgpu.linear_model._glm_base as glm_module
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0., 1., 2., 3., 4., 5.]}
    )
    weights = torch.tensor(
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        dtype=torch.float64,
        device="cuda",
    )
    original_to_numpy = glm_module._to_numpy

    def guarded_to_numpy(value):
        if (
            torch.is_tensor(value)
            and value.is_cuda
            and tuple(value.shape) == tuple(weights.shape)
            and bool(torch.allclose(value, weights))
        ):
            raise AssertionError("formula sample_weight copied to CPU")
        return original_to_numpy(value)

    monkeypatch.setattr(glm_module, "_to_numpy", guarded_to_numpy)
    model = GeneralizedLinearModel(
        family="gaussian",
        solver="irls",
        C=0.0,
        device="torch",
        compute_inference=True,
    ).fit(formula="y ~ x", data=data, sample_weight=weights)
    assert torch.is_tensor(model._sample_weight_inf)
    assert model._sample_weight_inf.is_cuda
    assert weights.is_cuda


def test_cupy_glm_formula_weight_inference_avoids_cpu_roundtrip(monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    pd = pytest.importorskip("pandas")
    import statgpu.linear_model._glm_base as glm_module
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "x": [0., 1., 2., 3., 4., 5.]}
    )
    weights = cp.asarray([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=cp.float64)
    original_to_numpy = glm_module._to_numpy

    def guarded_to_numpy(value):
        if (
            isinstance(value, cp.ndarray)
            and tuple(value.shape) == tuple(weights.shape)
            and bool(cp.allclose(value, weights))
        ):
            raise AssertionError("formula sample_weight copied to CPU")
        return original_to_numpy(value)

    monkeypatch.setattr(glm_module, "_to_numpy", guarded_to_numpy)
    model = GeneralizedLinearModel(
        family="gaussian",
        solver="irls",
        C=0.0,
        device="cuda",
        compute_inference=True,
    ).fit(formula="y ~ x", data=data, sample_weight=weights)
    assert isinstance(model._sample_weight_inf, cp.ndarray)
    assert int(model._sample_weight_inf.device.id) == int(weights.device.id)
    assert isinstance(weights, cp.ndarray)


# PR87_GLM_FISTA_WEIGHTED_INTERCEPT_TESTS
def _weighted_linear_reference(X, y, weights):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    design = np.column_stack([np.ones(X.shape[0]), X])
    root_w = np.sqrt(weights)
    return np.linalg.lstsq(
        design * root_w[:, None], y * root_w, rcond=None
    )[0]


def test_glm_fista_weighted_intercept_matches_closed_form_wls():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=np.float64
    )
    y = np.array([-1.0, 0.2, 1.1, 2.0, 8.0, 9.5], dtype=np.float64)
    weights = np.array([8.0, 7.0, 6.0, 2.0, 1.0, 0.5], dtype=np.float64)
    expected = _weighted_linear_reference(X, y, weights)

    model = GeneralizedLinearModel(
        family="gaussian",
        solver="fista",
        C=0.0,
        max_iter=4000,
        tol=1e-11,
        device="cpu",
        compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=2e-5, atol=2e-5)


def test_glm_formula_fista_weighted_intercept_matches_retained_wls():
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {
            "y": [-1.0, 0.2, 99.0, 2.0, 8.0, 9.5],
            "x": [-2.0, -1.0, np.nan, 1.0, 2.0, 3.0],
        }
    )
    weights = np.array([8.0, 7.0, 1000.0, 2.0, 1.0, 0.5])
    retained = np.array([0, 1, 3, 4, 5])
    expected = _weighted_linear_reference(
        data.loc[retained, ["x"]].to_numpy(),
        data.loc[retained, "y"].to_numpy(),
        weights[retained],
    )

    model = GeneralizedLinearModel(
        family="gaussian",
        solver="fista",
        C=0.0,
        max_iter=4000,
        tol=1e-11,
        device="cpu",
        compute_inference=False,
    ).fit(formula="y ~ x", data=data, sample_weight=weights)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=2e-5, atol=2e-5)


def test_torch_glm_formula_fista_weighted_intercept_matches_wls():
    torch = _require_modern_torch_cuda()
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {
            "y": [-1.0, 0.2, 99.0, 2.0, 8.0, 9.5],
            "x": [-2.0, -1.0, np.nan, 1.0, 2.0, 3.0],
        }
    )
    weights_np = np.array([8.0, 7.0, 1000.0, 2.0, 1.0, 0.5])
    weights = torch.as_tensor(weights_np, dtype=torch.float64, device="cuda")
    retained = np.array([0, 1, 3, 4, 5])
    expected = _weighted_linear_reference(
        data.loc[retained, ["x"]].to_numpy(),
        data.loc[retained, "y"].to_numpy(),
        weights_np[retained],
    )

    model = GeneralizedLinearModel(
        family="gaussian",
        solver="fista",
        C=0.0,
        max_iter=4000,
        tol=1e-11,
        device="torch",
        compute_inference=False,
    ).fit(formula="y ~ x", data=data, sample_weight=weights)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=3e-5, atol=3e-5)
    assert weights.is_cuda


def test_cupy_glm_formula_fista_weighted_intercept_matches_wls():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {
            "y": [-1.0, 0.2, 99.0, 2.0, 8.0, 9.5],
            "x": [-2.0, -1.0, np.nan, 1.0, 2.0, 3.0],
        }
    )
    weights_np = np.array([8.0, 7.0, 1000.0, 2.0, 1.0, 0.5])
    weights = cp.asarray(weights_np, dtype=cp.float64)
    retained = np.array([0, 1, 3, 4, 5])
    expected = _weighted_linear_reference(
        data.loc[retained, ["x"]].to_numpy(),
        data.loc[retained, "y"].to_numpy(),
        weights_np[retained],
    )

    model = GeneralizedLinearModel(
        family="gaussian",
        solver="fista",
        C=0.0,
        max_iter=4000,
        tol=1e-11,
        device="cuda",
        compute_inference=False,
    ).fit(formula="y ~ x", data=data, sample_weight=weights)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=3e-5, atol=3e-5)
    assert isinstance(weights, cp.ndarray)


# PR87_WEIGHTED_IRLS_AND_GLM_COMPILE_POLICY_TESTS
def test_weighted_irls_line_search_uses_weighted_objective_cpu():
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    X = np.ones((4, 1), dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float64)
    weights = np.array([1.0, 1.0, 1.0, 100.0], dtype=np.float64)
    params, _ = IRLSSolver(Gaussian(), max_iter=20, tol=1e-12).fit(
        X, y, sample_weight=weights, backend="numpy"
    )
    np.testing.assert_allclose(
        params[0], np.average(y, weights=weights), rtol=1e-10, atol=1e-10
    )


def test_glm_irls_weighted_ridge_matches_closed_form_and_weight_rescaling():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([-2.0, -0.5, 0.5, 2.0, 6.0], dtype=np.float64)
    weights = np.array([1.0, 2.0, 3.0, 7.0, 20.0], dtype=np.float64)
    C = 2.0
    lam = 1.0 / (2.0 * C)
    design = np.column_stack([np.ones(X.shape[0]), X])
    expected = np.linalg.solve(
        design.T @ (design * weights[:, None])
        + np.diag([0.0, weights.sum() * lam]),
        design.T @ (weights * y),
    )

    def fit(current_weights):
        return GeneralizedLinearModel(
            family="gaussian",
            solver="irls",
            C=C,
            max_iter=100,
            tol=1e-12,
            device="cpu",
            compute_inference=False,
        ).fit(X, y, sample_weight=current_weights)

    model = fit(weights)
    scaled = fit(17.0 * weights)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(scaled.intercept_, model.intercept_, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(scaled.coef_, model.coef_, rtol=1e-9, atol=1e-9)


def test_glm_weighted_loglikelihood_and_dispersion_match_manual_values():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([-0.5, 0.2, 1.8, 2.1, 5.5], dtype=np.float64)
    weights = np.array([1.0, 2.0, 4.0, 3.0, 8.0], dtype=np.float64)
    model = GeneralizedLinearModel(
        family="gaussian",
        solver="irls",
        C=0.0,
        max_iter=100,
        tol=1e-12,
        device="cpu",
        compute_inference=True,
    ).fit(X, y, sample_weight=weights)

    eta = model.intercept_ + X @ model.coef_
    resid_sq = (y - eta) ** 2
    expected_ll = -0.5 * X.shape[0] * float(
        np.sum(weights * resid_sq) / np.sum(weights)
    )
    k = 1 + X.shape[1]
    expected_dispersion = float(np.sum(weights * resid_sq)) / (X.shape[0] - k)
    np.testing.assert_allclose(model.loglikelihood, expected_ll, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        model._inference_result.metadata["dispersion"],
        expected_dispersion,
        rtol=1e-12,
        atol=1e-12,
    )

    no_inference = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0, device="cpu",
        compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    eta_no_inf = no_inference.intercept_ + X @ no_inference.coef_
    expected_no_inf = -0.5 * X.shape[0] * float(
        np.sum(weights * (y - eta_no_inf) ** 2) / np.sum(weights)
    )
    np.testing.assert_allclose(
        no_inference.loglikelihood, expected_no_inf, rtol=1e-12, atol=1e-12
    )


def test_active_glm_compile_helpers_use_central_policy_and_reraise():
    from pathlib import Path
    from statgpu.glm_core import _irls, _solver_utils

    for filename in (
        "statgpu/glm_core/_irls.py",
        "statgpu/glm_core/_solver_utils.py",
    ):
        source = Path(filename).read_text(encoding="utf-8")
        assert "torch.compile(" not in source
        assert "compile_torch(" in source

    def fail(*args):
        raise RuntimeError("unrelated runtime failure")

    with pytest.raises(RuntimeError, match="unrelated runtime failure"):
        _irls._irls_step_call(fail)
    with pytest.raises(RuntimeError, match="unrelated runtime failure"):
        _solver_utils._fista_step_call(fail)
    with pytest.raises(RuntimeError, match="unrelated runtime failure"):
        _solver_utils._newton_step_call(fail)


def test_torch_weighted_irls_compile_path_is_observable():
    torch = _require_modern_torch_cuda()
    from statgpu.backends._torch_compile import get_torch_compile_diagnostics
    from statgpu.glm_core import _irls
    from statgpu.linear_model import GeneralizedLinearModel

    _irls._IRLS_STEP_COMPILED = None
    torch._dynamo.reset()
    get_torch_compile_diagnostics(clear=True)
    before_graphs = _dynamo_unique_graphs(torch)

    X = np.arange(24.0, dtype=np.float64).reshape(12, 2)
    y = 1.5 + X @ np.array([0.2, -0.1])
    weights = torch.linspace(1.0, 3.0, X.shape[0], dtype=torch.float64, device="cuda")
    model = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0,
        max_iter=20, tol=1e-10, device="torch", compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    torch.cuda.synchronize()

    events = get_torch_compile_diagnostics(clear=True)
    assert _dynamo_unique_graphs(torch) > before_graphs
    assert any(event["status"] == "compiled" for event in events)
    assert not any("fallback" in event["status"] for event in events)
    assert np.isfinite(model.coef_).all()
    assert weights.is_cuda


def test_cupy_weighted_irls_matches_cpu_reference():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([-2.0, -0.5, 0.5, 2.0, 6.0], dtype=np.float64)
    weights_np = np.array([1.0, 2.0, 3.0, 7.0, 20.0], dtype=np.float64)
    weights = cp.asarray(weights_np)
    cpu = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=2.0,
        max_iter=100, tol=1e-12, device="cpu", compute_inference=False,
    ).fit(X, y, sample_weight=weights_np)
    gpu = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=2.0,
        max_iter=100, tol=1e-12, device="cuda", compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(gpu.intercept_, cpu.intercept_, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(gpu.coef_, cpu.coef_, rtol=1e-9, atol=1e-9)
    assert isinstance(weights, cp.ndarray)


# PR87_IRLS_OBJECTIVE_AND_EFFECTIVE_NOBS_TESTS
def test_irls_line_search_reuses_registered_loss_and_propagates_errors(monkeypatch):
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver
    from statgpu.glm_core._squared import SquaredErrorLoss

    def fail(self, eta, y):
        raise RuntimeError("objective evaluation failed")

    monkeypatch.setattr(SquaredErrorLoss, "per_sample_value", fail)
    with pytest.raises(RuntimeError, match="objective evaluation failed"):
        IRLSSolver(Gaussian(), max_iter=2).fit(
            np.ones((4, 1)), np.arange(4.0), backend="numpy"
        )


def test_irls_solve_only_falls_back_for_singular_systems(monkeypatch):
    from statgpu.glm_core import _irls

    singular = np.array([[1.0, 1.0], [2.0, 2.0]])
    rhs = np.array([1.0, 2.0])
    solution = _irls._solve(singular, rhs, backend="numpy")
    np.testing.assert_allclose(singular @ solution, rhs, rtol=1e-12, atol=1e-12)

    def invalid_solve(A, b):
        raise ValueError("shape/device contract failure")

    def forbidden_lstsq(*args, **kwargs):
        raise AssertionError("lstsq must not mask non-singularity failures")

    monkeypatch.setattr(np.linalg, "solve", invalid_solve)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden_lstsq)
    with pytest.raises(ValueError, match="shape/device contract failure"):
        _irls._solve(np.eye(2), np.ones(2), backend="numpy")


def test_irls_source_has_no_broad_objective_fallback_and_cupy_norm_is_native():
    from pathlib import Path

    source = Path("statgpu/glm_core/_irls.py").read_text(encoding="utf-8")
    line_search = source.split("# Backtracking line search", 1)[1].split(
        "# Convergence: normalized penalized score norm.", 1
    )[0]
    assert "except Exception" not in line_search
    assert "objective_loss.per_sample_value" in line_search
    norm_body = source.split("def _norm", 1)[1].split("def _zeros", 1)[0]
    assert "cp.linalg.norm" in norm_body


def test_glm_analytic_weight_diagnostics_are_scale_invariant():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [2.0], [4.0]], dtype=np.float64)
    y = np.array([-0.4, 0.5, 2.2, 5.1], dtype=np.float64)
    weights = np.array([0.5, 1.5, 2.0, 4.0], dtype=np.float64)

    def fit(current_weights, cov_type="nonrobust"):
        return GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            max_iter=100, tol=1e-12, device="cpu",
            compute_inference=True, cov_type=cov_type,
        ).fit(X, y, sample_weight=current_weights)

    weighted = fit(weights)
    scaled = fit(23.0 * weights)
    np.testing.assert_allclose(weighted.coef_, scaled.coef_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.intercept_, scaled.intercept_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.loglikelihood, scaled.loglikelihood, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.aic, scaled.aic, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.bic, scaled.bic, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted._bse, scaled._bse, rtol=1e-11, atol=1e-11)
    assert weighted._df_resid == X.shape[0] - (X.shape[1] + 1)

    robust = fit(weights, cov_type="hc0")
    robust_scaled = fit(23.0 * weights, cov_type="hc0")
    np.testing.assert_allclose(robust._bse, robust_scaled._bse, rtol=1e-11, atol=1e-11)


def test_glm_weighted_loglikelihood_uses_normalized_analytic_weights():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [1.0], [3.0]], dtype=np.float64)
    y = np.array([-0.2, 0.3, 1.4, 4.0], dtype=np.float64)
    weights = np.array([0.5, 2.0, 1.0, 6.0], dtype=np.float64)
    model = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0,
        device="cpu", compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    eta = model.intercept_ + X @ model.coef_
    expected = -X.shape[0] * np.sum(weights * 0.5 * (y - eta) ** 2) / np.sum(weights)
    np.testing.assert_allclose(model.loglikelihood, expected, rtol=1e-12, atol=1e-12)


def test_inference_solve_errors_only_downgrade_true_singularity(monkeypatch):
    import statgpu.inference._sandwich as sandwich
    from statgpu.glm_core._squared import SquaredErrorLoss

    X = np.column_stack([np.ones(5), np.arange(5.0)])
    y = np.arange(5.0)
    coef = np.array([0.0, 1.0])

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(np.linalg, "solve", oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        sandwich.compute_bread_avg(SquaredErrorLoss(), X, y, coef)

    assert sandwich._runtime_error_is_singular(
        RuntimeError("matrix is singular")
    )
    assert not sandwich._runtime_error_is_singular(
        RuntimeError("CUDA out of memory")
    )


def test_glm_parameter_count_does_not_use_numpy_array_conversion():
    from pathlib import Path

    source = Path("statgpu/linear_model/_glm_base.py").read_text(encoding="utf-8")
    block = source.split("# Parameter counts are backend-neutral", 1)[1].split(
        "# ---- Store design/loss", 1
    )[0]
    assert "self._params.shape[0]" in block
    assert "np.asarray(self._params)" not in block


# PR87_WEIGHTED_HELPER_SINGLE_SOURCE_TESTS
def test_solver_utils_weighted_helper_delegates_without_silent_unweighting(monkeypatch):
    from pathlib import Path
    import statgpu.glm_core._fused as fused
    import statgpu.glm_core._solver_utils as solver_utils

    sentinel = RuntimeError("weighted implementation failed")

    def fail(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(fused, "_weighted_loss_and_grad", fail)
    with pytest.raises(RuntimeError, match="weighted implementation failed"):
        solver_utils._weighted_loss_and_grad(
            object(), np.ones((2, 1)), np.ones(2), np.zeros(1), np.ones(2)
        )

    source = Path("statgpu/glm_core/_solver_utils.py").read_text(encoding="utf-8")
    block = source.split(
        "def _weighted_loss_and_grad(loss, X, y, coef, sample_weight):", 1
    )[1]
    assert "_to_numpy(sample_weight)" not in block
    assert "except TypeError" not in block
    assert "statgpu.glm_core._fused" in block


# PR87_GLM_RESPONSE_DOMAIN_MATRIX_TESTS
@pytest.mark.parametrize("solver", ["irls", "fista", "newton", "lbfgs"])
@pytest.mark.parametrize(
    "family,bad_y,message",
    [
        ("binomial", np.array([0.0, 1.0, -0.1, 0.5]), r"logistic response.*\[0, 1\]"),
        ("binomial", np.array([0.0, 1.0, 1.1, 0.5]), r"logistic response.*\[0, 1\]"),
        ("poisson", np.array([0.0, 1.0, -1.0, 2.0]), "poisson response.*non-negative"),
        ("gamma", np.array([1.0, 2.0, 0.0, 3.0]), "gamma response.*strictly positive"),
        ("inverse_gaussian", np.array([1.0, 2.0, -0.1, 3.0]), "inverse_gaussian response.*strictly positive"),
        ("negative_binomial", np.array([0.0, 1.0, -1.0, 2.0]), "negative_binomial response.*non-negative"),
        ("tweedie", np.array([0.0, 1.0, -0.1, 2.0]), "tweedie response.*non-negative"),
    ],
)
def test_glm_response_domain_is_validated_before_every_solver(family, bad_y, message, solver):
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.arange(8.0, dtype=np.float64).reshape(4, 2)
    with pytest.raises(ValueError, match=message):
        GeneralizedLinearModel(
            family=family,
            solver=solver,
            C=0.0,
            device="cpu",
            compute_inference=False,
        ).fit(X, bad_y)


def test_binomial_glm_accepts_fractional_responses_in_unit_interval():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0.0, 0.2, 0.5, 0.8, 1.0], dtype=np.float64)
    model = GeneralizedLinearModel(
        family="binomial", solver="irls", C=0.0,
        max_iter=100, device="cpu", compute_inference=False,
    ).fit(X, y)
    assert np.isfinite(model.coef_).all()


def test_formula_response_domain_validation_occurs_after_patsy_row_selection():
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {"y": [0.0, 1.0, -2.0, 3.0], "x": [0.0, 1.0, np.nan, 3.0]}
    )
    # The negative response belongs to the row Patsy removes.  Retained rows
    # are valid Poisson responses and must fit successfully.
    model = GeneralizedLinearModel(
        family="poisson", solver="irls", C=0.0,
        device="cpu", compute_inference=False,
    ).fit(formula="y ~ x", data=data)
    assert np.isfinite(model.coef_).all()

    data.loc[1, "y"] = -1.0
    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        ).fit(formula="y ~ x", data=data)


def test_direct_irls_solver_uses_loss_owned_response_validation():
    from statgpu.glm_core._family import Poisson
    from statgpu.glm_core._irls import IRLSSolver

    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        IRLSSolver(Poisson()).fit(
            np.ones((3, 1)), np.array([0.0, -1.0, 2.0]), backend="numpy"
        )


def test_torch_glm_response_domain_validation_stays_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model import GeneralizedLinearModel

    X = torch.arange(8.0, dtype=torch.float64, device="cuda").reshape(4, 2)
    y = torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="torch", compute_inference=False,
        ).fit(X, y)
    assert X.is_cuda and y.is_cuda


def test_cupy_glm_response_domain_validation_stays_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model import GeneralizedLinearModel

    X = cp.arange(8.0, dtype=cp.float64).reshape(4, 2)
    y = cp.asarray([1.0, 2.0, 0.0, 3.0], dtype=cp.float64)
    with pytest.raises(ValueError, match="gamma response.*strictly positive"):
        GeneralizedLinearModel(
            family="gamma", solver="irls", C=0.0,
            device="cuda", compute_inference=False,
        ).fit(X, y)
    assert isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)


# PR87_PENALIZED_GLM_RESPONSE_DOMAIN_TESTS
@pytest.mark.parametrize(
    "loss,bad_y,message",
    [
        ("logistic", [0.0, 1.0, 1.2, 0.5], r"logistic response.*\[0, 1\]"),
        ("poisson", [0.0, 1.0, -1.0, 2.0], "poisson response.*non-negative"),
        ("gamma", [1.0, 2.0, 0.0, 3.0], "gamma response.*strictly positive"),
        ("inverse_gaussian", [1.0, 2.0, -0.1, 3.0], "inverse_gaussian response.*strictly positive"),
        ("negative_binomial", [0.0, 1.0, -1.0, 2.0], "negative_binomial response.*non-negative"),
        ("tweedie", [0.0, 1.0, -0.1, 2.0], "tweedie response.*non-negative"),
    ],
)
def test_penalized_glm_validates_array_like_response_before_solver(loss, bad_y, message):
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = np.arange(8.0, dtype=np.float64).reshape(4, 2)
    with pytest.raises(ValueError, match=message):
        PenalizedGeneralizedLinearModel(
            loss=loss,
            penalty="l2",
            alpha=0.1,
            solver="fista",
            device="cpu",
            compute_inference=False,
        ).fit(X, bad_y)


def test_penalized_formula_response_validation_uses_retained_rows():
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    data = pd.DataFrame(
        {"y": [0.0, 1.0, -3.0, 2.0, 4.0], "x": [0.0, 1.0, np.nan, 3.0, 4.0]}
    )
    model = PenalizedGeneralizedLinearModel(
        loss="poisson", penalty="l2", alpha=0.1,
        solver="fista", max_iter=100, device="cpu",
        compute_inference=False,
    ).fit(formula="y ~ x", data=data)
    assert np.isfinite(model.coef_).all()

    data.loc[1, "y"] = -1.0
    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        ).fit(formula="y ~ x", data=data)


def test_penalized_glm_cv_rejects_invalid_response_before_folds_and_resets_state(monkeypatch):
    from statgpu.linear_model.penalized import PenalizedGLM_CV

    X = np.arange(12.0, dtype=np.float64).reshape(6, 2)
    y = np.array([0.0, 1.0, 2.0, -1.0, 3.0, 4.0])
    model = PenalizedGLM_CV(
        loss="poisson", penalty="l2", alpha_grid=[0.1, 1.0],
        cv=2, device="cpu", max_iter=20,
    )
    fold_called = False

    def forbidden(*args, **kwargs):
        nonlocal fold_called
        fold_called = True
        raise AssertionError("CV folds must not run for an invalid response")

    monkeypatch.setattr(model, "_fit_standard", forbidden)
    model._fitted = True
    model.alpha_ = 99.0
    model.coef_ = np.ones(2)
    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        model.fit(X, y)
    assert not fold_called
    assert model._fitted is False
    assert model.alpha_ is None
    assert model.coef_ is None


def test_torch_penalized_glm_response_validation_stays_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = torch.arange(8.0, dtype=torch.float64, device="cuda").reshape(4, 2)
    y = torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.1,
            solver="fista", device="torch", compute_inference=False,
        ).fit(X, y)
    assert X.is_cuda and y.is_cuda


def test_cupy_penalized_glm_response_validation_stays_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = cp.arange(8.0, dtype=cp.float64).reshape(4, 2)
    y = cp.asarray([1.0, 2.0, 0.0, 3.0], dtype=cp.float64)
    with pytest.raises(ValueError, match="gamma response.*strictly positive"):
        PenalizedGeneralizedLinearModel(
            loss="gamma", penalty="l2", alpha=0.1,
            solver="fista", device="cuda", compute_inference=False,
        ).fit(X, y)
    assert isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)


# PR87_GLM_RESPONSE_SHAPE_CONTRACT_TESTS
@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
def test_scalar_glm_rejects_multicolumn_response_before_solver(kind, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    X = np.arange(12.0, dtype=np.float64).reshape(6, 2)
    y = np.ones((6, 2), dtype=np.float64)
    if kind == "glm":
        model = GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        )
    elif kind == "penalized":
        model = PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        )
    else:
        model = PenalizedGLM_CV(
            loss="poisson", penalty="l2", alpha_grid=[0.1, 1.0],
            cv=2, device="cpu", max_iter=20,
        )
        monkeypatch.setattr(
            model,
            "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for multicolumn y")
            ),
        )
    with pytest.raises(ValueError, match="response must be one-dimensional"):
        model.fit(X, y)


@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
def test_scalar_glm_rejects_response_length_mismatch_before_solver(kind, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    X = np.arange(12.0, dtype=np.float64).reshape(6, 2)
    y = np.ones(5, dtype=np.float64)
    if kind == "glm":
        model = GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        )
    elif kind == "penalized":
        model = PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        )
    else:
        model = PenalizedGLM_CV(
            loss="poisson", penalty="l2", alpha_grid=[0.1, 1.0],
            cv=2, device="cpu", max_iter=20,
        )
        monkeypatch.setattr(
            model,
            "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for length mismatch")
            ),
        )
    with pytest.raises(ValueError, match=r"Response length must match (?:X\.shape\[0\]|the number of X rows)"):
        model.fit(X, y)


def test_scalar_glm_accepts_single_column_response_consistently():
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([[0.0], [1.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    glm = GeneralizedLinearModel(
        family="poisson", solver="irls", C=0.0,
        device="cpu", compute_inference=False,
    ).fit(X, y)
    penalized = PenalizedGeneralizedLinearModel(
        loss="poisson", penalty="l2", alpha=0.1,
        solver="fista", max_iter=100, device="cpu",
        compute_inference=False,
    ).fit(X, y)
    assert np.isfinite(glm.coef_).all()
    assert np.isfinite(penalized.coef_).all()


def test_direct_irls_rejects_multicolumn_and_length_mismatch():
    from statgpu.glm_core._family import Poisson
    from statgpu.glm_core._irls import IRLSSolver

    X = np.ones((4, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="response must be one-dimensional"):
        IRLSSolver(Poisson()).fit(X, np.ones((4, 2)), backend="numpy")
    with pytest.raises(ValueError, match=r"Response length must match (?:X\.shape\[0\]|the number of X rows)"):
        IRLSSolver(Poisson()).fit(X, np.ones(3), backend="numpy")


def test_glm_rejects_nonnumeric_response_with_public_value_error():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.ones((3, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="real numeric values"):
        GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        ).fit(X, np.array(["0", "one", "2"], dtype=object))


def test_torch_glm_multicolumn_response_rejected_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model import GeneralizedLinearModel

    X = torch.ones((4, 2), dtype=torch.float64, device="cuda")
    y = torch.ones((4, 2), dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="response must be one-dimensional"):
        GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="torch", compute_inference=False,
        ).fit(X, y)
    assert X.is_cuda and y.is_cuda


def test_cupy_penalized_glm_multicolumn_response_rejected_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = cp.ones((4, 2), dtype=cp.float64)
    y = cp.ones((4, 2), dtype=cp.float64)
    with pytest.raises(ValueError, match="response must be one-dimensional"):
        PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.1,
            solver="fista", device="cuda", compute_inference=False,
        ).fit(X, y)
    assert isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)


# PR87_GLM_LIST_DESIGN_LENGTH_TEST
def test_penalized_glm_cv_response_length_check_preserves_list_design_input(monkeypatch):
    from statgpu.linear_model.penalized import PenalizedGLM_CV

    X = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    y = [0.0, 1.0, 2.0, 3.0]
    model = PenalizedGLM_CV(
        loss="poisson", penalty="l2", alpha_grid=[0.1],
        cv=2, device="cpu", max_iter=10,
    )
    seen = {}

    def capture(X_arg, y_arg, sample_weight=None):
        seen["X"] = X_arg
        seen["y"] = y_arg
        return model

    monkeypatch.setattr(model, "_fit_standard", capture)
    result = model.fit(X, y)
    assert result is model
    assert seen["X"] is X
    assert isinstance(seen["y"], np.ndarray)
    assert seen["y"].shape == (len(X),)


# PR87_GLM_REAL_NONEMPTY_RESPONSE_TESTS
@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
@pytest.mark.parametrize(
    "bad_y",
    [
        np.array([0.0 + 0.0j, 1.0 + 0.0j, 2.0 + 1.0j]),
        np.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64[D]"),
        np.array(["0", "1", "two"], dtype=object),
    ],
)
def test_glm_entrypoints_reject_nonreal_response_with_value_error(kind, bad_y, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    X = np.arange(6.0, dtype=np.float64).reshape(3, 2)
    if kind == "glm":
        model = GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        )
    elif kind == "penalized":
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        )
    else:
        model = PenalizedGLM_CV(
            loss="squared_error", penalty="l2", alpha_grid=[0.1],
            cv=2, device="cpu", max_iter=10,
        )
        monkeypatch.setattr(
            model,
            "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for a nonreal response")
            ),
        )
    with pytest.raises(ValueError, match="response must contain real numeric"):
        model.fit(X, bad_y)


@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
def test_glm_entrypoints_reject_empty_response_before_solver(kind, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    X = np.empty((0, 2), dtype=np.float64)
    y = np.empty(0, dtype=np.float64)
    if kind == "glm":
        model = GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        )
    elif kind == "penalized":
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        )
    else:
        model = PenalizedGLM_CV(
            loss="squared_error", penalty="l2", alpha_grid=[0.1],
            cv=2, device="cpu", max_iter=10,
        )
        monkeypatch.setattr(
            model,
            "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for an empty response")
            ),
        )
    with pytest.raises(ValueError, match="at least one observation"):
        model.fit(X, y)


def test_direct_irls_rejects_complex_and_empty_response():
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    with pytest.raises(ValueError, match="real numeric"):
        IRLSSolver(Gaussian()).fit(
            np.ones((2, 1)), np.array([1.0 + 0.0j, 2.0 + 1.0j]),
            backend="numpy",
        )
    with pytest.raises(ValueError, match="at least one observation"):
        IRLSSolver(Gaussian()).fit(
            np.empty((0, 1)), np.empty(0), backend="numpy"
        )


def test_torch_glm_complex_response_rejected_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model import GeneralizedLinearModel

    X = torch.ones((3, 1), dtype=torch.float64, device="cuda")
    y = torch.tensor([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 1.0j], device="cuda")
    with pytest.raises(ValueError, match="real numeric"):
        GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="torch", compute_inference=False,
        ).fit(X, y)
    assert X.is_cuda and y.is_cuda


def test_cupy_penalized_glm_complex_response_rejected_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = cp.ones((3, 1), dtype=cp.float64)
    y = cp.asarray([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 1.0j])
    with pytest.raises(ValueError, match="real numeric"):
        PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cuda", compute_inference=False,
        ).fit(X, y)
    assert isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)


# PR87_GLM_DESIGN_AND_WEIGHT_CONTRACT_TESTS
@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
@pytest.mark.parametrize(
    "bad_X,message",
    [
        (np.ones(4), "two-dimensional design matrix"),
        (np.ones((2, 2, 1)), "two-dimensional design matrix"),
        (np.empty((0, 2)), "at least one observation"),
        (np.ones((4, 2), dtype=np.complex128), "real numeric values"),
        (np.array([["a"], ["b"], ["c"], ["d"]], dtype=object), "real numeric values"),
    ],
)
def test_glm_entrypoints_reject_invalid_design_before_solver(kind, bad_X, message, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    y = np.arange(len(bad_X) if getattr(bad_X, "ndim", 0) else 4, dtype=float)
    if bad_X.shape[0] == 0:
        y = np.empty(0)
    if kind == "glm":
        model = GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        )
    elif kind == "penalized":
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        )
    else:
        model = PenalizedGLM_CV(
            loss="squared_error", penalty="l2", alpha_grid=[0.1],
            cv=2, device="cpu", max_iter=10,
        )
        monkeypatch.setattr(
            model, "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for invalid X")
            ),
        )
    with pytest.raises(ValueError, match=message):
        model.fit(bad_X, y)


def test_glm_intercept_only_zero_feature_design_is_supported():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.empty((5, 0), dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0,
        device="cpu", compute_inference=False,
    ).fit(X, y)
    assert model.coef_.shape == (0,)
    np.testing.assert_allclose(model.intercept_, np.mean(y))


@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
@pytest.mark.parametrize(
    "bad_weight,message",
    [
        (np.ones((4, 1)), "one-dimensional"),
        (np.ones(3), "length n_samples"),
        (np.array([1.0, 1.0j, 1.0, 1.0]), "real numeric values"),
        (np.array(["1", "1", "1", "1"], dtype=object), "real numeric values"),
    ],
)
def test_glm_entrypoints_reject_invalid_sample_weight_consistently(kind, bad_weight, message, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    X = np.arange(8.0).reshape(4, 2)
    y = np.arange(4.0)
    if kind == "glm":
        model = GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        )
    elif kind == "penalized":
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        )
    else:
        model = PenalizedGLM_CV(
            loss="squared_error", penalty="l2", alpha_grid=[0.1],
            cv=2, device="cpu", max_iter=10,
        )
        monkeypatch.setattr(
            model, "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for invalid weights")
            ),
        )
    with pytest.raises(ValueError, match=message):
        model.fit(X, y, sample_weight=bad_weight)


def test_direct_irls_validates_design_and_weights_before_backend_math():
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    with pytest.raises(ValueError, match="two-dimensional design matrix"):
        IRLSSolver(Gaussian()).fit(np.ones(4), np.ones(4), backend="numpy")
    with pytest.raises(ValueError, match="real numeric values"):
        IRLSSolver(Gaussian()).fit(
            np.ones((4, 1)), np.ones(4),
            sample_weight=np.array([1.0, 1.0j, 1.0, 1.0]),
            backend="numpy",
        )


def test_penalized_cv_design_validation_preserves_list_X_identity(monkeypatch):
    from statgpu.linear_model.penalized import PenalizedGLM_CV

    X = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    y = [0.0, 1.0, 2.0, 3.0]
    model = PenalizedGLM_CV(
        loss="squared_error", penalty="l2", alpha_grid=[0.1],
        cv=2, device="cpu", max_iter=10,
    )
    seen = {}
    def capture(X_arg, y_arg, sample_weight=None):
        seen["X"] = X_arg
        seen["y"] = y_arg
        return model
    monkeypatch.setattr(model, "_fit_standard", capture)
    model.fit(X, y)
    assert seen["X"] is X
    assert isinstance(seen["y"], np.ndarray)


def test_torch_glm_complex_design_and_weight_rejected_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model import GeneralizedLinearModel

    X_complex = torch.ones((4, 2), dtype=torch.complex128, device="cuda")
    y = torch.arange(4.0, dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="real numeric values"):
        GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="torch", compute_inference=False,
        ).fit(X_complex, y)

    X = torch.ones((4, 2), dtype=torch.float64, device="cuda")
    weight = torch.ones(4, dtype=torch.complex128, device="cuda")
    with pytest.raises(ValueError, match="real numeric values"):
        GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="torch", compute_inference=False,
        ).fit(X, y, sample_weight=weight)
    assert X_complex.is_cuda and weight.is_cuda


def test_cupy_penalized_glm_complex_design_and_weight_rejected_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    y = cp.arange(4.0, dtype=cp.float64)
    X_complex = cp.ones((4, 2), dtype=cp.complex128)
    with pytest.raises(ValueError, match="real numeric values"):
        PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cuda", compute_inference=False,
        ).fit(X_complex, y)

    X = cp.ones((4, 2), dtype=cp.float64)
    weight = cp.ones(4, dtype=cp.complex128)
    with pytest.raises(ValueError, match="real numeric values"):
        PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cuda", compute_inference=False,
        ).fit(X, y, sample_weight=weight)
    assert isinstance(X_complex, cp.ndarray) and isinstance(weight, cp.ndarray)

# PR87_REVIEW_FIX_V37
def test_direct_fista_validates_weight_length_before_lipschitz():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import fista_solver

    class GuardedSquaredError(SquaredErrorLoss):
        def lipschitz(self, *args, **kwargs):
            raise AssertionError("lipschitz must not run before weight validation")

    X = np.ones((3, 1), dtype=np.float64)
    y = np.arange(3.0)
    with pytest.raises(ValueError, match="length n_samples"):
        fista_solver(
            GuardedSquaredError(),
            get_penalty("l2", alpha=0.0),
            X,
            y,
            sample_weight=np.ones(2),
        )


def test_solver_weight_validation_does_not_copy_torch_tensor(monkeypatch):
    torch = pytest.importorskip("torch")
    import statgpu.solvers._utils as solver_utils

    def forbidden(*args, **kwargs):
        raise AssertionError("sample_weight must not be copied through _to_numpy")

    monkeypatch.setattr(solver_utils, "_to_numpy", forbidden)
    weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    solver_utils._validate_sample_weight(weights, 3)
    assert torch.equal(weights, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))


def test_penalized_cv_uniform_weight_check_does_not_copy_torch_tensor(monkeypatch):
    torch = pytest.importorskip("torch")
    import statgpu.linear_model.penalized._penalized_cv as penalized_cv

    def forbidden(*args, **kwargs):
        raise AssertionError("uniform-weight check must stay backend-native")

    monkeypatch.setattr(penalized_cv, "_to_numpy", forbidden)
    assert penalized_cv._is_uniform_weight(torch.ones(4, dtype=torch.float64))
    assert not penalized_cv._is_uniform_weight(
        torch.tensor([1.0, 1.0, 2.0, 1.0], dtype=torch.float64)
    )


def test_glm_weight_validation_rejects_overflowing_total():
    from statgpu.glm_core._validation import validate_glm_sample_weight
    from statgpu.solvers._utils import _validate_sample_weight

    weights = np.array([np.finfo(np.float64).max, np.finfo(np.float64).max])
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="finite positive sum"):
            validate_glm_sample_weight(weights, 2)
        with pytest.raises(ValueError, match="finite positive sum"):
            _validate_sample_weight(weights, 2)


def test_glm_hc1_analytic_weight_diagnostics_are_scale_invariant():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [2.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([-0.4, 0.5, 2.2, 5.1, 5.8], dtype=np.float64)
    weights = np.array([0.5, 1.5, 2.0, 4.0, 3.0], dtype=np.float64)

    def fit(current_weights):
        return GeneralizedLinearModel(
            family="gaussian",
            solver="irls",
            C=0.0,
            max_iter=100,
            tol=1e-12,
            device="cpu",
            compute_inference=True,
            cov_type="hc1",
        ).fit(X, y, sample_weight=current_weights)

    weighted = fit(weights)
    scaled = fit(29.0 * weights)
    np.testing.assert_allclose(weighted._bse, scaled._bse, rtol=1e-11, atol=1e-11)

# PR87_REVIEW_FIX_V38
def test_solver_weight_reduction_is_computed_once():
    from pathlib import Path

    source = Path("statgpu/solvers/_utils.py").read_text(encoding="utf-8")
    block = source.split("def _validated_sample_weight", 1)[1].split(
        "def _validate_uniform_sample_weight", 1
    )[0]
    assert block.count("xp.sum(values)") == 1


def test_direct_fista_bb_validates_weight_length_before_lipschitz():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import fista_bb_solver

    class GuardedSquaredError(SquaredErrorLoss):
        def lipschitz(self, *args, **kwargs):
            raise AssertionError("lipschitz must not run before weight validation")

    X = np.ones((3, 1), dtype=np.float64)
    y = np.arange(3.0)
    with pytest.raises(ValueError, match="length n_samples"):
        fista_bb_solver(
            GuardedSquaredError(),
            get_penalty("l1", alpha=0.1),
            X,
            y,
            sample_weight=np.ones(2),
        )


def test_newton_does_not_mask_non_singular_solve_errors(monkeypatch):
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import newton_solver

    X = np.column_stack([np.ones(4), np.arange(4.0)])
    y = np.arange(4.0)

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    def forbidden(*args, **kwargs):
        raise AssertionError("lstsq must not mask infrastructure failures")

    monkeypatch.setattr(np.linalg, "solve", oom)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    with pytest.raises(RuntimeError, match="out of memory"):
        newton_solver(
            SquaredErrorLoss(), get_penalty("l2", alpha=0.1), X, y, max_iter=2
        )


def test_newton_validates_weights_before_constant_hessian():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import newton_solver

    class GuardedSquaredError(SquaredErrorLoss):
        def hessian(self, *args, **kwargs):
            raise AssertionError("hessian must not run before weight validation")

    with pytest.raises(ValueError, match="length n_samples"):
        newton_solver(
            GuardedSquaredError(),
            get_penalty("l2", alpha=0.1),
            np.ones((3, 1)),
            np.ones(3),
            sample_weight=np.ones(2),
        )


def test_admm_does_not_mask_non_singular_cholesky_errors(monkeypatch):
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import admm_solver

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(np.linalg, "cholesky", oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        admm_solver(
            SquaredErrorLoss(),
            get_penalty("l1", alpha=0.1),
            np.ones((4, 1)),
            np.arange(4.0),
            max_iter=2,
        )


def test_proximal_newton_validates_weight_length_before_curvature():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    class GuardedSquaredError(SquaredErrorLoss):
        def fused_gradient_and_hessian(self, *args, **kwargs):
            raise AssertionError("curvature must not run before weight validation")

    with pytest.raises(ValueError, match="length n_samples"):
        proximal_newton_solver(
            GuardedSquaredError(),
            get_penalty("l1", alpha=0.1),
            np.ones((3, 1)),
            np.ones(3),
            sample_weight=np.ones(2),
        )


def test_proximal_newton_preserves_torch_float32_dtype():
    torch = pytest.importorskip("torch")
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    coef, _ = proximal_newton_solver(
        SquaredErrorLoss(),
        get_penalty("l1", alpha=0.01),
        X,
        y,
        max_iter=3,
    )
    assert coef.dtype == torch.float32
    assert bool(torch.all(torch.isfinite(coef)).item())


def test_lbfgs_steepest_descent_uses_squared_norm_slope():
    from pathlib import Path

    lbfgs = Path("statgpu/solvers/_lbfgs.py").read_text(encoding="utf-8")
    lbfgsb = Path("statgpu/solvers/_lbfgs_b.py").read_text(encoding="utf-8")
    assert "gdd = -gn * gn" in lbfgs
    assert "direction = -proj_grad" in lbfgsb
    assert "gdd = -pg_norm * pg_norm" in lbfgsb


def test_lbfgsb_torch_bounds_and_projection_are_backend_native():
    torch = pytest.importorskip("torch")
    from statgpu.solvers._lbfgs_b import _clip_to_bounds, _projected_gradient

    params = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32)
    lb = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32)
    ub = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)
    clipped = _clip_to_bounds(params, lb, ub, "torch")
    torch.testing.assert_close(clipped, torch.tensor([-1.0, 0.5, 2.0]))
    grad = torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32)
    projected = _projected_gradient(grad, clipped, lb, ub, "torch")
    torch.testing.assert_close(projected, torch.tensor([0.0, -1.0, 0.0]))


def test_lbfgsb_cupy_bounds_and_projection_are_backend_native():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.solvers._lbfgs_b import _clip_to_bounds, _projected_gradient

    params = cp.asarray([-2.0, 0.5, 3.0], dtype=cp.float32)
    lb = cp.asarray([-1.0, 0.0, 0.0], dtype=cp.float32)
    ub = cp.asarray([1.0, 1.0, 2.0], dtype=cp.float32)
    clipped = _clip_to_bounds(params, lb, ub, "cupy")
    cp.testing.assert_allclose(clipped, cp.asarray([-1.0, 0.5, 2.0]))
    grad = cp.asarray([1.0, -1.0, -1.0], dtype=cp.float32)
    projected = _projected_gradient(grad, clipped, lb, ub, "cupy")
    cp.testing.assert_allclose(projected, cp.asarray([0.0, -1.0, 0.0]))

# PR87_REVIEW_FIX_V39
def test_admm_legitimate_cholesky_failure_initializes_iterative_fallback(monkeypatch):
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import admm_solver

    def not_positive_definite(*args, **kwargs):
        raise np.linalg.LinAlgError("not positive definite")

    monkeypatch.setattr(np.linalg, "cholesky", not_positive_definite)
    coef, n_iter = admm_solver(
        SquaredErrorLoss(),
        get_penalty("l1", alpha=0.05),
        np.column_stack([np.ones(6), np.arange(6.0)]),
        np.arange(6.0),
        max_iter=2,
    )
    assert n_iter >= 0
    assert np.all(np.isfinite(coef))


def test_proximal_newton_max_iter_zero_returns_initialized_coefficients():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    init = np.array([0.25, -0.5])
    coef, n_iter = proximal_newton_solver(
        SquaredErrorLoss(),
        get_penalty("l2", alpha=0.1),
        np.column_stack([np.ones(4), np.arange(4.0)]),
        np.arange(4.0),
        init_coef=init,
        max_iter=0,
    )
    np.testing.assert_allclose(coef, init)
    assert n_iter == 0


def test_proximal_newton_l2_matches_declared_closed_form_objective():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    X = np.array(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 4.0]],
        dtype=np.float64,
    )
    y = np.array([0.2, 1.0, 2.1, 2.7, 5.3], dtype=np.float64)
    alpha = 0.35
    expected = np.linalg.solve(
        X.T @ X / X.shape[0] + alpha * np.eye(X.shape[1]),
        X.T @ y / X.shape[0],
    )
    coef, _ = proximal_newton_solver(
        SquaredErrorLoss(),
        get_penalty("l2", alpha=alpha),
        X,
        y,
        max_iter=20,
        tol=1e-12,
    )
    np.testing.assert_allclose(coef, expected, rtol=1e-8, atol=1e-9)


def test_proximal_newton_nonsmooth_fallback_is_explicit_and_objective_preserving():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import fista_solver, proximal_newton_solver

    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    y = np.array([1.0, 1.8, 3.2, 3.9], dtype=np.float64)
    penalty = get_penalty("l1", alpha=0.05)
    with pytest.warns(RuntimeWarning, match="delegates non-smooth penalties"):
        delegated = proximal_newton_solver(
            SquaredErrorLoss(), penalty, X, y, max_iter=40, tol=1e-10
        )
    direct = fista_solver(
        SquaredErrorLoss(), penalty, X, y, max_iter=40, tol=1e-10
    )
    np.testing.assert_allclose(delegated[0], direct[0], rtol=0.0, atol=0.0)
    assert delegated[1] == direct[1]


def test_fista_lla_requires_explicit_metric_proximal_newton_capability():
    from pathlib import Path

    source = Path("statgpu/solvers/_fista_lla.py").read_text(encoding="utf-8")
    assert "_supports_metric_proximal_newton" in source


def test_lbfgsb_projects_quasi_newton_direction_and_rejects_nan_bounds():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import lbfgs_b_solver
    from statgpu.solvers._lbfgs_b import _project_direction

    params = np.array([-1.0, 0.5, 2.0])
    lb = np.array([-1.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 2.0])
    direction = np.array([-2.0, 0.25, 3.0])
    np.testing.assert_allclose(
        _project_direction(direction, params, lb, ub, "numpy"),
        np.array([0.0, 0.25, 0.0]),
    )

    with pytest.raises(ValueError, match="must not contain NaN"):
        lbfgs_b_solver(
            SquaredErrorLoss(),
            get_penalty("l2", alpha=0.1),
            np.ones((3, 1)),
            np.ones(3),
            lower_bounds=np.array([np.nan]),
            upper_bounds=np.array([1.0]),
        )

# PR87_REVIEW_FIX_V40
import warnings


def _run_warm_start_solver_matrix(X, y, init):
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import (
        admm_solver,
        fista_bb_solver,
        fista_solver,
        lbfgs_b_solver,
        lbfgs_solver,
        newton_solver,
        proximal_newton_solver,
    )

    loss = SquaredErrorLoss()
    l2 = get_penalty("l2", alpha=0.05)
    l1 = get_penalty("l1", alpha=0.05)
    return {
        "fista": fista_solver(
            loss, l1, X, y, init_coef=init, max_iter=2, tol=1e-8
        )[0],
        "fista_bb": fista_bb_solver(
            loss, l1, X, y, init_coef=init, max_iter=2, tol=1e-8
        )[0],
        "newton": newton_solver(
            loss, l2, X, y, init_coef=init, max_iter=2, tol=1e-8
        )[0],
        "proximal_newton": proximal_newton_solver(
            loss, l2, X, y, init_coef=init, max_iter=2, tol=1e-8
        )[0],
        "lbfgs": lbfgs_solver(
            loss, l2, X, y, init_coef=init, max_iter=2, tol=1e-8
        )[0],
        "lbfgs_b": lbfgs_b_solver(
            loss,
            l2,
            X,
            y,
            init_coef=init,
            lower_bounds=np.full(X.shape[1], -10.0),
            upper_bounds=np.full(X.shape[1], 10.0),
            max_iter=2,
            tol=1e-8,
        )[0],
        "admm": admm_solver(
            loss, l1, X, y, init_coef=init, max_iter=2, tol=1e-8
        )[0],
    }


def test_solver_numpy_warm_starts_follow_torch_cpu_backend_dtype():
    torch = pytest.importorskip("torch")

    X = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        dtype=torch.float32,
    )
    y = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    init = np.array([0.2, -0.1], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = _run_warm_start_solver_matrix(X, y, init)
    for name, coef in outputs.items():
        assert isinstance(coef, torch.Tensor), name
        assert coef.device == X.device, name
        assert coef.dtype == X.dtype, name
        assert bool(torch.all(torch.isfinite(coef)).item()), name


def test_torch_cuda_solver_numpy_warm_starts_stay_on_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires physical CUDA")

    X = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        dtype=torch.float64,
        device="cuda",
    )
    y = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64, device="cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = _run_warm_start_solver_matrix(
            X, y, np.array([0.2, -0.1], dtype=np.float64)
        )
    for name, coef in outputs.items():
        assert coef.device.type == "cuda", name
        assert coef.dtype == X.dtype, name
        assert bool(torch.all(torch.isfinite(coef)).item()), name


def test_cupy_solver_numpy_warm_starts_stay_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires physical CUDA")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")

    X = cp.asarray(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        dtype=cp.float64,
    )
    y = cp.asarray([0.0, 1.0, 2.0, 3.0], dtype=cp.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = _run_warm_start_solver_matrix(
            X, y, np.array([0.2, -0.1], dtype=np.float64)
        )
    for name, coef in outputs.items():
        assert isinstance(coef, cp.ndarray), name
        assert coef.dtype == X.dtype, name
        assert bool(cp.all(cp.isfinite(coef)).item()), name


def test_proximal_newton_none_penalty_has_no_spurious_warning():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.solvers import proximal_newton_solver

    X = np.column_stack([np.ones(4), np.arange(4.0)])
    y = np.arange(4.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coef, _ = proximal_newton_solver(
            SquaredErrorLoss(), None, X, y, max_iter=2
        )
    assert np.all(np.isfinite(coef))
    assert not any("has no value" in str(item.message) for item in caught)

# PR87_REVIEW_FIX_V41
def test_smooth_solvers_reject_elasticnet_before_numerical_work():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import lbfgs_b_solver, lbfgs_solver, newton_solver

    class GuardedLoss:
        def preprocess(self, *args, **kwargs):
            raise AssertionError("penalty validation must precede preprocessing")

    penalty = get_penalty("elasticnet", alpha=0.2, l1_ratio=0.5)
    X = np.ones((3, 1), dtype=np.float64)
    y = np.ones(3, dtype=np.float64)
    for solver in (newton_solver, lbfgs_solver, lbfgs_b_solver):
        with pytest.raises(ValueError, match="supports only l2/none"):
            solver(GuardedLoss(), penalty, X, y)

# PR87_REVIEW_FIX_V43
def test_solver_weight_validation_does_not_mask_runtime_failures(monkeypatch):
    import statgpu.solvers._utils as solver_utils

    class RuntimeFailingXP:
        @staticmethod
        def all(value):
            return value

        @staticmethod
        def isfinite(values):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        solver_utils,
        "_native_sample_weight",
        lambda sample_weight: ("numpy", RuntimeFailingXP(), np.ones(2)),
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        solver_utils._validate_sample_weight(np.ones(2), 2)


def test_solver_weight_validation_runtime_catches_are_narrow():
    from pathlib import Path

    source = Path("statgpu/solvers/_utils.py").read_text(encoding="utf-8")
    native_block = source.split("def _native_sample_weight", 1)[1].split(
        "def _validated_sample_weight", 1
    )[0]
    validated_block = source.split("def _validated_sample_weight", 1)[1].split(
        "def _validate_uniform_sample_weight", 1
    )[0]
    assert "except (TypeError, ValueError, RuntimeError)" not in native_block
    assert "except (TypeError, ValueError, RuntimeError)" not in validated_block


# PR87_REVIEW_FIX_V44
def test_newton_line_search_does_not_mask_runtime_failures():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import newton_solver

    class RuntimeFailingTrialLoss:
        name = "runtime_failing_trial"
        _has_constant_hessian = False

        def __init__(self):
            self.value_calls = 0

        def preprocess(self, X, y):
            return np.asarray(X), np.asarray(y)

        def gradient(self, X, y, coef):
            return np.ones(X.shape[1], dtype=np.float64)

        def hessian(self, X, y, coef):
            return np.eye(X.shape[1], dtype=np.float64)

        def fused_value_and_gradient(self, X, y, coef):
            self.value_calls += 1
            if self.value_calls == 1:
                return np.array(1.0), np.ones(X.shape[1], dtype=np.float64)
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        newton_solver(
            RuntimeFailingTrialLoss(),
            get_penalty("l2", alpha=0.1),
            np.ones((4, 1), dtype=np.float64),
            np.ones(4, dtype=np.float64),
            max_iter=2,
        )


def test_trial_error_classifier_is_narrow():
    from statgpu.solvers._utils import _trial_error_is_numerical

    assert _trial_error_is_numerical(RuntimeError("invalid value in log"))
    assert _trial_error_is_numerical(ValueError("domain error"))
    assert not _trial_error_is_numerical(RuntimeError("CUDA out of memory"))
    assert not _trial_error_is_numerical(RuntimeError("device-side assert"))

# PR87_REVIEW_FIX_V45
def test_fista_family_warm_starts_follow_preprocessed_dtype():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import fista_bb_solver, fista_solver

    class Float32PreprocessLoss:
        name = "float32_preprocess"
        _is_quadratic = False
        _prefer_fista_over_bb = False
        _lipschitz_uses_y = True

        def preprocess(self, X, y):
            return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

        def lipschitz(self, X, coef, y=None, sample_weight=None):
            return 1.0

        def gradient(self, X, y, coef, sample_weight=None):
            return np.zeros_like(coef)

        def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
            return np.asarray(0.0, dtype=coef.dtype), np.zeros_like(coef)

        def value(self, X, y, coef, sample_weight=None):
            return np.asarray(0.0, dtype=coef.dtype)

    X = np.ones((4, 2), dtype=np.float64)
    y = np.ones(4, dtype=np.float64)
    init = np.array([0.2, -0.1], dtype=np.float64)
    penalty = get_penalty("l2", alpha=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fista_coef, _ = fista_solver(
            Float32PreprocessLoss(), penalty, X, y, init_coef=init, max_iter=0
        )
        bb_coef, _ = fista_bb_solver(
            Float32PreprocessLoss(), penalty, X, y, init_coef=init, max_iter=0
        )
    assert fista_coef.dtype == np.float32
    assert bb_coef.dtype == np.float32


def test_proximal_newton_normalizes_weights_to_preprocessed_dtype():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    class RecordingLoss:
        name = "recording"
        has_hessian = True

        def __init__(self):
            self.seen_weight = None

        def preprocess(self, X, y):
            return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

        def fused_gradient_and_hessian(self, X, y, coef, sample_weight=None):
            self.seen_weight = sample_weight
            return np.zeros_like(coef), np.eye(coef.shape[0], dtype=coef.dtype)

    loss = RecordingLoss()
    coef, _ = proximal_newton_solver(
        loss,
        get_penalty("l2", alpha=0.0),
        np.ones((3, 1), dtype=np.float64),
        np.ones(3, dtype=np.float64),
        sample_weight=[1.0, 2.0, 3.0],
        max_iter=1,
    )
    assert coef.dtype == np.float32
    assert isinstance(loss.seen_weight, np.ndarray)
    assert loss.seen_weight.dtype == np.float32


def test_torch_cuda_proximal_newton_numpy_weights_stay_on_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires physical CUDA")
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device="cuda")
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    coef, _ = proximal_newton_solver(
        SquaredErrorLoss(),
        get_penalty("l2", alpha=0.05),
        X,
        y,
        sample_weight=np.array([1.0, 2.0, 3.0, 4.0]),
        max_iter=3,
    )
    assert coef.device.type == "cuda"
    assert coef.dtype == X.dtype
    assert bool(torch.all(torch.isfinite(coef)).item())


def test_cupy_proximal_newton_numpy_weights_stay_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires physical CUDA")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    X = cp.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=cp.float64)
    y = cp.asarray([1.0, 2.0, 3.0, 4.0], dtype=cp.float64)
    coef, _ = proximal_newton_solver(
        SquaredErrorLoss(),
        get_penalty("l2", alpha=0.05),
        X,
        y,
        sample_weight=np.array([1.0, 2.0, 3.0, 4.0]),
        max_iter=3,
    )
    assert isinstance(coef, cp.ndarray)
    assert coef.dtype == X.dtype
    assert bool(cp.all(cp.isfinite(coef)).item())

# PR87_REVIEW_FIX_V46
def test_numpy_backend_constructors_follow_floating_reference_dtype():
    from statgpu.backends._array_ops import _to_backend, _zeros

    ref32 = np.ones(3, dtype=np.float32)
    assert _zeros(3, "numpy", ref_tensor=ref32).dtype == np.float32
    assert _to_backend([1.0, 2.0], "numpy", ref_tensor=ref32).dtype == np.float32

    ref_int = np.ones(3, dtype=np.int64)
    assert _zeros(3, "numpy", ref_tensor=ref_int).dtype == np.float64
    assert _to_backend([1, 2], "numpy", ref_tensor=ref_int).dtype == np.float64

# PR87_REVIEW_FIX_V47
def test_shared_linear_solve_does_not_mask_runtime_failures(monkeypatch):
    from statgpu.backends._array_ops import _solve_linear_system

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    def forbidden(*args, **kwargs):
        raise AssertionError("lstsq must not mask infrastructure failures")

    monkeypatch.setattr(np.linalg, "solve", oom)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _solve_linear_system(np.eye(2), np.ones(2), backend="numpy")


def test_shared_linear_solve_retains_rank_failure_fallback(monkeypatch):
    from statgpu.backends._array_ops import _solve_linear_system

    expected = np.array([0.25, -0.5])

    def singular(*args, **kwargs):
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(np.linalg, "solve", singular)
    monkeypatch.setattr(np.linalg, "lstsq", lambda *args, **kwargs: (expected, None, None, None))
    result = _solve_linear_system(np.eye(2), np.ones(2), backend="numpy")
    np.testing.assert_allclose(result, expected)


def test_shared_linear_solve_runtime_classifier_is_narrow():
    from statgpu.backends._array_ops import _linear_solve_runtime_is_rank_failure

    assert _linear_solve_runtime_is_rank_failure(RuntimeError("singular matrix"))
    assert _linear_solve_runtime_is_rank_failure(RuntimeError("rank deficient"))
    assert not _linear_solve_runtime_is_rank_failure(RuntimeError("CUDA out of memory"))
    assert not _linear_solve_runtime_is_rank_failure(RuntimeError("device-side assert"))

# PR87_REVIEW_FIX_V48
def test_proximal_newton_backtracks_on_numeric_domain_value_error():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    class DomainTrialLoss:
        name = "domain_trial"
        has_hessian = True

        def __init__(self):
            self.value_calls = 0

        def preprocess(self, X, y):
            return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)

        def fused_gradient_and_hessian(self, X, y, coef, sample_weight=None):
            return np.ones_like(coef), np.eye(coef.shape[0], dtype=coef.dtype)

        def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
            self.value_calls += 1
            if self.value_calls == 1:
                return np.asarray(1.0), np.ones_like(coef)
            if self.value_calls == 2:
                raise ValueError("domain error at trial point")
            return np.asarray(0.25), np.ones_like(coef)

    coef, n_iter = proximal_newton_solver(
        DomainTrialLoss(),
        get_penalty("l2", alpha=0.0),
        np.ones((4, 1), dtype=np.float64),
        np.ones(4, dtype=np.float64),
        max_iter=1,
    )
    assert n_iter == 1
    assert np.all(np.isfinite(coef))
    assert not np.allclose(coef, 0.0)

# PR87_REVIEW_FIX_V49
def test_trial_error_classifier_does_not_mask_index_out_of_range():
    from statgpu.solvers._utils import _trial_error_is_numerical

    assert not _trial_error_is_numerical(
        RuntimeError("index out of range in self")
    )
    assert not _trial_error_is_numerical(
        ValueError("coefficient index out of range")
    )


def test_proximal_newton_propagates_index_out_of_range_trial_error():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    class IndexFailingTrialLoss:
        name = "index_failing_trial"
        has_hessian = True

        def __init__(self):
            self.value_calls = 0

        def preprocess(self, X, y):
            return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)

        def fused_gradient_and_hessian(self, X, y, coef, sample_weight=None):
            return np.ones_like(coef), np.eye(coef.shape[0], dtype=coef.dtype)

        def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
            self.value_calls += 1
            if self.value_calls == 1:
                return np.asarray(1.0), np.ones_like(coef)
            raise RuntimeError("index out of range in self")

    with pytest.raises(RuntimeError, match="index out of range"):
        proximal_newton_solver(
            IndexFailingTrialLoss(),
            get_penalty("l2", alpha=0.0),
            np.ones((4, 1), dtype=np.float64),
            np.ones(4, dtype=np.float64),
            max_iter=1,
        )



def test_backend_linalg_failure_classifier_is_narrow():
    from statgpu.backends._array_ops import _linalg_exception_is_rank_failure

    assert _linalg_exception_is_rank_failure(np.linalg.LinAlgError("singular matrix"))
    assert _linalg_exception_is_rank_failure(RuntimeError("matrix is singular"))
    assert _linalg_exception_is_rank_failure(RuntimeError("not positive definite"))
    assert not _linalg_exception_is_rank_failure(RuntimeError("CUDA out of memory"))
    assert not _linalg_exception_is_rank_failure(RuntimeError("index out of range"))
    assert not _linalg_exception_is_rank_failure(ValueError("incompatible dimensions"))


def test_glm_response_validation_preserves_backend_runtime_failure(monkeypatch):
    from types import SimpleNamespace
    from statgpu.glm_core import get_glm_loss
    import statgpu.backends._array_ops as array_ops

    fake_xp = SimpleNamespace(
        __name__="fake_gpu",
        asarray=lambda value: np.asarray(value),
        isfinite=np.isfinite,
        any=lambda value: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
    )
    monkeypatch.setattr(array_ops, "_xp", lambda value: fake_xp)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        get_glm_loss("squared_error").validate_response(np.array([1.0, 2.0]))


def test_penalized_exact_torch_preserves_nonrank_runtime_failure(monkeypatch):
    torch = pytest.importorskip("torch")
    from statgpu.linear_model.penalized._fit_mixin import _PenalizedFitMixin

    owner = object.__new__(_PenalizedFitMixin)
    owner._ridge_alpha_for_exact = lambda: 0.1
    monkeypatch.setattr(
        torch.linalg,
        "solve",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
    )
    monkeypatch.setattr(
        torch.linalg,
        "pinv",
        lambda *args, **kwargs: pytest.fail("pinv must not run after CUDA OOM"),
    )

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        owner._solve_exact_torch(torch.eye(2), torch.ones(2), normalization=3.0)


def test_kernel_ridge_retry_preserves_nonrank_runtime_failure(monkeypatch):
    from statgpu.nonparametric.kernel_smoothing._kernel_regression import (
        _solve_linear_system_with_ridge,
    )

    monkeypatch.setattr(
        np.linalg,
        "solve",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("CUDA out of memory")
        ),
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _solve_linear_system_with_ridge(np.eye(2), np.ones(2), np)



def test_penalized_cv_does_not_substitute_mse_for_non_gaussian_loss(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    owner = object.__new__(cv_mod.PenalizedGLM_CV)
    owner.loss = "poisson"

    class Loss:
        def value(self, *args, **kwargs):
            raise ValueError("generic poisson evaluation failed")

    class Model:
        coef_ = np.array([0.2])
        intercept_ = 0.1
        fit_intercept = True

        def predict(self, X):
            pytest.fail("non-Gaussian CV must not fall back to MSE predictions")

    monkeypatch.setattr(
        cv_mod,
        "_evaluate_loss_numpy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("registered poisson evaluation failed")
        ),
    )

    with pytest.raises(RuntimeError, match="Refusing to substitute mean squared error"):
        owner._evaluate_single(
            Model(),
            np.array([[1.0], [2.0]]),
            np.array([1.0, 3.0]),
            loss_fn=Loss(),
        )


def test_penalized_cv_squared_error_emergency_fallback_preserves_weights(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    owner = object.__new__(cv_mod.PenalizedGLM_CV)
    owner.loss = "squared_error"

    class Loss:
        def value(self, *args, **kwargs):
            raise ValueError("generic squared evaluation failed")

    class Model:
        coef_ = np.array([0.0])
        intercept_ = 0.0
        fit_intercept = True

        def predict(self, X):
            return np.array([0.0, 2.0])

    monkeypatch.setattr(
        cv_mod,
        "_evaluate_loss_numpy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("registered squared evaluation failed")
        ),
    )
    weights = np.array([1.0, 3.0])
    with pytest.warns(RuntimeWarning, match="weighted-MSE"):
        value = owner._evaluate_single(
            Model(),
            np.array([[1.0], [2.0]]),
            np.array([1.0, 4.0]),
            loss_fn=Loss(),
            sample_weight=weights,
        )
    expected = (1.0 * (1.0 - 0.0) ** 2 + 3.0 * (4.0 - 2.0) ** 2) / 4.0
    assert value == pytest.approx(expected)


def test_penalized_cv_loss_evaluation_preserves_infrastructure_failure(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    owner = object.__new__(cv_mod.PenalizedGLM_CV)
    owner.loss = "poisson"

    class Loss:
        def value(self, *args, **kwargs):
            pytest.fail("generic loss fallback must not run after CUDA OOM")

    class Model:
        coef_ = np.array([0.2])
        intercept_ = 0.1
        fit_intercept = True

    monkeypatch.setattr(
        cv_mod,
        "_evaluate_loss_numpy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("CUDA out of memory")
        ),
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        owner._evaluate_single(
            Model(),
            np.array([[1.0], [2.0]]),
            np.array([1.0, 3.0]),
            loss_fn=Loss(),
        )


def test_penalized_cv_infrastructure_classifier_is_narrow():
    from statgpu.linear_model.penalized._penalized_cv import (
        _cv_exception_is_infrastructure_failure,
    )

    assert _cv_exception_is_infrastructure_failure(RuntimeError("CUDA out of memory"))
    assert _cv_exception_is_infrastructure_failure(RuntimeError("index out of range"))
    assert _cv_exception_is_infrastructure_failure(MemoryError("allocation failed"))
    assert not _cv_exception_is_infrastructure_failure(
        np.linalg.LinAlgError("singular matrix")
    )
    assert not _cv_exception_is_infrastructure_failure(
        ValueError("numeric domain error")
    )



def test_penalized_cv_lipschitz_recovery_includes_cupy_linalg_only():
    from statgpu.linear_model.penalized._penalized_cv import (
        _cv_lipschitz_failure_is_recoverable,
    )

    CupyLinAlgError = type(
        "LinAlgError", (Exception,), {"__module__": "cupy.linalg._solve"}
    )
    assert _cv_lipschitz_failure_is_recoverable(
        CupyLinAlgError("singular matrix")
    )
    assert _cv_lipschitz_failure_is_recoverable(
        np.linalg.LinAlgError("singular matrix")
    )
    assert not _cv_lipschitz_failure_is_recoverable(
        RuntimeError("CUDA out of memory")
    )


def test_penalized_cv_alpha_grid_does_not_hide_memory_failure(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod
    import statgpu.linear_model.penalized._base as base_mod

    class FailingModel:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise MemoryError("host allocation failed")

    monkeypatch.setattr(base_mod, "PenalizedGeneralizedLinearModel", FailingModel)
    owner = object.__new__(cv_mod.PenalizedGLM_CV)
    owner.loss = "poisson"
    owner.penalty = "l1"
    owner.l1_ratio = 1.0
    owner._n_alphas = 5
    owner._loss_kwargs = None
    owner._penalty_kwargs = None

    with pytest.raises(MemoryError, match="host allocation failed"):
        owner._generate_alpha_grid(
            np.array([[1.0], [2.0], [3.0]]),
            np.array([1.0, 2.0, 3.0]),
        )


def _weighted_logistic_fixture():
    X = np.array(
        [
            [-1.2, 0.3],
            [-0.7, -0.4],
            [-0.1, 0.8],
            [0.4, -0.6],
            [0.9, 0.2],
            [1.5, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    weight = np.array([1, 3, 2, 4, 1, 2], dtype=np.float64)
    return X, y, weight


def test_weighted_logistic_cpu_matches_integer_row_replication():
    from statgpu.linear_model.wrappers._logistic import LogisticRegression

    X, y, weight = _weighted_logistic_fixture()
    weighted = LogisticRegression(
        C=2.5,
        max_iter=300,
        tol=1e-11,
        device="cpu",
        compute_inference=True,
    ).fit(X, y, sample_weight=weight)

    repeats = weight.astype(np.int64)
    replicated = LogisticRegression(
        C=2.5,
        max_iter=300,
        tol=1e-11,
        device="cpu",
        compute_inference=True,
    ).fit(np.repeat(X, repeats, axis=0), np.repeat(y, repeats, axis=0))

    np.testing.assert_allclose(weighted.intercept_, replicated.intercept_, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(weighted.coef_, replicated.coef_, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(weighted._bse, replicated._bse, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(weighted._loglik, replicated._loglik, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(weighted._loglik_null, replicated._loglik_null, rtol=1e-9, atol=1e-9)


def test_weighted_logistic_torch_matches_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    import statgpu.linear_model.wrappers._logistic as logistic_mod
    from statgpu.linear_model.wrappers._logistic import LogisticRegression

    X, y, weight = _weighted_logistic_fixture()
    cpu = LogisticRegression(
        C=2.5, max_iter=300, tol=1e-11, device="cpu", compute_inference=True
    ).fit(X, y, sample_weight=weight)

    # Exercise the Torch implementation on a CPU-only hosted runner
    # without weakening the public explicit-Torch CUDA contract.
    monkeypatch.setattr(logistic_mod, "_get_torch_device_str", lambda: "cpu")
    torch_model = LogisticRegression(
        C=2.5, max_iter=300, tol=1e-11, device="cpu", compute_inference=True
    )
    torch_model._y = y.astype(float)
    torch_model._sample_weight = weight.astype(float)
    torch_model._weight_sum = float(np.sum(weight))
    torch_model._fit_torch(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        torch.as_tensor(weight, dtype=torch.float64),
    )

    np.testing.assert_allclose(
        torch_model.intercept_, cpu.intercept_, rtol=1e-8, atol=1e-9
    )
    np.testing.assert_allclose(
        torch_model.coef_, cpu.coef_, rtol=1e-8, atol=1e-9
    )
    np.testing.assert_allclose(
        torch_model._bse, cpu._bse, rtol=1e-8, atol=1e-9
    )
    np.testing.assert_allclose(
        torch_model._loglik, cpu._loglik, rtol=1e-9, atol=1e-9
    )

@pytest.mark.parametrize(
    "weight, message",
    [
        (np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0]), "non-negative"),
        (np.zeros(6), "positive sum"),
        (np.array([1.0, np.inf, 1.0, 1.0, 1.0, 1.0]), "finite"),
    ],
)
def test_weighted_logistic_validates_weights_before_irls(weight, message):
    from statgpu.linear_model.wrappers._logistic import LogisticRegression

    X, y, _ = _weighted_logistic_fixture()
    with pytest.raises(ValueError, match=message):
        LogisticRegression(device="cpu").fit(X, y, sample_weight=weight)


def test_alpha_grid_fallback_classifier_is_narrow():
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    assert cv_mod._cv_alpha_grid_failure_is_recoverable(
        FloatingPointError("overflow")
    )
    assert cv_mod._cv_alpha_grid_failure_is_recoverable(
        np.linalg.LinAlgError("singular matrix")
    )
    assert not cv_mod._cv_alpha_grid_failure_is_recoverable(
        AttributeError("missing gradient")
    )
    assert not cv_mod._cv_alpha_grid_failure_is_recoverable(
        RuntimeError("unexpected implementation failure")
    )


def test_exact_cupy_ridge_does_not_mask_cuda_oom(monkeypatch):
    import sys
    import types
    from types import SimpleNamespace
    from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel

    fake_cp = types.ModuleType("cupy")
    fake_cp.eye = np.eye

    class FakeLinalg:
        @staticmethod
        def cholesky(_):
            raise RuntimeError("CUDA out of memory")

        @staticmethod
        def solve(*_):
            raise AssertionError("solve must not run after CUDA OOM")

        @staticmethod
        def pinv(*_):
            raise AssertionError("pinv must not run after CUDA OOM")

    fake_cp.linalg = FakeLinalg()
    fake_cupyx = types.ModuleType("cupyx")
    fake_cupyx_scipy = types.ModuleType("cupyx.scipy")
    fake_cupyx_linalg = types.ModuleType("cupyx.scipy.linalg")
    fake_cupyx_linalg.solve_triangular = lambda *args, **kwargs: np.asarray(args[1])

    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setitem(sys.modules, "cupyx", fake_cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", fake_cupyx_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.linalg", fake_cupyx_linalg)

    model = PenalizedGeneralizedLinearModel.__new__(PenalizedGeneralizedLinearModel)
    model._penalty = SimpleNamespace(alpha=0.1)
    model.alpha = 0.1
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        model._solve_exact_cupy(np.eye(2), np.ones(2), 4.0)


def test_dedicated_cv_device_resolution_preserves_explicit_backend(monkeypatch):
    import numpy as np
    import statgpu.linear_model.cv._device as cv_device
    from statgpu._config import Device

    calls = []

    class NumpyBackend:
        pass

    class CuPyBackend:
        pass

    class TorchBackend:
        pass

    classes = {
        "numpy": NumpyBackend,
        "cupy": CuPyBackend,
        "torch": TorchBackend,
        "auto": NumpyBackend,
    }

    def fake_get_backend(*, backend, device):
        calls.append((backend, device))
        return classes[backend]()

    monkeypatch.setattr(cv_device, "get_backend", fake_get_backend)
    X = np.zeros((4, 2))

    result = cv_device.resolve_cv_backend(Device.TORCH, X)
    assert result[0] == "torch"
    assert result[1] == "torch"
    assert result[3] is True
    assert calls[-1] == ("torch", "cuda")

    result = cv_device.resolve_cv_backend("cuda", X)
    assert result[1] == "cupy"
    assert calls[-1] == ("cupy", "cuda")


def test_dedicated_cv_device_resolution_rejects_cross_library_switch():
    import statgpu.linear_model.cv._device as cv_device

    FakeTorchCuda = type("FakeTorchCuda", (), {"__module__": "torch"})
    value = FakeTorchCuda()
    value.device = "cuda:0"

    with pytest.raises(ValueError, match="selects CuPy"):
        cv_device.resolve_cv_backend("cuda", value)


@pytest.mark.parametrize(
    "selector, kwargs",
    [
        ("logistic", {"Cs": [1.0], "cv_folds": 1}),
        ("ridge", {"alphas": [1.0], "cv_folds": 1}),
        (
            "elasticnet",
            {"alphas": [1.0], "l1_ratios": [0.5], "cv_folds": 1},
        ),
    ],
)
def test_dedicated_cv_validates_weights_before_degenerate_return(selector, kwargs):
    X = np.arange(12.0).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=float)

    if selector == "logistic":
        from statgpu.linear_model.cv._logistic_cv import _select_logistic_c_cv
        fn = _select_logistic_c_cv
    elif selector == "ridge":
        from statgpu.linear_model.cv._ridge_cv import _select_ridge_alpha_cv
        fn = _select_ridge_alpha_cv
    else:
        from statgpu.linear_model.cv._elasticnet_cv import _select_elasticnet_params_cv
        fn = _select_elasticnet_params_cv

    with pytest.raises(ValueError, match="non-negative"):
        fn(
            X,
            y,
            sample_weight=np.array([1, 1, 1, 1, 1, -1.0]),
            **kwargs,
        )
    with pytest.raises(ValueError, match="finite positive sum"):
        fn(X, y, sample_weight=np.zeros(6), **kwargs)


def test_logistic_default_grid_respects_integer_weight_replication():
    from statgpu.linear_model.cv._logistic_cv import _default_logistic_c_grid

    X = np.array([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [-1.0, 2.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    weight = np.array([1, 3, 2, 1])
    weighted = _default_logistic_c_grid(X, y, n_Cs=7, sample_weight=weight)
    replicated = _default_logistic_c_grid(
        np.repeat(X, weight, axis=0), np.repeat(y, weight), n_Cs=7
    )
    np.testing.assert_allclose(weighted, replicated, rtol=1e-12, atol=1e-12)


def test_elasticnet_default_grid_respects_integer_weight_replication():
    from statgpu.linear_model.cv._elasticnet_cv import _default_elasticnet_alpha_grid

    X = np.array([[0.0, 1.0], [1.0, -1.0], [2.0, 0.5], [-1.0, 2.0]])
    y = np.array([0.5, 2.0, -1.0, 1.5])
    weight = np.array([1, 3, 2, 1])
    weighted = _default_elasticnet_alpha_grid(
        X, y, l1_ratio=0.6, n_alphas=7, sample_weight=weight
    )
    replicated = _default_elasticnet_alpha_grid(
        np.repeat(X, weight, axis=0),
        np.repeat(y, weight),
        l1_ratio=0.6,
        n_alphas=7,
    )
    np.testing.assert_allclose(weighted, replicated, rtol=1e-12, atol=1e-12)


def test_cv_device_inspection_does_not_mask_runtime_failures():
    from statgpu.linear_model.cv._device import _array_gpu_backend

    class BrokenTorchArray:
        __module__ = "torch"

        @property
        def device(self):
            raise RuntimeError("CUDA device query failed")

    with pytest.raises(RuntimeError, match="device query failed"):
        _array_gpu_backend(BrokenTorchArray())



@pytest.mark.parametrize(
    "module_name,class_name,selector_name,y",
    [
        (
            "statgpu.linear_model.cv._ridge_cv",
            "RidgeCV",
            "_select_ridge_alpha_cv",
            np.arange(6, dtype=np.float64),
        ),
        (
            "statgpu.linear_model.cv._elasticnet_cv",
            "ElasticNetCV",
            "_select_elasticnet_params_cv",
            np.arange(6, dtype=np.float64),
        ),
        (
            "statgpu.linear_model.cv._logistic_cv",
            "LogisticRegressionCV",
            "_select_logistic_c_cv",
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        ),
    ],
)
def test_public_dedicated_cv_preserves_auto_device_request(
    monkeypatch, module_name, class_name, selector_name, y
):
    import importlib
    from statgpu.linear_model.cv._device import normalize_cv_device

    module = importlib.import_module(module_name)
    observed = []

    def probe(*args, **kwargs):
        observed.append(kwargs["device"])
        raise RuntimeError("device request probe")

    monkeypatch.setattr(module, selector_name, probe)
    estimator = getattr(module, class_name)(device="auto", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    with pytest.raises(RuntimeError, match="device request probe"):
        estimator.fit(X, y)

    assert len(observed) == 1
    assert normalize_cv_device(observed[0]) == "auto"


def test_logistic_cv_binary_validation_preserves_torch_response():
    torch = pytest.importorskip("torch")
    from statgpu.linear_model.cv._logistic_cv import _validate_binary_cv_response

    y = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    assert _validate_binary_cv_response(y) is y
    with pytest.raises(ValueError, match="binary y"):
        _validate_binary_cv_response(
            torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        )


def test_auto_cv_router_prefers_gpu_resident_input_backend(monkeypatch):
    import statgpu.linear_model.cv._device as device_mod

    class FakeTorchCudaArray:
        __module__ = "torch"
        device = "cuda:0"

    calls = []

    class FakeBackend:
        pass

    def backend_probe(*, backend, device):
        calls.append((backend, device))
        return FakeBackend()

    monkeypatch.setattr(device_mod, "get_backend", backend_probe)
    resolved = device_mod.resolve_cv_backend("auto", FakeTorchCudaArray())
    assert resolved[0] == "auto"
    assert resolved[1] == "torch"
    assert resolved[3] is True
    assert calls == [("torch", "cuda")]



def test_cv_refit_device_pins_auto_to_selected_backend():
    from statgpu._config import Device
    from statgpu.linear_model.cv._device import cv_refit_device

    assert cv_refit_device("auto", "numpy") == Device.CPU
    assert cv_refit_device("auto", "cupy") == Device.CUDA
    assert cv_refit_device("auto", "torch") == Device.TORCH
    assert cv_refit_device("cpu", "torch") == Device.CPU
    assert cv_refit_device("cuda", "torch") == Device.CUDA
    with pytest.raises(ValueError, match="Unknown CV backend"):
        cv_refit_device("auto", "mystery")


@pytest.mark.parametrize(
    "module_name,class_name,selector_name,model_name,y,details",
    [
        (
            "statgpu.linear_model.cv._ridge_cv",
            "RidgeCV",
            "_select_ridge_alpha_cv",
            "Ridge",
            np.arange(6, dtype=np.float64),
            {
                "alpha": 0.5,
                "alphas": np.array([0.5]),
                "mse_path": np.array([[1.0]]),
                "mean_mse": np.array([1.0]),
            },
        ),
        (
            "statgpu.linear_model.cv._logistic_cv",
            "LogisticRegressionCV",
            "_select_logistic_c_cv",
            "LogisticRegression",
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            {
                "C": 1.0,
                "Cs": np.array([1.0]),
                "loss_path": np.array([[0.5]]),
                "mean_loss": np.array([0.5]),
            },
        ),
    ],
)
def test_public_cv_auto_refit_uses_cv_selected_backend(
    monkeypatch,
    module_name,
    class_name,
    selector_name,
    model_name,
    y,
    details,
):
    import importlib
    from statgpu._config import Device

    module = importlib.import_module(module_name)
    backend = object()
    monkeypatch.setattr(
        module,
        "resolve_cv_backend",
        lambda device, X: ("auto", "torch", backend, True, False, True),
    )
    monkeypatch.setattr(module, selector_name, lambda *args, **kwargs: details)
    observed = []

    class FakeModel:
        def __init__(self, *args, device=None, **kwargs):
            observed.append(device)
            self.coef_ = np.zeros(2)
            self.intercept_ = 0.0
            self.n_iter_ = 1

        def fit(self, X, y, sample_weight=None):
            return self

    monkeypatch.setattr(module, model_name, FakeModel)
    estimator = getattr(module, class_name)(device="auto", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    estimator.fit(X, y)

    assert observed == [Device.TORCH]
    assert estimator.cv_selected_device_ == Device.TORCH


def test_elasticnet_cv_auto_refit_uses_cv_selected_backend(monkeypatch):
    import statgpu.linear_model.cv._elasticnet_cv as module
    from statgpu._config import Device

    backend = object()
    monkeypatch.setattr(
        module,
        "resolve_cv_backend",
        lambda device, X: ("auto", "cupy", backend, True, True, False),
    )
    details = {
        "mse_path": np.array([[[1.0]]]),
        "mean_mse": np.array([[1.0]]),
        "std_mse": np.array([[0.0]]),
        "alphas": {0: np.array([0.5])},
        "l1_ratios": np.array([0.5]),
        "best_mse": 1.0,
    }
    monkeypatch.setattr(
        module,
        "_select_elasticnet_params_cv",
        lambda *args, **kwargs: (0.5, 0.5, details),
    )
    observed = []

    class FakeElasticNet:
        def __init__(self, *args, device=None, **kwargs):
            observed.append(device)
            self.coef_ = np.zeros(2)
            self.intercept_ = 0.0
            self.n_iter_ = 1

        def fit(self, X, y, sample_weight=None):
            return self

    monkeypatch.setattr(module, "ElasticNet", FakeElasticNet)
    estimator = module.ElasticNetCV(device="auto", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    estimator.fit(X, np.arange(6, dtype=np.float64))

    assert observed == [Device.CUDA]
    assert estimator.cv_selected_device_ == Device.CUDA



@pytest.mark.parametrize(
    "module_name,class_name,selector_name,y,selected_attrs",
    [
        (
            "statgpu.linear_model.cv._ridge_cv",
            "RidgeCV",
            "_select_ridge_alpha_cv",
            np.arange(6, dtype=np.float64),
            ("alpha_", "alphas_", "mean_mse_"),
        ),
        (
            "statgpu.linear_model.cv._elasticnet_cv",
            "ElasticNetCV",
            "_select_elasticnet_params_cv",
            np.arange(6, dtype=np.float64),
            ("alpha_", "l1_ratio_"),
        ),
        (
            "statgpu.linear_model.cv._logistic_cv",
            "LogisticRegressionCV",
            "_select_logistic_c_cv",
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            ("C_", "Cs_", "mean_loss_"),
        ),
    ],
)
def test_dedicated_cv_failed_refit_clears_previous_state(
    monkeypatch, module_name, class_name, selector_name, y, selected_attrs
):
    import importlib
    from statgpu._config import Device

    module = importlib.import_module(module_name)
    estimator = getattr(module, class_name)(device="cpu", cv=2)
    estimator._fitted = True
    estimator.estimator_ = object()
    estimator.coef_ = np.array([9.0, 8.0])
    estimator.intercept_ = 7.0
    estimator.best_score_ = 6.0
    estimator.cv_results_ = {"stale": True}
    estimator.cv_selected_device_ = Device.TORCH
    for name in selected_attrs:
        setattr(estimator, name, np.array([5.0]) if name.endswith("s_") else 5.0)

    monkeypatch.setattr(
        module,
        selector_name,
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("selection failed")
        ),
    )
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    with pytest.raises(RuntimeError, match="selection failed"):
        estimator.fit(X, y)

    assert estimator._fitted is False
    assert estimator.estimator_ is None
    assert estimator.coef_ is None
    assert estimator.intercept_ is None
    assert estimator.best_score_ is None
    assert estimator.cv_results_ is None
    assert estimator.cv_selected_device_ is None
    for name in selected_attrs:
        assert getattr(estimator, name) is None


def test_ridge_cv_final_refit_failure_does_not_publish_partial_state(monkeypatch):
    import statgpu.linear_model.cv._ridge_cv as module

    details = {
        "alpha": 0.5,
        "alphas": np.array([0.5]),
        "mse_path": np.array([[1.0]]),
        "mean_mse": np.array([1.0]),
    }
    monkeypatch.setattr(module, "_select_ridge_alpha_cv", lambda *a, **k: details)

    class FailingRidge:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise RuntimeError("final refit failed")

    monkeypatch.setattr(module, "Ridge", FailingRidge)
    estimator = module.RidgeCV(device="cpu", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    with pytest.raises(RuntimeError, match="final refit failed"):
        estimator.fit(X, np.arange(6, dtype=np.float64))
    assert estimator._fitted is False
    assert estimator.alpha_ is None
    assert estimator.estimator_ is None
    assert estimator.cv_selected_device_ is None


def test_elasticnet_cv_final_refit_failure_does_not_publish_partial_state(monkeypatch):
    import statgpu.linear_model.cv._elasticnet_cv as module

    details = {
        "mse_path": np.array([[[1.0]]]),
        "mean_mse": np.array([[1.0]]),
        "std_mse": np.array([[0.0]]),
        "alphas": {0: np.array([0.5])},
        "l1_ratios": np.array([0.5]),
        "best_mse": 1.0,
    }
    monkeypatch.setattr(
        module,
        "_select_elasticnet_params_cv",
        lambda *a, **k: (0.5, 0.5, details),
    )

    class FailingElasticNet:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise RuntimeError("final refit failed")

    monkeypatch.setattr(module, "ElasticNet", FailingElasticNet)
    estimator = module.ElasticNetCV(device="cpu", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    with pytest.raises(RuntimeError, match="final refit failed"):
        estimator.fit(X, np.arange(6, dtype=np.float64))
    assert estimator._fitted is False
    assert estimator.alpha_ is None
    assert estimator.l1_ratio_ is None
    assert estimator.estimator_ is None
    assert estimator.cv_selected_device_ is None


def test_logistic_cv_final_refit_failure_does_not_publish_partial_state(monkeypatch):
    import statgpu.linear_model.cv._logistic_cv as module

    details = {
        "C": 1.0,
        "Cs": np.array([1.0]),
        "loss_path": np.array([[0.5]]),
        "mean_loss": np.array([0.5]),
    }
    monkeypatch.setattr(module, "_select_logistic_c_cv", lambda *a, **k: details)

    class FailingLogistic:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise RuntimeError("final refit failed")

    monkeypatch.setattr(module, "LogisticRegression", FailingLogistic)
    estimator = module.LogisticRegressionCV(device="cpu", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="final refit failed"):
        estimator.fit(X, y)
    assert estimator._fitted is False
    assert estimator.C_ is None
    assert estimator.estimator_ is None
    assert estimator.cv_selected_device_ is None



@pytest.mark.parametrize(
    "module_name,class_name,y,selected_name",
    [
        (
            "statgpu.linear_model.cv._ridge_cv",
            "RidgeCV",
            np.arange(6, dtype=np.float64),
            "alpha_",
        ),
        (
            "statgpu.linear_model.cv._elasticnet_cv",
            "ElasticNetCV",
            np.arange(6, dtype=np.float64),
            "alpha_",
        ),
        (
            "statgpu.linear_model.cv._logistic_cv",
            "LogisticRegressionCV",
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            "C_",
        ),
    ],
)
def test_cv_finite_guard_resets_stale_state_before_rejecting_input(
    module_name, class_name, y, selected_name
):
    import importlib
    from statgpu._config import Device

    module = importlib.import_module(module_name)
    estimator = getattr(module, class_name)(device="cpu", cv=2)
    estimator._fitted = True
    estimator.estimator_ = object()
    estimator.coef_ = np.array([3.0, 4.0])
    estimator.intercept_ = 2.0
    estimator.best_score_ = 1.0
    estimator.cv_results_ = {"stale": True}
    estimator.cv_selected_device_ = Device.TORCH
    setattr(estimator, selected_name, 0.5)

    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimator.fit(X, y)

    assert estimator._fitted is False
    assert estimator.estimator_ is None
    assert estimator.coef_ is None
    assert estimator.intercept_ is None
    assert estimator.best_score_ is None
    assert estimator.cv_results_ is None
    assert estimator.cv_selected_device_ is None
    assert getattr(estimator, selected_name) is None


def test_finite_guard_does_not_reset_cv_state_on_prediction_failure():
    from statgpu.linear_model.cv._ridge_cv import RidgeCV

    estimator = RidgeCV(device="cpu", cv=2)
    estimator._fitted = True

    class FittedModel:
        def predict(self, X):
            return np.zeros(int(X.shape[0]), dtype=np.float64)

    estimator.estimator_ = FittedModel()
    estimator.alpha_ = 0.5
    estimator.coef_ = np.array([1.0])
    estimator.intercept_ = 0.0

    with pytest.raises(ValueError, match="finite"):
        estimator.predict(np.array([[np.nan]], dtype=np.float64))

    assert estimator._fitted is True
    assert estimator.alpha_ == pytest.approx(0.5)
    assert estimator.estimator_ is not None



def _elasticnet_inference_fixture(seed=20260805):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(96, 3))
    beta = np.array([1.4, -0.9, 0.65])
    y = 0.35 + X @ beta + rng.normal(scale=0.35, size=X.shape[0])
    return X.astype(np.float64), y.astype(np.float64)


def _assert_elasticnet_inference_contract(model, n_features=3):
    n_params = n_features + 1
    assert model._compute_inference_enabled is True
    assert model._inference_result is not None
    assert model._inference_result.method == "debiased"
    assert np.asarray(model._params).shape == (n_params,)
    assert np.asarray(model._bse).shape == (n_params,)
    assert np.asarray(model._pvalues).shape == (n_params,)
    assert np.asarray(model._conf_int).shape == (n_params, 2)
    assert np.all(np.isfinite(np.asarray(model._params)))
    assert np.all(np.isfinite(np.asarray(model._bse)))
    assert np.all(np.isfinite(np.asarray(model._pvalues)))
    assert np.all(np.isfinite(np.asarray(model._conf_int)))
    model.summary()


def test_elasticnet_wrapper_cpu_debiased_inference_contract():
    from statgpu.linear_model import ElasticNet

    X, y = _elasticnet_inference_fixture()
    model = ElasticNet(
        alpha=0.02,
        l1_ratio=0.5,
        max_iter=2000,
        tol=1e-7,
        device="cpu",
        compute_inference=True,
        inference_method="debiased",
    ).fit(X, y)

    _assert_elasticnet_inference_contract(model)
    params = model.get_params(deep=False)
    assert params["compute_inference"] is True
    assert params["inference_method"] == "debiased"
    assert params["cov_type"] == "nonrobust"
    assert params["hac_maxlags"] is None


def test_elasticnet_cv_compute_inference_runs_on_final_refit():
    from statgpu.linear_model import ElasticNetCV

    X, y = _elasticnet_inference_fixture(seed=20260806)
    model = ElasticNetCV(
        l1_ratio=[0.5],
        alphas=[0.02],
        cv=2,
        max_iter=1500,
        tol=1e-7,
        device="cpu",
        compute_inference=True,
        random_state=17,
    ).fit(X, y)

    assert model.estimator_ is not None
    _assert_elasticnet_inference_contract(model.estimator_)
    model.summary()


def test_elasticnet_cv_passes_inference_flag_to_final_model(monkeypatch):
    import statgpu.linear_model.cv._elasticnet_cv as module

    details = {
        "mse_path": np.array([[[1.0]]]),
        "mean_mse": np.array([[1.0]]),
        "std_mse": np.array([[0.0]]),
        "alphas": {0: np.array([0.02])},
        "l1_ratios": np.array([0.5]),
        "best_mse": 1.0,
    }
    monkeypatch.setattr(
        module,
        "_select_elasticnet_params_cv",
        lambda *args, **kwargs: (0.02, 0.5, details),
    )
    observed = []

    class FakeElasticNet:
        def __init__(self, *args, **kwargs):
            observed.append(kwargs)
            self.coef_ = np.zeros(3)
            self.intercept_ = 0.0
            self.n_iter_ = 1

        def fit(self, X, y, sample_weight=None):
            return self

        def predict(self, X):
            return np.zeros(int(X.shape[0]))

    monkeypatch.setattr(module, "ElasticNet", FakeElasticNet)
    X, y = _elasticnet_inference_fixture(seed=20260807)
    model = module.ElasticNetCV(
        l1_ratio=[0.5],
        alphas=[0.02],
        cv=2,
        device="cpu",
        compute_inference=True,
        n_jobs=3,
    ).fit(X, y)

    assert len(observed) == 1
    assert observed[0]["compute_inference"] is True
    assert observed[0]["inference_method"] == "debiased"
    assert observed[0]["n_jobs"] == 3
    assert model.estimator_ is not None


def test_torch_cuda_elasticnet_inference_contract():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model import ElasticNet, ElasticNetCV

    X_np, y_np = _elasticnet_inference_fixture(seed=20260808)
    X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")
    y = torch.as_tensor(y_np, dtype=torch.float64, device="cuda")

    direct = ElasticNet(
        alpha=0.02,
        l1_ratio=0.5,
        max_iter=1200,
        tol=1e-6,
        device="torch",
        compute_inference=True,
    ).fit(X, y)
    _assert_elasticnet_inference_contract(direct)

    cv_model = ElasticNetCV(
        l1_ratio=[0.5],
        alphas=[0.02],
        cv=2,
        max_iter=800,
        tol=1e-6,
        device="torch",
        compute_inference=True,
        random_state=19,
    ).fit(X, y)
    _assert_elasticnet_inference_contract(cv_model.estimator_)


def _require_physical_cupy_v61():
    cp = pytest.importorskip("cupy")
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        pytest.skip(f"requires a physical CuPy CUDA backend: {exc}")
    if count < 1:
        pytest.skip("requires a physical CuPy CUDA backend")
    return cp


def test_cupy_elasticnet_inference_contract():
    cp = _require_physical_cupy_v61()
    from statgpu.linear_model import ElasticNet, ElasticNetCV

    X_np, y_np = _elasticnet_inference_fixture(seed=20260809)
    X = cp.asarray(X_np)
    y = cp.asarray(y_np)

    direct = ElasticNet(
        alpha=0.02,
        l1_ratio=0.5,
        max_iter=1200,
        tol=1e-6,
        device="cuda",
        compute_inference=True,
    ).fit(X, y)
    _assert_elasticnet_inference_contract(direct)

    cv_model = ElasticNetCV(
        l1_ratio=[0.5],
        alphas=[0.02],
        cv=2,
        max_iter=800,
        tol=1e-6,
        device="cuda",
        compute_inference=True,
        random_state=23,
    ).fit(X, y)
    _assert_elasticnet_inference_contract(cv_model.estimator_)



def test_elasticnet_zero_l1_ratio_matches_same_alpha_ridge():
    from statgpu.linear_model import ElasticNet, Ridge

    rng = np.random.default_rng(20260810)
    X = rng.normal(size=(80, 4))
    y = 0.45 + X @ np.array([1.2, -0.8, 0.0, 0.55])
    y = y + rng.normal(scale=0.2, size=X.shape[0])
    alpha = 0.17

    elastic = ElasticNet(
        alpha=alpha,
        l1_ratio=0.0,
        fit_intercept=True,
        solver="fista",
        max_iter=5000,
        tol=1e-10,
        device="cpu",
        compute_inference=False,
    ).fit(X, y)
    ridge = Ridge(
        alpha=alpha,
        fit_intercept=True,
        solver="exact",
        device="cpu",
        compute_inference=False,
    ).fit(X, y)

    np.testing.assert_allclose(
        elastic.coef_, ridge.coef_, rtol=2e-7, atol=2e-8
    )
    np.testing.assert_allclose(
        elastic.intercept_, ridge.intercept_, rtol=2e-7, atol=2e-8
    )
