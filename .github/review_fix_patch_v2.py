from __future__ import annotations

from pathlib import Path
import re


# ---------------------------------------------------------------------------
# 1. Central Torch compile policy: explicit diagnostics and narrow fallback.
# ---------------------------------------------------------------------------
Path("statgpu/backends/_torch_compile.py").write_text(
'''"""Safe, centralized policy for internal :func:`torch.compile` use.

statgpu iterative solvers reuse tensors across calls. PyTorch's
``reduce-overhead`` mode enables CUDA Graphs and can therefore expose
overwritten-output lifecycle errors on PyTorch 2.1 and newer. Internal
iterative call sites use ``default`` mode unless a user explicitly opts
into another mode through ``STATGPU_TORCH_COMPILE_MODE``.
"""

from __future__ import annotations

import functools
import os
import warnings
from typing import Callable, Optional

_ENV_NAME = "STATGPU_TORCH_COMPILE_MODE"
_ALLOWED_MODES = frozenset({"auto", "default", "reduce-overhead", "disable"})
_COMPILE_DIAGNOSTICS = []


def resolve_torch_compile_mode(
    *,
    workload: str = "general",
    requested_mode: Optional[str] = None,
) -> Optional[str]:
    """Resolve the mode for a statgpu-owned compiled callable."""
    configured = os.environ.get(_ENV_NAME, "auto").strip().lower()
    if configured not in _ALLOWED_MODES:
        allowed = ", ".join(sorted(_ALLOWED_MODES))
        raise ValueError(
            f"{_ENV_NAME} must be one of {allowed}; got {configured!r}"
        )
    if configured == "disable":
        return None
    if configured != "auto":
        return configured
    if workload.strip().lower() == "iterative":
        return "default"
    if requested_mode in (None, "reduce-overhead"):
        return "default"
    return requested_mode


def torch_compile_available() -> bool:
    """Return whether the local Torch installation can compile safely."""
    try:
        import torch
    except Exception:
        return False
    if not callable(getattr(torch, "compile", None)):
        return False
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()[0] >= 7
    except Exception:
        return False
    return True


def _record_compile_event(*, fn, status, mode, workload, error=None) -> None:
    _COMPILE_DIAGNOSTICS.append(
        {
            "function": getattr(
                fn, "__qualname__", getattr(fn, "__name__", repr(fn))
            ),
            "status": status,
            "mode": mode,
            "workload": workload,
            "error": error,
        }
    )


def get_torch_compile_diagnostics(*, clear: bool = False):
    """Return snapshots of internal Torch compile decisions.

    The returned dictionaries expose whether a callable is compiled, disabled,
    unavailable, or using an explicit construction/runtime eager fallback.
    """
    snapshot = tuple(dict(event) for event in _COMPILE_DIAGNOSTICS)
    if clear:
        _COMPILE_DIAGNOSTICS.clear()
    return snapshot


def _is_cudagraph_lifecycle_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    has_cudagraph = "cudagraph" in message
    has_overwrite = "overwrit" in message
    has_tensor_output = "tensor output" in message or "accessing tensor" in message
    return has_cudagraph and has_overwrite and has_tensor_output


def compile_torch(
    fn: Callable,
    *,
    workload: str = "general",
    mode: Optional[str] = None,
    **compile_kwargs,
) -> Callable:
    """Compile ``fn`` under the statgpu policy with observable eager fallback.

    A construction failure emits a warning and returns an eager wrapper carrying
    diagnostic attributes. At invocation time, only the known CUDA Graph tensor
    output lifecycle failure disables compilation; unrelated runtime errors are
    re-raised.
    """
    resolved_mode = resolve_torch_compile_mode(
        workload=workload,
        requested_mode=mode,
    )

    def eager_wrapper(status, error=None):
        @functools.wraps(fn)
        def eager(*args, **kwargs):
            return fn(*args, **kwargs)

        eager.__statgpu_compile_mode__ = resolved_mode
        eager.__statgpu_compile_workload__ = workload
        eager.__statgpu_compile_status__ = status
        eager.__statgpu_compile_error__ = error
        _record_compile_event(
            fn=fn,
            status=status,
            mode=resolved_mode,
            workload=workload,
            error=error,
        )
        return eager

    if resolved_mode is None:
        return eager_wrapper("disabled")
    if not torch_compile_available():
        return eager_wrapper("unavailable")

    try:
        import torch
        compiled = torch.compile(fn, mode=resolved_mode, **compile_kwargs)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        warnings.warn(
            "torch.compile construction failed; falling back to eager execution "
            f"for this statgpu kernel: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return eager_wrapper("construction-fallback", error)

    state = {"disabled": False}

    @functools.wraps(fn)
    def guarded(*args, **kwargs):
        if state["disabled"]:
            return fn(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except RuntimeError as exc:
            if not _is_cudagraph_lifecycle_error(exc):
                raise
            state["disabled"] = True
            error = f"{type(exc).__name__}: {exc}"
            guarded.__statgpu_compile_status__ = "runtime-fallback"
            guarded.__statgpu_compile_error__ = error
            _record_compile_event(
                fn=fn,
                status="runtime-fallback",
                mode=resolved_mode,
                workload=workload,
                error=error,
            )
            warnings.warn(
                "torch.compile CUDA Graph lifecycle failure; "
                "falling back to eager execution for this statgpu kernel",
                RuntimeWarning,
                stacklevel=2,
            )
            return fn(*args, **kwargs)

    guarded.__statgpu_compile_mode__ = resolved_mode
    guarded.__statgpu_compile_workload__ = workload
    guarded.__statgpu_compile_status__ = "compiled"
    guarded.__statgpu_compile_error__ = None
    _record_compile_event(
        fn=fn,
        status="compiled",
        mode=resolved_mode,
        workload=workload,
    )
    return guarded
''', encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. Backend-native finite validation, including sparse/pandas/object arrays.
# ---------------------------------------------------------------------------
Path("statgpu/backends/_validation.py").write_text(
'''"""Backend-native validation helpers for public numerical inputs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _raise_nonfinite(name: str) -> None:
    raise ValueError(
        f"{name} must contain only finite values; found NaN or infinite values"
    )


def check_finite(value: Any, *, name: str = "array") -> Any:
    """Reject NaN/Inf without transferring complete GPU arrays to CPU.

    NumPy, CuPy, Torch, scipy/cupyx sparse values, pandas numerical data,
    scalars, and nested/object sequences are checked. GPU arrays perform the
    reduction on device and synchronize only the final scalar boolean. The
    original object is returned unchanged.
    """
    if value is None:
        return value

    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        real = float(np.real(value))
        imag = float(np.imag(value))
        if not math.isfinite(real) or not math.isfinite(imag):
            _raise_nonfinite(name)
        return value
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return value

    module = type(value).__module__

    if module.startswith("scipy.sparse") or module.startswith("cupyx.scipy.sparse"):
        check_finite(value.data, name=name)
        return value

    if module.startswith("torch"):
        import torch

        tensor = value
        if getattr(tensor, "layout", torch.strided) != torch.strided:
            tensor = tensor.values()
        if not bool(torch.isfinite(tensor).all().item()):
            _raise_nonfinite(name)
        return value

    if module.startswith("cupy"):
        import cupy as cp

        if not bool(cp.isfinite(value).all().item()):
            _raise_nonfinite(name)
        return value

    if module.startswith("pandas"):
        try:
            array = value.to_numpy()
        except Exception:
            return value
        if array.dtype.kind in "biufc":
            if not np.isfinite(array).all():
                _raise_nonfinite(name)
            return value
        if array.dtype.kind == "O":
            for index, item in np.ndenumerate(array):
                check_finite(item, name=f"{name}{index}")
        return value

    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                check_finite(item, name=f"{name}[{index}]")
        return value

    if array.dtype.kind in "biufc":
        if not np.isfinite(array).all():
            _raise_nonfinite(name)
        return value

    if array.dtype.kind == "O":
        for index, item in np.ndenumerate(array):
            if item is value:
                continue
            check_finite(item, name=f"{name}{index}")
    return value
''', encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. BaseEstimator: expanded public matrix, formula ownership, tags, set_params.
# ---------------------------------------------------------------------------
p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
text = text.replace(
'''        "predict_cumulative_hazard",
    })''',
'''        "predict_cumulative_hazard",
        "inverse_transform",
        "score_samples",
        "bic",
        "aic",
        "predict_with_threshold",
        "confusion_matrix",
        "classification_table",
        "roc_curve",
        "roc_auc_score",
        "precision_recall_curve",
        "average_precision_score",
    })''', 1)
text = text.replace(
'''        "init_coef",
    })''',
'''        "init_coef",
        "initial_coef",
        "time_index",
        "entity_ids",
        "time_ids",
    })''', 1)
old_guard = '''                loss_value = getattr(self, "loss", "")
                loss_name = str(getattr(loss_value, "name", loss_value)).lower()
                for name, value in bound.arguments.items():
                    if name == "y" and loss_name in {"cox", "coxph", "cox_ph"}:
                        # Cox response matrices have stronger joint time/event
                        # contracts. Preserve model-specific errors and validate
                        # them before device selection inside the Cox estimator.
                        continue
                    if name in self._FINITE_PARAMETER_NAMES and value is not None:
                        check_finite(value, name=name)
                return original(self, *args, **kwargs)
'''
new_guard = '''                loss_value = getattr(self, "loss", "")
                loss_name = str(getattr(loss_value, "name", loss_value)).lower()
                formula_active = (
                    bound.arguments.get("formula") is not None
                    or bound.arguments.get("data") is not None
                    or getattr(self, "_design_info", None) is not None
                )
                for name, value in bound.arguments.items():
                    if name == "y" and loss_name in {"cox", "coxph", "cox_ph"}:
                        # Cox response matrices have stronger joint time/event
                        # contracts. Preserve model-specific errors and validate
                        # them before device selection inside the Cox estimator.
                        continue
                    if formula_active and type(value).__module__.startswith("pandas"):
                        # Formula/model-matrix code owns row dropping, categorical
                        # encoding, and aligned side-array error semantics.
                        continue
                    if name in self._FINITE_PARAMETER_NAMES and value is not None:
                        check_finite(value, name=name)
                return original(self, *args, **kwargs)
'''
if text.count(old_guard) != 1:
    raise RuntimeError("finite guard anchor mismatch")
text = text.replace(old_guard, new_guard, 1)

start = text.index("    def __sklearn_tags__(self):")
end = text.index("    def __sklearn_clone__(self):", start)
new_tags = '''    def _statgpu_estimator_type(self):
        """Infer sklearn estimator type without requiring sklearn at runtime."""
        explicit = getattr(self, "_estimator_type", None)
        if explicit in {"classifier", "regressor"}:
            return explicit
        name = type(self).__name__.lower()
        if "classifier" in name or "logistic" in name:
            return "classifier"
        if any(
            token in name
            for token in (
                "regression",
                "regressor",
                "ridge",
                "lasso",
                "elasticnet",
                "quantile",
                "cox",
                "panel",
                "ols",
                "effects",
                "fama",
                "kernelridge",
                "gam",
            )
        ):
            return "regressor"
        return None

    def __sklearn_tags__(self):
        """Return public estimator tags when sklearn >= 1.6 is installed."""
        estimator_type = self._statgpu_estimator_type()
        try:
            from sklearn.utils import (
                ClassifierTags,
                RegressorTags,
                Tags,
                TargetTags,
            )
        except ImportError:
            return self._more_tags()

        return Tags(
            estimator_type=estimator_type,
            target_tags=TargetTags(required=estimator_type is not None),
            classifier_tags=(
                ClassifierTags() if estimator_type == "classifier" else None
            ),
            regressor_tags=(
                RegressorTags() if estimator_type == "regressor" else None
            ),
            requires_fit=True,
        )

    def _more_tags(self):
        """Return the legacy sklearn tag dictionary."""
        estimator_type = self._statgpu_estimator_type()
        return {"requires_y": estimator_type in {"classifier", "regressor"}}

    def __sklearn_is_fitted__(self):
        """Expose statgpu fitted state to sklearn meta-estimators."""
        return bool(getattr(self, "_fitted", False))

'''
text = text[:start] + new_tags + text[end:]

start = text.index("    def set_params(self, **params):")
# set_params is the final method in the file at this head.
new_set_params = '''    def set_params(self, **params):
        """Set parameters and rebuild normalized runtime state transactionally."""
        if not params:
            return self

        import copy
        from collections.abc import Iterator

        valid_deep = self.get_params(deep=True)
        direct = self.get_params(deep=False)
        nested = {}
        for key, value in params.items():
            root, delimiter, sub_key = key.partition("__")
            if key not in valid_deep and root not in direct:
                valid_names = sorted(
                    name for name in valid_deep if "__" not in name
                )
                raise ValueError(
                    f"Invalid parameter {root!r} for estimator "
                    f"{type(self).__name__}. Valid parameters are: {valid_names}."
                )
            if delimiter:
                nested.setdefault(root, {})[sub_key] = value
            else:
                direct[root] = value

        for key, value in tuple(direct.items()):
            if isinstance(value, Iterator):
                snapshot = getattr(self, "_cox_cv_split_snapshot", None)
                if snapshot is None:
                    snapshot = list(value)
                direct[key] = copy.deepcopy(snapshot)

        # Constructor validation and normalization occur before mutating self.
        fresh = type(self)(**direct)
        for root, sub_params in nested.items():
            nested_estimator = getattr(fresh, root, None)
            if nested_estimator is None:
                nested_estimator = getattr(fresh, f"_{root}", None)
            if not hasattr(nested_estimator, "set_params"):
                raise ValueError(
                    f"Parameter {root!r} of {type(self).__name__} does not "
                    "support nested parameters."
                )
            nested_estimator.set_params(**sub_params)

        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        return self
'''
text = text[:start] + new_set_params
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Regression tests for every blocking finding and the remote matrix.
# ---------------------------------------------------------------------------
p = Path("dev/tests/test_maintenance_024_025.py")
text = p.read_text(encoding="utf-8")
old = '    assert calls == {"compiled": 1, "eager": 2}\n'
if old in text and "runtime-fallback\"\n    assert \"overwritten" not in text:
    text = text.replace(
        old,
        old
        + '    assert guarded.__statgpu_compile_status__ == "runtime-fallback"\n'
        + '    assert "overwritten" in guarded.__statgpu_compile_error__\n',
        1,
    )
marker = "def test_compile_construction_fallback_is_visible"
if marker not in text:
    text += r'''


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
    assert model.cov_type == "hac"
    assert model._fitted is False


def test_current_sklearn_classifier_and_regressor_tags():
    pytest.importorskip("sklearn")
    from sklearn.base import is_classifier, is_regressor
    from statgpu.linear_model import LogisticRegression, Ridge

    assert is_classifier(LogisticRegression())
    assert is_regressor(Ridge(compute_inference=False))


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

    compiled = compile_torch(add_one, workload="iterative")
    x = torch.arange(16, device="cuda", dtype=torch.float64)
    result = compiled(x)
    torch.cuda.synchronize()
    assert compiled.__statgpu_compile_status__ == "compiled"
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

    events = get_torch_compile_diagnostics(clear=True)
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
'''
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# 5. Scope documentation and defer performance claims explicitly.
# ---------------------------------------------------------------------------
for filename in ("docs/en/changelog.md", "docs/cn/changelog.md"):
    p = Path(filename)
    text = p.read_text(encoding="utf-8")
    text = text.replace(
        "Public estimator numerical inputs are checked for NaN/Inf using NumPy,\n  CuPy, or Torch reductions on the selected device.",
        "Maintained public numerical entry points are checked for NaN/Inf using\n  NumPy, CuPy, or Torch reductions on the selected device. The matrix includes\n  fit/predict/transform, inverse-transform, scoring, initialization arrays,\n  and panel identifiers while preserving formula-owned missing-row semantics.",
    )
    text = text.replace(
        "公共 estimator 的数值输入统一采用 NumPy、CuPy 或 Torch 原生 reduction\n  检查 NaN/Inf，不把完整 GPU 数组搬回 CPU。",
        "维护矩阵覆盖的公共 estimator 数值入口采用 NumPy、CuPy 或 Torch 原生\n  reduction 检查 NaN/Inf，不把完整 GPU 数组搬回 CPU；矩阵覆盖\n  fit/predict/transform、inverse-transform、scoring、初始化数组和 panel ID，\n  同时保留 formula 路径对缺失行的专属语义。",
    )
    p.write_text(text, encoding="utf-8")

p = Path("dev/manual/gpu_diagnostics/README.md")
text = p.read_text(encoding="utf-8")
if "## Torch compile performance note" not in text:
    text += '''

## Torch compile performance note

The maintenance release prioritizes correctness by defaulting iterative kernels
to Torch `default` compile mode. No claim is made that this matches the
steady-state latency of `reduce-overhead`; representative Lasso, ElasticNet,
nonconvex, adaptive, and group-penalty benchmarks remain an optimization task.
Users may opt into `reduce-overhead` explicitly, and construction/runtime
fallback decisions remain available through `get_torch_compile_diagnostics()`.
'''
p.write_text(text, encoding="utf-8")

p = Path("CHANGELOG.md")
text = p.read_text(encoding="utf-8")
text = text.replace(
    "  centralized policy that avoids CUDA Graph lifecycle hazards for iterative\n  solvers and falls back to eager execution for the known runtime failure.",
    "  centralized policy that avoids CUDA Graph lifecycle hazards for iterative\n  solvers; compile decisions are observable, and only the known lifecycle\n  failure falls back to eager execution. Performance comparison with\n  `reduce-overhead` remains explicitly deferred.",
    1,
)
p.write_text(text, encoding="utf-8")


import compileall
if not compileall.compile_dir("statgpu", quiet=1):
    raise SystemExit("statgpu compileall failed")
if not compileall.compile_file("dev/tests/test_maintenance_024_025.py", quiet=1):
    raise SystemExit("maintenance test compile failed")
