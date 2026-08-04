from __future__ import annotations

from pathlib import Path
from textwrap import dedent
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {text.count(old)}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Observable and narrowly scoped torch.compile fallback.
# ---------------------------------------------------------------------------
p = Path("statgpu/backends/_torch_compile.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    dedent(
        '''
        _CUDAGRAPH_RUNTIME_MARKERS = (
            "CUDAGraphs",
            "cudagraph",
            "overwritten by a subsequent run",
        )
        '''
    ).lstrip(),
    dedent(
        '''
        _CUDAGRAPH_RUNTIME_MARKERS = (
            "accessing tensor output of cudagraphs",
            "tensor output of cudagraphs",
            "overwritten by a subsequent run",
        )
        _COMPILE_DIAGNOSTICS = []


        def _record_compile_event(*, fn, status, mode, workload, error=None):
            _COMPILE_DIAGNOSTICS.append(
                {
                    "function": getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn))),
                    "status": status,
                    "mode": mode,
                    "workload": workload,
                    "error": error,
                }
            )


        def get_torch_compile_diagnostics(*, clear: bool = False):
            """Return immutable snapshots of internal Torch compile decisions."""
            snapshot = tuple(dict(event) for event in _COMPILE_DIAGNOSTICS)
            if clear:
                _COMPILE_DIAGNOSTICS.clear()
            return snapshot
        '''
    ).lstrip(),
    "compile diagnostics insertion",
)
text = replace_once(
    text,
    dedent(
        '''
        def _is_cudagraph_lifecycle_error(exc: BaseException) -> bool:
            message = str(exc)
            return any(marker.lower() in message.lower() for marker in _CUDAGRAPH_RUNTIME_MARKERS)
        '''
    ).lstrip(),
    dedent(
        '''
        def _is_cudagraph_lifecycle_error(exc: BaseException) -> bool:
            message = str(exc).lower()
            has_overwrite = "overwrit" in message
            has_cudagraph = "cudagraph" in message
            has_tensor_output = "tensor output" in message or "accessing tensor" in message
            return has_overwrite and has_cudagraph and has_tensor_output
        '''
    ).lstrip(),
    "narrow CUDA Graph matcher",
)
text = replace_once(
    text,
    dedent(
        '''
            if resolved_mode is None or not torch_compile_available():
                return fn

            try:
                import torch
                compiled = torch.compile(fn, mode=resolved_mode, **compile_kwargs)
            except Exception:
                return fn

            state = {"disabled": False}
        '''
    ),
    dedent(
        '''
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
        '''
    ),
    "observable compile construction fallback",
)
text = replace_once(
    text,
    dedent(
        '''
                    state["disabled"] = True
                    warnings.warn(
        '''
    ),
    dedent(
        '''
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
        '''
    ),
    "runtime fallback diagnostics",
)
text = replace_once(
    text,
    dedent(
        '''
            guarded.__statgpu_compile_mode__ = resolved_mode
            guarded.__statgpu_compile_workload__ = workload
            return guarded
        '''
    ),
    dedent(
        '''
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
        '''
    ),
    "compiled status diagnostics",
)
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Public finite-value matrix, sklearn tags, and normalized set_params rebuild.
# ---------------------------------------------------------------------------
p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    '        "predict_cumulative_hazard",\n    })',
    dedent(
        '''
                "predict_cumulative_hazard",
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
            })
        '''
    ).rstrip("\n"),
    "finite public method matrix",
)
text = replace_once(
    text,
    '        "init_coef",\n    })',
    dedent(
        '''
                "init_coef",
                "initial_coef",
                "time_index",
                "entity_ids",
                "time_ids",
            })
        '''
    ).rstrip("\n"),
    "finite parameter matrix",
)
old_guard = dedent(
    '''
                loss_value = getattr(self, "loss", "")
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
)
new_guard = dedent(
    '''
                loss_value = getattr(self, "loss", "")
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
                        # Formula/model-matrix code owns row dropping, category
                        # encoding, and aligned side-array errors.
                        continue
                    if name in self._FINITE_PARAMETER_NAMES and value is not None:
                        check_finite(value, name=name)
                return original(self, *args, **kwargs)
    '''
)
text = replace_once(text, old_guard, new_guard, "formula-aware finite guard")
pattern = re.compile(r"    def __sklearn_tags__\(self\):\n.*?    def __sklearn_clone__\(self\):", re.S)
replacement = dedent(
    '''
        def _statgpu_estimator_type(self):
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

        def __sklearn_clone__(self):
    '''
).lstrip("\n")
text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise RuntimeError(f"sklearn tag block: expected one match, found {count}")
start = text.index("    def set_params(self, **params):")
end = text.index("\n        return self", start) + len("\n        return self")
new_set_params = dedent(
    '''
        def set_params(self, **params):
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

            # One-shot split iterators cannot safely be passed through a fresh
            # constructor after use. Materialize a reusable snapshot before the
            # transactional rebuild.
            for key, value in tuple(direct.items()):
                if isinstance(value, Iterator):
                    snapshot = getattr(self, "_cox_cv_split_snapshot", None)
                    if snapshot is None:
                        snapshot = list(value)
                    direct[key] = copy.deepcopy(snapshot)

            fresh = type(self)(**direct)
            for root, sub_params in nested.items():
                nested_estimator = getattr(fresh, root, None)
                if nested_estimator is None or not hasattr(nested_estimator, "set_params"):
                    raise ValueError(
                        f"Parameter {root!r} of {type(self).__name__} does not "
                        "support nested parameters."
                    )
                nested_estimator.set_params(**sub_params)

            self.__dict__.clear()
            self.__dict__.update(fresh.__dict__)
            return self
    '''
).lstrip("\n")
text = text[:start] + new_set_params + text[end:]
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Backend-native finite validation for sparse, pandas, and object arrays.
# ---------------------------------------------------------------------------
p = Path("statgpu/backends/_validation.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    '    module = type(value).__module__\n',
    dedent(
        '''
            module = type(value).__module__
            if module.startswith("scipy.sparse") or module.startswith("cupyx.scipy.sparse"):
                check_finite(value.data, name=name)
                return value
        '''
    ),
    "sparse finite validation",
)
text = replace_once(
    text,
    '    if module.startswith("pandas"):\n        return value\n',
    dedent(
        '''
            if module.startswith("pandas"):
                try:
                    array = value.to_numpy()
                except Exception:
                    return value
                if array.dtype.kind in "biufc" and not np.isfinite(array).all():
                    _raise_nonfinite(name)
                if array.dtype.kind == "O":
                    for index, item in np.ndenumerate(array):
                        check_finite(item, name=f"{name}{index}")
                return value
        '''
    ),
    "pandas finite validation",
)
text = replace_once(
    text,
    dedent(
        '''
            if array.dtype.kind == "O" and isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    check_finite(item, name=f"{name}[{index}]")
            return value
        '''
    ),
    dedent(
        '''
            if array.dtype.kind == "O":
                for index, item in np.ndenumerate(array):
                    if item is value:
                        continue
                    check_finite(item, name=f"{name}{index}")
            return value
        '''
    ),
    "object finite validation",
)
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Regression tests for all blocking review findings and remote matrix.
# ---------------------------------------------------------------------------
p = Path("dev/tests/test_maintenance_024_025.py")
text = p.read_text(encoding="utf-8")
text = text.replace(
    '    assert calls == {"compiled": 1, "eager": 2}\n',
    '    assert calls == {"compiled": 1, "eager": 2}\n'
    '    assert guarded.__statgpu_compile_status__ == "runtime-fallback"\n'
    '    assert "overwritten" in guarded.__statgpu_compile_error__\n',
    1,
)
text += dedent(
    r'''


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
        events = get_torch_compile_diagnostics(clear=True)
        assert events[-1]["status"] == "construction-fallback"


    def test_set_params_rebuilds_normalized_panel_state():
        from statgpu.panel import PooledOLS

        model = PooledOLS()
        model._fitted = True
        model.set_params(cov_type="HAC", kernel="BARTLETT")
        assert model.get_params(deep=False)["cov_type"] == "HAC"
        assert model.cov_type == "hac"
        assert model.kernel == "BARTLETT"
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

        object_array = np.array([1.0, np.nan], dtype=object)
        with pytest.raises(ValueError, match="finite"):
            check_finite(object_array, name="X")

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
        events = get_torch_compile_diagnostics(clear=True)
        assert events[-1]["status"] == "compiled"


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
        compiled = [event for event in events if event["status"] == "compiled"]
        fallback = [event for event in events if "fallback" in event["status"]]
        assert len(compiled) >= len(penalties)
        assert fallback == []


    def test_cupy_finite_validation_stays_on_device():
        cp = pytest.importorskip("cupy")
        try:
            cp.cuda.runtime.getDeviceCount()
        except Exception:
            pytest.skip("requires a working CuPy CUDA backend")
        from statgpu.backends._validation import check_finite

        value = cp.asarray([1.0, 2.0])
        assert check_finite(value, name="X") is value
        with pytest.raises(ValueError, match="finite"):
            check_finite(cp.asarray([1.0, cp.inf]), name="X")
    '''
).lstrip("\n")
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Narrow public claims and record the deferred benchmark evidence.
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
    text += dedent(
        '''

        ## Torch compile performance note

        The maintenance release prioritizes correctness by defaulting iterative
        kernels to Torch `default` compile mode. No claim is made that this matches
        the steady-state latency of `reduce-overhead`; representative Lasso,
        ElasticNet, nonconvex, adaptive, and group-penalty benchmarks remain an
        optimization task. Users may opt into `reduce-overhead` explicitly, and
        construction/runtime fallback decisions remain available through
        `get_torch_compile_diagnostics()`.
        '''
    )
p.write_text(text, encoding="utf-8")


# The root changelog wording is already scoped to public estimator boundaries;
# append the explicit compile diagnostic and benchmark caveat.
p = Path("CHANGELOG.md")
text = p.read_text(encoding="utf-8")
needle = "  centralized policy that avoids CUDA Graph lifecycle hazards for iterative\n  solvers and falls back to eager execution for the known runtime failure."
replacement = (
    "  centralized policy that avoids CUDA Graph lifecycle hazards for iterative\n"
    "  solvers; compile decisions are observable, and only the known lifecycle\n"
    "  failure falls back to eager execution. Performance comparison with\n"
    "  `reduce-overhead` remains explicitly deferred."
)
if needle in text:
    text = text.replace(needle, replacement, 1)
p.write_text(text, encoding="utf-8")


# Fail before any commit when generated Python is malformed.
import compileall
if not compileall.compile_dir("statgpu", quiet=1):
    raise SystemExit("statgpu compileall failed")
if not compileall.compile_file("dev/tests/test_maintenance_024_025.py", quiet=1):
    raise SystemExit("maintenance test compile failed")
