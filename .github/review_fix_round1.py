from __future__ import annotations

import ast
import compileall
import re
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def rewrite_penalty_loader(path: str, function_name: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start = text.index(f"def {function_name}():")
    end = text.index("\n\nclass ", start)
    block = text[start:end]
    lines = block.splitlines()
    try_index = next(i for i, line in enumerate(lines) if line == "    try:")
    except_index = next(i for i, line in enumerate(lines) if line == "    except Exception:")
    prefix = lines[:try_index]
    # Remove the legacy availability pre-gate. compile_torch owns disabled and
    # unavailable states and returns an observable eager wrapper.
    gate_start = next(
        (i for i, line in enumerate(prefix) if "from statgpu.penalties import _torch_compile_ok" in line),
        None,
    )
    if gate_start is not None:
        prefix = prefix[:gate_start]
    body = [line[4:] if line.startswith("    ") else line for line in lines[try_index + 1:except_index]]
    final_return = lines[-1]
    new_block = "\n".join(prefix + body + [final_return])
    p.write_text(text[:start] + new_block + text[end:], encoding="utf-8")


for spec in (
    ("statgpu/penalties/_l1.py", "_get_l1_torch_compiled"),
    ("statgpu/penalties/_adaptive_l1.py", "_get_adaptive_l1_torch_compiled"),
    ("statgpu/penalties/_scad.py", "_get_scad_torch_compiled"),
    ("statgpu/penalties/_mcp.py", "_get_mcp_torch_compiled"),
    ("statgpu/penalties/_group_lasso.py", "_get_group_lasso_torch_compiled_equal"),
    ("statgpu/penalties/_group_scad.py", "_get_group_scad_torch_compiled"),
    ("statgpu/penalties/_group_mcp.py", "_get_group_mcp_torch_compiled"),
):
    rewrite_penalty_loader(*spec)


# Fix FISTA-LLA's invalid decorator use and remove caller-side compile swallowing.
p = Path("statgpu/solvers/_fista_lla.py")
text = p.read_text(encoding="utf-8")
old = '''        if _cap >= 7:
            try:
                @compile_torch(workload="iterative", backend='inductor')
                def _fused_update(y_current, grad, step, thresh, coef_old, beta):
                    w = y_current - step * grad
                    abs_w = w.abs()
                    sign_w = w.sign()
                    coef_new = sign_w * (abs_w - thresh).clamp(min=0.0)
                    y_k = coef_new + beta * (coef_new - coef_old)
                    return coef_new, y_k
                _SQERR_PROXIMAL_TORCH = _fused_update
            except (RuntimeError, TypeError):
                pass
        if _SQERR_PROXIMAL_TORCH is None:
'''
new = '''        if _cap >= 7:
            def _fused_update(y_current, grad, step, thresh, coef_old, beta):
                w = y_current - step * grad
                abs_w = w.abs()
                sign_w = w.sign()
                coef_new = sign_w * (abs_w - thresh).clamp(min=0.0)
                y_k = coef_new + beta * (coef_new - coef_old)
                return coef_new, y_k
            _SQERR_PROXIMAL_TORCH = compile_torch(
                _fused_update, workload="iterative", backend="inductor"
            )
        if _SQERR_PROXIMAL_TORCH is None:
'''
text = replace_once(text, old, new, "fista-lla squared-error compile")
old = '''        if _cap >= 7:
            try:
                _FUSED_PROXIMAL_CLIP_TORCH = compile_torch(
                    _fused, workload="iterative", backend='inductor')
            except (RuntimeError, TypeError):
                _FUSED_PROXIMAL_CLIP_TORCH = _fused
        else:
'''
new = '''        if _cap >= 7:
            _FUSED_PROXIMAL_CLIP_TORCH = compile_torch(
                _fused, workload="iterative", backend="inductor"
            )
        else:
'''
text = replace_once(text, old, new, "fista-lla generic compile")
p.write_text(text, encoding="utf-8")


# Let the centralized helper own availability/fallback semantics in the penalized FISTA path.
p = Path("statgpu/linear_model/penalized/_fit_mixin.py")
text = p.read_text(encoding="utf-8")
start = text.index("            if is_torch:\n", text.index("# Build fused element-wise kernel"))
end = text.index("            else:\n                import cupy as cp", start)
new_block = '''            if is_torch:
                import torch
                if _use_l2:
                    def _fista_elementwise_l2(
                        _y_k, _xtx_y, _step_over_n_Xty, _step_over_n,
                        _thresh, _l2_scale, _coef_old, _beta,
                    ):
                        w = _y_k - _step_over_n * _xtx_y + _step_over_n_Xty
                        c = _st_fn(w, _thresh, xp) / _l2_scale
                        y = c + _beta * (c - _coef_old)
                        return c, y
                    _fused_step_l2 = compile_torch(
                        _fista_elementwise_l2, workload="iterative"
                    )
                else:
                    def _fista_elementwise(
                        _y_k, _xtx_y, _step_over_n_Xty, _step_over_n,
                        _thresh, _coef_old, _beta,
                    ):
                        w = _y_k - _step_over_n * _xtx_y + _step_over_n_Xty
                        c = _st_fn(w, _thresh, xp)
                        y = c + _beta * (c - _coef_old)
                        return c, y
                    _fused_step = compile_torch(
                        _fista_elementwise, workload="iterative"
                    )
'''
text = text[:start] + new_block + text[end:]
p.write_text(text, encoding="utf-8")


# Do not globally suppress Dynamo errors: it makes helper diagnostics lie about fallback.
p = Path("statgpu/linear_model/legacy/_elasticnet_legacy.py")
text = p.read_text(encoding="utf-8")
old = '''    # Compile the proximal operator
    try:
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.guard_immutable_object = False
        _elastic_net_proximal_compiled = compile_torch(
            _elastic_net_proximal_torch, workload="iterative"
        )
    except (AttributeError, RuntimeError):
        _elastic_net_proximal_compiled = _elastic_net_proximal_torch

    return _elastic_net_proximal_compiled
'''
new = '''    # Compile through the centralized observable policy. Do not mutate
    # process-global Dynamo suppression settings here.
    return compile_torch(
        _elastic_net_proximal_torch, workload="iterative"
    )
'''
text = replace_once(text, old, new, "legacy elasticnet compile")
p.write_text(text, encoding="utf-8")


# Export compile diagnostics from the public backends namespace.
p = Path("statgpu/backends/__init__.py")
text = p.read_text(encoding="utf-8")
anchor = "from ._factory import get_backend\n"
insert = '''from ._factory import get_backend
from ._torch_compile import (
    compile_torch,
    get_torch_compile_diagnostics,
    resolve_torch_compile_mode,
    torch_compile_available,
)
'''
text = replace_once(text, anchor, insert, "backend compile export import")
anchor = '    "get_backend",\n'
insert = '''    "get_backend",
    "compile_torch",
    "get_torch_compile_diagnostics",
    "resolve_torch_compile_mode",
    "torch_compile_available",
'''
text = replace_once(text, anchor, insert, "backend compile export all")
p.write_text(text, encoding="utf-8")


# Strengthen the shared estimator contract.
p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    '        "time_ids",\n    })',
    '        "time_ids",\n        "pvalues",\n        "arrays",\n        "scores",\n        "thresholds",\n        "Xk",\n        "mu",\n        "Sigma",\n    })',
    "finite parameter expansion",
)
text = replace_once(
    text,
    '''            elif ("classifier" in name or "logistic" in name) and classifier_module:
                inferred_type = "classifier"
''',
    '''            elif (
                "classifier" in name
                or "logistic" in name
                or "logit" in name
                or "probit" in name
            ) and classifier_module:
                inferred_type = "classifier"
''',
    "classifier inference",
)
old = '''        for method_name in cls._FINITE_PUBLIC_METHODS:
            original = cls.__dict__.get(method_name)
            if original is None or not callable(original):
                continue
            if getattr(original, "__isabstractmethod__", False):
                continue
            if getattr(original, "__statgpu_finite_validation__", False):
                continue
            setattr(cls, method_name, wrap_method(original))
'''
new = '''        for method_name, original in tuple(cls.__dict__.items()):
            if method_name.startswith("_") or not callable(original):
                continue
            if getattr(original, "__isabstractmethod__", False):
                continue
            if getattr(original, "__statgpu_finite_validation__", False):
                continue
            try:
                signature = inspect.signature(original)
            except (TypeError, ValueError):
                continue
            numerical_parameters = set(signature.parameters) & cls._FINITE_PARAMETER_NAMES
            if method_name not in cls._FINITE_PUBLIC_METHODS and not numerical_parameters:
                continue
            setattr(cls, method_name, wrap_method(original))
'''
text = replace_once(text, old, new, "finite wrapper inventory")
old = '''            from sklearn.utils import (
                ClassifierTags,
                RegressorTags,
                Tags,
                TargetTags,
            )
'''
new = '''            from sklearn.utils import (
                ClassifierTags,
                RegressorTags,
                Tags,
                TargetTags,
                TransformerTags,
            )
'''
text = replace_once(text, old, new, "transformer tags import")
old = '''        return Tags(
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
'''
new = '''        has_transform = callable(getattr(self, "transform", None))
        return Tags(
            estimator_type=estimator_type,
            target_tags=TargetTags(required=estimator_type is not None),
            transformer_tags=TransformerTags() if has_transform else None,
            classifier_tags=(
                ClassifierTags() if estimator_type == "classifier" else None
            ),
            regressor_tags=(
                RegressorTags() if estimator_type == "regressor" else None
            ),
            requires_fit=True,
        )
'''
text = replace_once(text, old, new, "transformer tags result")
old_start = text.index("    def set_params(self, **params):\n")
old_end = len(text)
old_set_params = text[old_start:old_end]
new_set_params = '''    def set_params(self, **params):
        """Set parameters transactionally and refresh normalized state."""
        if not params:
            return self

        import copy
        from collections.abc import Iterator

        valid_deep = self.get_params(deep=True)
        direct = self.get_params(deep=False)
        direct_updates = {}
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
                direct_updates[root] = value

        explicitly_updated = set(direct_updates)
        for key, value in tuple(direct.items()):
            if isinstance(value, Iterator) and key not in explicitly_updated:
                snapshot = getattr(self, "_cox_cv_split_snapshot", None)
                if snapshot is None:
                    snapshot = list(value)
                direct[key] = copy.deepcopy(snapshot)

        try:
            fresh = type(self)(**direct)
        except (TypeError, ValueError):
            deferred = set(getattr(type(self), "_DEFERRED_SET_PARAMS", ()))
            if nested or not direct_updates or not set(direct_updates).issubset(deferred):
                raise
            # A small number of estimators intentionally validate selected
            # controls at fit time. Apply only those explicitly declared values
            # after the complete update has been classified as deferred.
            for key, value in direct_updates.items():
                setattr(self, key, value)
                raw_params = getattr(self, "_constructor_params_raw", None)
                if raw_params is None:
                    raw_params = {}
                    self._constructor_params_raw = raw_params
                raw_params[key] = value
            reset = getattr(self, "_reset_fit_state", None)
            if callable(reset):
                reset()
            else:
                self._fitted = False
            return self

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
text = text[:old_start] + new_set_params
# Install validation on BaseEstimator's own public inference helpers.
text += "\n\nBaseEstimator._install_public_finite_validation()\n"
p.write_text(text, encoding="utf-8")


# Preserve Cox's deliberately deferred boolean-control validation boundary.
p = Path("statgpu/survival/_cox.py")
text = p.read_text(encoding="utf-8")
anchor = '    _estimator_type = "regressor"\n'
insert = '''    _estimator_type = "regressor"
    _DEFERRED_SET_PARAMS = frozenset({
        "compute_inference", "compute_cindex", "gpu_memory_cleanup"
    })
'''
text = replace_once(text, anchor, insert, "cox deferred set params")
p.write_text(text, encoding="utf-8")


# Reject pandas extension-array missing values consistently.
p = Path("statgpu/backends/_validation.py")
text = p.read_text(encoding="utf-8")
old = '''    if module.startswith("pandas"):
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
'''
new = '''    if module.startswith("pandas"):
        import pandas as pd

        missing = pd.isna(value)
        if hasattr(missing, "to_numpy"):
            missing = missing.to_numpy()
        if bool(np.asarray(missing).any()):
            _raise_nonfinite(name)
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
'''
text = replace_once(text, old, new, "pandas finite validation")
p.write_text(text, encoding="utf-8")


# Give standalone knockoff selectors complete transformer/finite contracts.
p = Path("statgpu/feature_selection/_knockoff.py")
text = p.read_text(encoding="utf-8")
anchor = "from statgpu.feature_selection import _knockoff_utils as _kutils\n"
insert = '''from statgpu.feature_selection import _knockoff_utils as _kutils
from statgpu.backends._validation import check_finite
'''
text = replace_once(text, anchor, insert, "knockoff finite import")
anchor = "class KnockoffSelector:\n"
mixin = '''class _KnockoffSelectorContract:
    """Shared sklearn and finite-input contract for knockoff selectors."""

    def _more_tags(self):
        return {"requires_y": True}

    def __sklearn_tags__(self):
        try:
            from sklearn.utils import Tags, TargetTags, TransformerTags
        except ImportError:
            return self._more_tags()
        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=True),
            transformer_tags=TransformerTags(),
            requires_fit=True,
        )

    def __sklearn_is_fitted__(self):
        return getattr(self, "selected_features_", None) is not None

    @staticmethod
    def _validate_fit_inputs(X, y, Xk=None):
        check_finite(X, name="X")
        check_finite(y, name="y")
        if Xk is not None:
            check_finite(Xk, name="Xk")

    @staticmethod
    def _validate_transform_input(X):
        check_finite(X, name="X")


class KnockoffSelector(_KnockoffSelectorContract):
'''
text = replace_once(text, anchor, mixin, "knockoff selector mixin")
text = replace_once(
    text,
    '''    def fit(self, X, y, Xk=None):
        self.result_ = knockoff_filter(
''',
    '''    def fit(self, X, y, Xk=None):
        self._validate_fit_inputs(X, y, Xk)
        self.result_ = knockoff_filter(
''',
    "knockoff fit finite",
)
text = replace_once(
    text,
    '''    def transform(self, X):
        if self.selected_features_ is None:
''',
    '''    def transform(self, X):
        self._validate_transform_input(X)
        if self.selected_features_ is None:
''',
    "knockoff transform finite",
)
text = replace_once(
    text,
    "class FixedXKnockoffSelector:\n",
    "class FixedXKnockoffSelector(_KnockoffSelectorContract):\n",
    "fixed knockoff mixin",
)
text = replace_once(
    text,
    '''    def fit(self, X, y, Xk=None):
        self._selector.fit(X, y, Xk=Xk)
''',
    '''    def fit(self, X, y, Xk=None):
        self._validate_fit_inputs(X, y, Xk)
        self._selector.fit(X, y, Xk=Xk)
''',
    "fixed knockoff fit finite",
)
text = replace_once(
    text,
    '''    def transform(self, X):
        return self._selector.transform(X)
''',
    '''    def transform(self, X):
        self._validate_transform_input(X)
        return self._selector.transform(X)
''',
    "fixed knockoff transform finite",
)
p.write_text(text, encoding="utf-8")


# Add focused review regressions.
p = Path("dev/tests/test_maintenance_024_025.py")
text = p.read_text(encoding="utf-8")
text += r'''


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
'''
p.write_text(text, encoding="utf-8")


# Validate syntax and static compile-site contract before any production commit.
for path in (
    "statgpu/_base.py",
    "statgpu/backends/_validation.py",
    "statgpu/backends/_torch_compile.py",
    "statgpu/backends/__init__.py",
    "statgpu/solvers/_fista_lla.py",
    "statgpu/linear_model/penalized/_fit_mixin.py",
    "statgpu/linear_model/legacy/_elasticnet_legacy.py",
    "statgpu/feature_selection/_knockoff.py",
    "dev/tests/test_maintenance_024_025.py",
):
    if not compileall.compile_file(path, quiet=1):
        raise SystemExit(f"compile failed: {path}")

for path in Path("statgpu").rglob("*.py"):
    source = path.read_text(encoding="utf-8")
    if "compile_torch" not in source and "suppress_errors" not in source:
        continue
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "compile_torch":
                    raise SystemExit(f"compile_torch decorator misuse: {path}:{decorator.lineno}")
        if isinstance(node, ast.Try):
            if any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "compile_torch"
                for stmt in node.body
                for child in ast.walk(stmt)
            ):
                raise SystemExit(f"compile_torch caught by caller: {path}:{node.lineno}")
    if "torch._dynamo.config.suppress_errors" in source:
        raise SystemExit(f"Dynamo suppress_errors mutation remains: {path}")
