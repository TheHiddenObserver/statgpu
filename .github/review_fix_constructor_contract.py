from __future__ import annotations

import ast
import compileall
import copy
import inspect
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# Base constructor contract: public raw values, private normalized runtime.
# ---------------------------------------------------------------------------
p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
anchor = '''    _FINITE_PARAMETER_NAMES = frozenset({
'''
normalized_block = '''    _NORMALIZED_CONSTRUCTOR_PARAMS = frozenset({
        "device",
        "cov_type",
        "hac_maxlags",
        "gpu_memory_cleanup",
        "solver",
        "cpu_solver",
        "stopping",
        "inference_method",
        "simultaneous_method",
        "n_bootstrap",
        "enable_simultaneous_inference",
        "simultaneous_alpha",
        "simultaneous_n_bootstrap",
        "simultaneous_include_intercept",
        "method",
        "admm_rho",
        "alpha_min_ratio",
        "cd_kkt_check_every",
        "compute_inference",
        "cv",
        "fit_intercept",
        "gpu_cv_mixed_precision",
        "max_iter",
        "n_alphas",
        "tol",
        "n_Cs",
        "C_min_ratio",
        "penalty_kwargs",
        "loss_kwargs",
        "epsilon",
        "ties",
        "acknowledge_approx",
        "refine_top_k",
        "batch_size",
        "min_effective_weight",
        "quantile",
        "cv_strategy",
    })

    _FINITE_PARAMETER_NAMES = frozenset({
'''
text = replace_once(text, anchor, normalized_block, "normalized parameter set")
old = '''            original_init(self, *args, **kwargs)
            self._constructor_params_raw = raw_params
'''
new = '''            original_init(self, *args, **kwargs)
            normalized_names = type(self)._NORMALIZED_CONSTRUCTOR_PARAMS
            for name, raw_value in raw_params.items():
                private_name = f"_{name}"
                if name in normalized_names:
                    if hasattr(self, name):
                        runtime_value = getattr(self, name)
                    elif hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    else:
                        runtime_value = raw_value
                    if isinstance(runtime_value, (dict, list, set, np.ndarray)):
                        runtime_value = copy.deepcopy(runtime_value)
                    setattr(self, private_name, runtime_value)
                    setattr(self, name, raw_value)
                elif not hasattr(self, name):
                    # Parameters delegated to a superclass or represented only
                    # by a private runtime field must still exist publicly.
                    setattr(self, name, raw_value)
            self._constructor_params_raw = raw_params
'''
text = replace_once(text, old, new, "constructor capture normalization")
old = '''        self.device = device if isinstance(device, Device) else Device(device)
        self.n_jobs = n_jobs
'''
new = '''        self.device = device
        self._device = device if isinstance(device, Device) else Device(device)
        self.n_jobs = n_jobs
'''
text = replace_once(text, old, new, "base device storage")
p.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Internal estimator code uses private normalized values outside __init__.
# ---------------------------------------------------------------------------
NORMALIZED = {
    "device",
    "cov_type",
    "hac_maxlags",
    "gpu_memory_cleanup",
    "solver",
    "cpu_solver",
    "stopping",
    "inference_method",
    "simultaneous_method",
    "n_bootstrap",
    "enable_simultaneous_inference",
    "simultaneous_alpha",
    "simultaneous_n_bootstrap",
    "simultaneous_include_intercept",
    "method",
    "admm_rho",
    "alpha_min_ratio",
    "cd_kkt_check_every",
    "compute_inference",
    "cv",
    "fit_intercept",
    "gpu_cv_mixed_precision",
    "max_iter",
    "n_alphas",
    "tol",
    "n_Cs",
    "C_min_ratio",
    "penalty_kwargs",
    "loss_kwargs",
    "epsilon",
    "ties",
    "acknowledge_approx",
    "refine_top_k",
    "batch_size",
    "min_effective_weight",
    "quantile",
    "cv_strategy",
}

# Loaded estimator subclasses plus the mixins whose methods execute on them.
import statgpu
from statgpu._base import BaseEstimator


def descendants(cls):
    seen = set()
    stack = list(cls.__subclasses__())
    while stack:
        child = stack.pop()
        if child in seen:
            continue
        seen.add(child)
        stack.extend(child.__subclasses__())
    return seen


ESTIMATOR_CLASSES = {
    (cls.__module__, cls.__name__)
    for cls in descendants(BaseEstimator)
}
ESTIMATOR_CLASSES.add(("statgpu._base", "BaseEstimator"))
MIXIN_CLASSES = {
    ("statgpu.linear_model.penalized._fit_mixin", "_PenalizedFitMixin"),
    ("statgpu.linear_model.penalized._inference_mixin", "_PenalizedInferenceMixin"),
    ("statgpu.linear_model.penalized._predict_mixin", "_PenalizedPredictMixin"),
}
TARGET_CLASSES = ESTIMATOR_CLASSES | MIXIN_CLASSES


def module_name(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


def offset(lines, lineno, col):
    return sum(len(line) for line in lines[: lineno - 1]) + col


class RewriteVisitor(ast.NodeVisitor):
    def __init__(self, module, lines):
        self.module = module
        self.lines = lines
        self.class_stack = []
        self.function_stack = []
        self.replacements = []

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node):
        active_class = self.class_stack[-1] if self.class_stack else None
        active_function = self.function_stack[-1] if self.function_stack else None
        if (
            active_class is not None
            and (self.module, active_class) in TARGET_CLASSES
            and active_function != "__init__"
            and node.attr in NORMALIZED
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            start = offset(self.lines, node.lineno, node.col_offset)
            end = offset(self.lines, node.end_lineno, node.end_col_offset)
            self.replacements.append((start, end, f"self._{node.attr}"))
        self.generic_visit(node)


for path in Path("statgpu").rglob("*.py"):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    visitor = RewriteVisitor(module_name(path), lines)
    visitor.visit(tree)
    if not visitor.replacements:
        continue
    for start, end, replacement in sorted(visitor.replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    path.write_text(source, encoding="utf-8")


# ---------------------------------------------------------------------------
# Strong public contract tests.
# ---------------------------------------------------------------------------
p = Path("dev/tests/test_maintenance_024_025.py")
text = p.read_text(encoding="utf-8")
text += r'''


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
'''
p.write_text(text, encoding="utf-8")


# Compile and import gates before testing/committing.
for path in Path("statgpu").rglob("*.py"):
    if not compileall.compile_file(str(path), quiet=1):
        raise SystemExit(f"compile failed: {path}")
if not compileall.compile_file("dev/tests/test_maintenance_024_025.py", quiet=1):
    raise SystemExit("maintenance test compile failed")

# A direct post-patch structural check must already be zero for default exports.
import importlib
importlib.invalidate_caches()
