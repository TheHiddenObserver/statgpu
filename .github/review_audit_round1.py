from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import statgpu


def safe_repr(value):
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


def public_estimators():
    for name in statgpu.__all__:
        cls = getattr(statgpu, name, None)
        if not inspect.isclass(cls) or not hasattr(cls, "fit") or inspect.isabstract(cls):
            continue
        sig = inspect.signature(cls)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect._empty
            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        if required:
            continue
        try:
            yield name, cls, cls()
        except Exception as exc:
            print("DEFAULT_INIT_FAILURE", name, type(exc).__name__, str(exc))


def audit_constructor_contracts():
    mismatches = []
    for name, cls, estimator in public_estimators():
        params = estimator.get_params(deep=False)
        raw = getattr(estimator, "_constructor_params_raw", {})
        for param_name, param_value in params.items():
            if not hasattr(estimator, param_name):
                mismatches.append({
                    "estimator": name,
                    "parameter": param_name,
                    "kind": "missing-public-attribute",
                })
                continue
            attr_value = getattr(estimator, param_name)
            if attr_value is not param_value:
                mismatches.append({
                    "estimator": name,
                    "parameter": param_name,
                    "kind": "identity",
                    "param_type": type(param_value).__name__,
                    "attr_type": type(attr_value).__name__,
                    "param": safe_repr(param_value),
                    "attr": safe_repr(attr_value),
                    "in_raw_ledger": param_name in raw,
                })
    print("CONSTRUCTOR_MISMATCH_COUNT", len(mismatches))
    print("CONSTRUCTOR_MISMATCHES", json.dumps(mismatches, sort_keys=True))

    probes = []
    probe_specs = [
        ("PenalizedLinearRegression", {"penalty_kwargs": {"alpha": 7}, "loss_kwargs": {"scale": 2}}),
        ("PenalizedLogisticRegression", {"penalty_kwargs": {"alpha": 7}, "loss_kwargs": {"scale": 2}}),
        ("PenalizedGeneralizedLinearModel", {"penalty_kwargs": {"alpha": 7}, "loss_kwargs": {"scale": 2}}),
    ]
    for name, kwargs in probe_specs:
        cls = getattr(statgpu, name, None)
        if cls is None:
            continue
        originals = {key: value.copy() for key, value in kwargs.items()}
        try:
            estimator = cls(**originals)
        except Exception as exc:
            probes.append((name, "init-error", type(exc).__name__, str(exc)))
            continue
        params = estimator.get_params(deep=False)
        for key, original in originals.items():
            attr = getattr(estimator, key, None)
            before = safe_repr(attr)
            original["external_mutation"] = True
            probes.append((
                name,
                key,
                "param_is_original", params.get(key) is original,
                "attr_is_original", attr is original,
                "attr_before", before,
                "attr_after", safe_repr(getattr(estimator, key, None)),
                "param_after", safe_repr(estimator.get_params(deep=False).get(key)),
            ))
    print("MUTABLE_PARAMETER_PROBES", json.dumps(probes, default=str))


def audit_tags():
    try:
        from sklearn.utils import get_tags
    except ImportError:
        from sklearn.utils._tags import get_tags
    missing_transformer = []
    type_rows = []
    tag_errors = []
    for name, cls, estimator in public_estimators():
        try:
            tags = get_tags(estimator)
        except Exception as exc:
            tag_errors.append((name, type(exc).__name__, str(exc)))
            continue
        estimator_type = getattr(tags, "estimator_type", None)
        transformer_tags = getattr(tags, "transformer_tags", None)
        has_transform = callable(getattr(estimator, "transform", None))
        type_rows.append((name, estimator_type, has_transform, transformer_tags is not None))
        if has_transform and transformer_tags is None:
            missing_transformer.append(name)
    print("TAG_ROWS", json.dumps(type_rows, default=str))
    print("TAG_ERRORS", json.dumps(tag_errors, default=str))
    print("MISSING_TRANSFORMER_TAGS", json.dumps(missing_transformer))


def audit_finite_wrappers():
    candidate_names = {
        "X", "X_new", "x", "y", "sample_weight", "weights", "offset",
        "exposure", "entry", "start", "stop", "time", "event", "times",
        "cluster", "clusters", "strata", "subject", "subjects", "groups",
        "init", "init_coef", "initial_coef", "time_index", "entity_ids",
        "time_ids", "pvalues", "data", "values", "arrays", "scores",
        "labels", "thresholds",
    }
    missing = []
    for name, cls, estimator in public_estimators():
        for method_name in dir(cls):
            if method_name.startswith("_"):
                continue
            method = getattr(cls, method_name, None)
            if not callable(method):
                continue
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):
                continue
            relevant = sorted(set(sig.parameters) & candidate_names)
            if relevant and not getattr(method, "__statgpu_finite_validation__", False):
                missing.append((name, method_name, relevant))
    print("UNWRAPPED_NUMERIC_PUBLIC_METHODS", json.dumps(missing))


def caught_names(handler):
    typ = handler.type
    if typ is None:
        return {"bare"}
    nodes = typ.elts if isinstance(typ, ast.Tuple) else [typ]
    out = set()
    for node in nodes:
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def contains_compile_call(node):
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name) and func.id == "compile_torch":
                return True
    return False


def audit_compile_sites():
    findings = []
    for path in Path("statgpu").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "compile_torch" not in text and "suppress_errors" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "compile_torch":
                        findings.append((str(path), decorator.lineno, "decorator-factory-misuse"))
            if isinstance(node, ast.Try) and any(contains_compile_call(stmt) for stmt in node.body):
                for handler in node.handlers:
                    names = caught_names(handler)
                    if names & {"Exception", "RuntimeError", "TypeError", "AttributeError", "bare"}:
                        findings.append((str(path), node.lineno, "compile-error-swallowed", sorted(names)))
        for lineno, line in enumerate(text.splitlines(), 1):
            if "suppress_errors" in line:
                findings.append((str(path), lineno, "dynamo-suppress-errors", line.strip()))
    print("COMPILE_SITE_FINDINGS", json.dumps(findings))


def audit_runtime_edges():
    from statgpu.panel import PooledOLS
    pooled = PooledOLS()
    pooled._fitted = True
    pooled.marker_ = "must-survive-on-error"
    try:
        pooled.set_params(cov_type="invalid", kernel="PARZEN")
        print("POOLED_INVALID_SET_PARAMS_ACCEPTED", pooled.cov_type, pooled.kernel, pooled._fitted, getattr(pooled, "marker_", None), pooled.get_params(deep=False))
    except Exception as exc:
        print("POOLED_INVALID_SET_PARAMS_RAISED", type(exc).__name__, str(exc), pooled.cov_type, pooled.kernel, pooled._fitted, getattr(pooled, "marker_", None))

    from statgpu.backends._validation import check_finite
    try:
        import pandas as pd
        values = [
            pd.Series([True, pd.NA], dtype="boolean"),
            pd.Series([1, pd.NA], dtype="Int64"),
            pd.Series([1.0, pd.NA], dtype="Float64"),
        ]
        for value in values:
            try:
                check_finite(value, name="X")
                print("PANDAS_NULLABLE_MISSING_ACCEPTED", str(value.dtype))
            except Exception as exc:
                print("PANDAS_NULLABLE_MISSING_REJECTED", str(value.dtype), type(exc).__name__, str(exc))
    except ImportError:
        print("PANDAS_UNAVAILABLE")

    try:
        import os
        import statgpu.penalties._l1 as l1_module
        from statgpu.penalties import L1Penalty
        old = os.environ.get("STATGPU_TORCH_COMPILE_MODE")
        os.environ["STATGPU_TORCH_COMPILE_MODE"] = "definitely-invalid"
        l1_module._L1_PROXIMAL_TORCH_COMPILED = None
        try:
            import torch
            value = torch.tensor([1.0])
            try:
                L1Penalty(alpha=0.1).proximal(value, 0.1, backend="torch")
                print("INVALID_COMPILE_ENV_SWALLOWED")
            except Exception as exc:
                print("INVALID_COMPILE_ENV_RAISED", type(exc).__name__, str(exc))
        finally:
            if old is None:
                os.environ.pop("STATGPU_TORCH_COMPILE_MODE", None)
            else:
                os.environ["STATGPU_TORCH_COMPILE_MODE"] = old
    except ImportError:
        print("TORCH_UNAVAILABLE")


if __name__ == "__main__":
    audit_constructor_contracts()
    audit_tags()
    audit_finite_wrappers()
    audit_compile_sites()
    audit_runtime_edges()
