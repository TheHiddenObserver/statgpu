from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import statgpu

runtime_rows = []
for name in statgpu.__all__:
    cls = getattr(statgpu, name, None)
    if not inspect.isclass(cls) or not hasattr(cls, "fit") or inspect.isabstract(cls):
        continue
    signature = inspect.signature(cls)
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect._empty
        and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ]
    if required:
        continue
    try:
        estimator = cls()
    except Exception:
        continue
    for parameter, value in estimator.get_params(deep=False).items():
        if not hasattr(estimator, parameter) or getattr(estimator, parameter) is not value:
            runtime_rows.append(
                {
                    "estimator": name,
                    "module": cls.__module__,
                    "parameter": parameter,
                    "public_attribute": hasattr(estimator, parameter),
                    "runtime_type": None
                    if not hasattr(estimator, parameter)
                    else type(getattr(estimator, parameter)).__name__,
                    "raw_type": type(value).__name__,
                }
            )
print("CONSTRUCTOR_MAP", json.dumps(runtime_rows, sort_keys=True))


def is_direct_parameter(expr, parameter):
    return isinstance(expr, ast.Name) and expr.id == parameter


def target_attribute(target):
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return target.attr
    return None


static_rows = []
for path in Path("statgpu").rglob("*.py"):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for class_node in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        init = next(
            (
                node
                for node in class_node.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "__init__"
            ),
            None,
        )
        if init is None:
            continue
        parameters = {
            arg.arg
            for arg in (
                list(init.args.posonlyargs)
                + list(init.args.args)
                + list(init.args.kwonlyargs)
            )
            if arg.arg != "self"
        }
        assignments = {}
        for node in ast.walk(init):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    attribute = target_attribute(target)
                    if attribute is not None:
                        assignments.setdefault(attribute, []).append((node.value, node.lineno))
            elif isinstance(node, ast.AnnAssign):
                attribute = target_attribute(node.target)
                if attribute is not None and node.value is not None:
                    assignments.setdefault(attribute, []).append((node.value, node.lineno))

        for parameter in sorted(parameters):
            public_assignments = assignments.get(parameter, [])
            if not public_assignments:
                # Superclass-owned common parameters are expected to be absent.
                if parameter not in {"device", "n_jobs"}:
                    static_rows.append(
                        {
                            "path": path.as_posix(),
                            "class": class_node.name,
                            "parameter": parameter,
                            "kind": "missing-public-assignment",
                            "line": init.lineno,
                        }
                    )
                continue
            for expression, lineno in public_assignments:
                if not is_direct_parameter(expression, parameter):
                    static_rows.append(
                        {
                            "path": path.as_posix(),
                            "class": class_node.name,
                            "parameter": parameter,
                            "kind": "transformed-public-assignment",
                            "line": lineno,
                            "expression": ast.unparse(expression),
                        }
                    )
print("CONSTRUCTOR_STATIC_MAP", json.dumps(static_rows, sort_keys=True))
