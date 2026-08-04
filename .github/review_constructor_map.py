from __future__ import annotations

import inspect
import json

import statgpu

rows = []
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
            rows.append(
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
print("CONSTRUCTOR_MAP", json.dumps(rows, sort_keys=True))
