"""Public solver-API deprecation contract for direct penalized estimators.

The modern penalized engine uses ``solver`` as the backend-neutral direct-fit
solver selector.  ``cpu_solver`` is a legacy public constructor parameter from
an older CPU/GPU split API.  Keep it accepted for one compatibility cycle, but
make meaningful legacy use visible without changing numerical behavior.

LassoCV has a separate migration because its legacy ``cpu_solver`` really did
control the CPU cross-validation path; that API now uses ``cv_solver``.
"""

from __future__ import annotations

import functools
import inspect
import warnings

from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel


_CONSTRUCTOR_DEPTH_ATTR = "_statgpu_penalized_solver_deprecation_depth"
_WRAPPER_MARKER = "__statgpu_penalized_solver_deprecation__"


def _iter_subclasses(cls):
    seen = set()
    stack = [cls]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        yield current
        stack.extend(current.__subclasses__())


def _explicit_cpu_solver(signature, self, args, kwargs):
    """Return (explicit, value, default) for caller-owned cpu_solver input."""
    try:
        bound = signature.bind_partial(self, *args, **kwargs)
    except TypeError:
        return False, None, inspect.Parameter.empty

    if "cpu_solver" in bound.arguments:
        parameter = signature.parameters.get("cpu_solver")
        default = (
            inspect.Parameter.empty
            if parameter is None
            else parameter.default
        )
        return True, bound.arguments["cpu_solver"], default

    for name, parameter in signature.parameters.items():
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD:
            continue
        extra = bound.arguments.get(name, {})
        if isinstance(extra, dict) and "cpu_solver" in extra:
            return True, extra["cpu_solver"], inspect.Parameter.empty

    return False, None, inspect.Parameter.empty


def _install_constructor_warning(cls):
    original = cls.__dict__.get("__init__")
    if original is None or getattr(original, _WRAPPER_MARKER, False):
        return
    if cls.__module__.startswith("statgpu.linear_model.legacy"):
        return

    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        depth = int(getattr(self, _CONSTRUCTOR_DEPTH_ATTR, 0))
        explicit, value, default = _explicit_cpu_solver(
            signature, self, args, kwargs
        )

        # Forwarding through a typed wrapper into the shared base must not emit
        # the same warning repeatedly.  Also avoid warning for a constructor's
        # unchanged legacy default: sklearn clone/set_params reconstruct an
        # estimator with every constructor parameter explicitly supplied.
        if (
            depth == 0
            and explicit
            and value is not None
            and (default is inspect.Parameter.empty or value != default)
        ):
            warnings.warn(
                f"{cls.__name__}(cpu_solver=...) is deprecated for direct "
                "penalized estimators. cpu_solver does not select the direct "
                "fit algorithm in the unified solver engine; use solver=... "
                "instead. cpu_solver will be removed in a future breaking "
                "release.",
                FutureWarning,
                stacklevel=2,
            )

        setattr(self, _CONSTRUCTOR_DEPTH_ATTR, depth + 1)
        try:
            return original(self, *args, **kwargs)
        finally:
            if depth == 0:
                self.__dict__.pop(_CONSTRUCTOR_DEPTH_ATTR, None)
            else:
                setattr(self, _CONSTRUCTOR_DEPTH_ATTR, depth)

    setattr(wrapped, _WRAPPER_MARKER, True)
    cls.__init__ = wrapped


def install_penalized_solver_api_contract():
    """Install the direct-estimator deprecation wrapper once per public class."""
    for cls in _iter_subclasses(PenalizedGeneralizedLinearModel):
        if not cls.__module__.startswith("statgpu.linear_model"):
            continue
        _install_constructor_warning(cls)


__all__ = ["install_penalized_solver_api_contract"]
