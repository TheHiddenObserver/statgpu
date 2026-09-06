"""Public solver-API deprecation contract for direct penalized estimators.

The modern penalized engine uses ``solver`` as the backend-neutral direct-fit
solver selector. ``cpu_solver`` is a legacy public constructor parameter from
an older CPU/GPU split API. Keep it accepted for one compatibility cycle, but
make caller-owned legacy use visible without changing numerical behavior.

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
    """Return whether the caller explicitly supplied cpu_solver and its value."""
    try:
        bound = signature.bind_partial(self, *args, **kwargs)
    except TypeError:
        return False, None

    if "cpu_solver" in bound.arguments:
        return True, bound.arguments["cpu_solver"]

    for name, parameter in signature.parameters.items():
        if parameter.kind is not inspect.Parameter.VAR_KEYWORD:
            continue
        extra = bound.arguments.get(name, {})
        if isinstance(extra, dict) and "cpu_solver" in extra:
            return True, extra["cpu_solver"]

    return False, None


def _inside_non_user_constructor_replay() -> bool:
    """Return True for framework or statgpu-internal constructor replay."""
    frame = inspect.currentframe()
    try:
        for _ in range(20):
            if frame is None:
                return False
            frame = frame.f_back
            if frame is None:
                return False
            module_name = str(frame.f_globals.get("__name__", ""))
            function_name = frame.f_code.co_name

            # statgpu's own reconstruction contract (sklearn >=1.3 and
            # transactional set_params) should not turn a previously omitted
            # legacy default into a user-facing warning.
            if (
                module_name == "statgpu._base"
                and function_name in {"__sklearn_clone__", "set_params"}
            ):
                return True

            # sklearn <=1.2 reconstructs estimators directly from
            # get_params(deep=False) -> klass(**params), so __sklearn_clone__
            # is never on the stack.
            if (
                module_name == "sklearn.base"
                and function_name in {"clone", "_clone_parametrized"}
            ):
                return True

            # Internal statgpu algorithms may construct helper penalized
            # estimators (for example node-wise Lasso in debiased inference).
            # Those calls are implementation details, not caller-owned use of
            # the deprecated public argument.
            if module_name.startswith("statgpu.") and module_name != __name__:
                return True
        return False
    finally:
        del frame


def _install_base_docstring_contract():
    """Keep runtime help() aligned with the staged cpu_solver deprecation."""
    doc = PenalizedGeneralizedLinearModel.__doc__
    if not doc:
        return
    old = (
        "    cpu_solver : str, default='fista'\n"
        "        CPU solver: 'fista', 'fista_bb', or 'coordinate_descent'."
    )
    new = (
        "    cpu_solver : str, deprecated\n"
        "        Historical compatibility parameter from the earlier CPU/GPU "
        "split API. It no longer selects the direct-fit algorithm; use "
        "solver instead."
    )
    if old in doc:
        PenalizedGeneralizedLinearModel.__doc__ = doc.replace(old, new)


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
        explicit, value = _explicit_cpu_solver(signature, self, args, kwargs)

        # Forwarding through a typed wrapper into the shared base must not emit
        # the same warning repeatedly. Framework/internal reconstruction must
        # not convert an omitted legacy default into a user-facing warning.
        # Any caller-owned explicit value, including the historical default,
        # still receives the deprecation warning.
        if (
            depth == 0
            and explicit
            and value is not None
            and not _inside_non_user_constructor_replay()
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
    """Install the staged direct-estimator deprecation contract."""
    _install_base_docstring_contract()
    for cls in _iter_subclasses(PenalizedGeneralizedLinearModel):
        if not cls.__module__.startswith("statgpu.linear_model"):
            continue
        _install_constructor_warning(cls)


__all__ = ["install_penalized_solver_api_contract"]
