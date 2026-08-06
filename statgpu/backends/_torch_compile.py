"""Safe, centralized policy for internal :func:`torch.compile` use.

statgpu iterative solvers reuse tensors across calls. PyTorch's
``reduce-overhead`` mode enables CUDA Graphs and can therefore expose
overwritten-output lifecycle errors on PyTorch 2.1 and newer. Internal
kernels therefore remain eager by default. Users can explicitly opt in
to ``default`` or ``reduce-overhead`` compilation through
``STATGPU_TORCH_COMPILE_MODE``.
"""

from __future__ import annotations

import functools
import os
from collections import deque
import warnings
from typing import Callable, Optional

_ENV_NAME = "STATGPU_TORCH_COMPILE_MODE"
_ALLOWED_MODES = frozenset({"auto", "default", "reduce-overhead", "disable"})
_COMPILE_DIAGNOSTICS = deque(maxlen=256)


def resolve_torch_compile_mode(
    *,
    workload: str = "general",
    requested_mode: Optional[str] = None,
) -> Optional[str]:
    """Resolve an explicitly opted-in mode for a statgpu callable.

    ``auto`` is intentionally eager. ``workload`` and ``requested_mode``
    remain part of the internal call-site contract, but neither can
    silently enable compilation when the environment is unset or set to
    ``auto``.
    """
    configured = os.environ.get(_ENV_NAME, "auto").strip().lower()
    if configured not in _ALLOWED_MODES:
        allowed = ", ".join(sorted(_ALLOWED_MODES))
        raise ValueError(
            f"{_ENV_NAME} must be one of {allowed}; got {configured!r}"
        )
    if configured in {"auto", "disable"}:
        return None
    return configured


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
