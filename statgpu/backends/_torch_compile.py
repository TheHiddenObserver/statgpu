"""Safe, centralized policy for internal :func:`torch.compile` use.

statgpu iterative solvers reuse tensors across calls.  PyTorch's
``reduce-overhead`` mode enables CUDA Graphs and can therefore expose
overwritten-output lifecycle errors on PyTorch 2.1 and newer.  Internal
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
_CUDAGRAPH_RUNTIME_MARKERS = (
    "CUDAGraphs",
    "cudagraph",
    "overwritten by a subsequent run",
)


def resolve_torch_compile_mode(
    *,
    workload: str = "general",
    requested_mode: Optional[str] = None,
) -> Optional[str]:
    """Resolve the mode for a statgpu-owned compiled callable.

    ``None`` means eager execution.  ``auto`` selects ``default`` for
    iterative workloads because they retain and reuse tensors between
    calls; other workloads preserve an explicitly requested safe mode
    and otherwise use ``default``.
    """
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


def _is_cudagraph_lifecycle_error(exc: BaseException) -> bool:
    message = str(exc)
    return any(marker.lower() in message.lower() for marker in _CUDAGRAPH_RUNTIME_MARKERS)


def compile_torch(
    fn: Callable,
    *,
    workload: str = "general",
    mode: Optional[str] = None,
    **compile_kwargs,
) -> Callable:
    """Compile ``fn`` under the statgpu policy, with eager fallback.

    Construction failures retain the historical eager fallback.  A
    known CUDA Graph output-lifecycle failure at invocation time also
    disables the compiled callable permanently for that function.  All
    unrelated runtime errors are re-raised.
    """
    resolved_mode = resolve_torch_compile_mode(
        workload=workload,
        requested_mode=mode,
    )
    if resolved_mode is None or not torch_compile_available():
        return fn

    try:
        import torch
        compiled = torch.compile(fn, mode=resolved_mode, **compile_kwargs)
    except Exception:
        return fn

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
            warnings.warn(
                "torch.compile CUDA Graph lifecycle failure; "
                "falling back to eager execution for this statgpu kernel",
                RuntimeWarning,
                stacklevel=2,
            )
            return fn(*args, **kwargs)

    guarded.__statgpu_compile_mode__ = resolved_mode
    guarded.__statgpu_compile_workload__ = workload
    return guarded
