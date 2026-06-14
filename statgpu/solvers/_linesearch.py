"""Compiled step functions for FISTA and Newton solvers (torch.compile)."""

_FISTA_STEP_COMPILED = None
_NEWTON_STEP_COMPILED = None

from statgpu.backends._utils import torch_compile_supported as _torch_compile_supported


def _get_fista_step_compiled():
    global _FISTA_STEP_COMPILED
    if _FISTA_STEP_COMPILED is not None:
        return _FISTA_STEP_COMPILED
    import torch
    def _fista_step(y_k, grad, step, coef_old, coef, beta_t):
        w_tilde = y_k - step * grad
        y_k_new = coef + beta_t * (coef - coef_old)
        return w_tilde, y_k_new
    if _torch_compile_supported():
        try:
            _FISTA_STEP_COMPILED = torch.compile(_fista_step, dynamic=True, fullgraph=False)
        except RuntimeError:
            _FISTA_STEP_COMPILED = _fista_step
    else:
        _FISTA_STEP_COMPILED = _fista_step
    return _FISTA_STEP_COMPILED


def _fista_step_call(compiled_fn, *args):
    try:
        return compiled_fn(*args)
    except (RuntimeError, TypeError):
        def _fista_eager(y_k, grad, step, coef_old, coef, beta_t):
            w_tilde = y_k - step * grad
            y_k_new = coef + beta_t * (coef - coef_old)
            return w_tilde, y_k_new
        return _fista_eager(*args)


def _get_newton_step_compiled():
    global _NEWTON_STEP_COMPILED
    if _NEWTON_STEP_COMPILED is not None:
        return _NEWTON_STEP_COMPILED
    import torch
    def _newton_step(params, direction, params_old):
        params_new = params - direction
        diff_norm = torch.linalg.norm(params_new - params_old)
        return params_new, diff_norm
    if _torch_compile_supported():
        try:
            _NEWTON_STEP_COMPILED = torch.compile(_newton_step, dynamic=True, fullgraph=False)
        except RuntimeError:
            _NEWTON_STEP_COMPILED = _newton_step
    else:
        _NEWTON_STEP_COMPILED = _newton_step
    return _NEWTON_STEP_COMPILED


def _newton_step_call(compiled_fn, *args):
    try:
        return compiled_fn(*args)
    except (RuntimeError, TypeError):
        def _newton_eager(params, direction, params_old):
            params_new = params - direction
            diff_norm = torch.linalg.norm(params_new - params_old)
            return params_new, diff_norm
        return _newton_eager(*args)
