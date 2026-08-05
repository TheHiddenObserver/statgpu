from pathlib import Path


def replace_once(path, old, new):
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path, marker, addition):
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if marker in text:
        return
    file_path.write_text(text.rstrip() + "\n\n" + addition.strip() + "\n", encoding="utf-8")


# Shared solver validation/error-classification helpers.
replace_once(
    "statgpu/solvers/_utils.py",
    '''def _scalar_bool(value):
    return bool(value.item() if hasattr(value, "item") else value)


def _native_sample_weight(sample_weight):
''',
    '''def _scalar_bool(value):
    return bool(value.item() if hasattr(value, "item") else value)


def _runtime_error_is_singular(exc):
    """Return whether a backend RuntimeError reports a solvable rank failure."""
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "singular",
            "not invertible",
            "zero pivot",
            "rank deficient",
            "ill-conditioned",
            "not positive-definite",
            "not positive definite",
        )
    )


def _native_sample_weight(sample_weight):
''',
)
replace_once(
    "statgpu/solvers/_utils.py",
    '''        finite = xp.all(xp.isfinite(values))
        negative = xp.any(values < 0)
        total = float(xp.sum(values).item() if hasattr(xp.sum(values), "item") else xp.sum(values))
''',
    '''        finite = xp.all(xp.isfinite(values))
        negative = xp.any(values < 0)
        total_dev = xp.sum(values)
        total = float(total_dev.item() if hasattr(total_dev, "item") else total_dev)
''',
)

# FISTA-BB must validate direct-solver weights before conversion/Lipschitz work.
replace_once(
    "statgpu/solvers/_fista_bb.py",
    '''    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]
    _pen_name = _penalty_name(penalty)

    # Convert sample_weight to backend-native (prevent CPU/CUDA mismatch)
''',
    '''    X_proc, y_proc = loss.preprocess(X, y)
    _validate_sample_weight(sample_weight, X_proc.shape[0])
    n_features = X_proc.shape[1]
    _pen_name = _penalty_name(penalty)

    # Convert sample_weight to backend-native (prevent CPU/CUDA mismatch)
''',
)
replace_once(
    "statgpu/solvers/_fista_bb.py",
    '''    step_max = step_L * step_max_factor
    step_min = step_L * step_min_factor
    _validate_sample_weight(sample_weight, X_proc.shape[0])

    # Gradient at initial point for first BB difference
''',
    '''    step_max = step_L * step_max_factor
    step_min = step_L * step_min_factor

    # Gradient at initial point for first BB difference
''',
)

# Newton: validate before Hessian work and only downgrade true rank failures.
replace_once(
    "statgpu/solvers/_newton.py",
    '''    _smooth_penalty_value_dev,
)
''',
    '''    _smooth_penalty_value_dev,
    _runtime_error_is_singular,
)
''',
)
replace_once(
    "statgpu/solvers/_newton.py",
    '''    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]

    if init_coef is not None:
''',
    '''    X_proc, y_proc = loss.preprocess(X, y)
    _validate_uniform_sample_weight(sample_weight, X_proc.shape[0], "newton_solver")
    n_features = X_proc.shape[1]

    if init_coef is not None:
''',
)
replace_once(
    "statgpu/solvers/_newton.py",
    '''    _validate_uniform_sample_weight(sample_weight, X_proc.shape[0], "newton_solver")
    iteration = -1
''',
    '''    iteration = -1
''',
)
old_newton_solve = '''        try:
            if backend == "numpy":
                direction = np.linalg.solve(hess_reg, grad)
            elif backend == "cupy":
                import cupy as cp

                direction = cp.linalg.solve(hess_reg, grad)
            else:
                import torch

                direction = torch.linalg.solve(hess_reg, grad.unsqueeze(1))
                direction = direction.squeeze(1)
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            if backend == "numpy":
                direction = np.linalg.lstsq(hess_reg, grad, rcond=None)[0]
            elif backend == "cupy":
                import cupy as cp

                direction = cp.linalg.lstsq(hess_reg, grad)[0]
            else:
                import torch

                direction = torch.linalg.lstsq(hess_reg, grad.unsqueeze(1)).solution
                direction = direction.squeeze(1)
'''
new_newton_solve = '''        try:
            if backend == "numpy":
                direction = np.linalg.solve(hess_reg, grad)
            elif backend == "cupy":
                import cupy as cp

                direction = cp.linalg.solve(hess_reg, grad)
            else:
                import torch

                direction = torch.linalg.solve(hess_reg, grad.unsqueeze(1))
                direction = direction.squeeze(1)
        except np.linalg.LinAlgError:
            if backend == "numpy":
                direction = np.linalg.lstsq(hess_reg, grad, rcond=None)[0]
            elif backend == "cupy":
                import cupy as cp

                direction = cp.linalg.lstsq(hess_reg, grad)[0]
            else:
                import torch

                direction = torch.linalg.lstsq(hess_reg, grad.unsqueeze(1)).solution
                direction = direction.squeeze(1)
        except RuntimeError as exc:
            if not _runtime_error_is_singular(exc):
                raise
            if backend == "torch":
                import torch

                direction = torch.linalg.lstsq(hess_reg, grad.unsqueeze(1)).solution
                direction = direction.squeeze(1)
            elif backend == "cupy":
                import cupy as cp

                direction = cp.linalg.lstsq(hess_reg, grad)[0]
            else:
                direction = np.linalg.lstsq(hess_reg, grad, rcond=None)[0]
'''
replace_once("statgpu/solvers/_newton.py", old_newton_solve, new_newton_solve)

# Proximal Newton: shared validation, dtype-consistent ridge, narrow fallbacks.
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''from ._utils import _smooth_penalty_gradient, _smooth_penalty_hessian
''',
    '''from ._utils import (
    _runtime_error_is_singular,
    _smooth_penalty_gradient,
    _smooth_penalty_hessian,
    _validate_sample_weight,
)
''',
)
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]

    if init_coef is not None:
''',
    '''    X_proc, y_proc = loss.preprocess(X, y)
    _validate_sample_weight(sample_weight, X_proc.shape[0])
    n_features = X_proc.shape[1]

    if init_coef is not None:
''',
)
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''    # Pre-allocate ridge matrix (reused every iteration)
    _n = n_features
    if backend == "numpy":
        _ridge = 1e-10 * np.eye(_n, dtype=np.float64)
    elif backend == "cupy":
        import cupy as cp
        _ridge = 1e-10 * cp.eye(_n, dtype=cp.float64)
    else:
        import torch
        _ridge = 1e-10 * torch.eye(_n, dtype=torch.float64,
                                    device=params.device if hasattr(params, 'device') else 'cpu')
''',
    '''    # Pre-allocate a dtype/device-compatible ridge matrix.
    _n = n_features
    _dtype = getattr(params, "dtype", np.float64)
    if backend == "numpy":
        _ridge = 1e-10 * np.eye(_n, dtype=_dtype)
    elif backend == "cupy":
        import cupy as cp
        _ridge = 1e-10 * cp.eye(_n, dtype=_dtype)
    else:
        import torch
        _ridge = 1e-10 * torch.eye(
            _n,
            dtype=_dtype,
            device=params.device if hasattr(params, "device") else "cpu",
        )
''',
)
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''        except (np.linalg.LinAlgError, ValueError) as e:
            # Fallback to gradient descent if Hessian is singular/ill-conditioned
            direction = grad
        except RuntimeError as e:
            # Only catch singular/ill-conditioned errors, re-raise others (OOM, device mismatch, etc.)
            err_msg = str(e).lower()
            if "singular" in err_msg or "ill-conditioned" in err_msg or "not invertible" in err_msg:
                direction = grad
            else:
                raise
''',
    '''        except np.linalg.LinAlgError:
            # Fall back only for a true singular/ill-conditioned Hessian.
            direction = grad
        except RuntimeError as exc:
            if not _runtime_error_is_singular(exc):
                raise
            direction = grad
''',
)
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''            except (ValueError, FloatingPointError):
                pass
            except RuntimeError as e:
                # Only swallow numerical errors, re-raise infrastructure bugs
                err_msg = str(e).lower()
                if any(kw in err_msg for kw in ("singular", "ill-conditioned",
                        "not invertible", "overflow", "invalid value", "nan")):
                    pass
                else:
                    raise
                pass
''',
    '''            except FloatingPointError:
                pass
            except RuntimeError as exc:
                # Only swallow trial-point numerical failures; infrastructure
                # and device errors remain visible to the caller.
                err_msg = str(exc).lower()
                if not any(
                    marker in err_msg
                    for marker in ("overflow", "invalid value", "nan")
                ):
                    raise
''',
)

# ADMM: validate before initialization/curvature and narrow Cholesky fallback.
replace_once(
    "statgpu/solvers/_admm.py",
    '''    _nesterov_momentum,
    _validate_uniform_sample_weight,
)
''',
    '''    _nesterov_momentum,
    _runtime_error_is_singular,
    _validate_uniform_sample_weight,
)
''',
)
replace_once(
    "statgpu/solvers/_admm.py",
    '''    X_proc, y_proc = loss.preprocess(X, y)
    n_features = X_proc.shape[1]

    # Initialize
''',
    '''    X_proc, y_proc = loss.preprocess(X, y)
    _validate_uniform_sample_weight(sample_weight, X_proc.shape[0], "admm_solver")
    n_features = X_proc.shape[1]

    # Initialize
''',
)
replace_once(
    "statgpu/solvers/_admm.py",
    '''    if sample_weight is not None:
        _validate_uniform_sample_weight(sample_weight, X_proc.shape[0], "admm_solver")

    def _grad_w(w_vec, z_cur, u_cur):
''',
    '''    def _grad_w(w_vec, z_cur, u_cur):
''',
)
replace_once(
    "statgpu/solvers/_admm.py",
    '''            except (np.linalg.LinAlgError, ValueError, RuntimeError):
                # Matrix not positive-definite (numerical issues, collinear features)
                # Fall back to CG solver below
                _cholesky_ok = False
''',
    '''            except np.linalg.LinAlgError:
                # A genuinely non-positive-definite system may use the
                # iterative fallback below.
                _cholesky_ok = False
            except RuntimeError as exc:
                if not _runtime_error_is_singular(exc):
                    raise
                _cholesky_ok = False
''',
)

# Correct steepest-descent Armijo slopes in both L-BFGS variants.
replace_once(
    "statgpu/solvers/_lbfgs.py",
    '''        if gdd >= 0:
            direction = -grad
            gdd = -gn  # -||grad||^2
''',
    '''        if gdd >= 0:
            direction = -grad
            gdd = -gn * gn  # grad'(-grad) = -||grad||^2
''',
)

# L-BFGS-B: backend-native bounds/projection and correct projected fallback.
replace_once(
    "statgpu/solvers/_lbfgs_b.py",
    '''    # Initialize bounds
    if backend == "torch":
        import torch
        _neg_inf = torch.full((n_features,), float("-inf"), dtype=torch.float64, device=params.device)
        _pos_inf = torch.full((n_features,), float("inf"), dtype=torch.float64, device=params.device)
    else:
        _neg_inf = np.full(n_features, float("-inf"))
        _pos_inf = np.full(n_features, float("inf"))

    lb = _neg_inf if lower_bounds is None else (
        lower_bounds if hasattr(lower_bounds, "shape") else np.array(lower_bounds)
    )
    ub = _pos_inf if upper_bounds is None else (
        upper_bounds if hasattr(upper_bounds, "shape") else np.array(upper_bounds)
    )
''',
    '''    # Initialize bounds on the same backend/device/dtype as params.
    if backend == "torch":
        import torch

        _neg_inf = torch.full(
            (n_features,), float("-inf"), dtype=params.dtype, device=params.device
        )
        _pos_inf = torch.full(
            (n_features,), float("inf"), dtype=params.dtype, device=params.device
        )
        _as_bound = lambda value: torch.as_tensor(
            value, dtype=params.dtype, device=params.device
        )
    elif backend == "cupy":
        import cupy as cp

        _neg_inf = cp.full((n_features,), float("-inf"), dtype=params.dtype)
        _pos_inf = cp.full((n_features,), float("inf"), dtype=params.dtype)
        _as_bound = lambda value: cp.asarray(value, dtype=params.dtype)
    else:
        _neg_inf = np.full(n_features, float("-inf"), dtype=params.dtype)
        _pos_inf = np.full(n_features, float("inf"), dtype=params.dtype)
        _as_bound = lambda value: np.asarray(value, dtype=params.dtype)

    lb = _neg_inf if lower_bounds is None else _as_bound(lower_bounds)
    ub = _pos_inf if upper_bounds is None else _as_bound(upper_bounds)
    if lb.shape != params.shape or ub.shape != params.shape:
        raise ValueError("lower_bounds and upper_bounds must match coefficient shape")
    if _device_gt(lb, ub):
        raise ValueError("lower_bounds must not exceed upper_bounds")
''',
)
# The preceding scalar helper cannot compare vectors; replace with backend-safe reduction.
replace_once(
    "statgpu/solvers/_lbfgs_b.py",
    '''    if _device_gt(lb, ub):
        raise ValueError("lower_bounds must not exceed upper_bounds")
''',
    '''    if backend == "torch":
        invalid_bounds = bool((lb > ub).any().item())
    elif backend == "cupy":
        invalid_bounds = bool((lb > ub).any().item())
    else:
        invalid_bounds = bool(np.any(lb > ub))
    if invalid_bounds:
        raise ValueError("lower_bounds must not exceed upper_bounds")
''',
)
replace_once(
    "statgpu/solvers/_lbfgs_b.py",
    '''        proj_grad = _projected_gradient(grad, params, lb, ub)
''',
    '''        proj_grad = _projected_gradient(grad, params, lb, ub, backend)
''',
)
replace_once(
    "statgpu/solvers/_lbfgs_b.py",
    '''        if gdd >= 0:
            direction = -grad
            gdd = -_norm2_dev(grad)
            gdd = float(gdd) if not hasattr(gdd, "item") else float(gdd.item())
''',
    '''        if gdd >= 0:
            direction = -proj_grad
            gdd = -pg_norm * pg_norm
''',
)
replace_once(
    "statgpu/solvers/_lbfgs_b.py",
    '''def _clip_to_bounds(params, lb, ub, backend):
    """Clip parameters to [lb, ub]. Works on all backends."""
    if backend == "torch":
        import torch
        return torch.clamp(params, min=lb, max=ub)
    else:
        xp = np
        return xp.maximum(xp.minimum(params, ub), lb)


def _projected_gradient(grad, params, lb, ub):
''',
    '''def _clip_to_bounds(params, lb, ub, backend):
    """Clip parameters to [lb, ub] on their current backend."""
    if backend == "torch":
        import torch
        return torch.maximum(torch.minimum(params, ub), lb)
    if backend == "cupy":
        import cupy as cp
        return cp.maximum(cp.minimum(params, ub), lb)
    return np.maximum(np.minimum(params, ub), lb)


def _projected_gradient(grad, params, lb, ub, backend):
''',
)
replace_once(
    "statgpu/solvers/_lbfgs_b.py",
    '''    backend = "torch" if hasattr(params, "device") else "numpy"
    if backend == "torch":
        import torch
        at_lower = (params <= lb) & (grad > 0)
        at_upper = (params >= ub) & (grad < 0)
        at_bound = at_lower | at_upper
        return grad * (~at_bound).to(grad.dtype)
    else:
        at_lower = (params <= lb) & (grad > 0)
        at_upper = (params >= ub) & (grad < 0)
        at_bound = at_lower | at_upper
        mask = (~at_bound).astype(grad.dtype)
        return grad * mask
''',
    '''    at_lower = (params <= lb) & (grad > 0)
    at_upper = (params >= ub) & (grad < 0)
    at_bound = at_lower | at_upper
    if backend == "torch":
        return grad * (~at_bound).to(grad.dtype)
    return grad * (~at_bound).astype(grad.dtype)
''',
)


tests = r'''
# PR87_REVIEW_FIX_V38
def test_solver_weight_reduction_is_computed_once():
    from pathlib import Path

    source = Path("statgpu/solvers/_utils.py").read_text(encoding="utf-8")
    block = source.split("def _validated_sample_weight", 1)[1].split(
        "def _validate_uniform_sample_weight", 1
    )[0]
    assert block.count("xp.sum(values)") == 1


def test_direct_fista_bb_validates_weight_length_before_lipschitz():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import fista_bb_solver

    class GuardedSquaredError(SquaredErrorLoss):
        def lipschitz(self, *args, **kwargs):
            raise AssertionError("lipschitz must not run before weight validation")

    X = np.ones((3, 1), dtype=np.float64)
    y = np.arange(3.0)
    with pytest.raises(ValueError, match="length n_samples"):
        fista_bb_solver(
            GuardedSquaredError(),
            get_penalty("l1", alpha=0.1),
            X,
            y,
            sample_weight=np.ones(2),
        )


def test_newton_does_not_mask_non_singular_solve_errors(monkeypatch):
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import newton_solver

    X = np.column_stack([np.ones(4), np.arange(4.0)])
    y = np.arange(4.0)

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    def forbidden(*args, **kwargs):
        raise AssertionError("lstsq must not mask infrastructure failures")

    monkeypatch.setattr(np.linalg, "solve", oom)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    with pytest.raises(RuntimeError, match="out of memory"):
        newton_solver(
            SquaredErrorLoss(), get_penalty("l2", alpha=0.1), X, y, max_iter=2
        )


def test_newton_validates_weights_before_constant_hessian():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import newton_solver

    class GuardedSquaredError(SquaredErrorLoss):
        def hessian(self, *args, **kwargs):
            raise AssertionError("hessian must not run before weight validation")

    with pytest.raises(ValueError, match="length n_samples"):
        newton_solver(
            GuardedSquaredError(),
            get_penalty("l2", alpha=0.1),
            np.ones((3, 1)),
            np.ones(3),
            sample_weight=np.ones(2),
        )


def test_admm_does_not_mask_non_singular_cholesky_errors(monkeypatch):
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import admm_solver

    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(np.linalg, "cholesky", oom)
    with pytest.raises(RuntimeError, match="out of memory"):
        admm_solver(
            SquaredErrorLoss(),
            get_penalty("l1", alpha=0.1),
            np.ones((4, 1)),
            np.arange(4.0),
            max_iter=2,
        )


def test_proximal_newton_validates_weight_length_before_curvature():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    class GuardedSquaredError(SquaredErrorLoss):
        def fused_gradient_and_hessian(self, *args, **kwargs):
            raise AssertionError("curvature must not run before weight validation")

    with pytest.raises(ValueError, match="length n_samples"):
        proximal_newton_solver(
            GuardedSquaredError(),
            get_penalty("l1", alpha=0.1),
            np.ones((3, 1)),
            np.ones(3),
            sample_weight=np.ones(2),
        )


def test_proximal_newton_preserves_torch_float32_dtype():
    torch = pytest.importorskip("torch")
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    coef, _ = proximal_newton_solver(
        SquaredErrorLoss(),
        get_penalty("l1", alpha=0.01),
        X,
        y,
        max_iter=3,
    )
    assert coef.dtype == torch.float32
    assert bool(torch.all(torch.isfinite(coef)).item())


def test_lbfgs_steepest_descent_uses_squared_norm_slope():
    from pathlib import Path

    lbfgs = Path("statgpu/solvers/_lbfgs.py").read_text(encoding="utf-8")
    lbfgsb = Path("statgpu/solvers/_lbfgs_b.py").read_text(encoding="utf-8")
    assert "gdd = -gn * gn" in lbfgs
    assert "direction = -proj_grad" in lbfgsb
    assert "gdd = -pg_norm * pg_norm" in lbfgsb


def test_lbfgsb_torch_bounds_and_projection_are_backend_native():
    torch = pytest.importorskip("torch")
    from statgpu.solvers._lbfgs_b import _clip_to_bounds, _projected_gradient

    params = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32)
    lb = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32)
    ub = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)
    clipped = _clip_to_bounds(params, lb, ub, "torch")
    torch.testing.assert_close(clipped, torch.tensor([-1.0, 0.5, 2.0]))
    grad = torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32)
    projected = _projected_gradient(grad, clipped, lb, ub, "torch")
    torch.testing.assert_close(projected, torch.tensor([0.0, -1.0, 0.0]))


def test_lbfgsb_cupy_bounds_and_projection_are_backend_native():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.solvers._lbfgs_b import _clip_to_bounds, _projected_gradient

    params = cp.asarray([-2.0, 0.5, 3.0], dtype=cp.float32)
    lb = cp.asarray([-1.0, 0.0, 0.0], dtype=cp.float32)
    ub = cp.asarray([1.0, 1.0, 2.0], dtype=cp.float32)
    clipped = _clip_to_bounds(params, lb, ub, "cupy")
    cp.testing.assert_allclose(clipped, cp.asarray([-1.0, 0.5, 2.0]))
    grad = cp.asarray([1.0, -1.0, -1.0], dtype=cp.float32)
    projected = _projected_gradient(grad, clipped, lb, ub, "cupy")
    cp.testing.assert_allclose(projected, cp.asarray([0.0, -1.0, 0.0]))
'''
append_once("dev/tests/test_maintenance_024_025.py", "# PR87_REVIEW_FIX_V38", tests)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Hardened adjacent Newton, proximal-Newton, ADMM, FISTA-BB, L-BFGS, "
    "and L-BFGS-B contracts: validate weights before curvature work, only "
    "downgrade true singular systems, preserve dtype/device for proximal "
    "Newton and CuPy bounds, and use the correct squared-gradient Armijo slope.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- Adjacent Newton, proximal-Newton, ADMM, FISTA-BB, L-BFGS, and "
    "L-BFGS-B paths now validate weights before curvature work, narrow "
    "singular-system fallbacks, preserve dtype/device for proximal Newton and "
    "CuPy bounds, and use the correct squared-gradient Armijo slope.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- 相邻的 Newton、proximal-Newton、ADMM、FISTA-BB、L-BFGS 与 "
    "L-BFGS-B 路径现在会在曲率计算前校验权重，仅对真正的奇异系统降级，"
    "保持 proximal Newton 与 CuPy bounds 的 dtype/device，并采用正确的"
    "梯度平方 Armijo 斜率。\n",
)
