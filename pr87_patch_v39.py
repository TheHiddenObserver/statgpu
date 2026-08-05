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


# ADMM: initialize the iterative fallback after a legitimate Cholesky failure.
replace_once(
    "statgpu/solvers/_admm.py",
    '''        if not _cholesky_ok:
            use_cholesky = False

        # Precompute -grad_f(0) = Xty/n for squared_error (the constant part)
        _zero_coef = _zeros_like(w)
        _neg_grad_zero = -loss.gradient(X_proc, y_proc, _zero_coef, sample_weight=sample_weight)  # Xty/n

    else:
        # Gradient descent step: 1/(L_f + rho)
        L_f = loss.lipschitz(X_proc, w, y=y_proc)
        if L_f <= 0:
            L_f = 1.0
        lr_sub = 1.0 / (L_f + rho + 1e-8)
''',
    '''        if not _cholesky_ok:
            use_cholesky = False

        if use_cholesky:
            # Precompute -grad_f(0) = Xty/n for squared_error.
            _zero_coef = _zeros_like(w)
            _neg_grad_zero = -loss.gradient(
                X_proc,
                y_proc,
                _zero_coef,
                sample_weight=sample_weight,
            )

    if not use_cholesky:
        # Gradient descent step: 1/(L_f + rho). This must also be
        # initialized when a requested Cholesky path legitimately falls back.
        L_f = loss.lipschitz(X_proc, w, y=y_proc, sample_weight=sample_weight)
        if L_f <= 0:
            L_f = 1.0
        lr_sub = 1.0 / (L_f + rho + 1e-8)
''',
)

# Proximal Newton: do not optimize a duplicated/wrong composite objective.
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''Solves: min f(x) + g(x)
where f is smooth (loss) and g is non-smooth (penalty).

Algorithm:
1. Compute Newton direction: d = -H^-1 @ (grad_f + prox_grad_g)
2. Line search: find step that decreases f(x + step*d) + g(x + step*d)
3. Update: x = x + step * d

Much faster than FISTA for problems where:
- f has a Hessian (Huber, Bisquare, Fair, CoxPH)
- g is non-smooth but has a proximal operator (L1, SCAD/MCP via LLA)

Typical convergence: 5-10 iterations vs 300+ for FISTA.
''',
    '''Solves smooth loss plus a smooth penalty with Newton updates.

A general non-smooth proximal-Newton step requires solving the Hessian-metric
proximal subproblem. The historical Euclidean-prox approximation optimized a
different objective (and double-counted L2/ElasticNet curvature). Until a
metric proximal subproblem solver is implemented, non-smooth penalties are
explicitly delegated to the backend-native FISTA solver with a warning.
''',
)
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    _validate_sample_weight(sample_weight, X_proc.shape[0])
    n_features = X_proc.shape[1]
''',
    '''    _pen_name = str(getattr(penalty, "name", "none")).lower()
    _is_smooth_pen = _pen_name in ("l2", "none", "null", "")
    if not _is_smooth_pen:
        warnings.warn(
            "proximal_newton_solver delegates non-smooth penalties to "
            "fista_solver because the Hessian-metric proximal subproblem is "
            "not implemented; this preserves the declared objective.",
            RuntimeWarning,
            stacklevel=2,
        )
        from ._fista import fista_solver

        return fista_solver(
            loss,
            penalty,
            X,
            y,
            max_iter=max_iter,
            tol=tol,
            init_coef=init_coef,
            sample_weight=sample_weight,
        )

    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    _validate_sample_weight(sample_weight, X_proc.shape[0])
    n_features = X_proc.shape[1]
''',
)
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''    # Check if loss supports fused gradient+hessian
    _has_fused = hasattr(loss, 'fused_gradient_and_hessian')

    for iteration in range(max_iter):
''',
    '''    # Check if loss supports fused gradient+hessian
    _has_fused = hasattr(loss, 'fused_gradient_and_hessian')
    iteration = -1  # max_iter=0 returns the initialized coefficient vector

    for iteration in range(max_iter):
''',
)
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''        # Add smooth penalty gradient/hessian only for smooth penalties.
        # Non-smooth penalties (L1, AdaptiveL1) are handled by proximal operator.
        _pen_name = getattr(penalty, 'name', '')
        _is_smooth_pen = _pen_name in ('l2', 'none', 'null', '', 'elasticnet')
        if _is_smooth_pen:
            grad = loss_grad + _smooth_penalty_gradient(penalty, params)
            hess = loss_hess + _smooth_penalty_hessian(penalty, params)
        else:
            grad = loss_grad
            hess = loss_hess
''',
    '''        # Only smooth penalties reach this path. Their gradient and
        # curvature are included exactly once in the Newton system.
        grad = loss_grad + _smooth_penalty_gradient(penalty, params)
        hess = loss_hess + _smooth_penalty_hessian(penalty, params)
''',
)
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''            # Apply proximal operator (handles non-smooth penalty)
            if hasattr(penalty, 'proximal'):
                # For weighted L1 (AdaptiveL1 from LLA): proximal is soft-threshold
                params_try = penalty.proximal(params_try, step, backend=backend)

            try:
''',
    '''            # Smooth penalty terms are already represented in the Newton
            # direction; applying their proximal operator here would count the
            # same penalty a second time.
            try:
''',
)

# FISTA-LLA: disable the incorrect Euclidean-prox Newton shortcut unless a
# loss explicitly opts into a future, correct Hessian-metric implementation.
replace_once(
    "statgpu/solvers/_fista_lla.py",
    '''        _has_hessian = (
            getattr(loss, 'has_hessian', False)
            and not _is_quadratic
            and getattr(loss, 'name', '') != 'cox_ph'
        )
''',
    '''        _has_hessian = (
            getattr(loss, "has_hessian", False)
            and getattr(loss, "_supports_metric_proximal_newton", False)
            and not _is_quadratic
            and getattr(loss, "name", "") != "cox_ph"
        )
''',
)
replace_once(
    "statgpu/solvers/_fista_lla.py",
    '''        # Generic path: fixed-step FISTA for quadratic/no-Hessian losses and
        # proximal Newton for genuinely non-quadratic Hessian-equipped losses.
        # For losses with Hessian: use Proximal Newton (5-10 iter per LLA step).
        # For losses without Hessian: use FISTA (300+ iter per LLA step).
''',
    '''        # Generic path: fixed-step FISTA is the correctness-preserving
        # default for composite penalties. A proximal-Newton branch is used
        # only when a loss explicitly advertises a correct Hessian-metric
        # proximal subproblem implementation.
''',
)

# L-BFGS-B: keep quasi-Newton directions feasible and reject NaN bounds.
replace_once(
    "statgpu/solvers/_lbfgs_b.py",
    '''    if backend == "torch":
        invalid_bounds = bool((lb > ub).any().item())
    elif backend == "cupy":
        invalid_bounds = bool((lb > ub).any().item())
    else:
        invalid_bounds = bool(np.any(lb > ub))
    if invalid_bounds:
        raise ValueError("lower_bounds must not exceed upper_bounds")
''',
    '''    if backend == "torch":
        invalid_nan = bool((lb.isnan().any() | ub.isnan().any()).item())
        invalid_bounds = bool((lb > ub).any().item())
    elif backend == "cupy":
        import cupy as cp

        invalid_nan = bool((cp.isnan(lb).any() | cp.isnan(ub).any()).item())
        invalid_bounds = bool((lb > ub).any().item())
    else:
        invalid_nan = bool(np.isnan(lb).any() or np.isnan(ub).any())
        invalid_bounds = bool(np.any(lb > ub))
    if invalid_nan:
        raise ValueError("lower_bounds and upper_bounds must not contain NaN")
    if invalid_bounds:
        raise ValueError("lower_bounds must not exceed upper_bounds")
''',
)
replace_once(
    "statgpu/solvers/_lbfgs_b.py",
    '''        direction = -r
        gdd_dev = _dot_dev(grad, direction)
''',
    '''        direction = _project_direction(-r, params, lb, ub, backend)
        gdd_dev = _dot_dev(grad, direction)
''',
)
append_helper = '''

def _project_direction(direction, params, lb, ub, backend):
    """Remove direction components that would leave the feasible box."""
    blocked = ((params <= lb) & (direction < 0)) | (
        (params >= ub) & (direction > 0)
    )
    if backend == "torch":
        return direction * (~blocked).to(direction.dtype)
    return direction * (~blocked).astype(direction.dtype)
'''
append_once("statgpu/solvers/_lbfgs_b.py", "def _project_direction(", append_helper)


tests = r'''
# PR87_REVIEW_FIX_V39
def test_admm_legitimate_cholesky_failure_initializes_iterative_fallback(monkeypatch):
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import admm_solver

    def not_positive_definite(*args, **kwargs):
        raise np.linalg.LinAlgError("not positive definite")

    monkeypatch.setattr(np.linalg, "cholesky", not_positive_definite)
    coef, n_iter = admm_solver(
        SquaredErrorLoss(),
        get_penalty("l1", alpha=0.05),
        np.column_stack([np.ones(6), np.arange(6.0)]),
        np.arange(6.0),
        max_iter=2,
    )
    assert n_iter >= 0
    assert np.all(np.isfinite(coef))


def test_proximal_newton_max_iter_zero_returns_initialized_coefficients():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    init = np.array([0.25, -0.5])
    coef, n_iter = proximal_newton_solver(
        SquaredErrorLoss(),
        get_penalty("l2", alpha=0.1),
        np.column_stack([np.ones(4), np.arange(4.0)]),
        np.arange(4.0),
        init_coef=init,
        max_iter=0,
    )
    np.testing.assert_allclose(coef, init)
    assert n_iter == 0


def test_proximal_newton_l2_matches_declared_closed_form_objective():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    X = np.array(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 4.0]],
        dtype=np.float64,
    )
    y = np.array([0.2, 1.0, 2.1, 2.7, 5.3], dtype=np.float64)
    alpha = 0.35
    expected = np.linalg.solve(
        X.T @ X / X.shape[0] + alpha * np.eye(X.shape[1]),
        X.T @ y / X.shape[0],
    )
    coef, _ = proximal_newton_solver(
        SquaredErrorLoss(),
        get_penalty("l2", alpha=alpha),
        X,
        y,
        max_iter=20,
        tol=1e-12,
    )
    np.testing.assert_allclose(coef, expected, rtol=1e-8, atol=1e-9)


def test_proximal_newton_nonsmooth_fallback_is_explicit_and_objective_preserving():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import fista_solver, proximal_newton_solver

    X = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64)
    y = np.array([1.0, 1.8, 3.2, 3.9], dtype=np.float64)
    penalty = get_penalty("l1", alpha=0.05)
    with pytest.warns(RuntimeWarning, match="delegates non-smooth penalties"):
        delegated = proximal_newton_solver(
            SquaredErrorLoss(), penalty, X, y, max_iter=40, tol=1e-10
        )
    direct = fista_solver(
        SquaredErrorLoss(), penalty, X, y, max_iter=40, tol=1e-10
    )
    np.testing.assert_allclose(delegated[0], direct[0], rtol=0.0, atol=0.0)
    assert delegated[1] == direct[1]


def test_fista_lla_requires_explicit_metric_proximal_newton_capability():
    from pathlib import Path

    source = Path("statgpu/solvers/_fista_lla.py").read_text(encoding="utf-8")
    assert "_supports_metric_proximal_newton" in source


def test_lbfgsb_projects_quasi_newton_direction_and_rejects_nan_bounds():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import lbfgs_b_solver
    from statgpu.solvers._lbfgs_b import _project_direction

    params = np.array([-1.0, 0.5, 2.0])
    lb = np.array([-1.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 2.0])
    direction = np.array([-2.0, 0.25, 3.0])
    np.testing.assert_allclose(
        _project_direction(direction, params, lb, ub, "numpy"),
        np.array([0.0, 0.25, 0.0]),
    )

    with pytest.raises(ValueError, match="must not contain NaN"):
        lbfgs_b_solver(
            SquaredErrorLoss(),
            get_penalty("l2", alpha=0.1),
            np.ones((3, 1)),
            np.ones(3),
            lower_bounds=np.array([np.nan]),
            upper_bounds=np.array([1.0]),
        )
'''
append_once("dev/tests/test_maintenance_024_025.py", "# PR87_REVIEW_FIX_V39", tests)

# Maintained documentation: describe the correctness gate and actual backends.
replace_once(
    "docs/en/guides/solver-algorithms.md",
    "| Proximal Newton | Huber/Bisquare/Cox + SCAD/MCP | numpy, cupy, torch |",
    "| Proximal Newton | smooth loss + smooth penalty; non-smooth explicitly uses FISTA | numpy, cupy, torch |",
)
replace_once(
    "docs/en/guides/solver-algorithms.md",
    "| L-BFGS-B | box-constrained problems | numpy |",
    "| L-BFGS-B | box-constrained problems | numpy, cupy, torch |",
)
replace_once(
    "docs/en/guides/solver-algorithms.md",
    '''**Use case**: Smooth losses with Hessian (Huber, Bisquare, Cox PH) + non-smooth penalties (SCAD/MCP via LLA). Converges in 5-10 iterations.

### Algorithm

1. Compute Hessian H = X'WX and gradient g = X'ψ / n
2. Newton direction: d = -H⁻¹·g
3. Armijo line search (max 25 retries):
   a. Trial point: β_try = proximal(β − step·d, step)
   b. Check composite Armijo: f(β_try) + g(β_try) ≤ f(β) + g(β) + c·step·g'd
   c. Halve step if not satisfied
4. If Hessian singular or g'd ≤ 0: fall back to gradient descent
''',
    '''**Use case**: Smooth losses with a smooth L2/no penalty Newton system.

A general non-smooth proximal-Newton update requires a Hessian-metric proximal
subproblem. The previous Euclidean-prox shortcut optimized the wrong composite
objective. Direct non-smooth requests now emit a warning and use FISTA; the
FISTA-LLA path likewise stays on its backend-native FISTA implementation until
a metric proximal subproblem is implemented and explicitly advertised.

### Algorithm

1. Compute the loss and smooth-penalty gradient/Hessian exactly once.
2. Solve the Newton system, using least squares only for a genuine rank failure.
3. Run Armijo backtracking on the declared full objective.
4. If the Newton direction is not a descent direction, use steepest descent.
''',
)
replace_once(
    "docs/en/guides/solver-algorithms.md",
    '''   b. **Inner solver**:
      - Losses with Hessian → Proximal Newton (5-10 iter)
      - Losses without Hessian → FISTA (300+ iter)
''',
    '''   b. **Inner solver**:
      - backend-native FISTA for composite LLA subproblems
      - a future proximal-Newton path is gated on an explicit, correct
        Hessian-metric proximal capability
''',
)

replace_once(
    "docs/cn/guides/solver-algorithms.md",
    "| Proximal Newton | Huber/Bisquare/Cox + SCAD/MCP | numpy, cupy, torch |",
    "| Proximal Newton | 光滑损失 + 光滑惩罚；非光滑情形显式使用 FISTA | numpy, cupy, torch |",
)
replace_once(
    "docs/cn/guides/solver-algorithms.md",
    "| L-BFGS | 光滑损失，中低维度 | numpy, cupy, torch |\n| exact |",
    "| L-BFGS | 光滑损失，中低维度 | numpy, cupy, torch |\n| L-BFGS-B | box-constrained 问题 | numpy, cupy, torch |\n| ADMM | 可分惩罚 | numpy, cupy, torch |\n| exact |",
)
replace_once(
    "docs/cn/guides/solver-algorithms.md",
    '''**用途**: 有 Hessian 的光滑损失（Huber、Bisquare、Cox PH）+ 非光滑惩罚（SCAD/MCP 通过 LLA）。5-10 次迭代收敛。

### 算法

1. 计算 Hessian H = X'WX 和梯度 g = X'ψ / n
2. Newton 方向 d = -H⁻¹·g
3. Armijo 线搜索（最多 25 次回退）：
   a. 尝试点: β_try = proximal(β − step·d, step)
   b. 检查复合 Armijo: f(β_try) + g(β_try) ≤ f(β) + g(β) + c·step·g'd
   c. 不满足则步长减半
4. Hessian 奇异或 g'd ≤ 0 → 回退到梯度下降
''',
    '''**用途**: 对光滑损失与 L2/无惩罚目标执行 Newton 更新。

一般非光滑 proximal-Newton 需要求解 Hessian metric 下的 proximal 子问题；
旧的 Euclidean-prox 快捷路径会优化错误目标。现在 direct 非光滑调用会明确告警并
使用 FISTA；FISTA-LLA 也保持 backend-native FISTA，直到实现并显式声明正确的
metric proximal 能力。

### 算法

1. 对损失和光滑惩罚各计入一次梯度与 Hessian。
2. 仅在真正的秩失败时使用 least-squares 降级。
3. 对完整声明目标执行 Armijo 回溯。
4. Newton 方向不是下降方向时使用最速下降。
''',
)
replace_once(
    "docs/cn/guides/solver-algorithms.md",
    '''   b. **内层求解器**：
      - 有 Hessian → Proximal Newton（5-10 次迭代）
      - 无 Hessian → FISTA（300+ 次迭代）
''',
    '''   b. **内层求解器**：
      - 复合 LLA 子问题统一使用 backend-native FISTA
      - 未来的 proximal-Newton 路径必须显式提供正确的 Hessian-metric proximal 能力
''',
)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Removed the incorrect Euclidean-prox Newton shortcut that duplicated "
    "smooth penalty terms and solved the wrong non-smooth objective. Smooth "
    "L2/no-penalty requests retain Newton updates; non-smooth requests now "
    "explicitly use FISTA, and FISTA-LLA requires a future metric-prox capability.\n"
    "- Completed ADMM's legitimate Cholesky-to-iterative fallback and kept "
    "L-BFGS-B directions/bounds feasible and backend-native.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- Removed the wrong Euclidean-prox Newton shortcut that duplicated "
    "smooth penalties. Smooth objectives retain Newton; non-smooth objectives "
    "explicitly use FISTA until a Hessian-metric proximal solver exists.\n"
    "- Completed ADMM's Cholesky fallback initialization and hardened "
    "L-BFGS-B feasible directions and NaN-bound validation.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- 删除会重复计入光滑惩罚、从而优化错误目标的 Euclidean-prox Newton "
    "快捷路径；光滑目标保留 Newton，非光滑目标在 Hessian-metric proximal "
    "求解器完成前显式使用 FISTA。\n"
    "- 补全 ADMM 的 Cholesky 降级初始化，并强化 L-BFGS-B 的可行方向与 "
    "NaN bounds 校验。\n",
)
