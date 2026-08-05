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


SOLVERS = {
    "statgpu/solvers/_newton.py": (
        '''    _runtime_error_is_singular,\n)\n''',
        '''    _runtime_error_is_singular,\n    _as_backend_vector,\n)\n''',
        '''    if init_coef is not None:\n        params = (\n            _copy_arr(init_coef)\n            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")\n            else np.array(init_coef).copy()\n        )\n    else:\n        params = _zeros(n_features, backend, ref_tensor=X_proc)\n''',
        '''    if init_coef is not None:\n        params = _as_backend_vector(init_coef, backend, X_proc)\n    else:\n        params = _zeros(n_features, backend, ref_tensor=X_proc)\n''',
    ),
    "statgpu/solvers/_proximal_newton.py": (
        '''    _validate_sample_weight,\n)\n''',
        '''    _validate_sample_weight,\n    _as_backend_vector,\n)\n''',
        '''    if init_coef is not None:\n        params = (\n            _copy_arr(init_coef)\n            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")\n            else np.array(init_coef).copy()\n        )\n    else:\n        params = _zeros(n_features, backend, ref_tensor=X_proc)\n''',
        '''    if init_coef is not None:\n        params = _as_backend_vector(init_coef, backend, X_proc)\n    else:\n        params = _zeros(n_features, backend, ref_tensor=X_proc)\n''',
    ),
    "statgpu/solvers/_lbfgs.py": (
        '''    _validate_uniform_sample_weight,\n)\n''',
        '''    _validate_uniform_sample_weight,\n    _as_backend_vector,\n)\n''',
        '''    if init_coef is not None:\n        params = (\n            _copy_arr(init_coef)\n            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")\n            else np.array(init_coef).copy()\n        )\n    else:\n        params = _zeros(n_features, backend, ref_tensor=X)\n''',
        '''    if init_coef is not None:\n        params = _as_backend_vector(init_coef, backend, X_proc)\n    else:\n        params = _zeros(n_features, backend, ref_tensor=X_proc)\n''',
    ),
    "statgpu/solvers/_lbfgs_b.py": (
        '''    _validate_uniform_sample_weight,\n)\n''',
        '''    _validate_uniform_sample_weight,\n    _as_backend_vector,\n)\n''',
        '''    # Initialize params\n    if init_coef is not None:\n        params = (\n            _copy_arr(init_coef)\n            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")\n            else np.array(init_coef).copy()\n        )\n    else:\n        params = _zeros(n_features, backend, ref_tensor=X)\n''',
        '''    # Initialize params on the preprocessed design backend/device/dtype.\n    if init_coef is not None:\n        params = _as_backend_vector(init_coef, backend, X_proc)\n    else:\n        params = _zeros(n_features, backend, ref_tensor=X_proc)\n''',
    ),
    "statgpu/solvers/_admm.py": (
        '''    _validate_uniform_sample_weight,\n)\n''',
        '''    _validate_uniform_sample_weight,\n    _as_backend_vector,\n)\n''',
        '''    # Initialize\n    if init_coef is not None:\n        w = (\n            _copy_arr(init_coef)\n            if hasattr(init_coef, "copy") or hasattr(init_coef, "clone")\n            else np.array(init_coef).copy()\n        )\n    else:\n        w = _zeros(n_features, backend, ref_tensor=X)\n''',
        '''    # Initialize on the preprocessed design backend/device/dtype.\n    if init_coef is not None:\n        w = _as_backend_vector(init_coef, backend, X_proc)\n    else:\n        w = _zeros(n_features, backend, ref_tensor=X_proc)\n''',
    ),
}

for path, (old_import, new_import, old_init, new_init) in SOLVERS.items():
    replace_once(path, old_import, new_import)
    replace_once(path, old_init, new_init)

# A true no-penalty Newton fit must not emit a misleading missing-value warning.
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''            if iteration == 0:\n                warnings.warn(\n                    f"proximal_newton: penalty '{getattr(penalty, 'name', '?')}' "\n                    f"has no value() method. Armijo condition ignores penalty value.",\n                    RuntimeWarning, stacklevel=2,\n                )\n''',
    '''            if iteration == 0 and _pen_name not in ("none", "null", ""):\n                warnings.warn(\n                    f"proximal_newton: penalty '{getattr(penalty, 'name', '?')}' "\n                    f"has no value() method. Armijo condition ignores penalty value.",\n                    RuntimeWarning, stacklevel=2,\n                )\n''',
)

# The maintained solver matrix lists eleven public solvers.
replace_once(
    "docs/en/guides/solver-algorithms.md",
    "statgpu provides 10 solvers for penalized loss minimization.",
    "statgpu provides 11 solvers for penalized loss minimization.",
)
replace_once(
    "docs/cn/guides/solver-algorithms.md",
    "statgpu 提供 10 种求解器用于惩罚损失最小化。",
    "statgpu 提供 11 种求解器用于惩罚损失最小化。",
)


tests = r'''
# PR87_REVIEW_FIX_V40
def _run_warm_start_solver_matrix(X, y, init):
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import (
        admm_solver,
        lbfgs_b_solver,
        lbfgs_solver,
        newton_solver,
        proximal_newton_solver,
    )

    loss = SquaredErrorLoss()
    l2 = get_penalty("l2", alpha=0.05)
    l1 = get_penalty("l1", alpha=0.05)
    return {
        "newton": newton_solver(
            loss, l2, X, y, init_coef=init, max_iter=2, tol=1e-8
        )[0],
        "proximal_newton": proximal_newton_solver(
            loss, l2, X, y, init_coef=init, max_iter=2, tol=1e-8
        )[0],
        "lbfgs": lbfgs_solver(
            loss, l2, X, y, init_coef=init, max_iter=2, tol=1e-8
        )[0],
        "lbfgs_b": lbfgs_b_solver(
            loss,
            l2,
            X,
            y,
            init_coef=init,
            lower_bounds=np.full(X.shape[1], -10.0),
            upper_bounds=np.full(X.shape[1], 10.0),
            max_iter=2,
            tol=1e-8,
        )[0],
        "admm": admm_solver(
            loss, l1, X, y, init_coef=init, max_iter=2, tol=1e-8
        )[0],
    }


def test_solver_numpy_warm_starts_follow_torch_cpu_backend_dtype():
    torch = pytest.importorskip("torch")

    X = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        dtype=torch.float32,
    )
    y = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    init = np.array([0.2, -0.1], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = _run_warm_start_solver_matrix(X, y, init)
    for name, coef in outputs.items():
        assert isinstance(coef, torch.Tensor), name
        assert coef.device == X.device, name
        assert coef.dtype == X.dtype, name
        assert bool(torch.all(torch.isfinite(coef)).item()), name


def test_torch_cuda_solver_numpy_warm_starts_stay_on_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires physical CUDA")

    X = torch.tensor(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        dtype=torch.float64,
        device="cuda",
    )
    y = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64, device="cuda")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = _run_warm_start_solver_matrix(
            X, y, np.array([0.2, -0.1], dtype=np.float64)
        )
    for name, coef in outputs.items():
        assert coef.device.type == "cuda", name
        assert coef.dtype == X.dtype, name
        assert bool(torch.all(torch.isfinite(coef)).item()), name


def test_cupy_solver_numpy_warm_starts_stay_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires physical CUDA")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")

    X = cp.asarray(
        [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]],
        dtype=cp.float64,
    )
    y = cp.asarray([0.0, 1.0, 2.0, 3.0], dtype=cp.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = _run_warm_start_solver_matrix(
            X, y, np.array([0.2, -0.1], dtype=np.float64)
        )
    for name, coef in outputs.items():
        assert isinstance(coef, cp.ndarray), name
        assert coef.dtype == X.dtype, name
        assert bool(cp.all(cp.isfinite(coef)).item()), name


def test_proximal_newton_none_penalty_has_no_spurious_warning():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.solvers import proximal_newton_solver

    X = np.column_stack([np.ones(4), np.arange(4.0)])
    y = np.arange(4.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coef, _ = proximal_newton_solver(
            SquaredErrorLoss(), None, X, y, max_iter=2
        )
    assert np.all(np.isfinite(coef))
    assert not any("has no value" in str(item.message) for item in caught)
'''
append_once("dev/tests/test_maintenance_024_025.py", "# PR87_REVIEW_FIX_V40", tests)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Normalized Newton, proximal-Newton, L-BFGS, L-BFGS-B, and ADMM "
    "warm starts onto the preprocessed design backend, device, and dtype; "
    "added physical Torch/CuPy regression entry points.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- Newton-family, L-BFGS-family, and ADMM warm starts now follow the "
    "preprocessed design backend, device, and dtype rather than retaining the "
    "caller's original array placement.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- Newton 系列、L-BFGS 系列与 ADMM 的 warm start 现在统一跟随预处理"
    "设计矩阵的 backend、device 与 dtype，不再保留调用方原始数组的位置。\n",
)
