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


# FISTA-family warm starts and zero vectors must follow the preprocessed
# design, not the caller's pre-preprocessing dtype/device.
for old, new in (
    ("_as_backend_vector(init_coef, backend, X)", "_as_backend_vector(init_coef, backend, X_proc)"),
    ("_zeros(n_features, backend, ref_tensor=X)", "_zeros(n_features, backend, ref_tensor=X_proc)"),
):
    path = Path("statgpu/solvers/_fista.py")
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count < 1:
        raise RuntimeError(f"expected at least one FISTA match for {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")

for old, new in (
    ("_as_backend_vector(init_coef, backend, X)", "_as_backend_vector(init_coef, backend, X_proc)"),
    ("_zeros(n_features, backend, ref_tensor=X)", "_zeros(n_features, backend, ref_tensor=X_proc)"),
):
    path = Path("statgpu/solvers/_fista_bb.py")
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count < 1:
        raise RuntimeError(f"expected at least one FISTA-BB match for {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")

# Proximal Newton validates the caller's weight object, then normalizes it to
# the loss backend/device/dtype before any loss method receives it.
replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''    _validate_sample_weight(sample_weight, X_proc.shape[0])\n    n_features = X_proc.shape[1]\n''',
    '''    _validate_sample_weight(sample_weight, X_proc.shape[0])\n    _sw_arr = (\n        None\n        if sample_weight is None\n        else _as_backend_vector(sample_weight, backend, X_proc)\n    )\n    n_features = X_proc.shape[1]\n''',
)
for old, new in (
    ("sample_weight=sample_weight\n            )", "sample_weight=_sw_arr\n            )"),
    ("sample_weight=sample_weight)\n            loss_hess", "sample_weight=_sw_arr)\n            loss_hess"),
    ("sample_weight=sample_weight)\n\n        # Only smooth penalties", "sample_weight=_sw_arr)\n\n        # Only smooth penalties"),
    ("params_old, sample_weight=sample_weight)", "params_old, sample_weight=_sw_arr)"),
    ("params_try, sample_weight=sample_weight)", "params_try, sample_weight=_sw_arr)"),
):
    replace_once("statgpu/solvers/_proximal_newton.py", old, new)

# Constant Hessians are reused, but Armijo remains active.
replace_once(
    "statgpu/solvers/_newton.py",
    '''    For losses with constant Hessian (e.g. Gamma log link), the Hessian\n    doesn't change across iterations, so the Newton step is always valid\n    and line search is skipped.\n''',
    '''    For losses with constant Hessian, the Hessian is computed once and\n    reused across iterations; Armijo backtracking still verifies each step.\n''',
)

# Expand the shared warm-start matrix to include both FISTA variants.
replace_once(
    "dev/tests/test_maintenance_024_025.py",
    '''        admm_solver,\n        lbfgs_b_solver,\n''',
    '''        admm_solver,\n        fista_bb_solver,\n        fista_solver,\n        lbfgs_b_solver,\n''',
)
replace_once(
    "dev/tests/test_maintenance_024_025.py",
    '''    return {\n        "newton": newton_solver(\n''',
    '''    return {\n        "fista": fista_solver(\n            loss, l1, X, y, init_coef=init, max_iter=2, tol=1e-8\n        )[0],\n        "fista_bb": fista_bb_solver(\n            loss, l1, X, y, init_coef=init, max_iter=2, tol=1e-8\n        )[0],\n        "newton": newton_solver(\n''',
)


tests = r'''
# PR87_REVIEW_FIX_V45
def test_fista_family_warm_starts_follow_preprocessed_dtype():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import fista_bb_solver, fista_solver

    class Float32PreprocessLoss:
        name = "float32_preprocess"
        _is_quadratic = False
        _prefer_fista_over_bb = False
        _lipschitz_uses_y = True

        def preprocess(self, X, y):
            return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

        def lipschitz(self, X, coef, y=None, sample_weight=None):
            return 1.0

        def gradient(self, X, y, coef, sample_weight=None):
            return np.zeros_like(coef)

        def fused_value_and_gradient(self, X, y, coef, sample_weight=None):
            return np.asarray(0.0, dtype=coef.dtype), np.zeros_like(coef)

        def value(self, X, y, coef, sample_weight=None):
            return np.asarray(0.0, dtype=coef.dtype)

    X = np.ones((4, 2), dtype=np.float64)
    y = np.ones(4, dtype=np.float64)
    init = np.array([0.2, -0.1], dtype=np.float64)
    penalty = get_penalty("l2", alpha=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fista_coef, _ = fista_solver(
            Float32PreprocessLoss(), penalty, X, y, init_coef=init, max_iter=0
        )
        bb_coef, _ = fista_bb_solver(
            Float32PreprocessLoss(), penalty, X, y, init_coef=init, max_iter=0
        )
    assert fista_coef.dtype == np.float32
    assert bb_coef.dtype == np.float32


def test_proximal_newton_normalizes_weights_to_preprocessed_dtype():
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    class RecordingLoss:
        name = "recording"
        has_hessian = True

        def __init__(self):
            self.seen_weight = None

        def preprocess(self, X, y):
            return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

        def fused_gradient_and_hessian(self, X, y, coef, sample_weight=None):
            self.seen_weight = sample_weight
            return np.zeros_like(coef), np.eye(coef.shape[0], dtype=coef.dtype)

    loss = RecordingLoss()
    coef, _ = proximal_newton_solver(
        loss,
        get_penalty("l2", alpha=0.0),
        np.ones((3, 1), dtype=np.float64),
        np.ones(3, dtype=np.float64),
        sample_weight=[1.0, 2.0, 3.0],
        max_iter=1,
    )
    assert coef.dtype == np.float32
    assert isinstance(loss.seen_weight, np.ndarray)
    assert loss.seen_weight.dtype == np.float32


def test_torch_cuda_proximal_newton_numpy_weights_stay_on_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires physical CUDA")
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device="cuda")
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    coef, _ = proximal_newton_solver(
        SquaredErrorLoss(),
        get_penalty("l2", alpha=0.05),
        X,
        y,
        sample_weight=np.array([1.0, 2.0, 3.0, 4.0]),
        max_iter=3,
    )
    assert coef.device.type == "cuda"
    assert coef.dtype == X.dtype
    assert bool(torch.all(torch.isfinite(coef)).item())


def test_cupy_proximal_newton_numpy_weights_stay_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires physical CUDA")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import proximal_newton_solver

    X = cp.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=cp.float64)
    y = cp.asarray([1.0, 2.0, 3.0, 4.0], dtype=cp.float64)
    coef, _ = proximal_newton_solver(
        SquaredErrorLoss(),
        get_penalty("l2", alpha=0.05),
        X,
        y,
        sample_weight=np.array([1.0, 2.0, 3.0, 4.0]),
        max_iter=3,
    )
    assert isinstance(coef, cp.ndarray)
    assert coef.dtype == X.dtype
    assert bool(cp.all(cp.isfinite(coef)).item())
'''
append_once("dev/tests/test_maintenance_024_025.py", "# PR87_REVIEW_FIX_V45", tests)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Normalized FISTA/FISTA-BB warm starts to the preprocessed design and "
    "converted smooth proximal-Newton sample weights to the active backend, "
    "device, and dtype before loss evaluation.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- FISTA-family warm starts now follow the preprocessed design, and smooth "
    "proximal-Newton weights are normalized to the active backend/device/dtype "
    "before loss evaluation.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- FISTA 系列 warm start 现在跟随预处理设计矩阵；smooth proximal-Newton "
    "权重会在 loss 计算前转换到当前 backend/device/dtype。\n",
)
