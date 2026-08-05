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


old_validators = '''def _validate_uniform_sample_weight(sample_weight, n_samples, solver_name):
    if sample_weight is None:
        return
    _sw = _to_numpy(sample_weight)
    if _sw.ndim != 1 or _sw.shape[0] != n_samples:
        raise ValueError("sample_weight must be a 1D array with length n_samples")
    if not np.all(np.isfinite(_sw)):
        raise ValueError("sample_weight must contain only finite values")
    if np.any(_sw < 0):
        raise ValueError("sample_weight must be non-negative")
    if np.sum(_sw) <= 0.0:
        raise ValueError("sample_weight must contain at least one positive value")
    if not np.allclose(_sw, _sw[0]):
        raise ValueError(
            f"{solver_name} does not support non-uniform sample_weight yet; "
            "use solver='irls' for weighted GLM fits."
        )


def _validate_sample_weight(sample_weight, n_samples):
    if sample_weight is None:
        return
    _sw = _to_numpy(sample_weight)
    if _sw.ndim != 1 or _sw.shape[0] != n_samples:
        raise ValueError("sample_weight must be 1D with length n_samples")
    if not np.all(np.isfinite(_sw)):
        raise ValueError("sample_weight must contain only finite values")
    if np.any(_sw < 0):
        raise ValueError("sample_weight must be non-negative")
    if np.sum(_sw) <= 0:
        raise ValueError("sample_weight must contain at least one positive value")
'''

new_validators = '''def _scalar_bool(value):
    return bool(value.item() if hasattr(value, "item") else value)


def _native_sample_weight(sample_weight):
    """Return sample weights on their current backend without a full D2H copy."""
    backend = _resolve_backend("auto", sample_weight)
    xp = _get_xp(backend)
    if backend == "torch":
        import torch

        try:
            values = (
                sample_weight
                if torch.is_tensor(sample_weight)
                else torch.as_tensor(sample_weight)
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("sample_weight must be a real numeric array-like") from exc
        if torch.is_complex(values):
            raise ValueError("sample_weight must contain real numeric values")
    else:
        try:
            values = xp.asarray(sample_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("sample_weight must be a real numeric array-like") from exc
        if getattr(values.dtype, "kind", "") not in "biuf":
            raise ValueError("sample_weight must contain real numeric values")
    return backend, xp, values


def _validated_sample_weight(sample_weight, n_samples):
    backend, xp, values = _native_sample_weight(sample_weight)
    if int(values.ndim) != 1 or int(values.shape[0]) != int(n_samples):
        raise ValueError("sample_weight must be 1D with length n_samples")
    try:
        finite = xp.all(xp.isfinite(values))
        negative = xp.any(values < 0)
        total = float(xp.sum(values).item() if hasattr(xp.sum(values), "item") else xp.sum(values))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("sample_weight must contain real finite values") from exc
    if not _scalar_bool(finite):
        raise ValueError("sample_weight must contain only finite values")
    if _scalar_bool(negative):
        raise ValueError("sample_weight must be non-negative")
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("sample_weight must have a finite positive sum")
    return backend, xp, values


def _validate_uniform_sample_weight(sample_weight, n_samples, solver_name):
    if sample_weight is None:
        return
    backend, xp, values = _validated_sample_weight(sample_weight, n_samples)
    if backend == "torch":
        import torch

        uniform = (
            torch.allclose(values, values[0])
            if torch.is_floating_point(values)
            else torch.all(values == values[0])
        )
    else:
        uniform = (
            xp.allclose(values, values[0])
            if getattr(values.dtype, "kind", "") == "f"
            else xp.all(values == values[0])
        )
    if not _scalar_bool(uniform):
        raise ValueError(
            f"{solver_name} does not support non-uniform sample_weight yet; "
            "use solver='irls' for weighted GLM fits."
        )


def _validate_sample_weight(sample_weight, n_samples):
    if sample_weight is not None:
        _validated_sample_weight(sample_weight, n_samples)
'''
replace_once("statgpu/solvers/_utils.py", old_validators, new_validators)

replace_once(
    "statgpu/solvers/_fista.py",
    '''    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    _is_quadratic = getattr(loss, '_is_quadratic', False)
''',
    '''    backend = _resolve_backend("auto", X)
    X_proc, y_proc = loss.preprocess(X, y)
    # Validate before any weighted Lipschitz or matrix operation so direct
    # solver callers receive the public contract error rather than a backend
    # broadcast, NaN, or device-mismatch failure.
    _validate_sample_weight(sample_weight, X_proc.shape[0])
    _is_quadratic = getattr(loss, '_is_quadratic', False)
''',
)
replace_once(
    "statgpu/solvers/_fista.py",
    '''    if _use_gpu_loop:
        _conv_interval = 10
        _div_interval = 25
        _lip_interval = 25
    _validate_sample_weight(sample_weight, X_proc.shape[0])

    # Convert sample_weight to backend-native array (prevent CPU/CUDA mismatch)
''',
    '''    if _use_gpu_loop:
        _conv_interval = 10
        _div_interval = 25
        _lip_interval = 25

    # Convert sample_weight to backend-native array (prevent CPU/CUDA mismatch)
''',
)

replace_once(
    "statgpu/glm_core/_validation.py",
    '''    if total <= 0.0:
        raise ValueError(f"{name} must have a positive sum")
''',
    '''    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have a finite positive sum")
''',
)

replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''    if total <= 0.0:
        raise ValueError("sample_weight must have a positive sum")
''',
    '''    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("sample_weight must have a finite positive sum")
''',
)

replace_once(
    "statgpu/linear_model/penalized/_penalized_cv.py",
    '''def _is_uniform_weight(sample_weight) -> bool:
    """Check if sample_weight is uniform (all elements equal) or None."""
    if sample_weight is None:
        return True
    sw_np = np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
    return not sw_np.size or np.allclose(sw_np, sw_np[0])
''',
    '''def _is_uniform_weight(sample_weight) -> bool:
    """Check uniformity on the current backend and synchronize one boolean."""
    if sample_weight is None:
        return True
    module = type(sample_weight).__module__
    if module.startswith("torch"):
        import torch

        values = sample_weight.reshape(-1)
        if int(values.numel()) == 0:
            return True
        uniform = (
            torch.allclose(values, values[0])
            if torch.is_floating_point(values)
            else torch.all(values == values[0])
        )
        return bool(uniform.item() if hasattr(uniform, "item") else uniform)
    if module.startswith("cupy"):
        import cupy as cp

        values = sample_weight.reshape(-1)
        if int(values.size) == 0:
            return True
        uniform = (
            cp.allclose(values, values[0])
            if getattr(values.dtype, "kind", "") == "f"
            else cp.all(values == values[0])
        )
        return bool(uniform.item())
    values = np.asarray(sample_weight).reshape(-1)
    return not values.size or bool(np.allclose(values, values[0]))
''',
)

replace_once(
    "statgpu/inference/_sandwich.py",
    '''def assemble_cov_avg(
    bread_avg,
    meat_avg,
    n_eff,
    k,
    cov_type,
) -> np.ndarray:
''',
    '''def assemble_cov_avg(
    bread_avg,
    meat_avg,
    n_eff,
    k,
    cov_type,
    *,
    hc1_n=None,
) -> np.ndarray:
''',
)
replace_once(
    "statgpu/inference/_sandwich.py",
    '''    HC1: multiply by n_eff / (n_eff - k).
''',
    '''    HC1: multiply by n / (n - k), where ``n`` is the observation count.
    For analytic weights this keeps the correction invariant to a global
    rescaling of the weights; ``n_eff`` remains the sandwich normalization.
''',
)
replace_once(
    "statgpu/inference/_sandwich.py",
    '''    cov_type : str

    Returns
''',
    '''    cov_type : str
    hc1_n : int or float, optional
        Observation count used by the HC1 finite-sample correction. Defaults
        to ``n_eff`` for backward-compatible direct calls.

    Returns
''',
)
replace_once(
    "statgpu/inference/_sandwich.py",
    '''    if cov_type == "hc1" and n_eff > k:
        cov = cov * (n_eff / (n_eff - k))
''',
    '''    correction_n = float(n_eff if hc1_n is None else hc1_n)
    if cov_type == "hc1" and correction_n > k:
        cov = cov * (correction_n / (correction_n - k))
''',
)
replace_once(
    "statgpu/inference/_sandwich.py",
    '''        cov = assemble_cov_avg(bread_avg, meat_avg, n_eff, k, cov_type)
''',
    '''        cov = assemble_cov_avg(
            bread_avg,
            meat_avg,
            n_eff,
            k,
            cov_type,
            hc1_n=X.shape[0],
        )
''',
)

tests = r'''
# PR87_REVIEW_FIX_V37
def test_direct_fista_validates_weight_length_before_lipschitz():
    from statgpu.glm_core._squared import SquaredErrorLoss
    from statgpu.penalties import get_penalty
    from statgpu.solvers import fista_solver

    class GuardedSquaredError(SquaredErrorLoss):
        def lipschitz(self, *args, **kwargs):
            raise AssertionError("lipschitz must not run before weight validation")

    X = np.ones((3, 1), dtype=np.float64)
    y = np.arange(3.0)
    with pytest.raises(ValueError, match="length n_samples"):
        fista_solver(
            GuardedSquaredError(),
            get_penalty("l2", alpha=0.0),
            X,
            y,
            sample_weight=np.ones(2),
        )


def test_solver_weight_validation_does_not_copy_torch_tensor(monkeypatch):
    torch = pytest.importorskip("torch")
    import statgpu.solvers._utils as solver_utils

    def forbidden(*args, **kwargs):
        raise AssertionError("sample_weight must not be copied through _to_numpy")

    monkeypatch.setattr(solver_utils, "_to_numpy", forbidden)
    weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    solver_utils._validate_sample_weight(weights, 3)
    assert torch.equal(weights, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))


def test_penalized_cv_uniform_weight_check_does_not_copy_torch_tensor(monkeypatch):
    torch = pytest.importorskip("torch")
    import statgpu.linear_model.penalized._penalized_cv as penalized_cv

    def forbidden(*args, **kwargs):
        raise AssertionError("uniform-weight check must stay backend-native")

    monkeypatch.setattr(penalized_cv, "_to_numpy", forbidden)
    assert penalized_cv._is_uniform_weight(torch.ones(4, dtype=torch.float64))
    assert not penalized_cv._is_uniform_weight(
        torch.tensor([1.0, 1.0, 2.0, 1.0], dtype=torch.float64)
    )


def test_glm_weight_validation_rejects_overflowing_total():
    from statgpu.glm_core._validation import validate_glm_sample_weight
    from statgpu.solvers._utils import _validate_sample_weight

    weights = np.array([np.finfo(np.float64).max, np.finfo(np.float64).max])
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="finite positive sum"):
            validate_glm_sample_weight(weights, 2)
        with pytest.raises(ValueError, match="finite positive sum"):
            _validate_sample_weight(weights, 2)


def test_glm_hc1_analytic_weight_diagnostics_are_scale_invariant():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [2.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([-0.4, 0.5, 2.2, 5.1, 5.8], dtype=np.float64)
    weights = np.array([0.5, 1.5, 2.0, 4.0, 3.0], dtype=np.float64)

    def fit(current_weights):
        return GeneralizedLinearModel(
            family="gaussian",
            solver="irls",
            C=0.0,
            max_iter=100,
            tol=1e-12,
            device="cpu",
            compute_inference=True,
            cov_type="hc1",
        ).fit(X, y, sample_weight=current_weights)

    weighted = fit(weights)
    scaled = fit(29.0 * weights)
    np.testing.assert_allclose(weighted._bse, scaled._bse, rtol=1e-11, atol=1e-11)
'''
append_once("dev/tests/test_maintenance_024_025.py", "# PR87_REVIEW_FIX_V37", tests)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Kept direct solver and penalized-CV sample-weight checks backend-native, "
    "validated weights before weighted Lipschitz operations, rejected overflowing "
    "weight totals, and made HC1 analytic-weight inference invariant to global "
    "weight rescaling.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- Direct solver and penalized-CV sample-weight checks now remain on the "
    "selected backend, run before weighted Lipschitz operations, reject "
    "overflowing totals, and preserve HC1 analytic-weight scale invariance.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- direct solver 与 penalized-CV 的 sample-weight 检查现在保持在所选 "
    "backend，并在 weighted Lipschitz 运算前执行；权重总和溢出会被拒绝，"
    "HC1 analytic-weight inference 对全局权重缩放保持不变。\n",
)
