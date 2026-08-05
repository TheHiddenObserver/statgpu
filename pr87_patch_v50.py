from __future__ import annotations

from pathlib import Path


def replace(path: str, old: str, new: str, count: int = 1) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"{path}: expected {count} matches, found {actual}: {old[:120]!r}")
    p.write_text(text.replace(old, new, count), encoding="utf-8")


# Shared classification: only genuine rank/definiteness failures may fall back.
replace(
    "statgpu/backends/_array_ops.py",
    '''def _linear_solve_runtime_is_rank_failure(exc):
    """Classify backend solve errors that may safely use least squares."""
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
''',
    '''def _linear_solve_runtime_is_rank_failure(exc):
    """Classify backend solve errors that may safely use least squares."""
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


def _linalg_exception_is_rank_failure(exc):
    """Return whether a backend linalg exception permits a numeric fallback.

    NumPy/CuPy expose dedicated ``LinAlgError`` classes, whereas Torch reports
    rank and definiteness failures as ``RuntimeError``.  Runtime failures are
    therefore message-classified so CUDA OOM, device, index, and programming
    errors remain visible to callers.
    """
    if isinstance(exc, np.linalg.LinAlgError):
        return True
    exc_type = type(exc)
    if exc_type.__name__ == "LinAlgError" and "linalg" in exc_type.__module__.lower():
        return True
    return isinstance(exc, RuntimeError) and _linear_solve_runtime_is_rank_failure(exc)
''',
)
replace(
    "statgpu/backends/_array_ops.py",
    '''    except np.linalg.LinAlgError:
        pass
    except RuntimeError as exc:
        if not _linear_solve_runtime_is_rank_failure(exc):
            raise
''',
    '''    except Exception as exc:
        if not _linalg_exception_is_rank_failure(exc):
            raise
''',
)

# Response validation must not relabel CUDA/device failures as bad user data.
replace(
    "statgpu/glm_core/_base.py",
    '''        try:
            invalid = xp.any(~xp.isfinite(values))
        except (TypeError, RuntimeError) as exc:
            raise ValueError(
                f"{self.name} response must contain real numeric finite values."
            ) from exc
''',
    '''        try:
            invalid = xp.any(~xp.isfinite(values))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.name} response must contain real numeric finite values."
            ) from exc
''',
)

# GLM initialization, ordered fitting, and ordered inference.
replace(
    "statgpu/linear_model/_glm_base.py",
    "from statgpu.backends import _to_numpy, _resolve_backend, _is_torch_array\n",
    "from statgpu.backends import _to_numpy, _resolve_backend, _is_torch_array\nfrom statgpu.backends._array_ops import _linalg_exception_is_rank_failure\n",
)
replace(
    "statgpu/linear_model/_glm_base.py",
    '''                    try:
                        init_t = torch.linalg.lstsq(X_t, eta_target).solution
                    except RuntimeError:
                        init_t = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
''',
    '''                    try:
                        init_t = torch.linalg.lstsq(X_t, eta_target).solution
                    except RuntimeError as exc:
                        if not _linalg_exception_is_rank_failure(exc):
                            raise
                        init_t = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
''',
)
replace(
    "statgpu/linear_model/_glm_base.py",
    '''                try:
                    delta = xp.linalg.solve(H_reg, -grad)
                except (np.linalg.LinAlgError, RuntimeError):
                    ridge *= 10; continue
                except Exception:
                    if is_cupy:
                        ridge *= 10; continue
                    raise
''',
    '''                try:
                    delta = xp.linalg.solve(H_reg, -grad)
                except Exception as exc:
                    if not _linalg_exception_is_rank_failure(exc):
                        raise
                    ridge *= 10
                    continue
''',
)
replace(
    "statgpu/linear_model/_glm_base.py",
    '''        try:
            H_inv = xp.linalg.solve(H, eye)
        except (np.linalg.LinAlgError, RuntimeError) as e:
            raise np.linalg.LinAlgError(
                "Ordered model Hessian is singular — cannot compute standard errors. "
                "This may indicate quasi-complete separation or redundant thresholds. "
                "Consider using inference_method='bootstrap' or reducing n_categories."
            ) from e
        except Exception as e:
            if is_cupy:
                raise np.linalg.LinAlgError(
                    "Ordered model Hessian is singular — cannot compute standard errors. "
                    "This may indicate quasi-complete separation or redundant thresholds. "
                    "Consider using inference_method='bootstrap' or reducing n_categories."
                ) from e
            raise
''',
    '''        try:
            H_inv = xp.linalg.solve(H, eye)
        except Exception as exc:
            if not _linalg_exception_is_rank_failure(exc):
                raise
            raise np.linalg.LinAlgError(
                "Ordered model Hessian is singular — cannot compute standard errors. "
                "This may indicate quasi-complete separation or redundant thresholds. "
                "Consider using inference_method='bootstrap' or reducing n_categories."
            ) from exc
''',
)

# Penalized exact and group-block solves.
replace(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    "from statgpu.solvers._utils import _nesterov_momentum, _nesterov_update\n",
    "from statgpu.solvers._utils import _nesterov_momentum, _nesterov_update\nfrom statgpu.backends._array_ops import _linalg_exception_is_rank_failure\n",
)
replace(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''        try:
            # torch.linalg.solve is faster than Cholesky + solve_triangular
            # on PyTorch due to kernel launch overhead for small matrices
            return torch.linalg.solve(A, Xty)
        except RuntimeError:
            return torch.linalg.pinv(A) @ Xty
''',
    '''        try:
            # torch.linalg.solve is faster than Cholesky + solve_triangular
            # on PyTorch due to kernel launch overhead for small matrices
            return torch.linalg.solve(A, Xty)
        except RuntimeError as exc:
            if not _linalg_exception_is_rank_failure(exc):
                raise
            return torch.linalg.pinv(A) @ Xty
''',
)
replace(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''                try:
                    w_mat = xp.linalg.solve(_XtX_batched, rho_mat)  # (G, gs)
                except Exception:
                    w_mat = xp.zeros_like(rho_mat)
''',
    '''                try:
                    w_mat = xp.linalg.solve(_XtX_batched, rho_mat)  # (G, gs)
                except Exception as exc:
                    if not _linalg_exception_is_rank_failure(exc):
                        raise
                    w_mat = xp.zeros_like(rho_mat)
''',
)
replace(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''                    try:
                        w_g = xp.linalg.solve(_XtX_blocks[g], rho_g)
                        if xp.any(xp.isnan(w_g)) or xp.any(xp.isinf(w_g)):
                            w_g = _xp_zeros(len(g_idx), X_work.dtype, X_work)
                    except Exception:
                        w_g = _xp_zeros(len(g_idx), X_work.dtype, X_work)
''',
    '''                    try:
                        w_g = xp.linalg.solve(_XtX_blocks[g], rho_g)
                        if xp.any(xp.isnan(w_g)) or xp.any(xp.isinf(w_g)):
                            w_g = _xp_zeros(len(g_idx), X_work.dtype, X_work)
                    except Exception as exc:
                        if not _linalg_exception_is_rank_failure(exc):
                            raise
                        w_g = _xp_zeros(len(g_idx), X_work.dtype, X_work)
''',
)

# Penalized inference inversion/cholesky fallbacks.
replace(
    "statgpu/linear_model/penalized/_inference_mixin.py",
    "from statgpu.backends import _to_numpy\n",
    "from statgpu.backends import _to_numpy\nfrom statgpu.backends._array_ops import _linalg_exception_is_rank_failure\n",
)
replace(
    "statgpu/linear_model/penalized/_inference_mixin.py",
    '''            try:
                XtX_inv = xp.linalg.inv(X_full.T @ X_full)
            except Exception:
                XtX_inv = xp.linalg.pinv(X_full.T @ X_full)
''',
    '''            try:
                XtX_inv = xp.linalg.inv(X_full.T @ X_full)
            except Exception as exc:
                if not _linalg_exception_is_rank_failure(exc):
                    raise
                XtX_inv = xp.linalg.pinv(X_full.T @ X_full)
''',
)
replace(
    "statgpu/linear_model/penalized/_inference_mixin.py",
    '''        try:
            chol = cp.linalg.cholesky(bread)
            bread_inv = cp.linalg.solve(chol.T, cp.linalg.solve(chol, cp.eye(bread.shape[0], dtype=bread.dtype)))
        except Exception:
            bread_inv = cp.linalg.pinv(bread)
''',
    '''        try:
            chol = cp.linalg.cholesky(bread)
            bread_inv = cp.linalg.solve(chol.T, cp.linalg.solve(chol, cp.eye(bread.shape[0], dtype=bread.dtype)))
        except Exception as exc:
            if not _linalg_exception_is_rank_failure(exc):
                raise
            bread_inv = cp.linalg.pinv(bread)
''',
)
replace(
    "statgpu/linear_model/penalized/_inference_mixin.py",
    '''        try:
            chol = torch.linalg.cholesky(bread)
            bread_inv = torch.cholesky_inverse(chol)
        except RuntimeError:
            bread_inv = torch.linalg.pinv(bread)
''',
    '''        try:
            chol = torch.linalg.cholesky(bread)
            bread_inv = torch.cholesky_inverse(chol)
        except RuntimeError as exc:
            if not _linalg_exception_is_rank_failure(exc):
                raise
            bread_inv = torch.linalg.pinv(bread)
''',
)

# Public linear/logistic wrappers: singular fallback only.
for path in ("statgpu/linear_model/wrappers/_linear.py", "statgpu/linear_model/wrappers/_logistic.py"):
    replace(
        path,
        "from statgpu._config import Device\n",
        "from statgpu._config import Device\nfrom statgpu.backends._array_ops import _linalg_exception_is_rank_failure\n",
    )
replace(
    "statgpu/linear_model/wrappers/_linear.py",
    '''        except Exception:
            lstsq_result = cp.linalg.lstsq(X_design, y, rcond=None)
''',
    '''        except Exception as exc:
            if not _linalg_exception_is_rank_failure(exc):
                raise
            lstsq_result = cp.linalg.lstsq(X_design, y, rcond=None)
''',
)
replace(
    "statgpu/linear_model/wrappers/_linear.py",
    '''                try:
                    XtX_inv = cp.linalg.inv(XtX_cov)
                except Exception:
                    XtX_inv = cp.linalg.pinv(XtX_cov)
''',
    '''                try:
                    XtX_inv = cp.linalg.inv(XtX_cov)
                except Exception as exc:
                    if not _linalg_exception_is_rank_failure(exc):
                        raise
                    XtX_inv = cp.linalg.pinv(XtX_cov)
''',
)
replace(
    "statgpu/linear_model/wrappers/_linear.py",
    '''        except Exception:
            coef = torch.linalg.lstsq(X_design, y).solution
''',
    '''        except Exception as exc:
            if not _linalg_exception_is_rank_failure(exc):
                raise
            coef = torch.linalg.lstsq(X_design, y).solution
''',
)
replace(
    "statgpu/linear_model/wrappers/_linear.py",
    '''                try:
                    XtX_inv = torch.linalg.inv(XtX_cov)
                except Exception:
                    XtX_inv = torch.linalg.pinv(XtX_cov)
''',
    '''                try:
                    XtX_inv = torch.linalg.inv(XtX_cov)
                except Exception as exc:
                    if not _linalg_exception_is_rank_failure(exc):
                        raise
                    XtX_inv = torch.linalg.pinv(XtX_cov)
''',
)
replace(
    "statgpu/linear_model/wrappers/_logistic.py",
    '''            try:
                params = cp.linalg.solve(XtWX, Xtz)
            except Exception:
                params = cp.linalg.lstsq(XtWX, Xtz)[0]
''',
    '''            try:
                params = cp.linalg.solve(XtWX, Xtz)
            except Exception as exc:
                if not _linalg_exception_is_rank_failure(exc):
                    raise
                params = cp.linalg.lstsq(XtWX, Xtz)[0]
''',
)
replace(
    "statgpu/linear_model/wrappers/_logistic.py",
    '''            try:
                eye = cp.eye(H.shape[0], dtype=H.dtype)
                bread = cp.linalg.solve(H, eye)
            except Exception:
                bread = cp.linalg.pinv(H)
''',
    '''            try:
                eye = cp.eye(H.shape[0], dtype=H.dtype)
                bread = cp.linalg.solve(H, eye)
            except Exception as exc:
                if not _linalg_exception_is_rank_failure(exc):
                    raise
                bread = cp.linalg.pinv(H)
''',
)
replace(
    "statgpu/linear_model/wrappers/_logistic.py",
    '''            try:
                params = torch.linalg.solve(XtWX, Xtz)
            except Exception:
                params = torch.linalg.lstsq(XtWX, Xtz)[0]
''',
    '''            try:
                params = torch.linalg.solve(XtWX, Xtz)
            except Exception as exc:
                if not _linalg_exception_is_rank_failure(exc):
                    raise
                params = torch.linalg.lstsq(XtWX, Xtz)[0]
''',
)
replace(
    "statgpu/linear_model/wrappers/_logistic.py",
    '''            try:
                eye = torch.eye(H.shape[0], dtype=H.dtype, device=torch_device)
                bread = torch.linalg.solve(H, eye)
            except Exception:
                bread = torch.linalg.pinv(H)
''',
    '''            try:
                eye = torch.eye(H.shape[0], dtype=H.dtype, device=torch_device)
                bread = torch.linalg.solve(H, eye)
            except Exception as exc:
                if not _linalg_exception_is_rank_failure(exc):
                    raise
                bread = torch.linalg.pinv(H)
''',
)

# Kernel local-linear ridge retries must not suppress device/programming errors.
replace(
    "statgpu/nonparametric/kernel_smoothing/_kernel_regression.py",
    "from statgpu.backends._array_ops import ",
    "from statgpu.backends._array_ops import ",
    count=1,
)
# Insert a direct import without disturbing the existing multiline import.
p = Path("statgpu/nonparametric/kernel_smoothing/_kernel_regression.py")
text = p.read_text(encoding="utf-8")
needle = "import numpy as np\n"
if needle not in text:
    raise RuntimeError("kernel regression numpy import not found")
text = text.replace(
    needle,
    needle + "from statgpu.backends._array_ops import _linalg_exception_is_rank_failure\n",
    1,
)
p.write_text(text, encoding="utf-8")
replace(
    "statgpu/nonparametric/kernel_smoothing/_kernel_regression.py",
    '''                    except Exception:
                        A_work = A_work + ridge_work[:, None, None] * eye_p1[None, :, :]
                        ridge_work = ridge_work * 10.0
''',
    '''                    except Exception as exc:
                        if not _linalg_exception_is_rank_failure(exc):
                            raise
                        A_work = A_work + ridge_work[:, None, None] * eye_p1[None, :, :]
                        ridge_work = ridge_work * 10.0
''',
)
replace(
    "statgpu/nonparametric/kernel_smoothing/_kernel_regression.py",
    '''        except Exception:
            A_work = A_work + ridge * eye
            ridge *= 10.0
''',
    '''        except Exception as exc:
            if not _linalg_exception_is_rank_failure(exc):
                raise
            A_work = A_work + ridge * eye
            ridge *= 10.0
''',
)

# CV IRLS solve fallback.
replace(
    "statgpu/linear_model/cv/_logistic_cv.py",
    "from statgpu.backends import get_backend, _torch_dev\n",
    "from statgpu.backends import get_backend, _torch_dev\nfrom statgpu.backends._array_ops import _linalg_exception_is_rank_failure\n",
)
replace(
    "statgpu/linear_model/cv/_logistic_cv.py",
    '''                try:
                    params = backend.solve(XtWX, Xtz)
                except Exception:
                    params = backend.lstsq(XtWX, Xtz)[0]
''',
    '''                try:
                    params = backend.solve(XtWX, Xtz)
                except Exception as exc:
                    if not _linalg_exception_is_rank_failure(exc):
                        raise
                    params = backend.lstsq(XtWX, Xtz)[0]
''',
)

# Regression tests: shared classifier and representative public/internal paths.
test_path = Path("dev/tests/test_maintenance_024_025.py")
test_text = test_path.read_text(encoding="utf-8")
append = r'''


def test_backend_linalg_failure_classifier_is_narrow():
    from statgpu.backends._array_ops import _linalg_exception_is_rank_failure

    assert _linalg_exception_is_rank_failure(np.linalg.LinAlgError("singular matrix"))
    assert _linalg_exception_is_rank_failure(RuntimeError("matrix is singular"))
    assert _linalg_exception_is_rank_failure(RuntimeError("not positive definite"))
    assert not _linalg_exception_is_rank_failure(RuntimeError("CUDA out of memory"))
    assert not _linalg_exception_is_rank_failure(RuntimeError("index out of range"))
    assert not _linalg_exception_is_rank_failure(ValueError("incompatible dimensions"))


def test_glm_response_validation_preserves_backend_runtime_failure(monkeypatch):
    from types import SimpleNamespace
    from statgpu.glm_core import get_glm_loss
    import statgpu.backends._array_ops as array_ops

    fake_xp = SimpleNamespace(
        __name__="fake_gpu",
        asarray=lambda value: np.asarray(value),
        isfinite=np.isfinite,
        any=lambda value: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
    )
    monkeypatch.setattr(array_ops, "_xp", lambda value: fake_xp)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        get_glm_loss("squared_error").validate_response(np.array([1.0, 2.0]))


def test_penalized_exact_torch_preserves_nonrank_runtime_failure(monkeypatch):
    torch = pytest.importorskip("torch")
    from statgpu.linear_model.penalized._fit_mixin import _PenalizedFitMixin

    owner = object.__new__(_PenalizedFitMixin)
    owner._ridge_alpha_for_exact = lambda: 0.1
    monkeypatch.setattr(
        torch.linalg,
        "solve",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
    )
    monkeypatch.setattr(
        torch.linalg,
        "pinv",
        lambda *args, **kwargs: pytest.fail("pinv must not run after CUDA OOM"),
    )

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        owner._solve_exact_torch(torch.eye(2), torch.ones(2), normalization=3.0)


def test_kernel_ridge_retry_preserves_nonrank_runtime_failure(monkeypatch):
    from types import SimpleNamespace
    from statgpu.nonparametric.kernel_smoothing._kernel_regression import (
        _solve_linear_system_with_ridge,
    )

    fake_xp = SimpleNamespace(
        float64=np.float64,
        trace=np.trace,
        linalg=SimpleNamespace(
            solve=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("CUDA out of memory")
            )
        ),
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        _solve_linear_system_with_ridge(np.eye(2), np.ones(2), fake_xp)
'''
if "test_backend_linalg_failure_classifier_is_narrow" in test_text:
    raise RuntimeError("v50 tests already present")
test_path.write_text(test_text + append, encoding="utf-8")

# Record the behavioral contract in maintained changelogs.
for path, bullet in (
    ("CHANGELOG.md", "- Narrowed GPU linear-algebra fallbacks so only genuine rank/definiteness failures use least-squares, pseudo-inverse, ridge, or zero-block recovery; CUDA OOM, device, index, and programming errors now propagate.\n"),
    ("docs/en/changelog.md", "- Narrowed GPU linear-algebra fallbacks so only genuine rank/definiteness failures use least-squares, pseudo-inverse, ridge, or zero-block recovery; CUDA OOM, device, index, and programming errors now propagate.\n"),
    ("docs/cn/changelog.md", "- 收窄 GPU 线性代数降级条件：仅真实的秩亏/非正定失败可转用最小二乘、伪逆、ridge 或零块恢复；CUDA OOM、设备、索引与实现错误将原样抛出。\n"),
):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    marker = "## Unreleased\n"
    if marker not in text:
        marker = "# Changelog\n"
    if bullet.strip() not in text:
        text = text.replace(marker, marker + "\n" + bullet, 1)
    p.write_text(text, encoding="utf-8")
