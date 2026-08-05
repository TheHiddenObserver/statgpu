from __future__ import annotations

from pathlib import Path


def replace(path: str, old: str, new: str, count: int = 1) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"{path}: expected {count} matches, found {actual}: {old[:120]!r}")
    p.write_text(text.replace(old, new, count), encoding="utf-8")


path = "statgpu/linear_model/penalized/_penalized_cv.py"
replace(
    path,
    "from statgpu.backends._array_ops import _copy_arr, _zeros, _xp_zeros, _soft_threshold\n",
    "from statgpu.backends._array_ops import (\n"
    "    _copy_arr,\n"
    "    _linalg_exception_is_rank_failure,\n"
    "    _soft_threshold,\n"
    "    _xp_zeros,\n"
    "    _zeros,\n"
    ")\n",
)
replace(
    path,
    '''def _is_squared_error_loss_name(loss_name) -> bool:
    return str(loss_name).lower() in ("squared_error", "gaussian", "normal")


def _is_uniform_weight(sample_weight) -> bool:
''',
    '''def _is_squared_error_loss_name(loss_name) -> bool:
    return str(loss_name).lower() in ("squared_error", "gaussian", "normal")


def _cv_lipschitz_failure_is_recoverable(exc) -> bool:
    """Return whether an optional Lipschitz hint may defer to the solver."""
    return isinstance(
        exc,
        (NotImplementedError, ValueError, FloatingPointError, OverflowError),
    ) or _linalg_exception_is_rank_failure(exc)


def _is_uniform_weight(sample_weight) -> bool:
''',
)
replace(
    path,
    '''        except (NotImplementedError, ValueError, FloatingPointError,
                OverflowError, np.linalg.LinAlgError):
            # A solver may estimate L internally when the optional closed-form
            # Lipschitz hint is unavailable or numerically invalid.
            lipschitz_L = None
''',
    '''        except Exception as exc:
            if not _cv_lipschitz_failure_is_recoverable(exc):
                raise
            # A solver may estimate L internally when the optional closed-form
            # Lipschitz hint is unavailable or numerically invalid.  The shared
            # classifier includes NumPy, CuPy, and Torch rank failures without
            # treating OOM/device errors as recoverable.
            lipschitz_L = None
''',
)
replace(
    path,
    '''            except Exception as e:
                warnings.warn(
                    f"Alpha grid estimation failed ({e}), using alpha_max=1.0",
''',
    '''            except Exception as e:
                _raise_cv_infrastructure_failure(e)
                warnings.warn(
                    f"Alpha grid estimation failed ({e}), using alpha_max=1.0",
''',
)

test_path = Path("dev/tests/test_maintenance_024_025.py")
test_text = test_path.read_text(encoding="utf-8")
append = r'''


def test_penalized_cv_lipschitz_recovery_includes_cupy_linalg_only():
    from statgpu.linear_model.penalized._penalized_cv import (
        _cv_lipschitz_failure_is_recoverable,
    )

    CupyLinAlgError = type(
        "LinAlgError", (Exception,), {"__module__": "cupy.linalg._solve"}
    )
    assert _cv_lipschitz_failure_is_recoverable(
        CupyLinAlgError("singular matrix")
    )
    assert _cv_lipschitz_failure_is_recoverable(
        np.linalg.LinAlgError("singular matrix")
    )
    assert not _cv_lipschitz_failure_is_recoverable(
        RuntimeError("CUDA out of memory")
    )


def test_penalized_cv_alpha_grid_does_not_hide_memory_failure(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod
    import statgpu.linear_model.penalized._base as base_mod

    class FailingModel:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise MemoryError("host allocation failed")

    monkeypatch.setattr(base_mod, "PenalizedGeneralizedLinearModel", FailingModel)
    owner = object.__new__(cv_mod.PenalizedGLM_CV)
    owner.loss = "poisson"
    owner.penalty = "l1"
    owner.l1_ratio = 1.0
    owner._n_alphas = 5
    owner._loss_kwargs = None
    owner._penalty_kwargs = None

    with pytest.raises(MemoryError, match="host allocation failed"):
        owner._generate_alpha_grid(
            np.array([[1.0], [2.0], [3.0]]),
            np.array([1.0, 2.0, 3.0]),
        )
'''
if "test_penalized_cv_lipschitz_recovery_includes_cupy_linalg_only" in test_text:
    raise RuntimeError("v52 tests already present")
test_path.write_text(test_text + append, encoding="utf-8")

for changelog, bullet in (
    ("CHANGELOG.md", "- Completed penalized-CV fallback hardening: optional Lipschitz recovery now recognizes NumPy/CuPy/Torch rank failures consistently, while alpha-grid estimation no longer hides memory or GPU infrastructure failures.\n"),
    ("docs/en/changelog.md", "- Completed penalized-CV fallback hardening: optional Lipschitz recovery now recognizes NumPy/CuPy/Torch rank failures consistently, while alpha-grid estimation no longer hides memory or GPU infrastructure failures.\n"),
    ("docs/cn/changelog.md", "- 完成惩罚 CV 降级边界加固：可选 Lipschitz 提示统一识别 NumPy/CuPy/Torch 的秩失败，而 alpha 网格估计不再隐藏内存或 GPU 基础设施错误。\n"),
):
    p = Path(changelog)
    text = p.read_text(encoding="utf-8")
    marker = "# Changelog\n"
    if bullet.strip() not in text:
        text = text.replace(marker, marker + "\n" + bullet, 1)
    p.write_text(text, encoding="utf-8")
