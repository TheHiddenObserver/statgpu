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
    '''class ApproximateCVWarning(UserWarning):
    """Warning emitted when approximate two-stage CV screening is enabled."""


def _is_uniform_weight(sample_weight) -> bool:
''',
    '''class ApproximateCVWarning(UserWarning):
    """Warning emitted when approximate two-stage CV screening is enabled."""


def _cv_exception_is_infrastructure_failure(exc) -> bool:
    """Return whether a CV exception must never be converted to fallback data.

    Candidate-level numerical failures may be represented by ``NaN`` or routed
    to a slower implementation. Hardware/runtime failures must remain visible,
    otherwise CV can silently continue after CUDA OOM, device mismatch, illegal
    memory access, or an indexing/programming error.
    """
    if isinstance(exc, MemoryError):
        return True
    exc_type = type(exc)
    type_text = f"{exc_type.__module__}.{exc_type.__name__}".lower()
    if "outofmemory" in type_text or "out_of_memory" in type_text:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cuda error",
            "hip error",
            "device-side assert",
            "illegal memory access",
            "invalid device ordinal",
            "expected all tensors to be on the same device",
            "device mismatch",
            "index out of range",
            "cublas",
            "cudnn",
            "nvrtc",
        )
    )


def _raise_cv_infrastructure_failure(exc) -> None:
    """Re-raise failures that make a CV fallback unsafe or misleading."""
    if _cv_exception_is_infrastructure_failure(exc):
        raise exc


def _weighted_mse_fallback(y_true, y_pred, sample_weight=None) -> float:
    """Squared-error-only emergency evaluator preserving validation weights."""
    residual_sq = (np.asarray(y_true, dtype=np.float64).ravel() -
                   np.asarray(y_pred, dtype=np.float64).ravel()) ** 2
    if sample_weight is None:
        return float(np.mean(residual_sq))
    weights = np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
    if weights.shape != residual_sq.shape:
        raise ValueError("validation sample_weight must match validation rows")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("validation sample_weight must be finite and non-negative")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("validation sample_weight must have a finite positive sum")
    return float(np.dot(weights, residual_sq) / total)


def _is_squared_error_loss_name(loss_name) -> bool:
    return str(loss_name).lower() in ("squared_error", "gaussian", "normal")


def _is_uniform_weight(sample_weight) -> bool:
''',
)

replace(
    path,
    '''    lipschitz_L = None
    if not getattr(loss_fn, "_lipschitz_at_init", False):
        try:
            zero_lip = _zeros(n_features + 1, backend, ref_tensor=X_work)
            lipschitz_L = float(_to_numpy(loss_fn.lipschitz(X_work, zero_lip, y=yb)))
            if not np.isfinite(lipschitz_L) or lipschitz_L <= 0.0:
                lipschitz_L = None
        except Exception:
            lipschitz_L = None
''',
    '''    lipschitz_L = None
    if not getattr(loss_fn, "_lipschitz_at_init", False):
        try:
            zero_lip = _zeros(n_features + 1, backend, ref_tensor=X_work)
            lipschitz_L = float(_to_numpy(loss_fn.lipschitz(X_work, zero_lip, y=yb)))
            if not np.isfinite(lipschitz_L) or lipschitz_L <= 0.0:
                lipschitz_L = None
        except (NotImplementedError, ValueError, FloatingPointError,
                OverflowError, np.linalg.LinAlgError):
            # A solver may estimate L internally when the optional closed-form
            # Lipschitz hint is unavailable or numerically invalid.
            lipschitz_L = None
''',
)

replace(
    path,
    '''        try:
            val_loss = _evaluate_loss_numpy(
                self.loss,
                loss_fn,
                X_val_np,
                y_val_np,
                _to_numpy(model.coef_).ravel(),
                float(model.intercept_),
                model.fit_intercept,
                sample_weight=sample_weight,
            )
        except Exception:
            # Fallback: use loss_fn.value() for correct loss, not raw MSE
            try:
                if model.fit_intercept:
                    X_design = np.column_stack([np.ones(n_val), X_val_np])
                    coef_full = np.concatenate([[float(model.intercept_)], _to_numpy(model.coef_).ravel()])
                else:
                    X_design = X_val_np
                    coef_full = _to_numpy(model.coef_).ravel()
                val_loss = float(loss_fn.value(X_design, y_val_np, coef_full))
            except Exception:
                y_pred_np = _to_numpy(model.predict(X_val_np)).ravel()
                val_loss = float(np.mean((y_val_np - y_pred_np) ** 2))
                warnings.warn(
                    f"_evaluate_single: loss evaluation failed for '{self.loss}', "
                    f"falling back to MSE. CV scores may be inaccurate for non-Gaussian losses.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return val_loss
''',
    '''        try:
            val_loss = _evaluate_loss_numpy(
                self.loss,
                loss_fn,
                X_val_np,
                y_val_np,
                _to_numpy(model.coef_).ravel(),
                float(model.intercept_),
                model.fit_intercept,
                sample_weight=sample_weight,
            )
        except Exception as primary_exc:
            _raise_cv_infrastructure_failure(primary_exc)
            # Preserve the declared objective by retrying through the generic
            # loss interface, including analytic validation weights.
            try:
                if model.fit_intercept:
                    X_design = np.column_stack([np.ones(n_val), X_val_np])
                    coef_full = np.concatenate(
                        [[float(model.intercept_)], _to_numpy(model.coef_).ravel()]
                    )
                else:
                    X_design = X_val_np
                    coef_full = _to_numpy(model.coef_).ravel()
                val_loss = float(
                    loss_fn.value(
                        X_design,
                        y_val_np,
                        coef_full,
                        sample_weight=(
                            None
                            if sample_weight is None
                            else np.asarray(_to_numpy(sample_weight), dtype=np.float64).ravel()
                        ),
                    )
                )
            except Exception as fallback_exc:
                _raise_cv_infrastructure_failure(fallback_exc)
                if not _is_squared_error_loss_name(self.loss):
                    raise RuntimeError(
                        f"Could not evaluate declared validation loss '{self.loss}'. "
                        "Refusing to substitute mean squared error because that "
                        "could select a different regularization parameter."
                    ) from fallback_exc
                y_pred_np = _to_numpy(model.predict(X_val_np)).ravel()
                val_loss = _weighted_mse_fallback(
                    y_val_np, y_pred_np, sample_weight=sample_weight
                )
                warnings.warn(
                    "_evaluate_single: both squared-error evaluators failed; "
                    "using an equivalent weighted-MSE calculation.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return val_loss
''',
)

# Every layered CV fallback may recover from candidate-specific numerical
# failures, but never from hardware/device/index failures.
replace(
    path,
    '''                except Exception as e:
                    warnings.warn(
                        f"Ridge eig batch failed for fold {fold_idx}: {e}",
''',
    '''                except Exception as e:
                    _raise_cv_infrastructure_failure(e)
                    warnings.warn(
                        f"Ridge eig batch failed for fold {fold_idx}: {e}",
''',
)
replace(
    path,
    '''            except Exception as e:
                warnings.warn(
                    f"Fold-batched {loss_name} sparse CV failed on {device_name}; "
''',
    '''            except Exception as e:
                _raise_cv_infrastructure_failure(e)
                warnings.warn(
                    f"Fold-batched {loss_name} sparse CV failed on {device_name}; "
''',
)
replace(
    path,
    '''                except Exception as e:
                    warnings.warn(
                        f"{path_fn.__name__} failed for {loss_name}+{penalty_name} "
''',
    '''                except Exception as e:
                    _raise_cv_infrastructure_failure(e)
                    warnings.warn(
                        f"{path_fn.__name__} failed for {loss_name}+{penalty_name} "
''',
)
replace(
    path,
    '''            except Exception:
                # Same as path-is-None: keep _cv_cache for warm-start fallback.
''',
    '''            except Exception as exc:
                _raise_cv_infrastructure_failure(exc)
                # Same as path-is-None: keep _cv_cache for warm-start fallback.
''',
)
replace(
    path,
    '''            except Exception as exc:
                orig_idx = sort_idx[alpha_idx_sorted]
                all_scores[fold_idx, orig_idx] = np.nan
''',
    '''            except Exception as exc:
                _raise_cv_infrastructure_failure(exc)
                orig_idx = sort_idx[alpha_idx_sorted]
                all_scores[fold_idx, orig_idx] = np.nan
''',
)

# Tests exercise objective preservation, weighting, and infrastructure errors.
test_path = Path("dev/tests/test_maintenance_024_025.py")
test_text = test_path.read_text(encoding="utf-8")
append = r'''


def test_penalized_cv_does_not_substitute_mse_for_non_gaussian_loss(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    owner = object.__new__(cv_mod.PenalizedGLM_CV)
    owner.loss = "poisson"

    class Loss:
        def value(self, *args, **kwargs):
            raise ValueError("generic poisson evaluation failed")

    class Model:
        coef_ = np.array([0.2])
        intercept_ = 0.1
        fit_intercept = True

        def predict(self, X):
            pytest.fail("non-Gaussian CV must not fall back to MSE predictions")

    monkeypatch.setattr(
        cv_mod,
        "_evaluate_loss_numpy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("registered poisson evaluation failed")
        ),
    )

    with pytest.raises(RuntimeError, match="Refusing to substitute mean squared error"):
        owner._evaluate_single(
            Model(),
            np.array([[1.0], [2.0]]),
            np.array([1.0, 3.0]),
            loss_fn=Loss(),
        )


def test_penalized_cv_squared_error_emergency_fallback_preserves_weights(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    owner = object.__new__(cv_mod.PenalizedGLM_CV)
    owner.loss = "squared_error"

    class Loss:
        def value(self, *args, **kwargs):
            raise ValueError("generic squared evaluation failed")

    class Model:
        coef_ = np.array([0.0])
        intercept_ = 0.0
        fit_intercept = True

        def predict(self, X):
            return np.array([0.0, 2.0])

    monkeypatch.setattr(
        cv_mod,
        "_evaluate_loss_numpy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("registered squared evaluation failed")
        ),
    )
    weights = np.array([1.0, 3.0])
    with pytest.warns(RuntimeWarning, match="weighted-MSE"):
        value = owner._evaluate_single(
            Model(),
            np.array([[1.0], [2.0]]),
            np.array([1.0, 4.0]),
            loss_fn=Loss(),
            sample_weight=weights,
        )
    expected = (1.0 * (1.0 - 0.0) ** 2 + 3.0 * (4.0 - 2.0) ** 2) / 4.0
    assert value == pytest.approx(expected)


def test_penalized_cv_loss_evaluation_preserves_infrastructure_failure(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    owner = object.__new__(cv_mod.PenalizedGLM_CV)
    owner.loss = "poisson"

    class Loss:
        def value(self, *args, **kwargs):
            pytest.fail("generic loss fallback must not run after CUDA OOM")

    class Model:
        coef_ = np.array([0.2])
        intercept_ = 0.1
        fit_intercept = True

    monkeypatch.setattr(
        cv_mod,
        "_evaluate_loss_numpy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("CUDA out of memory")
        ),
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        owner._evaluate_single(
            Model(),
            np.array([[1.0], [2.0]]),
            np.array([1.0, 3.0]),
            loss_fn=Loss(),
        )


def test_penalized_cv_infrastructure_classifier_is_narrow():
    from statgpu.linear_model.penalized._penalized_cv import (
        _cv_exception_is_infrastructure_failure,
    )

    assert _cv_exception_is_infrastructure_failure(RuntimeError("CUDA out of memory"))
    assert _cv_exception_is_infrastructure_failure(RuntimeError("index out of range"))
    assert _cv_exception_is_infrastructure_failure(MemoryError("allocation failed"))
    assert not _cv_exception_is_infrastructure_failure(
        np.linalg.LinAlgError("singular matrix")
    )
    assert not _cv_exception_is_infrastructure_failure(
        ValueError("numeric domain error")
    )
'''
if "test_penalized_cv_does_not_substitute_mse_for_non_gaussian_loss" in test_text:
    raise RuntimeError("v51 tests already present")
test_path.write_text(test_text + append, encoding="utf-8")

for changelog, bullet in (
    ("CHANGELOG.md", "- Preserved the declared validation objective in penalized CV: non-Gaussian losses no longer silently fall back to MSE, weighted squared-error fallback retains validation weights, and GPU infrastructure failures propagate through layered CV fallbacks.\n"),
    ("docs/en/changelog.md", "- Preserved the declared validation objective in penalized CV: non-Gaussian losses no longer silently fall back to MSE, weighted squared-error fallback retains validation weights, and GPU infrastructure failures propagate through layered CV fallbacks.\n"),
    ("docs/cn/changelog.md", "- 保持惩罚 CV 的声明验证目标：非 Gaussian 损失不再静默退化为 MSE，平方损失应急路径保留验证权重，GPU 基础设施错误会穿透多层 CV 降级并原样抛出。\n"),
):
    p = Path(changelog)
    text = p.read_text(encoding="utf-8")
    marker = "# Changelog\n"
    if bullet.strip() not in text:
        text = text.replace(marker, marker + "\n" + bullet, 1)
    p.write_text(text, encoding="utf-8")
