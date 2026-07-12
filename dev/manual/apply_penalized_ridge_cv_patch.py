"""Temporary patch for weighted PenalizedGLM_CV Ridge consistency."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


path = ROOT / "statgpu/linear_model/penalized/_penalized_cv.py"
text = path.read_text()

text = replace_once(
    text,
    '''def _device_to_name(device):
    if isinstance(device, Device):
        return device.value
    return str(device).lower()
''',
    '''def _device_to_name(device):
    if isinstance(device, Device):
        return device.value
    return str(device).lower()


def _should_build_squared_error_cv_cache(loss_name, penalty_name, solver_name, device_name):
    """Return whether the general CV fallback can consume a Gram cache."""
    if str(loss_name).lower() != "squared_error":
        return False
    if str(device_name).lower() not in ("cuda", "torch"):
        return False
    penalty_name = str(penalty_name).lower()
    solver_name = str(solver_name).lower()
    # The default GPU Ridge route is Newton and does not read _cv_cache.
    # Explicit exact Ridge and sparse squared-error paths do consume it.
    return not (penalty_name == "l2" and solver_name != "exact")
''',
    "cache-consumer helper",
)

text = replace_once(
    text,
    '''    def _generate_alpha_grid(self, X, y):
        """Auto-generate alpha grid based on loss and penalty type."""
        from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel

        X_np = _to_numpy(X).astype(np.float64)
        y_np = _to_numpy(y).astype(np.float64).ravel()
        n = X_np.shape[0]

        if self.loss == 'squared_error':
            # Gradient at null model (intercept = mean(y)): X'(y - mean(y)) / n
            alpha_max = float(np.max(np.abs(X_np.T @ (y_np - np.mean(y_np))))) / n
        elif self.loss == 'logistic':
            # Null model prediction: mu_null = mean(y)
            mu_null = np.mean(y_np)
            alpha_max = float(np.max(np.abs(X_np.T @ (y_np - mu_null)))) / n
        else:
            try:
                model = PenalizedGeneralizedLinearModel(
                    loss=self.loss, penalty='l2', alpha=0.0,
                    device='cpu', compute_inference=False, max_iter=5,
                    loss_kwargs=getattr(self, '_loss_kwargs', None),
                    penalty_kwargs=getattr(self, '_penalty_kwargs', None),
                )
                model.fit(X_np, y_np)
                grad = X_np.T @ (y_np - _to_numpy(model.predict(X_np))) / n
                alpha_max = float(np.max(np.abs(grad)))
            except Exception as e:
                warnings.warn(
                    f"Alpha grid estimation failed ({e}), using alpha_max=1.0",
                    RuntimeWarning,
                    stacklevel=2,
                )
                alpha_max = 1.0
''',
    '''    def _generate_alpha_grid(self, X, y, sample_weight=None):
        """Auto-generate an alpha grid on the fitted average-loss scale."""
        from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel

        X_np = _to_numpy(X).astype(np.float64)
        y_np = _to_numpy(y).astype(np.float64).ravel()
        n = X_np.shape[0]
        if sample_weight is None:
            sw_np = None
            normalization = float(n)
        else:
            sw_np = np.asarray(_to_numpy(sample_weight), dtype=np.float64).reshape(-1)
            if sw_np.shape[0] != n:
                raise ValueError("sample_weight must have length n_samples")
            if not np.all(np.isfinite(sw_np)):
                raise ValueError("sample_weight must be finite")
            if np.any(sw_np < 0):
                raise ValueError("sample_weight must be non-negative")
            normalization = float(np.sum(sw_np))
            if normalization <= 0.0:
                raise ValueError("sample_weight must have a positive sum")

        if self.loss == 'squared_error':
            if sw_np is None:
                x_mean = np.mean(X_np, axis=0)
                y_mean = float(np.mean(y_np))
                grad = (X_np - x_mean).T @ (y_np - y_mean) / normalization
            else:
                x_mean = np.sum(X_np * sw_np[:, None], axis=0) / normalization
                y_mean = float(np.sum(y_np * sw_np) / normalization)
                grad = (X_np - x_mean).T @ (sw_np * (y_np - y_mean)) / normalization
            alpha_max = float(np.max(np.abs(grad)))
        elif self.loss == 'logistic':
            if sw_np is None:
                mu_null = float(np.mean(y_np))
                grad = X_np.T @ (y_np - mu_null) / normalization
            else:
                mu_null = float(np.sum(y_np * sw_np) / normalization)
                grad = X_np.T @ (sw_np * (y_np - mu_null)) / normalization
            alpha_max = float(np.max(np.abs(grad)))
        else:
            try:
                model = PenalizedGeneralizedLinearModel(
                    loss=self.loss, penalty='l2', alpha=0.0,
                    device='cpu', compute_inference=False, max_iter=5,
                    loss_kwargs=getattr(self, '_loss_kwargs', None),
                    penalty_kwargs=getattr(self, '_penalty_kwargs', None),
                )
                model.fit(X_np, y_np, sample_weight=sw_np)
                residual = y_np - _to_numpy(model.predict(X_np))
                if sw_np is None:
                    grad = X_np.T @ residual / normalization
                else:
                    grad = X_np.T @ (sw_np * residual) / normalization
                alpha_max = float(np.max(np.abs(grad)))
            except Exception as e:
                warnings.warn(
                    f"Alpha grid estimation failed ({e}), using alpha_max=1.0",
                    RuntimeWarning,
                    stacklevel=2,
                )
                alpha_max = 1.0
''',
    "weighted alpha-grid generation",
)

text = replace_once(
    text,
    '''        # Precompute XtX/Xty for squared-error GPU cache
        cv_cache, L_np = self._build_cv_cache(
            loss_name, device_name, X_train, y_train, sw_train
        )
''',
    '''        # Precompute a Gram cache only for solver paths that consume it.
        if _should_build_squared_error_cv_cache(
            loss_name, penalty_name, cv_solver, device_name
        ):
            cv_cache, L_np = self._build_cv_cache(
                loss_name, device_name, X_train, y_train, sw_train
            )
        else:
            cv_cache, L_np = None, None
''',
    "cache construction guard",
)

text = replace_once(
    text,
    '''        if self._alpha_grid_input is not None:
            alpha_grid = np.asarray(self._alpha_grid_input, dtype=np.float64)
        else:
            alpha_grid = self._generate_alpha_grid(X, y)
''',
    '''        if self._alpha_grid_input is not None:
            alpha_grid = np.asarray(self._alpha_grid_input, dtype=np.float64)
        else:
            alpha_grid = self._generate_alpha_grid(
                X, y, sample_weight=sample_weight
            )
''',
    "fit alpha-grid call",
)

path.write_text(text)


test_path = ROOT / "dev/tests/test_ridge_weighted_consistency.py"
test = test_path.read_text()
test += '''


def test_penalized_glm_cv_weighted_alpha_grid_matches_null_gradient():
    from statgpu.linear_model.penalized._penalized_cv import PenalizedGLM_CV

    rng = np.random.default_rng(1209)
    X = rng.normal(size=(160, 5))
    y = 0.7 + X @ rng.normal(size=5) + rng.normal(scale=0.4, size=160)
    w = rng.uniform(0.1, 3.0, size=160)
    cv = PenalizedGLM_CV(
        loss="squared_error", penalty="l2", n_alphas=6,
        cv=3, random_state=4, device="cpu",
    )
    grid = cv._generate_alpha_grid(X, y, sample_weight=w)

    total = float(np.sum(w))
    x_mean = np.sum(X * w[:, None], axis=0) / total
    y_mean = float(np.sum(y * w) / total)
    expected_max = float(
        np.max(np.abs((X - x_mean).T @ (w * (y - y_mean)) / total))
    )
    np.testing.assert_allclose(grid[0], expected_max, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        grid,
        cv._generate_alpha_grid(X, y, sample_weight=13.0 * w),
        rtol=1e-12,
        atol=1e-12,
    )


def test_penalized_glm_cv_weighted_ridge_is_weight_scale_invariant():
    from statgpu.linear_model.penalized._penalized_cv import PenalizedGLM_CV

    rng = np.random.default_rng(1210)
    X = rng.normal(size=(150, 5))
    y = -0.3 + X @ rng.normal(size=5) + rng.normal(scale=0.45, size=150)
    w = rng.uniform(0.15, 2.8, size=150)

    kwargs = dict(
        loss="squared_error", penalty="l2", n_alphas=7,
        cv=3, random_state=7, device="cpu", max_iter=3000, tol=1e-10,
    )
    first = PenalizedGLM_CV(**kwargs).fit(X, y, sample_weight=w)
    second = PenalizedGLM_CV(**kwargs).fit(X, y, sample_weight=8.0 * w)

    np.testing.assert_allclose(first.alpha_grid_, second.alpha_grid_, rtol=1e-12, atol=1e-12)
    assert first.alpha_ == second.alpha_
    np.testing.assert_allclose(first.coef_, second.coef_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(first.intercept_, second.intercept_, rtol=1e-10, atol=1e-10)


def test_gpu_newton_ridge_cv_does_not_request_unused_gram_cache():
    from statgpu.linear_model.penalized._penalized_cv import (
        _should_build_squared_error_cv_cache,
    )

    assert not _should_build_squared_error_cv_cache(
        "squared_error", "l2", "newton", "torch"
    )
    assert not _should_build_squared_error_cv_cache(
        "squared_error", "l2", "newton", "cuda"
    )
    assert _should_build_squared_error_cv_cache(
        "squared_error", "l2", "exact", "torch"
    )
    assert _should_build_squared_error_cv_cache(
        "squared_error", "l1", "fista", "cuda"
    )
'''
test_path.write_text(test)

print("Penalized Ridge CV consistency patch applied")
