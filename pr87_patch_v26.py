from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:180]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# IRLS backend correctness and narrow solve fallback.
# ---------------------------------------------------------------------------
replace_once(
    "statgpu/glm_core/_irls.py",
    '''def _solve(A, b, backend="auto"):
    """Solve linear system, fallback to lstsq if singular."""
    if backend == "auto":
        backend = _infer_backend(A)

    try:
        if backend == "torch":
            import torch
            b_col = b.unsqueeze(1) if b.ndim == 1 else b
            sol = torch.linalg.solve(A, b_col)
            return sol.squeeze(1) if b.ndim == 1 else sol
        elif backend == "cupy":
            import cupy as cp
            return cp.linalg.solve(A, b)
        else:
            return np.linalg.solve(A, b)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        if backend == "torch":
            import torch
            b_col = b.unsqueeze(1) if b.ndim == 1 else b
            sol = torch.linalg.lstsq(A, b_col).solution
            return sol.squeeze(1) if b.ndim == 1 else sol
        elif backend == "cupy":
            import cupy as cp
            return cp.linalg.lstsq(A, b)[0]
        return np.linalg.lstsq(A, b, rcond=None)[0]
''',
    '''def _solve(A, b, backend="auto"):
    """Solve a linear system, using least squares only for singular systems."""
    if backend == "auto":
        backend = _infer_backend(A)

    if backend == "torch":
        import torch

        b_col = b.unsqueeze(1) if b.ndim == 1 else b
        try:
            sol = torch.linalg.solve(A, b_col)
        except RuntimeError as exc:
            message = str(exc).lower()
            singular_markers = (
                "singular",
                "not invertible",
                "zero pivot",
                "rank deficient",
            )
            if not any(marker in message for marker in singular_markers):
                raise
            sol = torch.linalg.lstsq(A, b_col).solution
        return sol.squeeze(1) if b.ndim == 1 else sol

    if backend == "cupy":
        import cupy as cp

        try:
            return cp.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return cp.linalg.lstsq(A, b)[0]

    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]
''',
)
replace_once(
    "statgpu/glm_core/_irls.py",
    '''def _norm(x, backend):
    if backend == "torch":
        import torch

        return float(torch.linalg.norm(x).item())
    return float(np.linalg.norm(x))
''',
    '''def _norm(x, backend):
    if backend == "torch":
        import torch

        return float(torch.linalg.norm(x).item())
    if backend == "cupy":
        import cupy as cp

        return float(cp.linalg.norm(x).item())
    return float(np.linalg.norm(x))
''',
)

# Reuse registered loss formulas as the single source of truth for line search.
irls_path = Path("statgpu/glm_core/_irls.py")
irls_text = irls_path.read_text(encoding="utf-8")
compile_anchor = '''from statgpu.backends._torch_compile import compile_torch


def _get_irls_step_compiled():
'''
compile_new = '''from statgpu.backends._torch_compile import compile_torch


def _objective_loss_for_family(family):
    """Return the registered loss matching an IRLS family."""
    from statgpu.glm_core._base import get_glm_loss

    family_name = str(getattr(family, "name", "")).lower()
    loss_names = {
        "gaussian": "squared_error",
        "squared_error": "squared_error",
        "binomial": "logistic",
        "logistic": "logistic",
        "poisson": "poisson",
        "gamma": "gamma",
        "inverse_gaussian": "inverse_gaussian",
        "negative_binomial": "negative_binomial",
        "tweedie": "tweedie",
    }
    if family_name not in loss_names:
        raise NotImplementedError(
            "IRLS line search requires a registered objective for family "
            f"{family_name!r}."
        )
    kwargs = {}
    if family_name == "gamma":
        kwargs["link"] = str(getattr(family.link, "name", "log")).lower()
    elif family_name == "negative_binomial":
        kwargs["alpha"] = float(getattr(family, "alpha", 1.0))
    elif family_name == "tweedie":
        kwargs["power"] = float(getattr(family, "power", 1.5))
    return get_glm_loss(loss_names[family_name], **kwargs)


def _get_irls_step_compiled():
'''
if compile_anchor not in irls_text:
    raise RuntimeError("IRLS objective helper anchor missing")
irls_text = irls_text.replace(compile_anchor, compile_new, 1)

family_anchor = '''    family_name = getattr(family, "name", "")
    if backend == "torch":
'''
family_new = '''    family_name = getattr(family, "name", "")
    objective_loss = _objective_loss_for_family(family)
    if backend == "torch":
'''
if family_anchor not in irls_text:
    raise RuntimeError("IRLS family objective anchor missing")
irls_text = irls_text.replace(family_anchor, family_new, 1)

start = irls_text.index("        # Armijo backtracking line search:")
end = irls_text.index("        # Convergence: normalized penalized score norm.", start)
line_search = '''        # Backtracking line search on the same registered loss used by the
        # public GLM objective.  Loss classes own link/domain clipping, so the
        # identity-link Gaussian path is never spuriously clipped to [-30, 30].
        def _loss_val(eta_arr):
            terms = objective_loss.per_sample_value(eta_arr, y_work)
            if sw_work is not None:
                terms = terms * sw_work
            if backend == "torch":
                import torch

                return torch.sum(terms)
            if backend == "cupy":
                import cupy as cp

                return cp.sum(terms)
            return np.sum(terms)

        def _penalty_val(params_arr):
            value = 0.0
            if ridge_alpha > 0:
                penalized = (
                    params_arr if ridge_penalize_intercept else params_arr[1:]
                )
                if backend == "torch":
                    import torch

                    value = value + 0.5 * ridge_alpha * torch.sum(penalized ** 2)
                elif backend == "cupy":
                    import cupy as cp

                    value = value + 0.5 * ridge_alpha * cp.sum(penalized ** 2)
                else:
                    value = value + 0.5 * ridge_alpha * np.sum(penalized ** 2)
            if penalty_matrix_work is not None:
                value = value + 0.5 * (
                    params_arr @ penalty_matrix_work @ params_arr
                )
            return value

        def _objective_val(eta_arr, params_arr):
            return _loss_val(eta_arr) + _penalty_val(params_arr)

        def _scalar_float(value):
            return float(value.item() if hasattr(value, "item") else value)

        def _scalar_is_finite(value):
            if backend == "torch":
                import torch

                return bool(torch.isfinite(value).item())
            if backend == "cupy":
                import cupy as cp

                return bool(cp.isfinite(value).item())
            return bool(np.isfinite(value))

        eta_cur = X @ params_old
        objective_old = _objective_val(eta_cur, params_old)
        if not _scalar_is_finite(objective_old):
            raise FloatingPointError(
                "IRLS objective became non-finite at the current iterate."
            )
        objective_old_float = _scalar_float(objective_old)
        objective_tolerance = max(
            abs(objective_old_float) * 1e-10,
            1e-6,
        )

        def _objective_accept(objective_try):
            if not _scalar_is_finite(objective_try):
                return False
            return _scalar_float(objective_try) <= (
                objective_old_float + objective_tolerance
            )

        direction = params_new - params_old
        is_constant_weight = (
            family_name in ("gaussian", "squared_error")
            or (
                family_name == "gamma"
                and str(getattr(family.link, "name", "")).lower() == "log"
            )
        )

        if is_constant_weight:
            objective_new = _objective_val(X @ params_new, params_new)
            if _objective_accept(objective_new):
                params = params_new
            else:
                step = 1.0
                accepted = False
                for _ in range(30):
                    params_try = params_old + step * direction
                    objective_try = _objective_val(X @ params_try, params_try)
                    if _objective_accept(objective_try):
                        accepted = True
                        break
                    step *= 0.5
                if accepted:
                    params = params_try
                else:
                    params = params_old
                    line_search_failed = True
                    break
        else:
            step = 1.0
            accepted = False
            for _ in range(30):
                params_try = params_old + step * direction
                objective_try = _objective_val(X @ params_try, params_try)
                if _objective_accept(objective_try):
                    accepted = True
                    break
                step *= 0.5
            if accepted:
                params = params_try
            else:
                params = params_old
                line_search_failed = True
                break

'''
irls_text = irls_text[:start] + line_search + irls_text[end:]
irls_path.write_text(irls_text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Frequency-weight-consistent fitted diagnostics and residual degrees of freedom.
# ---------------------------------------------------------------------------
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        self._nobs = None
        self._df_resid = None
''',
    '''        self._nobs = None
        self._effective_nobs = None
        self._df_resid = None
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''            if weight_sum <= 0.0:
                raise ValueError("sample_weight must have a positive sum")

        family = self._get_family()
''',
    '''            if weight_sum <= 0.0:
                raise ValueError("sample_weight must have a positive sum")

        self._effective_nobs = (
            float(weight_sum) if sample_weight is not None else float(self._nobs)
        )

        family = self._get_family()
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        else:
            raise ValueError(
                "solver must be one of: 'auto', 'irls', 'fista', 'newton', 'lbfgs'"
            )

        # ---- Store design/loss for loglikelihood/aic/bic (always) ----
''',
    '''        else:
            raise ValueError(
                "solver must be one of: 'auto', 'irls', 'fista', 'newton', 'lbfgs'"
            )

        # Keep displayed/inference degrees of freedom consistent with the
        # frequency-weight likelihood convention used by diagnostics.
        parameter_count = int(np.asarray(self._params).shape[0])
        self._df_resid = self._effective_nobs - parameter_count

        # ---- Store design/loss for loglikelihood/aic/bic (always) ----
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        lines.append(f"  No. Observations: {self._nobs}")
        lines.append(f"  Df Residuals: {self._df_resid}")
''',
    '''        lines.append(f"  No. Observations: {self._nobs}")
        if (
            self._effective_nobs is not None
            and not np.isclose(self._effective_nobs, float(self._nobs))
        ):
            lines.append(f"  Effective Observations: {self._effective_nobs:g}")
        lines.append(f"  Df Residuals: {self._df_resid:g}")
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        n = self._nobs if self._nobs else 0
        return -2.0 * ll + k * np.log(max(n, 1))
''',
    '''        n = self._effective_nobs if self._effective_nobs is not None else self._nobs
        return -2.0 * ll + k * np.log(max(float(n or 0), 1.0))
''',
)

# ---------------------------------------------------------------------------
# Regression tests.
# ---------------------------------------------------------------------------
test_path = Path("dev/tests/test_maintenance_024_025.py")
test_text = test_path.read_text(encoding="utf-8")
marker = "# PR87_IRLS_OBJECTIVE_AND_EFFECTIVE_NOBS_TESTS"
if marker not in test_text:
    test_text += '''

# PR87_IRLS_OBJECTIVE_AND_EFFECTIVE_NOBS_TESTS
def test_irls_line_search_reuses_registered_loss_and_propagates_errors(monkeypatch):
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver
    from statgpu.glm_core._squared import SquaredErrorLoss

    def fail(self, eta, y):
        raise RuntimeError("objective evaluation failed")

    monkeypatch.setattr(SquaredErrorLoss, "per_sample_value", fail)
    with pytest.raises(RuntimeError, match="objective evaluation failed"):
        IRLSSolver(Gaussian(), max_iter=2).fit(
            np.ones((4, 1)), np.arange(4.0), backend="numpy"
        )


def test_irls_solve_only_falls_back_for_singular_systems(monkeypatch):
    from statgpu.glm_core import _irls

    singular = np.array([[1.0, 1.0], [2.0, 2.0]])
    rhs = np.array([1.0, 2.0])
    solution = _irls._solve(singular, rhs, backend="numpy")
    np.testing.assert_allclose(singular @ solution, rhs, rtol=1e-12, atol=1e-12)

    def invalid_solve(A, b):
        raise ValueError("shape/device contract failure")

    def forbidden_lstsq(*args, **kwargs):
        raise AssertionError("lstsq must not mask non-singularity failures")

    monkeypatch.setattr(np.linalg, "solve", invalid_solve)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden_lstsq)
    with pytest.raises(ValueError, match="shape/device contract failure"):
        _irls._solve(np.eye(2), np.ones(2), backend="numpy")


def test_irls_source_has_no_broad_objective_fallback_and_cupy_norm_is_native():
    from pathlib import Path

    source = Path("statgpu/glm_core/_irls.py").read_text(encoding="utf-8")
    line_search = source.split("# Backtracking line search", 1)[1].split(
        "# Convergence: normalized penalized score norm.", 1
    )[0]
    assert "except Exception" not in line_search
    assert "objective_loss.per_sample_value" in line_search
    norm_body = source.split("def _norm", 1)[1].split("def _zeros", 1)[0]
    assert "cp.linalg.norm" in norm_body


def test_glm_frequency_weights_match_expanded_data_diagnostics():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [2.0], [4.0]], dtype=np.float64)
    y = np.array([-0.4, 0.5, 2.2, 5.1], dtype=np.float64)
    weights = np.array([1, 3, 2, 4], dtype=np.float64)
    repeat_index = np.repeat(np.arange(X.shape[0]), weights.astype(int))

    weighted = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0,
        max_iter=100, tol=1e-12, device="cpu", compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    expanded = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0,
        max_iter=100, tol=1e-12, device="cpu", compute_inference=False,
    ).fit(X[repeat_index], y[repeat_index])

    np.testing.assert_allclose(weighted.coef_, expanded.coef_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.intercept_, expanded.intercept_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.loglikelihood, expanded.loglikelihood, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.aic, expanded.aic, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weighted.bic, expanded.bic, rtol=1e-12, atol=1e-12)
    assert weighted._effective_nobs == weights.sum()
    assert weighted._df_resid == expanded._df_resid
'''
    test_path.write_text(test_text, encoding="utf-8")
