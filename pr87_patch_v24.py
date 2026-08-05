from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:160]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Centralize all active GLM torch.compile paths and remove broad fallback.
# ---------------------------------------------------------------------------
replace_once(
    "statgpu/glm_core/_irls.py",
    '''def _torch_compile_supported():
    """Check if torch.compile is safe (CUDA Capability >= 7.0)."""
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return cap[0] >= 7
    except Exception:
        pass
    return True


def _get_irls_step_compiled():
''',
    '''from statgpu.backends._torch_compile import compile_torch


def _get_irls_step_compiled():
''',
)
replace_once(
    "statgpu/glm_core/_irls.py",
    '''    if _torch_compile_supported():
        try:
            _IRLS_STEP_COMPILED = torch.compile(_irls_weighted_gemm, dynamic=True, fullgraph=False)
        except Exception:
            _IRLS_STEP_COMPILED = _irls_weighted_gemm
    else:
        _IRLS_STEP_COMPILED = _irls_weighted_gemm

    return _IRLS_STEP_COMPILED


def _irls_step_call(compiled_fn, *args):
    """Call compiled IRLS step, falling back to eager on GPU arch mismatch."""
    try:
        return compiled_fn(*args)
    except Exception:
        def _irls_gemm_eager(X, W, z):
            W_col = W.unsqueeze(1)
            XtWX = X.T @ (X * W_col)
            Xtz = X.T @ (W * z)
            return XtWX, Xtz
        return _irls_gemm_eager(*args)
''',
    '''    _IRLS_STEP_COMPILED = compile_torch(
        _irls_weighted_gemm,
        workload="iterative",
        dynamic=True,
        fullgraph=False,
    )
    return _IRLS_STEP_COMPILED


def _irls_step_call(compiled_fn, *args):
    """Call the centrally managed compiled IRLS step."""
    return compiled_fn(*args)
''',
)

replace_once(
    "statgpu/glm_core/_solver_utils.py",
    '''from statgpu.backends._utils import torch_compile_supported as _torch_compile_supported


def _get_fista_step_compiled():
''',
    '''from statgpu.backends._torch_compile import compile_torch


def _get_fista_step_compiled():
''',
)
replace_once(
    "statgpu/glm_core/_solver_utils.py",
    '''    if _torch_compile_supported():
        try:
            _FISTA_STEP_COMPILED = torch.compile(_fista_step, dynamic=True, fullgraph=False)
        except RuntimeError:
            _FISTA_STEP_COMPILED = _fista_step
    else:
        _FISTA_STEP_COMPILED = _fista_step
    return _FISTA_STEP_COMPILED


def _fista_step_call(compiled_fn, *args):
    try:
        return compiled_fn(*args)
    except (RuntimeError, TypeError):
        def _fista_eager(y_k, grad, step, coef_old, coef, beta_t):
            w_tilde = y_k - step * grad
            y_k_new = coef + beta_t * (coef - coef_old)
            return w_tilde, y_k_new
        return _fista_eager(*args)
''',
    '''    _FISTA_STEP_COMPILED = compile_torch(
        _fista_step,
        workload="iterative",
        dynamic=True,
        fullgraph=False,
    )
    return _FISTA_STEP_COMPILED


def _fista_step_call(compiled_fn, *args):
    return compiled_fn(*args)
''',
)
replace_once(
    "statgpu/glm_core/_solver_utils.py",
    '''    if _torch_compile_supported():
        try:
            _NEWTON_STEP_COMPILED = torch.compile(_newton_step, dynamic=True, fullgraph=False)
        except RuntimeError:
            _NEWTON_STEP_COMPILED = _newton_step
    else:
        _NEWTON_STEP_COMPILED = _newton_step
    return _NEWTON_STEP_COMPILED


def _newton_step_call(compiled_fn, *args):
    try:
        return compiled_fn(*args)
    except (RuntimeError, TypeError):
        def _newton_eager(params, direction, params_old):
            import torch

            params_new = params - direction
            diff_norm = torch.linalg.norm(params_new - params_old)
            return params_new, diff_norm
        return _newton_eager(*args)
''',
    '''    _NEWTON_STEP_COMPILED = compile_torch(
        _newton_step,
        workload="iterative",
        dynamic=True,
        fullgraph=False,
    )
    return _NEWTON_STEP_COMPILED


def _newton_step_call(compiled_fn, *args):
    return compiled_fn(*args)
''',
)

# ---------------------------------------------------------------------------
# Make IRLS line search evaluate the same weighted objective as the WLS step.
# ---------------------------------------------------------------------------
irls_path = Path("statgpu/glm_core/_irls.py")
irls_text = irls_path.read_text(encoding="utf-8")
start = irls_text.index("        def _dev_val(mu_arr):")
end = irls_text.index("\n        def _penalty_val(params_arr):", start)
new_dev = '''        def _dev_val(mu_arr):
            """Return weighted family deviance on the active backend."""
            _y = y_work
            if backend == "torch":
                import torch as xp
            elif backend == "cupy":
                import cupy as xp
            else:
                xp = np

            if _fname in ("gaussian", "squared_error"):
                terms = 0.5 * (_y - mu_arr) ** 2
            elif _fname in ("binomial", "logistic"):
                _mu_c = _clip(mu_arr, 1e-10, 1.0 - 1e-10, backend)
                terms = -_y * xp.log(_mu_c) - (1.0 - _y) * xp.log1p(-_mu_c)
            elif _fname == "gamma":
                terms = _y / mu_arr + xp.log(mu_arr)
            elif _fname == "inverse_gaussian":
                terms = _y / (2.0 * mu_arr ** 2) - 1.0 / mu_arr
            elif _fname == "negative_binomial":
                _mu_c = _clip(mu_arr, 1e-10, None, backend)
                _y_c = _clip(_y, 1e-10, None, backend)
                _a = _nb_alpha
                terms = (
                    _y_c * xp.log(_y_c / _mu_c)
                    - (_y_c + 1.0 / _a)
                    * xp.log((1.0 + _a * _y_c) / (1.0 + _a * _mu_c))
                )
            elif _fname == "tweedie":
                p = _tweedie_power
                if abs(p - 1.0) < 0.01:
                    terms = mu_arr - _y * xp.log(mu_arr)
                elif abs(p - 2.0) < 0.01:
                    terms = _y / mu_arr - xp.log(_y / mu_arr) - 1.0
                else:
                    terms = (
                        -_y * xp.pow(mu_arr, 1.0 - p) / (1.0 - p)
                        + xp.pow(mu_arr, 2.0 - p) / (2.0 - p)
                    )
            else:
                terms = mu_arr - _y * xp.log(mu_arr)

            if sw_work is not None:
                terms = terms * sw_work
            return xp.sum(terms)
'''
irls_path.write_text(irls_text[:start] + new_dev + irls_text[end:], encoding="utf-8")

# ---------------------------------------------------------------------------
# Correct weighted ridge scale and preserve weights for likelihood/inference.
# ---------------------------------------------------------------------------
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        # IRLSSolver solves the unnormalized WLS normal equations
        # X'WX + lambda I, while _get_penalty_alpha() is the normalized
        # objective penalty.  Scale by n to keep C semantics consistent.
        ridge_alpha = X.shape[0] * self._get_penalty_alpha()
''',
    '''        # IRLSSolver solves unnormalized WLS normal equations.  The public
        # objective is normalized by n without weights and by sum(w) with
        # weights, so the normal-equation ridge term must use the same scale.
        if sample_weight is None:
            objective_scale = float(X.shape[0])
        else:
            scale_value = sample_weight.sum()
            objective_scale = float(
                scale_value.item() if hasattr(scale_value, "item") else scale_value
            )
        ridge_alpha = objective_scale * self._get_penalty_alpha()
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        # ---- Compute inference if requested ----
        if self._compute_inference_enabled:
            if sample_weight is not None:
                if is_gpu:
                    # sample_weight is already validated on the selected backend;
                    # preserve device residency instead of copying the full vector
                    # to NumPy and immediately transferring it back to the GPU.
                    self._sample_weight_inf = self._to_array(
                        sample_weight, backend=inf_backend
                    )
                else:
                    self._sample_weight_inf = np.asarray(
                        sample_weight, dtype=float
                    ).ravel()
            else:
                self._sample_weight_inf = None

            self._fit_metadata = {
''',
    '''        # Preserve fit weights even when inference is disabled because
        # loglikelihood/AIC/BIC are public fitted-model diagnostics.  GPU
        # weights stay on their selected backend.
        if sample_weight is not None:
            if is_gpu:
                self._sample_weight_inf = self._to_array(
                    sample_weight, backend=inf_backend
                )
            else:
                self._sample_weight_inf = np.asarray(
                    sample_weight, dtype=float
                ).ravel()
        else:
            self._sample_weight_inf = None

        # ---- Compute inference if requested ----
        if self._compute_inference_enabled:
            self._fit_metadata = {
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        params = xp_asarray(self._params, xp=xp, ref_arr=self._X_design)
        eta = self._X_design @ params
        return -float(xp.sum(self._loss.per_sample_value(eta, self._y_inf)))
''',
    '''        params = xp_asarray(self._params, xp=xp, ref_arr=self._X_design)
        eta = self._X_design @ params
        values = self._loss.per_sample_value(eta, self._y_inf)
        if self._sample_weight_inf is not None:
            weights = xp_asarray(
                self._sample_weight_inf, xp=xp, ref_arr=self._X_design
            )
            values = values * weights
        return -float(xp.sum(values))
''',
)

# ---------------------------------------------------------------------------
# Weighted dispersion must use the same retained weights as fitting.
# ---------------------------------------------------------------------------
replace_once(
    "statgpu/inference/_sandwich.py",
    '''        dispersion = _default_dispersion(loss, X, y, coef, n_eff, k)
''',
    '''        dispersion = _default_dispersion(
            loss, X, y, coef, n_eff, k, sample_weight=sample_weight
        )
''',
)
replace_once(
    "statgpu/inference/_sandwich.py",
    '''def _default_dispersion(loss, X, y, coef, n_eff, k):
''',
    '''def _default_dispersion(
    loss, X, y, coef, n_eff, k, *, sample_weight=None
):
''',
)
replace_once(
    "statgpu/inference/_sandwich.py",
    '''        eta = X @ coef; mu = eta
        resid = y - mu; rss = float(xp.sum(resid ** 2))
        return rss / max(n_eff - k, 1)
''',
    '''        eta = X @ coef; mu = eta
        resid_sq = (y - mu) ** 2
        if sample_weight is not None:
            resid_sq = resid_sq * sample_weight
        rss = float(xp.sum(resid_sq))
        return rss / max(n_eff - k, 1)
''',
)
replace_once(
    "statgpu/inference/_sandwich.py",
    '''        resid_sq = (y - mu) ** 2
        from statgpu.backends._utils import xp_maximum
        pearson = float(xp.sum(resid_sq / xp_maximum(V, 1e-10, xp)))
        return pearson / df
''',
    '''        resid_sq = (y - mu) ** 2
        from statgpu.backends._utils import xp_maximum
        pearson_terms = resid_sq / xp_maximum(V, 1e-10, xp)
        if sample_weight is not None:
            pearson_terms = pearson_terms * sample_weight
        pearson = float(xp.sum(pearson_terms))
        return pearson / df
''',
)

# ---------------------------------------------------------------------------
# Regression tests for weighted objective and compile-policy ownership.
# ---------------------------------------------------------------------------
test_path = Path("dev/tests/test_maintenance_024_025.py")
test_text = test_path.read_text(encoding="utf-8")
marker = "# PR87_WEIGHTED_IRLS_AND_GLM_COMPILE_POLICY_TESTS"
if marker not in test_text:
    test_text += '''

# PR87_WEIGHTED_IRLS_AND_GLM_COMPILE_POLICY_TESTS
def test_weighted_irls_line_search_uses_weighted_objective_cpu():
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    X = np.ones((4, 1), dtype=np.float64)
    y = np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float64)
    weights = np.array([1.0, 1.0, 1.0, 100.0], dtype=np.float64)
    params, _ = IRLSSolver(Gaussian(), max_iter=20, tol=1e-12).fit(
        X, y, sample_weight=weights, backend="numpy"
    )
    np.testing.assert_allclose(
        params[0], np.average(y, weights=weights), rtol=1e-10, atol=1e-10
    )


def test_glm_irls_weighted_ridge_matches_closed_form_and_weight_rescaling():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([-2.0, -0.5, 0.5, 2.0, 6.0], dtype=np.float64)
    weights = np.array([1.0, 2.0, 3.0, 7.0, 20.0], dtype=np.float64)
    C = 2.0
    lam = 1.0 / (2.0 * C)
    design = np.column_stack([np.ones(X.shape[0]), X])
    expected = np.linalg.solve(
        design.T @ (design * weights[:, None])
        + np.diag([0.0, weights.sum() * lam]),
        design.T @ (weights * y),
    )

    def fit(current_weights):
        return GeneralizedLinearModel(
            family="gaussian",
            solver="irls",
            C=C,
            max_iter=100,
            tol=1e-12,
            device="cpu",
            compute_inference=False,
        ).fit(X, y, sample_weight=current_weights)

    model = fit(weights)
    scaled = fit(17.0 * weights)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(scaled.intercept_, model.intercept_, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(scaled.coef_, model.coef_, rtol=1e-9, atol=1e-9)


def test_glm_weighted_loglikelihood_and_dispersion_match_manual_values():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([-0.5, 0.2, 1.8, 2.1, 5.5], dtype=np.float64)
    weights = np.array([1.0, 2.0, 4.0, 3.0, 8.0], dtype=np.float64)
    model = GeneralizedLinearModel(
        family="gaussian",
        solver="irls",
        C=0.0,
        max_iter=100,
        tol=1e-12,
        device="cpu",
        compute_inference=True,
    ).fit(X, y, sample_weight=weights)

    eta = model.intercept_ + X @ model.coef_
    resid_sq = (y - eta) ** 2
    expected_ll = -0.5 * float(np.sum(weights * resid_sq))
    k = 1 + X.shape[1]
    expected_dispersion = float(np.sum(weights * resid_sq)) / (weights.sum() - k)
    np.testing.assert_allclose(model.loglikelihood, expected_ll, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        model._inference_result.metadata["dispersion"],
        expected_dispersion,
        rtol=1e-12,
        atol=1e-12,
    )

    no_inference = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0, device="cpu",
        compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    eta_no_inf = no_inference.intercept_ + X @ no_inference.coef_
    expected_no_inf = -0.5 * float(np.sum(weights * (y - eta_no_inf) ** 2))
    np.testing.assert_allclose(
        no_inference.loglikelihood, expected_no_inf, rtol=1e-12, atol=1e-12
    )


def test_active_glm_compile_helpers_use_central_policy_and_reraise():
    from pathlib import Path
    from statgpu.glm_core import _irls, _solver_utils

    for filename in (
        "statgpu/glm_core/_irls.py",
        "statgpu/glm_core/_solver_utils.py",
    ):
        source = Path(filename).read_text(encoding="utf-8")
        assert "torch.compile(" not in source
        assert "compile_torch(" in source

    def fail(*args):
        raise RuntimeError("unrelated runtime failure")

    with pytest.raises(RuntimeError, match="unrelated runtime failure"):
        _irls._irls_step_call(fail)
    with pytest.raises(RuntimeError, match="unrelated runtime failure"):
        _solver_utils._fista_step_call(fail)
    with pytest.raises(RuntimeError, match="unrelated runtime failure"):
        _solver_utils._newton_step_call(fail)


def test_torch_weighted_irls_compile_path_is_observable():
    torch = _require_modern_torch_cuda()
    from statgpu.backends._torch_compile import get_torch_compile_diagnostics
    from statgpu.glm_core import _irls
    from statgpu.linear_model import GeneralizedLinearModel

    _irls._IRLS_STEP_COMPILED = None
    torch._dynamo.reset()
    get_torch_compile_diagnostics(clear=True)
    before_graphs = _dynamo_unique_graphs(torch)

    X = np.arange(24.0, dtype=np.float64).reshape(12, 2)
    y = 1.5 + X @ np.array([0.2, -0.1])
    weights = torch.linspace(1.0, 3.0, X.shape[0], dtype=torch.float64, device="cuda")
    model = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0,
        max_iter=20, tol=1e-10, device="torch", compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    torch.cuda.synchronize()

    events = get_torch_compile_diagnostics(clear=True)
    assert _dynamo_unique_graphs(torch) > before_graphs
    assert any(event["status"] == "compiled" for event in events)
    assert not any("fallback" in event["status"] for event in events)
    assert np.isfinite(model.coef_).all()
    assert weights.is_cuda


def test_cupy_weighted_irls_matches_cpu_reference():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    y = np.array([-2.0, -0.5, 0.5, 2.0, 6.0], dtype=np.float64)
    weights_np = np.array([1.0, 2.0, 3.0, 7.0, 20.0], dtype=np.float64)
    weights = cp.asarray(weights_np)
    cpu = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=2.0,
        max_iter=100, tol=1e-12, device="cpu", compute_inference=False,
    ).fit(X, y, sample_weight=weights_np)
    gpu = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=2.0,
        max_iter=100, tol=1e-12, device="cuda", compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(gpu.intercept_, cpu.intercept_, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(gpu.coef_, cpu.coef_, rtol=1e-9, atol=1e-9)
    assert isinstance(weights, cp.ndarray)
'''
    test_path.write_text(test_text, encoding="utf-8")

# Add IRLS to the physical compile benchmark so central-policy coverage is
# machine-readable rather than relying only on a direct smoke test.
benchmark = Path("dev/benchmarks/benchmark_torch_compile_maintenance.py")
bench_text = benchmark.read_text(encoding="utf-8")
bench_text = bench_text.replace(
    '''    from statgpu.linear_model import ElasticNet, Lasso, PenalizedLinearRegression
''',
    '''    from statgpu.linear_model import (
        ElasticNet,
        GeneralizedLinearModel,
        Lasso,
        PenalizedLinearRegression,
    )
''',
    1,
)
bench_text = bench_text.replace(
    '''    cases = {
        "lasso": lambda: Lasso(
''',
    '''    cases = {
        "glm_irls": lambda: GeneralizedLinearModel(
            family="gaussian",
            solver="irls",
            C=0.0,
            max_iter=50,
            tol=1e-7,
            device="torch",
            compute_inference=False,
        ),
        "lasso": lambda: Lasso(
''',
    1,
)
benchmark.write_text(bench_text, encoding="utf-8")
