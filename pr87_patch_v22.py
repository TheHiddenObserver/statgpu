from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


old_branch = '''        else:
            # Squared error: centering X and y preserves the objective.
            from statgpu.backends._utils import _get_xp
            xp = _get_xp(backend_name)
            if backend_name == "cupy":
                X_centered = X - xp.mean(X, axis=0)
                y_centered = y - xp.mean(y)
            elif backend_name == "torch":
                import torch
                x_dtype = _torch_promoted_float_dtype(X, y)
                X_float = X.to(dtype=x_dtype)
                y_float = y.to(X.device).to(x_dtype)
                X_centered = X_float - torch.mean(X_float, dim=0)
                y_centered = y_float - torch.mean(y_float)
            else:
                X_centered = X - X.mean(axis=0)
                y_centered = y - y.mean()

            coef, n_iter = fista_solver(
                loss, L2Penalty(alpha=0.0), X_centered, y_centered,
                max_iter=self._max_iter, tol=self._tol,
                init_coef=None, sample_weight=sample_weight,
            )

            _xp_mod = _get_xp(backend_name) if backend_name != "numpy" else np
            X_mean = _to_numpy(_xp_mod.mean(X, axis=0))
            y_mean = float(_xp_mod.mean(y))
            self.coef_ = _to_numpy(coef)
            self.intercept_ = float(y_mean - X_mean @ self.coef_)
            self.n_iter_ = n_iter
            self._params = np.concatenate([[self.intercept_], self.coef_])
'''
new_branch = '''        else:
            # Squared error with an intercept can be profiled by centering.  For
            # weighted loss the centering constants must be weighted means;
            # ordinary means optimize a different objective whenever weights
            # are unequal.  Keep all reductions on the selected backend.
            from statgpu.backends._utils import _get_xp

            xp = _get_xp(backend_name)
            if backend_name == "cupy":
                x_dtype = X.dtype if xp.issubdtype(X.dtype, xp.floating) else xp.float64
                X_float = X.astype(x_dtype, copy=False)
                y_float = xp.asarray(y, dtype=x_dtype)
                if sample_weight is None:
                    X_mean_native = xp.mean(X_float, axis=0)
                    y_mean_native = xp.mean(y_float)
                else:
                    weights = xp.asarray(sample_weight, dtype=x_dtype)
                    weight_sum = xp.sum(weights)
                    X_mean_native = xp.sum(
                        X_float * weights[:, None], axis=0
                    ) / weight_sum
                    y_mean_native = xp.sum(y_float * weights) / weight_sum
                X_centered = X_float - X_mean_native
                y_centered = y_float - y_mean_native
            elif backend_name == "torch":
                import torch

                x_dtype = _torch_promoted_float_dtype(X, y)
                if sample_weight is not None:
                    weight_dtype = (
                        sample_weight.dtype
                        if sample_weight.is_floating_point()
                        else torch.float64
                    )
                    x_dtype = torch.promote_types(x_dtype, weight_dtype)
                X_float = X.to(dtype=x_dtype)
                y_float = y.to(X.device).to(x_dtype)
                if sample_weight is None:
                    X_mean_native = torch.mean(X_float, dim=0)
                    y_mean_native = torch.mean(y_float)
                else:
                    weights = sample_weight.to(X.device).to(x_dtype)
                    weight_sum = torch.sum(weights)
                    X_mean_native = torch.sum(
                        X_float * weights[:, None], dim=0
                    ) / weight_sum
                    y_mean_native = torch.sum(y_float * weights) / weight_sum
                X_centered = X_float - X_mean_native
                y_centered = y_float - y_mean_native
            else:
                X_float = np.asarray(X, dtype=np.float64)
                y_float = np.asarray(y, dtype=np.float64)
                if sample_weight is None:
                    X_mean_native = np.mean(X_float, axis=0)
                    y_mean_native = np.mean(y_float)
                else:
                    weights = np.asarray(sample_weight, dtype=np.float64)
                    weight_sum = np.sum(weights)
                    X_mean_native = np.sum(
                        X_float * weights[:, None], axis=0
                    ) / weight_sum
                    y_mean_native = np.sum(y_float * weights) / weight_sum
                X_centered = X_float - X_mean_native
                y_centered = y_float - y_mean_native

            coef, n_iter = fista_solver(
                loss, L2Penalty(alpha=0.0), X_centered, y_centered,
                max_iter=self._max_iter, tol=self._tol,
                init_coef=None, sample_weight=sample_weight,
            )

            X_mean = _to_numpy(X_mean_native)
            if backend_name == "torch":
                y_mean = float(y_mean_native.item())
            elif backend_name == "cupy":
                y_mean = float(y_mean_native.item())
            else:
                y_mean = float(y_mean_native)
            self.coef_ = _to_numpy(coef)
            self.intercept_ = float(y_mean - X_mean @ self.coef_)
            self.n_iter_ = n_iter
            self._params = np.concatenate([[self.intercept_], self.coef_])
'''
replace_once("statgpu/linear_model/_glm_base.py", old_branch, new_branch)

path = Path("dev/tests/test_maintenance_024_025.py")
text = path.read_text(encoding="utf-8")
marker = "# PR87_GLM_FISTA_WEIGHTED_INTERCEPT_TESTS"
if marker not in text:
    text += '''

# PR87_GLM_FISTA_WEIGHTED_INTERCEPT_TESTS
def _weighted_linear_reference(X, y, weights):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    design = np.column_stack([np.ones(X.shape[0]), X])
    root_w = np.sqrt(weights)
    return np.linalg.lstsq(
        design * root_w[:, None], y * root_w, rcond=None
    )[0]


def test_glm_fista_weighted_intercept_matches_closed_form_wls():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.array(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=np.float64
    )
    y = np.array([-1.0, 0.2, 1.1, 2.0, 8.0, 9.5], dtype=np.float64)
    weights = np.array([8.0, 7.0, 6.0, 2.0, 1.0, 0.5], dtype=np.float64)
    expected = _weighted_linear_reference(X, y, weights)

    model = GeneralizedLinearModel(
        family="gaussian",
        solver="fista",
        C=0.0,
        max_iter=4000,
        tol=1e-11,
        device="cpu",
        compute_inference=False,
    ).fit(X, y, sample_weight=weights)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=2e-5, atol=2e-5)


def test_glm_formula_fista_weighted_intercept_matches_retained_wls():
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {
            "y": [-1.0, 0.2, 99.0, 2.0, 8.0, 9.5],
            "x": [-2.0, -1.0, np.nan, 1.0, 2.0, 3.0],
        }
    )
    weights = np.array([8.0, 7.0, 1000.0, 2.0, 1.0, 0.5])
    retained = np.array([0, 1, 3, 4, 5])
    expected = _weighted_linear_reference(
        data.loc[retained, ["x"]].to_numpy(),
        data.loc[retained, "y"].to_numpy(),
        weights[retained],
    )

    model = GeneralizedLinearModel(
        family="gaussian",
        solver="fista",
        C=0.0,
        max_iter=4000,
        tol=1e-11,
        device="cpu",
        compute_inference=False,
    ).fit(formula="y ~ x", data=data, sample_weight=weights)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=2e-5, atol=2e-5)


def test_torch_glm_formula_fista_weighted_intercept_matches_wls():
    torch = _require_modern_torch_cuda()
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {
            "y": [-1.0, 0.2, 99.0, 2.0, 8.0, 9.5],
            "x": [-2.0, -1.0, np.nan, 1.0, 2.0, 3.0],
        }
    )
    weights_np = np.array([8.0, 7.0, 1000.0, 2.0, 1.0, 0.5])
    weights = torch.as_tensor(weights_np, dtype=torch.float64, device="cuda")
    retained = np.array([0, 1, 3, 4, 5])
    expected = _weighted_linear_reference(
        data.loc[retained, ["x"]].to_numpy(),
        data.loc[retained, "y"].to_numpy(),
        weights_np[retained],
    )

    model = GeneralizedLinearModel(
        family="gaussian",
        solver="fista",
        C=0.0,
        max_iter=4000,
        tol=1e-11,
        device="torch",
        compute_inference=False,
    ).fit(formula="y ~ x", data=data, sample_weight=weights)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=3e-5, atol=3e-5)
    assert weights.is_cuda


def test_cupy_glm_formula_fista_weighted_intercept_matches_wls():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import GeneralizedLinearModel

    data = pd.DataFrame(
        {
            "y": [-1.0, 0.2, 99.0, 2.0, 8.0, 9.5],
            "x": [-2.0, -1.0, np.nan, 1.0, 2.0, 3.0],
        }
    )
    weights_np = np.array([8.0, 7.0, 1000.0, 2.0, 1.0, 0.5])
    weights = cp.asarray(weights_np, dtype=cp.float64)
    retained = np.array([0, 1, 3, 4, 5])
    expected = _weighted_linear_reference(
        data.loc[retained, ["x"]].to_numpy(),
        data.loc[retained, "y"].to_numpy(),
        weights_np[retained],
    )

    model = GeneralizedLinearModel(
        family="gaussian",
        solver="fista",
        C=0.0,
        max_iter=4000,
        tol=1e-11,
        device="cuda",
        compute_inference=False,
    ).fit(formula="y ~ x", data=data, sample_weight=weights)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=3e-5, atol=3e-5)
    assert isinstance(weights, cp.ndarray)
'''
    path.write_text(text, encoding="utf-8")
