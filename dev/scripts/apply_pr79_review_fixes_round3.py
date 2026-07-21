#!/usr/bin/env python3
"""Apply weighted LinearRegression fixes found in PR79 review round three."""

from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one match, found {count}: {old[:120]!r}"
        )
    p.write_text(text.replace(old, new, 1))


def patch_linear_weighted_fit():
    path = "statgpu/linear_model/wrappers/_linear.py"
    replace_once(
        path,
        '''        self._effective_fit_intercept = bool(fit_intercept)

    def _clear_inference_result(self):
''',
        '''        self._effective_fit_intercept = bool(fit_intercept)
        self._sample_weight_fit = None
        self._raw_resid = None

    def _clear_inference_result(self):
''',
    )
    replace_once(
        path,
        '''        self._clear_inference_result()

        # Formula syntax controls the fitted design without mutating the
''',
        '''        self._clear_inference_result()
        self._sample_weight_fit = None
        self._raw_resid = None

        # Formula syntax controls the fitted design without mutating the
''',
    )

    old_cpu = '''        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        self._nobs = n_samples
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            sqrt_sw = np.sqrt(sample_weight)
            X = X * sqrt_sw[:, np.newaxis]
            y = y * sqrt_sw
        
        if self._effective_fit_intercept:
            self._X_design = np.column_stack([np.ones(n_samples, dtype=X.dtype), X])
        else:
            self._X_design = X.copy()
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        coef, _, _, _ = np.linalg.lstsq(self._X_design, y, rcond=None)
'''
    new_cpu = '''        X_raw = np.asarray(X)
        y_raw = np.asarray(y)

        n_samples, n_features = X_raw.shape
        self._nobs = n_samples
        y_2d = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float).reshape(-1)
            if sw.shape[0] != n_samples:
                raise ValueError("sample_weight must have length n_samples")
            if not np.all(np.isfinite(sw)) or np.any(sw < 0) or float(sw.sum()) <= 0:
                raise ValueError("sample_weight must be finite, non-negative, and have positive sum")
            sqrt_sw = np.sqrt(sw)
            X_fit = X_raw * sqrt_sw[:, None]
            y_fit = y_2d * sqrt_sw[:, None]
            intercept_column = sqrt_sw[:, None]
            self._sample_weight_fit = sw.copy()
        else:
            X_fit = X_raw
            y_fit = y_2d
            intercept_column = np.ones((n_samples, 1), dtype=X_raw.dtype)

        if self._effective_fit_intercept:
            self._X_design = np.column_stack([intercept_column, X_fit])
        else:
            self._X_design = X_fit.copy()

        coef, _, _, _ = np.linalg.lstsq(self._X_design, y_fit, rcond=None)
'''
    replace_once(path, old_cpu, new_cpu)
    replace_once(
        path,
        '''        y_pred = self._X_design @ coef
        self._resid = y - y_pred
        if self._resid.shape[1] == 1:
            self._resid = self._resid[:, 0]
''',
        '''        y_pred = self._X_design @ coef
        self._resid = y_fit - y_pred
        raw_pred = (
            coef[0] + X_raw @ coef[1:]
            if self._effective_fit_intercept
            else X_raw @ coef
        )
        raw_resid = y_2d - raw_pred
        self._raw_resid = raw_resid[:, 0] if raw_resid.shape[1] == 1 else raw_resid
        if self._resid.shape[1] == 1:
            self._resid = self._resid[:, 0]
''',
    )

    old_gpu = '''        # Ensure CuPy arrays
        X = cp.asarray(X)
        y = cp.asarray(y)
        
        if sample_weight is not None:
            sample_weight = cp.asarray(sample_weight)
            sqrt_sw = cp.sqrt(sample_weight)
            X = X * sqrt_sw[:, cp.newaxis]
            y = y * sqrt_sw
        
        if self._effective_fit_intercept:
            X_design = cp.column_stack([cp.ones(n_samples, dtype=X.dtype), X])
        else:
            X_design = X
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
'''
    new_gpu = '''        # Ensure CuPy arrays and retain raw arrays for weighted diagnostics.
        X_raw = cp.asarray(X)
        y_raw = cp.asarray(y)
        y_2d = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

        sw = None
        if sample_weight is not None:
            sw = cp.asarray(sample_weight, dtype=cp.float64).reshape(-1)
            if sw.shape[0] != n_samples:
                raise ValueError("sample_weight must have length n_samples")
            valid = cp.all(cp.isfinite(sw)) & cp.all(sw >= 0) & (cp.sum(sw) > 0)
            if not bool(valid.item()):
                raise ValueError("sample_weight must be finite, non-negative, and have positive sum")
            sqrt_sw = cp.sqrt(sw)
            X_fit = X_raw * sqrt_sw[:, cp.newaxis]
            y_fit = y_2d * sqrt_sw[:, cp.newaxis]
            intercept_column = sqrt_sw[:, cp.newaxis]
        else:
            X_fit = X_raw
            y_fit = y_2d
            intercept_column = cp.ones((n_samples, 1), dtype=X_raw.dtype)

        if self._effective_fit_intercept:
            X_design = cp.column_stack([intercept_column, X_fit])
        else:
            X_design = X_fit
        y = y_fit
'''
    replace_once(path, old_gpu, new_gpu)
    replace_once(
        path,
        '''        except Exception:
            coef = cp.linalg.solve(XtX, Xty)
        
        # Compute predictions and residuals on GPU
        y_pred = X_design @ coef
        resid = y - y_pred
''',
        '''        except Exception:
            coef = cp.linalg.lstsq(X_design, y, rcond=None)[0]

        # Compute weighted inference residuals and raw diagnostic residuals.
        y_pred = X_design @ coef
        resid = y - y_pred
        raw_pred = (
            coef[0] + X_raw @ coef[1:]
            if self._effective_fit_intercept
            else X_raw @ coef
        )
        raw_resid = y_2d - raw_pred
''',
    )
    replace_once(
        path,
        '''        coef_np = coef.get()
        resid_np = resid.get()
''',
        '''        coef_np = coef.get()
        resid_np = resid.get()
        raw_resid_np = raw_resid.get()
        self._sample_weight_fit = None if sw is None else sw.get()
''',
    )
    replace_once(
        path,
        '''        if resid_np.shape[1] == 1:
            self._resid = resid_np[:, 0]
        else:
            self._resid = resid_np
''',
        '''        if resid_np.shape[1] == 1:
            self._resid = resid_np[:, 0]
        else:
            self._resid = resid_np
        self._raw_resid = (
            raw_resid_np[:, 0] if raw_resid_np.shape[1] == 1 else raw_resid_np
        )
''',
    )

    old_torch = '''        if sample_weight is not None:
            if not isinstance(sample_weight, torch.Tensor):
                sample_weight = torch.from_numpy(np.asarray(sample_weight)).to(torch_device)
            if sample_weight.dtype != torch.float64:
                sample_weight = sample_weight.to(torch.float64)
            sqrt_sw = torch.sqrt(sample_weight)
            X = X * sqrt_sw[:, None]
            y = y * sqrt_sw

        if self._effective_fit_intercept:
            X_design = torch.cat([torch.ones(n_samples, 1, dtype=X.dtype, device=torch_device), X], dim=1)
        else:
            X_design = X.clone()

        if y.ndim == 1:
            y = y.reshape(-1, 1)
'''
    new_torch = '''        X_raw = X
        y_raw = y
        y_2d = y_raw.reshape(-1, 1) if y_raw.ndim == 1 else y_raw

        sw = None
        if sample_weight is not None:
            sw = torch.as_tensor(sample_weight, dtype=torch.float64, device=torch_device).reshape(-1)
            if sw.shape[0] != n_samples:
                raise ValueError("sample_weight must have length n_samples")
            valid = torch.all(torch.isfinite(sw)) & torch.all(sw >= 0) & (torch.sum(sw) > 0)
            if not bool(valid.item()):
                raise ValueError("sample_weight must be finite, non-negative, and have positive sum")
            sqrt_sw = torch.sqrt(sw)
            X_fit = X_raw * sqrt_sw[:, None]
            y_fit = y_2d * sqrt_sw[:, None]
            intercept_column = sqrt_sw[:, None]
        else:
            X_fit = X_raw
            y_fit = y_2d
            intercept_column = torch.ones(
                n_samples, 1, dtype=X_raw.dtype, device=X_raw.device
            )

        if self._effective_fit_intercept:
            X_design = torch.cat([intercept_column, X_fit], dim=1)
        else:
            X_design = X_fit.clone()
        y = y_fit
'''
    replace_once(path, old_torch, new_torch)
    replace_once(
        path,
        '''        except Exception:
            coef = torch.linalg.solve(XtX, Xty)

        # Compute predictions and residuals on Torch
        y_pred = X_design @ coef
        resid = y - y_pred
''',
        '''        except Exception:
            coef = torch.linalg.lstsq(X_design, y).solution

        # Compute weighted inference residuals and raw diagnostic residuals.
        y_pred = X_design @ coef
        resid = y - y_pred
        raw_pred = (
            coef[0] + X_raw @ coef[1:]
            if self._effective_fit_intercept
            else X_raw @ coef
        )
        raw_resid = y_2d - raw_pred
''',
    )
    replace_once(
        path,
        '''        coef_np = coef.detach().cpu().numpy()
        resid_np = resid.detach().cpu().numpy()
''',
        '''        coef_np = coef.detach().cpu().numpy()
        resid_np = resid.detach().cpu().numpy()
        raw_resid_np = raw_resid.detach().cpu().numpy()
        self._sample_weight_fit = (
            None if sw is None else sw.detach().cpu().numpy()
        )
''',
    )
    # The same residual-storage block appears once more in the Torch path now.
    p = Path(path)
    text = p.read_text()
    old_store = '''        if resid_np.shape[1] == 1:
            self._resid = resid_np[:, 0]
        else:
            self._resid = resid_np
        self._df_resid = df_resid
'''
    new_store = '''        if resid_np.shape[1] == 1:
            self._resid = resid_np[:, 0]
        else:
            self._resid = resid_np
        self._raw_resid = (
            raw_resid_np[:, 0] if raw_resid_np.shape[1] == 1 else raw_resid_np
        )
        self._df_resid = df_resid
'''
    if text.count(old_store) != 1:
        raise RuntimeError(f"{path}: Torch residual storage block mismatch")
    p.write_text(text.replace(old_store, new_store, 1))

    # Weighted R-squared/F-test use raw residuals and weighted centering.
    replace_once(
        path,
        '''        y_mean = np.mean(self._y)
        ss_tot = np.sum((self._y - y_mean) ** 2)
        ss_res = np.sum(self._resid ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
''',
        '''        y = np.asarray(self._y, dtype=float)
        resid = np.asarray(
            self._raw_resid if self._raw_resid is not None else self._resid,
            dtype=float,
        )
        weights = self._sample_weight_fit
        if weights is None:
            y_mean = np.mean(y, axis=0) if y.ndim > 1 else np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = np.sum(resid ** 2)
        else:
            weights = np.asarray(weights, dtype=float)
            y_mean = np.average(y, axis=0, weights=weights)
            weight_shape = (weights.shape[0],) + (1,) * (y.ndim - 1)
            w = weights.reshape(weight_shape)
            ss_tot = np.sum(w * (y - y_mean) ** 2)
            ss_res = np.sum(w * resid ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
''',
    )
    replace_once(
        path,
        '''        y = np.asarray(self._y, dtype=float)
        resid = np.asarray(self._resid, dtype=float)
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        ss_res = float(np.sum(resid ** 2))
''',
        '''        y = np.asarray(self._y, dtype=float)
        resid = np.asarray(
            self._raw_resid if self._raw_resid is not None else self._resid,
            dtype=float,
        )
        weights = self._sample_weight_fit
        if weights is None:
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            ss_res = float(np.sum(resid ** 2))
        else:
            weights = np.asarray(weights, dtype=float)
            y_mean = np.average(y, weights=weights)
            ss_tot = float(np.sum(weights * (y - y_mean) ** 2))
            ss_res = float(np.sum(weights * resid ** 2))
''',
    )


def strengthen_weighted_tests():
    path = Path("dev/tests/test_pr79_final_review_fixes.py")
    text = path.read_text()
    insertion = '''

def test_weighted_linear_regression_matches_sklearn_and_statsmodels():
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(7903)
    X = rng.normal(size=(120, 4))
    y = 1.4 + X @ np.array([0.8, -1.1, 0.25, 0.6]) + rng.normal(scale=0.3, size=120)
    weights = np.linspace(0.2, 3.0, X.shape[0]) ** 2

    model = LinearRegression().fit(X, y, sample_weight=weights)
    sk = SkLinearRegression().fit(X, y, sample_weight=weights)
    reference = sm.WLS(y, sm.add_constant(X), weights=weights).fit()

    assert np.isclose(model.intercept_, sk.intercept_, rtol=1e-10, atol=1e-10)
    assert_allclose(model.coef_, sk.coef_, rtol=1e-10, atol=1e-10)
    assert_allclose(model._bse, reference.bse, rtol=1e-8, atol=1e-10)
    assert np.isclose(model.rsquared, sk.score(X, y, sample_weight=weights), atol=1e-12)


def test_weighted_linear_multioutput_broadcasts_weights_by_row():
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(7904)
    X = rng.normal(size=(70, 3))
    beta = np.array([[0.5, -0.2, 0.8], [-0.7, 1.2, 0.1]])
    y = X @ beta.T + np.array([1.0, -2.0]) + rng.normal(scale=0.1, size=(70, 2))
    weights = np.linspace(0.1, 2.0, X.shape[0])

    model = LinearRegression(compute_inference=False).fit(X, y, sample_weight=weights)
    reference = SkLinearRegression().fit(X, y, sample_weight=weights)
    assert_allclose(model.intercept_, reference.intercept_, rtol=1e-10, atol=1e-10)
    assert_allclose(model.coef_, reference.coef_, rtol=1e-10, atol=1e-10)


def test_weighted_linear_rejects_invalid_weights():
    X = np.arange(30.0).reshape(10, 3)
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="sample_weight"):
        LinearRegression().fit(X, y, sample_weight=np.ones(9))
    with pytest.raises(ValueError, match="sample_weight"):
        LinearRegression().fit(X, y, sample_weight=-np.ones(10))
    with pytest.raises(ValueError, match="sample_weight"):
        LinearRegression().fit(X, y, sample_weight=np.zeros(10))
'''
    anchor = '''

def test_pipefail_propagates_the_failing_pytest_side_of_a_pipeline():
'''
    if text.count(anchor) != 1:
        raise RuntimeError("weighted test insertion anchor mismatch")
    text = text.replace(anchor, insertion + anchor, 1)

    old_gpu_data = '''    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy CUDA device unavailable")
        X = cp.arange(60, dtype=cp.float64).reshape(20, 3)
        y = X @ cp.asarray([0.5, -0.2, 0.1])
        model = LinearRegression(device="cuda", compute_inference=False)
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA device unavailable")
        X = torch.arange(60, dtype=torch.float64, device="cuda").reshape(20, 3)
        y = X @ torch.tensor([0.5, -0.2, 0.1], dtype=torch.float64, device="cuda")
        model = LinearRegression(device="torch", compute_inference=False)
'''
    new_gpu_data = '''    rng = np.random.default_rng(7905)
    X_np = rng.normal(size=(40, 3))
    y_np = 0.7 + X_np @ np.array([0.5, -0.2, 0.1])
    weights_np = np.linspace(0.25, 2.0, X_np.shape[0])
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy CUDA device unavailable")
        X = cp.asarray(X_np)
        y = cp.asarray(y_np)
        weights = cp.asarray(weights_np)
        model = LinearRegression(device="cuda", compute_inference=False)
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA device unavailable")
        X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")
        y = torch.as_tensor(y_np, dtype=torch.float64, device="cuda")
        weights = torch.as_tensor(weights_np, dtype=torch.float64, device="cuda")
        model = LinearRegression(device="torch", compute_inference=False)
'''
    if text.count(old_gpu_data) != 1:
        raise RuntimeError("GPU test data block mismatch")
    text = text.replace(old_gpu_data, new_gpu_data, 1)
    text = text.replace(
        '''    model.fit(X, y)
    pred = model.predict(X[:3])
    assert tuple(pred.shape) == (3,)
''',
        '''    model.fit(X, y, sample_weight=weights)
    pred = model.predict(X[:3])
    assert tuple(pred.shape) == (3,)
    cpu = LinearRegression(compute_inference=False).fit(
        X_np, y_np, sample_weight=weights_np
    )
    assert_allclose(model.coef_, cpu.coef_, rtol=1e-8, atol=1e-9)
    assert np.isclose(model.intercept_, cpu.intercept_, rtol=1e-8, atol=1e-9)
''',
        1,
    )
    path.write_text(text)


def main():
    patch_linear_weighted_fit()
    strengthen_weighted_tests()


if __name__ == "__main__":
    main()
