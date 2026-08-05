from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one match, found {count}: {old[:180]!r}"
        )
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# RidgeCV lifecycle and transactional state publication.
ridge = "statgpu/linear_model/cv/_ridge_cv.py"
replace_once(
    ridge,
    '''        self.estimator_ = None
        self.cv_selected_device_ = None

    def fit(self, X, y, sample_weight=None):
''',
    '''        self.estimator_ = None
        self.cv_selected_device_ = None

    def _reset_cv_fit_state(self):
        """Clear all fitted outputs before a new CV attempt."""
        self._fitted = False
        self.alpha_ = None
        self.alphas_ = None
        self.cv_results_ = None
        self.mean_mse_ = None
        self.best_score_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.estimator_ = None
        self.cv_selected_device_ = None

    def fit(self, X, y, sample_weight=None):
''',
)
replace_once(
    ridge,
    '''        from statgpu.cross_validation._base import validate_cv_sample_weight
''',
    '''        self._reset_cv_fit_state()
        from statgpu.cross_validation._base import validate_cv_sample_weight
''',
)
replace_once(
    ridge,
    '''        refit_device = cv_refit_device(device_name, cv_backend_name)
        self.cv_selected_device_ = refit_device

        # Run CV to select alpha
''',
    '''        refit_device = cv_refit_device(device_name, cv_backend_name)

        # Run CV to select alpha
''',
)
replace_once(
    ridge,
    '''        # Store CV results
        self.alpha_ = float(details["alpha"])
        self.alphas_ = np.asarray(details["alphas"], dtype=np.float64)
        mse_path = np.asarray(details["mse_path"], dtype=np.float64)
        mean_mse = np.asarray(details["mean_mse"], dtype=np.float64)

        self.cv_results_ = {"mse_path": mse_path}
        self.mean_mse_ = mean_mse

        if np.any(np.isfinite(mean_mse)):
            # sklearn convention: best_score_ is negative MSE (higher is better)
            self.best_score_ = -float(np.nanmin(mean_mse))
        else:
            self.best_score_ = np.nan

        # Fit final model with selected alpha.
''',
    '''        # Keep candidate results local until the final refit succeeds.
        selected_alpha = float(details["alpha"])
        selected_alphas = np.asarray(details["alphas"], dtype=np.float64)
        mse_path = np.asarray(details["mse_path"], dtype=np.float64)
        mean_mse = np.asarray(details["mean_mse"], dtype=np.float64)
        best_score = (
            -float(np.nanmin(mean_mse))
            if np.any(np.isfinite(mean_mse))
            else np.nan
        )

        # Fit final model with selected alpha.
''',
)
replace_once(
    ridge,
    '''        estimator = Ridge(
            alpha=self.alpha_,
''',
    '''        estimator = Ridge(
            alpha=selected_alpha,
''',
)
replace_once(
    ridge,
    '''        estimator.fit(X, y, sample_weight=sample_weight)

        self.estimator_ = estimator
        self.coef_ = np.asarray(estimator.coef_)
        self.intercept_ = estimator.intercept_
        self.n_iter_ = getattr(estimator, 'n_iter_', None)

        self._fitted = True
''',
    '''        estimator.fit(X, y, sample_weight=sample_weight)

        self.alpha_ = selected_alpha
        self.alphas_ = selected_alphas
        self.cv_results_ = {"mse_path": mse_path}
        self.mean_mse_ = mean_mse
        self.best_score_ = best_score
        self.estimator_ = estimator
        self.coef_ = np.asarray(estimator.coef_)
        self.intercept_ = estimator.intercept_
        self.n_iter_ = getattr(estimator, 'n_iter_', None)
        self.cv_selected_device_ = refit_device
        self._fitted = True
''',
)

# ElasticNetCV lifecycle and transactional state publication.
elastic = "statgpu/linear_model/cv/_elasticnet_cv.py"
replace_once(
    elastic,
    '''        self.estimator_ = None
        self.cv_selected_device_ = None

    def _fit_cv(self, X, y, sample_weight=None):
''',
    '''        self.estimator_ = None
        self.cv_selected_device_ = None

    def _reset_cv_fit_state(self):
        """Clear all fitted outputs before a new CV attempt."""
        self._fitted = False
        self.alpha_ = None
        self.l1_ratio_ = None
        self.coef_ = None
        self.intercept_ = None
        self.cv_results_ = None
        self.best_score_ = None
        self.n_iter_ = None
        self.estimator_ = None
        self.cv_selected_device_ = None

    def _fit_cv(self, X, y, sample_weight=None):
''',
)
replace_once(
    elastic,
    '''        device_request = self._device
''',
    '''        self._reset_cv_fit_state()
        device_request = self._device
''',
)
replace_once(
    elastic,
    '''        refit_device = cv_refit_device(device_request, cv_backend_name)
        self.cv_selected_device_ = refit_device

        # Normalize l1_ratio to list
''',
    '''        refit_device = cv_refit_device(device_request, cv_backend_name)

        # Normalize l1_ratio to list
''',
)
replace_once(
    elastic,
    '''        # Store CV results
        self.alpha_ = best_alpha
        self.l1_ratio_ = best_l1_ratio
        self.cv_results_ = {
            "mse_path": details["mse_path"],
            "mean_mse": details["mean_mse"],
            "std_mse": details["std_mse"],
            "alphas": details["alphas"],
            "l1_ratios": details["l1_ratios"],
            "best_alpha": self.alpha_,
            "best_l1_ratio": self.l1_ratio_,
        }
        # sklearn convention: best_score_ is negative MSE (higher is better)
        self.best_score_ = -float(details["best_mse"])

        # Fit final model on full data with best parameters
        final_model = ElasticNet(
            alpha=self.alpha_,
            l1_ratio=self.l1_ratio_,
''',
    '''        # Keep candidate results local until the final refit succeeds.
        selected_alpha = float(best_alpha)
        selected_l1_ratio = float(best_l1_ratio)
        cv_results = {
            "mse_path": details["mse_path"],
            "mean_mse": details["mean_mse"],
            "std_mse": details["std_mse"],
            "alphas": details["alphas"],
            "l1_ratios": details["l1_ratios"],
            "best_alpha": selected_alpha,
            "best_l1_ratio": selected_l1_ratio,
        }
        best_score = -float(details["best_mse"])

        # Fit final model on full data with best parameters
        final_model = ElasticNet(
            alpha=selected_alpha,
            l1_ratio=selected_l1_ratio,
''',
)
replace_once(
    elastic,
    '''        final_model.fit(X, y, sample_weight=sample_weight)

        self.coef_ = final_model.coef_.copy()
        self.intercept_ = final_model.intercept_
        self.n_iter_ = final_model.n_iter_
        self.estimator_ = final_model
        self._fitted = True
''',
    '''        final_model.fit(X, y, sample_weight=sample_weight)

        self.alpha_ = selected_alpha
        self.l1_ratio_ = selected_l1_ratio
        self.cv_results_ = cv_results
        self.best_score_ = best_score
        self.coef_ = final_model.coef_.copy()
        self.intercept_ = final_model.intercept_
        self.n_iter_ = final_model.n_iter_
        self.estimator_ = final_model
        self.cv_selected_device_ = refit_device
        self._fitted = True
''',
)

# LogisticRegressionCV lifecycle and transactional state publication.
logistic = "statgpu/linear_model/cv/_logistic_cv.py"
replace_once(
    logistic,
    '''        self.estimator_ = None
        self.cv_selected_device_ = None

    def fit(self, X, y, sample_weight=None):
''',
    '''        self.estimator_ = None
        self.cv_selected_device_ = None

    def _reset_cv_fit_state(self):
        """Clear all fitted outputs before a new CV attempt."""
        self._fitted = False
        self.C_ = None
        self.Cs_ = None
        self.cv_results_ = None
        self.mean_loss_ = None
        self.best_score_ = None
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        self.estimator_ = None
        self.cv_selected_device_ = None

    def fit(self, X, y, sample_weight=None):
''',
)
replace_once(
    logistic,
    '''        # Preserve response residency; only a scalar validity decision syncs.
        _validate_binary_cv_response(y)
''',
    '''        self._reset_cv_fit_state()
        # Preserve response residency; only a scalar validity decision syncs.
        _validate_binary_cv_response(y)
''',
)
replace_once(
    logistic,
    '''        refit_device = cv_refit_device(device_name, cv_backend_name)
        self.cv_selected_device_ = refit_device

        # Run CV to select C
''',
    '''        refit_device = cv_refit_device(device_name, cv_backend_name)

        # Run CV to select C
''',
)
replace_once(
    logistic,
    '''        # Store CV results
        self.C_ = float(details["C"])
        self.Cs_ = np.asarray(details["Cs"], dtype=np.float64)
        loss_path = np.asarray(details["loss_path"], dtype=np.float64)
        mean_loss = np.asarray(details["mean_loss"], dtype=np.float64)

        self.cv_results_ = {"loss_path": loss_path}
        self.mean_loss_ = mean_loss

        if np.any(np.isfinite(mean_loss)):
            # sklearn convention: best_score_ is negative loss (higher is better)
            self.best_score_ = -float(np.nanmin(mean_loss))
        else:
            self.best_score_ = np.nan

        # Fit final model with selected C
        estimator = LogisticRegression(
            C=self.C_,
''',
    '''        # Keep candidate results local until the final refit succeeds.
        selected_C = float(details["C"])
        selected_Cs = np.asarray(details["Cs"], dtype=np.float64)
        loss_path = np.asarray(details["loss_path"], dtype=np.float64)
        mean_loss = np.asarray(details["mean_loss"], dtype=np.float64)
        best_score = (
            -float(np.nanmin(mean_loss))
            if np.any(np.isfinite(mean_loss))
            else np.nan
        )

        # Fit final model with selected C
        estimator = LogisticRegression(
            C=selected_C,
''',
)
replace_once(
    logistic,
    '''        estimator.fit(X, y, sample_weight=sample_weight)

        self.estimator_ = estimator
        self.coef_ = np.asarray(estimator.coef_)
        self.intercept_ = estimator.intercept_
        self.n_iter_ = getattr(estimator, 'n_iter_', None)

        self._fitted = True
''',
    '''        estimator.fit(X, y, sample_weight=sample_weight)

        self.C_ = selected_C
        self.Cs_ = selected_Cs
        self.cv_results_ = {"loss_path": loss_path}
        self.mean_loss_ = mean_loss
        self.best_score_ = best_score
        self.estimator_ = estimator
        self.coef_ = np.asarray(estimator.coef_)
        self.intercept_ = estimator.intercept_
        self.n_iter_ = getattr(estimator, 'n_iter_', None)
        self.cv_selected_device_ = refit_device
        self._fitted = True
''',
)

# Regression tests for stale-state and transactional refit behavior.
test_path = Path("dev/tests/test_maintenance_024_025.py")
test_text = test_path.read_text(encoding="utf-8")
if "test_dedicated_cv_failed_refit_clears_previous_state" in test_text:
    raise RuntimeError("v59 tests already present")
test_text += r'''


@pytest.mark.parametrize(
    "module_name,class_name,selector_name,y,selected_attrs",
    [
        (
            "statgpu.linear_model.cv._ridge_cv",
            "RidgeCV",
            "_select_ridge_alpha_cv",
            np.arange(6, dtype=np.float64),
            ("alpha_", "alphas_", "mean_mse_"),
        ),
        (
            "statgpu.linear_model.cv._elasticnet_cv",
            "ElasticNetCV",
            "_select_elasticnet_params_cv",
            np.arange(6, dtype=np.float64),
            ("alpha_", "l1_ratio_"),
        ),
        (
            "statgpu.linear_model.cv._logistic_cv",
            "LogisticRegressionCV",
            "_select_logistic_c_cv",
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            ("C_", "Cs_", "mean_loss_"),
        ),
    ],
)
def test_dedicated_cv_failed_refit_clears_previous_state(
    monkeypatch, module_name, class_name, selector_name, y, selected_attrs
):
    import importlib
    from statgpu._config import Device

    module = importlib.import_module(module_name)
    estimator = getattr(module, class_name)(device="cpu", cv=2)
    estimator._fitted = True
    estimator.estimator_ = object()
    estimator.coef_ = np.array([9.0, 8.0])
    estimator.intercept_ = 7.0
    estimator.best_score_ = 6.0
    estimator.cv_results_ = {"stale": True}
    estimator.cv_selected_device_ = Device.TORCH
    for name in selected_attrs:
        setattr(estimator, name, np.array([5.0]) if name.endswith("s_") else 5.0)

    monkeypatch.setattr(
        module,
        selector_name,
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("selection failed")
        ),
    )
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    with pytest.raises(RuntimeError, match="selection failed"):
        estimator.fit(X, y)

    assert estimator._fitted is False
    assert estimator.estimator_ is None
    assert estimator.coef_ is None
    assert estimator.intercept_ is None
    assert estimator.best_score_ is None
    assert estimator.cv_results_ is None
    assert estimator.cv_selected_device_ is None
    for name in selected_attrs:
        assert getattr(estimator, name) is None


def test_ridge_cv_final_refit_failure_does_not_publish_partial_state(monkeypatch):
    import statgpu.linear_model.cv._ridge_cv as module

    details = {
        "alpha": 0.5,
        "alphas": np.array([0.5]),
        "mse_path": np.array([[1.0]]),
        "mean_mse": np.array([1.0]),
    }
    monkeypatch.setattr(module, "_select_ridge_alpha_cv", lambda *a, **k: details)

    class FailingRidge:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise RuntimeError("final refit failed")

    monkeypatch.setattr(module, "Ridge", FailingRidge)
    estimator = module.RidgeCV(device="cpu", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    with pytest.raises(RuntimeError, match="final refit failed"):
        estimator.fit(X, np.arange(6, dtype=np.float64))
    assert estimator._fitted is False
    assert estimator.alpha_ is None
    assert estimator.estimator_ is None
    assert estimator.cv_selected_device_ is None


def test_elasticnet_cv_final_refit_failure_does_not_publish_partial_state(monkeypatch):
    import statgpu.linear_model.cv._elasticnet_cv as module

    details = {
        "mse_path": np.array([[[1.0]]]),
        "mean_mse": np.array([[1.0]]),
        "std_mse": np.array([[0.0]]),
        "alphas": {0: np.array([0.5])},
        "l1_ratios": np.array([0.5]),
        "best_mse": 1.0,
    }
    monkeypatch.setattr(
        module,
        "_select_elasticnet_params_cv",
        lambda *a, **k: (0.5, 0.5, details),
    )

    class FailingElasticNet:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise RuntimeError("final refit failed")

    monkeypatch.setattr(module, "ElasticNet", FailingElasticNet)
    estimator = module.ElasticNetCV(device="cpu", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    with pytest.raises(RuntimeError, match="final refit failed"):
        estimator.fit(X, np.arange(6, dtype=np.float64))
    assert estimator._fitted is False
    assert estimator.alpha_ is None
    assert estimator.l1_ratio_ is None
    assert estimator.estimator_ is None
    assert estimator.cv_selected_device_ is None


def test_logistic_cv_final_refit_failure_does_not_publish_partial_state(monkeypatch):
    import statgpu.linear_model.cv._logistic_cv as module

    details = {
        "C": 1.0,
        "Cs": np.array([1.0]),
        "loss_path": np.array([[0.5]]),
        "mean_loss": np.array([0.5]),
    }
    monkeypatch.setattr(module, "_select_logistic_c_cv", lambda *a, **k: details)

    class FailingLogistic:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            raise RuntimeError("final refit failed")

    monkeypatch.setattr(module, "LogisticRegression", FailingLogistic)
    estimator = module.LogisticRegressionCV(device="cpu", cv=2)
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="final refit failed"):
        estimator.fit(X, y)
    assert estimator._fitted is False
    assert estimator.C_ is None
    assert estimator.estimator_ is None
    assert estimator.cv_selected_device_ is None
'''
test_path.write_text(test_text, encoding="utf-8")

for changelog, bullet in (
    (
        "CHANGELOG.md",
        "- Made dedicated RidgeCV, ElasticNetCV, and LogisticRegressionCV refits failure-safe: every fit attempt clears stale fitted state, and CV selections are published only after the final model refit succeeds.\n",
    ),
    (
        "docs/en/changelog.md",
        "- Made dedicated RidgeCV, ElasticNetCV, and LogisticRegressionCV refits failure-safe: every fit attempt clears stale fitted state, and CV selections are published only after the final model refit succeeds.\n",
    ),
    (
        "docs/cn/changelog.md",
        "- 使专用 RidgeCV、ElasticNetCV 与 LogisticRegressionCV 的重拟合具备失败安全语义：每次 fit 均先清除旧拟合状态，仅在最终模型重拟合成功后发布 CV 选择结果。\n",
    ),
):
    p = Path(changelog)
    text = p.read_text(encoding="utf-8")
    marker = "# Changelog\n"
    if bullet.strip() not in text:
        text = text.replace(marker, marker + "\n" + bullet, 1)
    p.write_text(text, encoding="utf-8")
