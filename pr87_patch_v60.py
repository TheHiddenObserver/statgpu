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


replace_once(
    "statgpu/_base.py",
    '''            @functools.wraps(original)
            def guarded(self, *args, **kwargs):
                try:
                    bound = signature.bind(self, *args, **kwargs)
''',
    '''            @functools.wraps(original)
            def guarded(self, *args, **kwargs):
                # A rejected refit must not leave a previously fitted CV model
                # usable.  Reset only estimators that explicitly expose the
                # transactional CV lifecycle hook; other estimator families keep
                # their existing validation behavior.
                if method_name == "fit":
                    reset_cv_state = getattr(self, "_reset_cv_fit_state", None)
                    if callable(reset_cv_state):
                        reset_cv_state()
                try:
                    bound = signature.bind(self, *args, **kwargs)
''',
)

test_path = Path("dev/tests/test_maintenance_024_025.py")
test_text = test_path.read_text(encoding="utf-8")
if "test_cv_finite_guard_resets_stale_state_before_rejecting_input" in test_text:
    raise RuntimeError("v60 tests already present")
test_text += r'''


@pytest.mark.parametrize(
    "module_name,class_name,y,selected_name",
    [
        (
            "statgpu.linear_model.cv._ridge_cv",
            "RidgeCV",
            np.arange(6, dtype=np.float64),
            "alpha_",
        ),
        (
            "statgpu.linear_model.cv._elasticnet_cv",
            "ElasticNetCV",
            np.arange(6, dtype=np.float64),
            "alpha_",
        ),
        (
            "statgpu.linear_model.cv._logistic_cv",
            "LogisticRegressionCV",
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            "C_",
        ),
    ],
)
def test_cv_finite_guard_resets_stale_state_before_rejecting_input(
    module_name, class_name, y, selected_name
):
    import importlib
    from statgpu._config import Device

    module = importlib.import_module(module_name)
    estimator = getattr(module, class_name)(device="cpu", cv=2)
    estimator._fitted = True
    estimator.estimator_ = object()
    estimator.coef_ = np.array([3.0, 4.0])
    estimator.intercept_ = 2.0
    estimator.best_score_ = 1.0
    estimator.cv_results_ = {"stale": True}
    estimator.cv_selected_device_ = Device.TORCH
    setattr(estimator, selected_name, 0.5)

    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimator.fit(X, y)

    assert estimator._fitted is False
    assert estimator.estimator_ is None
    assert estimator.coef_ is None
    assert estimator.intercept_ is None
    assert estimator.best_score_ is None
    assert estimator.cv_results_ is None
    assert estimator.cv_selected_device_ is None
    assert getattr(estimator, selected_name) is None


def test_finite_guard_does_not_reset_cv_state_on_prediction_failure():
    from statgpu.linear_model.cv._ridge_cv import RidgeCV

    estimator = RidgeCV(device="cpu", cv=2)
    estimator._fitted = True

    class FittedModel:
        def predict(self, X):
            return np.zeros(int(X.shape[0]), dtype=np.float64)

    estimator.estimator_ = FittedModel()
    estimator.alpha_ = 0.5
    estimator.coef_ = np.array([1.0])
    estimator.intercept_ = 0.0

    with pytest.raises(ValueError, match="finite"):
        estimator.predict(np.array([[np.nan]], dtype=np.float64))

    assert estimator._fitted is True
    assert estimator.alpha_ == pytest.approx(0.5)
    assert estimator.estimator_ is not None
'''
test_path.write_text(test_text, encoding="utf-8")

for changelog, bullet in (
    (
        "CHANGELOG.md",
        "- Integrated transactional CV reset with the shared public finite-input guard, so NaN/Inf refit attempts invalidate stale RidgeCV, ElasticNetCV, LogisticRegressionCV, and unified penalized-CV state before validation raises.\n",
    ),
    (
        "docs/en/changelog.md",
        "- Integrated transactional CV reset with the shared public finite-input guard, so NaN/Inf refit attempts invalidate stale RidgeCV, ElasticNetCV, LogisticRegressionCV, and unified penalized-CV state before validation raises.\n",
    ),
    (
        "docs/cn/changelog.md",
        "- 将事务式 CV 重置接入共享的公开有限值校验，使 NaN/Inf 重拟合在抛错前先使旧的 RidgeCV、ElasticNetCV、LogisticRegressionCV 与统一 penalized-CV 状态失效。\n",
    ),
):
    p = Path(changelog)
    text = p.read_text(encoding="utf-8")
    marker = "# Changelog\n"
    if bullet.strip() not in text:
        text = text.replace(marker, marker + "\n" + bullet, 1)
    p.write_text(text, encoding="utf-8")
