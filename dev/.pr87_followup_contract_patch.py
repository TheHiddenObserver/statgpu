from pathlib import Path
import textwrap

logistic_path = Path("statgpu/linear_model/wrappers/_logistic.py")
logistic = logistic_path.read_text(encoding="utf-8")

old_imports = '''from statgpu.metrics import (
    binary_average_precision_score,
    binary_precision_recall_curve,
    binary_roc_auc_score,
    binary_roc_curve,
    evaluate_binary_classification,
)
'''
new_imports = '''from statgpu.metrics import (
    binary_average_precision_score,
    binary_classification_table,
    binary_confusion_matrix,
    binary_precision_recall_curve,
    binary_roc_auc_score,
    binary_roc_curve,
    evaluate_binary_classification,
)
'''
if logistic.count(old_imports) != 1:
    raise RuntimeError(f"metrics import count={logistic.count(old_imports)}")
logistic = logistic.replace(old_imports, new_imports, 1)

old_confusion = '''    def confusion_matrix(self, X, y, threshold: float = 0.5) -> np.ndarray:
        """Compute binary confusion matrix on a dataset."""
        threshold = self._validate_threshold(threshold)
        if self._get_compute_device() == Device.CUDA:
            cp = _require_cupy("confusion_matrix")

            y_true = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X))[:, 1]
            out = evaluate_binary_classification(
                y_true,
                y_score,
                threshold=threshold,
                include_curves=False,
                backend="cupy",
            )
            return out["confusion_matrix"]
        if self._get_compute_device() == Device.TORCH:
            y_true = self._to_array(y, Device.TORCH, backend="torch").reshape(-1)
            y_score = self.predict_proba(X)[:, 1]
            out = evaluate_binary_classification(
                y_true,
                y_score,
                threshold=threshold,
                include_curves=False,
                backend="torch",
            )
            return out["confusion_matrix"]

        y_true = self._to_numpy(y)
        y_score = self._to_numpy(self.predict_proba(X))[:, 1]
        out = evaluate_binary_classification(
            y_true,
            y_score,
            threshold=threshold,
            include_curves=False,
            backend="numpy",
        )
        return out["confusion_matrix"]
'''
new_confusion = '''    def confusion_matrix(self, X, y, threshold: float = 0.5) -> np.ndarray:
        """Compute binary confusion matrix on a dataset."""
        threshold = self._validate_threshold(threshold)
        y_pred = self.predict_with_threshold(X, threshold=threshold)
        if self._get_compute_device() == Device.CUDA:
            cp = _require_cupy("confusion_matrix")

            y_true = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            return binary_confusion_matrix(
                y_true, y_pred, backend="cupy"
            )
        if self._get_compute_device() == Device.TORCH:
            y_true = self._to_array(
                y, Device.TORCH, backend="torch"
            ).reshape(-1)
            return binary_confusion_matrix(
                y_true, y_pred, backend="torch"
            )

        y_true = self._to_numpy(y)
        return binary_confusion_matrix(
            y_true, y_pred, backend="numpy"
        )
'''
if logistic.count(old_confusion) != 1:
    raise RuntimeError(f"confusion block count={logistic.count(old_confusion)}")
logistic = logistic.replace(old_confusion, new_confusion, 1)

old_table = '''    def classification_table(self, X, y, threshold: float = 0.5) -> Dict[str, float]:
        """Return a compact classification table on a dataset."""
        threshold = self._validate_threshold(threshold)
        if self._get_compute_device() == Device.CUDA:
            cp = _require_cupy("classification_table")

            y_true = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            y_score = cp.asarray(self.predict_proba(X))[:, 1]
            out = evaluate_binary_classification(
                y_true,
                y_score,
                threshold=threshold,
                include_curves=False,
                backend="cupy",
            )
            return out["classification_table"]
        if self._get_compute_device() == Device.TORCH:
            y_true = self._to_array(y, Device.TORCH, backend="torch").reshape(-1)
            y_score = self.predict_proba(X)[:, 1]
            out = evaluate_binary_classification(
                y_true,
                y_score,
                threshold=threshold,
                include_curves=False,
                backend="torch",
            )
            return out["classification_table"]

        y_true = self._to_numpy(y)
        y_score = self._to_numpy(self.predict_proba(X))[:, 1]
        out = evaluate_binary_classification(
            y_true,
            y_score,
            threshold=threshold,
            include_curves=False,
            backend="numpy",
        )
        return out["classification_table"]
'''
new_table = '''    def classification_table(self, X, y, threshold: float = 0.5) -> Dict[str, float]:
        """Return a compact classification table on a dataset."""
        threshold = self._validate_threshold(threshold)
        y_pred = self.predict_with_threshold(X, threshold=threshold)
        if self._get_compute_device() == Device.CUDA:
            cp = _require_cupy("classification_table")

            y_true = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            return binary_classification_table(
                y_true, y_pred, backend="cupy"
            )
        if self._get_compute_device() == Device.TORCH:
            y_true = self._to_array(
                y, Device.TORCH, backend="torch"
            ).reshape(-1)
            return binary_classification_table(
                y_true, y_pred, backend="torch"
            )

        y_true = self._to_numpy(y)
        return binary_classification_table(
            y_true, y_pred, backend="numpy"
        )
'''
if logistic.count(old_table) != 1:
    raise RuntimeError(f"classification block count={logistic.count(old_table)}")
logistic = logistic.replace(old_table, new_table, 1)

fit_start = logistic.index("    def fit(self, X, y, sample_weight=None):\n")
fit_end = logistic.index("    def _fit_cpu(self, X, y, sample_weight=None):\n", fit_start)
fit_block = logistic[fit_start:fit_end]
reset_marker = "        self._reset_fit_state()\n"
if fit_block.count(reset_marker) != 1:
    raise RuntimeError(f"fit reset marker count={fit_block.count(reset_marker)}")
prefix, body = fit_block.split(reset_marker, 1)
wrapped_fit = (
    prefix
    + reset_marker
    + "        try:\n"
    + textwrap.indent(body, "    ")
    + "        except Exception:\n"
    + "            self._reset_fit_state()\n"
    + "            raise\n\n"
)
logistic = logistic[:fit_start] + wrapped_fit + logistic[fit_end:]
logistic_path.write_text(logistic, encoding="utf-8")

cv_path = Path("statgpu/linear_model/penalized/_penalized_cv.py")
cv = cv_path.read_text(encoding="utf-8")
old_unknown = '''    # Fallback: unweighted loss. Weighted mean cannot be derived from
    # unweighted mean, so weights are ignored for unknown loss types.
    if sw is not None:
        import warnings
        warnings.warn(
            f"_evaluate_loss_numpy: loss '{loss_name}' not in dispatch table, "
            f"falling back to unweighted loss_fn.value(). Sample weights ignored.",
            RuntimeWarning,
            stacklevel=2,
        )
    return float(loss_fn.value(X_design, y_val_np, coef_with_intercept))
'''
new_unknown = '''    # Unknown/custom losses use the same public LossBase contract as the
    # optimized dispatch table, including analytic validation weights.
    return float(
        loss_fn.value(
            X_design,
            y_val_np,
            coef_with_intercept,
            sample_weight=sw,
        )
    )
'''
if cv.count(old_unknown) != 1:
    raise RuntimeError(f"unknown loss fallback count={cv.count(old_unknown)}")
cv = cv.replace(old_unknown, new_unknown, 1)
cv_path.write_text(cv, encoding="utf-8")

classifier_test_path = Path("dev/tests/test_pr87_classifier_output_contracts.py")
classifier_tests = classifier_test_path.read_text(encoding="utf-8")
classifier_addition = '''

def test_logistic_confusion_and_table_support_single_class_targets():
    model, X, _ = _cpu_logistic_fixture()
    y_single = np.zeros(X.shape[0], dtype=np.int64)

    matrix = model.confusion_matrix(X, y_single)
    table = model.classification_table(X, y_single)

    assert matrix.shape == (2, 2)
    assert int(matrix.sum()) == X.shape[0]
    assert table["support_negative"] == X.shape[0]
    assert table["support_positive"] == 0


def test_logistic_failed_backend_fit_clears_partial_publication(monkeypatch):
    model, X, y = _cpu_logistic_fixture()

    def fail_after_partial_publication(*args, **kwargs):
        model.coef_ = np.array([99.0])
        model.intercept_ = 99.0
        model._params = np.array([99.0, 99.0])
        model._loglik = -1.0
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(model, "_fit_cpu", fail_after_partial_publication)
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        model.fit(X, y)

    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._loglik is None


def test_logistic_failed_inference_clears_fitted_outputs(monkeypatch):
    from statgpu.linear_model import LogisticRegression

    _, X, y = _cpu_logistic_fixture()
    model = LogisticRegression(
        C=1.0,
        max_iter=200,
        tol=1e-10,
        device="cpu",
        compute_inference=True,
    )

    def fail_inference():
        model._bse = np.array([1.0, 1.0])
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(model, "_compute_inference", fail_inference)
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        model.fit(X, y)

    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._bse is None
    assert model._loglik is None
'''
if "test_logistic_confusion_and_table_support_single_class_targets" not in classifier_tests:
    classifier_tests += classifier_addition
classifier_test_path.write_text(classifier_tests, encoding="utf-8")

review_test_path = Path("dev/tests/test_pr87_code_review_fix_cycle.py")
review_tests = review_test_path.read_text(encoding="utf-8")
review_addition = '''

def test_unknown_cv_loss_preserves_validation_sample_weight():
    from statgpu.linear_model.penalized._penalized_cv import _evaluate_loss_numpy

    observed = {}

    class CustomLoss:
        def value(self, X, y, coef, sample_weight=None):
            observed["sample_weight"] = np.asarray(sample_weight).copy()
            residual = np.asarray(y) - np.asarray(X) @ np.asarray(coef)
            weights = np.asarray(sample_weight, dtype=np.float64)
            return float(np.dot(weights, residual ** 2) / weights.sum())

    X = np.array([[1.0], [2.0], [4.0]])
    y = np.array([1.0, 1.0, 5.0])
    weights = np.array([1.0, 3.0, 7.0])
    value = _evaluate_loss_numpy(
        "custom_weighted_loss",
        CustomLoss(),
        X,
        y,
        np.array([1.0]),
        0.0,
        False,
        sample_weight=weights,
    )

    expected = np.dot(weights, (y - X[:, 0]) ** 2) / weights.sum()
    assert value == pytest.approx(expected)
    np.testing.assert_array_equal(observed["sample_weight"], weights)
'''
if "test_unknown_cv_loss_preserves_validation_sample_weight" not in review_tests:
    review_tests += review_addition
review_test_path.write_text(review_tests, encoding="utf-8")

for changelog, entry in [
    (
        Path("docs/en/changelog.md"),
        "- Closed follow-up review gaps in direct LogisticRegression and penalized CV: failed fits clear partial state, single-class confusion/table metrics remain available, and custom validation losses retain analytic weights.\n\n",
    ),
    (
        Path("docs/cn/changelog.md"),
        "- 闭合直接 LogisticRegression 与惩罚 CV 的后续审查缺口：失败拟合会清除半发布状态，单一类别仍可计算混淆矩阵/分类表，自定义验证损失保留解析权重。\n\n",
    ),
]:
    current = changelog.read_text(encoding="utf-8")
    heading = "# Changelog\n\n"
    if not current.startswith(heading):
        raise RuntimeError(f"unexpected changelog heading: {changelog}")
    if entry not in current:
        current = heading + entry + current[len(heading):]
    changelog.write_text(current, encoding="utf-8")
