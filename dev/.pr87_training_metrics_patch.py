from pathlib import Path

path = Path("statgpu/linear_model/wrappers/_logistic.py")
text = path.read_text(encoding="utf-8")

start = text.index("    def _train_classification_table(self):\n")
end = text.index("    @staticmethod\n    def _validate_threshold", start)
new_method = '''    def _train_classification_table(self):
        """Return and cache hard-label training metrics on the active backend.

        Confusion-based metrics are intentionally independent of ROC and
        precision-recall curves so they remain available for one-class targets.
        Ranking metrics are computed lazily by ``auc`` and
        ``average_precision`` because their class-support requirements differ.
        """
        if self._y is None or not self._fitted:
            return None

        if self._train_eval_cache is not None:
            cached = self._train_eval_cache.get("classification_table")
            if cached is not None:
                return cached

        X_train = self._X_design[:, 1:] if self._fit_intercept else self._X_design
        y_pred = self.predict(X_train)
        device = self._get_compute_device()
        if device == Device.CUDA:
            cp = _require_cupy("_train_classification_table")
            y_true = cp.asarray(
                self._to_array(self._y, Device.CUDA)
            ).reshape(-1)
            table = binary_classification_table(
                y_true, y_pred, backend="cupy"
            )
        elif device == Device.TORCH:
            y_true = self._to_array(
                self._y, Device.TORCH, backend="torch"
            ).reshape(-1)
            table = binary_classification_table(
                y_true, y_pred, backend="torch"
            )
        else:
            table = binary_classification_table(
                self._y, self._to_numpy(y_pred), backend="numpy"
            )

        if self._train_eval_cache is None:
            self._train_eval_cache = {}
        self._train_eval_cache["classification_table"] = table
        return table

'''
text = text[:start] + new_method + text[end:]

old_auc = '''    @property
    def auc(self):
        """ROC-AUC on training data."""
        if self._y is None or not self._fitted:
            return None
        # Use cached eval result if available (populated by _train_classification_table)
        if self._train_eval_cache is not None:
            return self._train_eval_cache.get("roc_auc")
        # Trigger cache population via _train_classification_table
        self._train_classification_table()
        if self._train_eval_cache is not None:
            return self._train_eval_cache.get("roc_auc")
        return None

    @property
    def average_precision(self):
        """Average precision on training data."""
        if self._y is None or not self._fitted:
            return None
        # Use cached eval result if available (populated by _train_classification_table)
        if self._train_eval_cache is not None:
            return self._train_eval_cache.get("average_precision")
        # Trigger cache population via _train_classification_table
        self._train_classification_table()
        if self._train_eval_cache is not None:
            return self._train_eval_cache.get("average_precision")
        return None
'''
new_auc = '''    @property
    def auc(self):
        """ROC-AUC on training data."""
        if self._y is None or not self._fitted:
            return None
        if (
            self._train_eval_cache is not None
            and "roc_auc" in self._train_eval_cache
        ):
            return self._train_eval_cache["roc_auc"]

        X_train = self._X_design[:, 1:] if self._fit_intercept else self._X_design
        value = self.roc_auc_score(X_train, self._y)
        if self._train_eval_cache is None:
            self._train_eval_cache = {}
        self._train_eval_cache["roc_auc"] = value
        return value

    @property
    def average_precision(self):
        """Average precision on training data."""
        if self._y is None or not self._fitted:
            return None
        if (
            self._train_eval_cache is not None
            and "average_precision" in self._train_eval_cache
        ):
            return self._train_eval_cache["average_precision"]

        X_train = self._X_design[:, 1:] if self._fit_intercept else self._X_design
        value = self.average_precision_score(X_train, self._y)
        if self._train_eval_cache is None:
            self._train_eval_cache = {}
        self._train_eval_cache["average_precision"] = value
        return value
'''
if text.count(old_auc) != 1:
    raise RuntimeError(f"training ranking property anchor count={text.count(old_auc)}")
text = text.replace(old_auc, new_auc, 1)
path.write_text(text, encoding="utf-8")

test_path = Path("dev/tests/test_pr87_classifier_output_contracts.py")
tests = test_path.read_text(encoding="utf-8")
addition = '''

def test_logistic_training_hard_metrics_support_one_class_targets():
    from statgpu.linear_model import LogisticRegression

    X = np.ones((8, 1), dtype=np.float64)
    y = np.zeros(8, dtype=np.int64)
    model = LogisticRegression(
        fit_intercept=False,
        C=1.0,
        max_iter=200,
        tol=1e-10,
        device="cpu",
        compute_inference=False,
    ).fit(X, y)

    assert model.accuracy == pytest.approx(1.0)
    assert model.precision == pytest.approx(0.0)
    assert model.recall == pytest.approx(0.0)
    assert model.f1 == pytest.approx(0.0)
    with pytest.raises(ValueError, match="both positive and negative"):
        _ = model.auc
    with pytest.raises(ValueError, match="no positive class"):
        _ = model.average_precision


def test_logistic_training_metric_caches_are_independent(monkeypatch):
    model, _, _ = _cpu_logistic_fixture()
    calls = {"auc": 0, "ap": 0}
    original_auc = model.roc_auc_score
    original_ap = model.average_precision_score

    def counted_auc(X, y):
        calls["auc"] += 1
        return original_auc(X, y)

    def counted_ap(X, y):
        calls["ap"] += 1
        return original_ap(X, y)

    monkeypatch.setattr(model, "roc_auc_score", counted_auc)
    monkeypatch.setattr(model, "average_precision_score", counted_ap)

    assert model.accuracy is not None
    assert calls == {"auc": 0, "ap": 0}
    first_auc = model.auc
    first_ap = model.average_precision
    assert model.auc == first_auc
    assert model.average_precision == first_ap
    assert calls == {"auc": 1, "ap": 1}
'''
if "test_logistic_training_hard_metrics_support_one_class_targets" not in tests:
    tests += addition
test_path.write_text(tests, encoding="utf-8")

for changelog, entry in [
    (
        Path("docs/en/changelog.md"),
        "- Decoupled direct LogisticRegression training confusion metrics from ROC/PR evaluation, so accuracy, precision, recall, and F1 remain available for one-class targets while ranking metrics keep their explicit support requirements.\n\n",
    ),
    (
        Path("docs/cn/changelog.md"),
        "- 将直接 LogisticRegression 的训练集混淆指标与 ROC/PR 评估解耦，使单一类别目标仍可获得 accuracy、precision、recall 与 F1，同时排序指标保留其显式类别支持要求。\n\n",
    ),
]:
    current = changelog.read_text(encoding="utf-8")
    heading = "# Changelog\n\n"
    if not current.startswith(heading):
        raise RuntimeError(f"unexpected changelog heading: {changelog}")
    if entry not in current:
        current = heading + entry + current[len(heading):]
    changelog.write_text(current, encoding="utf-8")
