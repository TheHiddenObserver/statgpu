from pathlib import Path

path = Path("statgpu/linear_model/wrappers/_logistic.py")
text = path.read_text(encoding="utf-8")
old = '''        try:
            auc = self.auc
        except ValueError:
            auc = None
        auc_display = self._to_python_float(auc)
        print(f"ROC-AUC:                    {auc_display:>15.4f}")
        try:
            ap = self.average_precision
        except ValueError:
            ap = None
        ap_display = self._to_python_float(ap)
'''
new = '''        try:
            auc = self.auc
        except ValueError as exc:
            if "only one class" not in str(exc).lower():
                raise
            auc = None
        auc_display = self._to_python_float(auc)
        print(f"ROC-AUC:                    {auc_display:>15.4f}")
        try:
            ap = self.average_precision
        except ValueError as exc:
            if "no positive class" not in str(exc).lower():
                raise
            ap = None
        ap_display = self._to_python_float(ap)
'''
if text.count(old) != 1:
    raise RuntimeError(f"summary error boundary anchor count={text.count(old)}")
text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")

test_path = Path("dev/tests/test_pr87_classifier_output_contracts.py")
tests = test_path.read_text(encoding="utf-8")
addition = '''

@pytest.mark.parametrize("metric", ["auc", "average_precision"])
def test_logistic_summary_propagates_unrelated_metric_value_errors(
    monkeypatch, metric
):
    from statgpu.linear_model import LogisticRegression

    X = np.array(
        [[-2.0], [-1.0], [-0.25], [0.25], [1.0], [2.0]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = LogisticRegression(
        C=1.0,
        max_iter=200,
        tol=1e-10,
        device="cpu",
        compute_inference=True,
    ).fit(X, y)

    if metric == "auc":
        monkeypatch.setattr(
            model,
            "roc_auc_score",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("programming shape bug")
            ),
        )
    else:
        _ = model.auc
        monkeypatch.setattr(
            model,
            "average_precision_score",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                ValueError("programming shape bug")
            ),
        )

    with pytest.raises(ValueError, match="programming shape bug"):
        model.summary()
'''
if "test_logistic_summary_propagates_unrelated_metric_value_errors" not in tests:
    tests += addition
test_path.write_text(tests, encoding="utf-8")
