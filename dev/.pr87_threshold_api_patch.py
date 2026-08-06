from pathlib import Path

path = Path("statgpu/linear_model/wrappers/_logistic.py")
text = path.read_text(encoding="utf-8")

old_confusion = '''    def confusion_matrix(self, X, y, threshold: float = 0.5) -> np.ndarray:
        """Compute binary confusion matrix on a dataset."""
        if self._get_compute_device() == Device.CUDA:
'''
new_confusion = '''    def confusion_matrix(self, X, y, threshold: float = 0.5) -> np.ndarray:
        """Compute binary confusion matrix on a dataset."""
        threshold = self._validate_threshold(threshold)
        if self._get_compute_device() == Device.CUDA:
'''
if text.count(old_confusion) != 1:
    raise RuntimeError(f"confusion anchor count={text.count(old_confusion)}")
text = text.replace(old_confusion, new_confusion, 1)

old_table = '''    def classification_table(self, X, y, threshold: float = 0.5) -> Dict[str, float]:
        """Return a compact classification table on a dataset."""
        if self._get_compute_device() == Device.CUDA:
'''
new_table = '''    def classification_table(self, X, y, threshold: float = 0.5) -> Dict[str, float]:
        """Return a compact classification table on a dataset."""
        threshold = self._validate_threshold(threshold)
        if self._get_compute_device() == Device.CUDA:
'''
if text.count(old_table) != 1:
    raise RuntimeError(f"classification anchor count={text.count(old_table)}")
text = text.replace(old_table, new_table, 1)

old_evaluate = '''        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")

        if self._get_compute_device() == Device.CUDA:
'''
new_evaluate = '''        threshold = self._validate_threshold(threshold)

        if self._get_compute_device() == Device.CUDA:
'''
if text.count(old_evaluate) != 1:
    raise RuntimeError(f"evaluate anchor count={text.count(old_evaluate)}")
text = text.replace(old_evaluate, new_evaluate, 1)
path.write_text(text, encoding="utf-8")

test_path = Path("dev/tests/test_pr87_classifier_output_contracts.py")
tests = test_path.read_text(encoding="utf-8")
addition = '''

@pytest.mark.parametrize(
    "method_name",
    ["confusion_matrix", "classification_table", "evaluate_classification"],
)
@pytest.mark.parametrize("threshold", [np.nan, np.inf, -0.01, 1.01, True, "0.5"])
def test_logistic_evaluation_threshold_contract_is_consistent(
    method_name, threshold
):
    model, X, y = _cpu_logistic_fixture()
    method = getattr(model, method_name)

    with pytest.raises(
        ValueError, match=r"finite real number in \\[0, 1\\]"
    ):
        method(X, y, threshold=threshold)


@pytest.mark.parametrize(
    "method_name",
    ["confusion_matrix", "classification_table", "evaluate_classification"],
)
def test_logistic_evaluation_threshold_accepts_numpy_real(method_name):
    model, X, y = _cpu_logistic_fixture()
    method = getattr(model, method_name)

    result = method(X, y, threshold=np.float64(0.5))
    assert result is not None
'''
if "test_logistic_evaluation_threshold_contract_is_consistent" not in tests:
    tests += addition
test_path.write_text(tests, encoding="utf-8")
