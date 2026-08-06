from pathlib import Path

path = Path("statgpu/linear_model/wrappers/_logistic.py")
text = path.read_text(encoding="utf-8")

anchor = '''    @staticmethod
    def _to_python_float(value):
        """Convert scalar-like values (including CuPy scalars) to float."""
'''
replacement = '''    @staticmethod
    def _validate_threshold(threshold):
        """Return a finite binary decision threshold as a Python float."""
        if (
            isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, Real)
        ):
            raise ValueError(
                "threshold must be a finite real number in [0, 1]"
            )
        threshold = float(threshold)
        if (
            not np.isfinite(threshold)
            or threshold < 0.0
            or threshold > 1.0
        ):
            raise ValueError(
                "threshold must be a finite real number in [0, 1]"
            )
        return threshold

    @staticmethod
    def _to_python_float(value):
        """Convert scalar-like values (including CuPy scalars) to float."""
'''
if text.count(anchor) != 1:
    raise RuntimeError(f"threshold helper anchor count={text.count(anchor)}")
text = text.replace(anchor, replacement, 1)

old_predict = '''        proba = self.predict_proba(X)
        if hasattr(proba, 'is_floating_point'):  # torch tensor
            return (proba[:, 1] >= 0.5).to(dtype=proba.dtype)
        return (proba[:, 1] >= 0.5).astype(int)
'''
new_predict = '''        proba = self.predict_proba(X)
        if type(proba).__module__.startswith("torch"):
            import torch

            return (proba[:, 1] >= 0.5).to(dtype=torch.int64)
        return (proba[:, 1] >= 0.5).astype(np.int64)
'''
if text.count(old_predict) != 1:
    raise RuntimeError(f"predict anchor count={text.count(old_predict)}")
text = text.replace(old_predict, new_predict, 1)

old_threshold = '''        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")
        proba = self.predict_proba(X)
        if hasattr(proba, "to") and hasattr(proba, "dtype"):
            return (proba[:, 1] >= threshold).to(dtype=proba.dtype)
        return (proba[:, 1] >= threshold).astype(int)
'''
new_threshold = '''        threshold = self._validate_threshold(threshold)
        proba = self.predict_proba(X)
        if type(proba).__module__.startswith("torch"):
            import torch

            return (proba[:, 1] >= threshold).to(dtype=torch.int64)
        return (proba[:, 1] >= threshold).astype(np.int64)
'''
if text.count(old_threshold) != 1:
    raise RuntimeError(
        f"predict_with_threshold anchor count={text.count(old_threshold)}"
    )
text = text.replace(old_threshold, new_threshold, 1)

old_score = '''        y_pred = self.predict(X)
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp

            yb = cp.asarray(self._to_array(y, Device.CUDA)).reshape(-1)
            return float(cp.mean(y_pred.reshape(-1) == yb).item())
        if device == Device.TORCH:
            import torch

            yb = self._to_array(y, Device.TORCH, backend="torch").reshape(-1)
            return float(torch.mean((y_pred.reshape(-1) == yb).to(torch.float64)).item())
        y_pred = self._to_numpy(y_pred)
        y = self._to_numpy(y)
        return np.mean(y_pred == y)
'''
new_score = '''        y_pred = self.predict(X).reshape(-1)
        y_validated = validate_binary_response(
            y,
            int(y_pred.shape[0]),
            context="LogisticRegression.score",
        )
        device = self._get_compute_device()
        if device == Device.CUDA:
            import cupy as cp

            yb = cp.asarray(
                self._to_array(y_validated, Device.CUDA)
            ).reshape(-1)
            return float(cp.mean(y_pred == yb).item())
        if device == Device.TORCH:
            import torch

            yb = self._to_array(
                y_validated, Device.TORCH, backend="torch"
            ).reshape(-1)
            return float(
                torch.mean((y_pred == yb).to(torch.float64)).item()
            )
        y_pred_np = np.asarray(self._to_numpy(y_pred)).reshape(-1)
        y_np = np.asarray(self._to_numpy(y_validated)).reshape(-1)
        return float(np.mean(y_pred_np == y_np))
'''
if text.count(old_score) != 1:
    raise RuntimeError(f"score anchor count={text.count(old_score)}")
text = text.replace(old_score, new_score, 1)

path.write_text(text, encoding="utf-8")

test_path = Path("dev/tests/test_pr87_classifier_output_contracts.py")
test_path.write_text('''from __future__ import annotations

import numpy as np
import pytest


def _cpu_logistic_fixture():
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
        compute_inference=False,
    ).fit(X, y)
    return model, X, y


def test_logistic_predict_returns_integer_labels_on_numpy():
    model, X, _ = _cpu_logistic_fixture()
    prediction = model.predict(X)

    assert prediction.dtype == np.int64
    assert prediction.shape == (X.shape[0],)
    assert set(np.unique(prediction)).issubset({0, 1})


def test_logistic_score_flattens_single_column_binary_response():
    model, X, y = _cpu_logistic_fixture()
    expected = float(np.mean(model.predict(X) == y))

    assert model.score(X, y) == pytest.approx(expected)
    assert model.score(X, y[:, None]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "threshold",
    [np.nan, np.inf, -np.inf, -0.01, 1.01, True, "0.5"],
)
def test_logistic_predict_threshold_rejects_invalid_controls(threshold):
    model, X, _ = _cpu_logistic_fixture()

    with pytest.raises(
        ValueError, match=r"finite real number in \\[0, 1\\]"
    ):
        model.predict_with_threshold(X, threshold=threshold)


def test_logistic_predict_threshold_accepts_numpy_real_scalar():
    model, X, _ = _cpu_logistic_fixture()
    prediction = model.predict_with_threshold(
        X, threshold=np.float64(0.5)
    )

    assert prediction.dtype == np.int64
    assert prediction.shape == (X.shape[0],)


def test_logistic_torch_prediction_labels_are_int64(monkeypatch):
    torch = pytest.importorskip("torch")

    from statgpu._config import Device
    from statgpu.linear_model import LogisticRegression

    model = LogisticRegression(
        device="cpu", compute_inference=False
    )
    model._fitted = True
    model.coef_ = np.array([1.0], dtype=np.float64)
    model.intercept_ = 0.0

    monkeypatch.setattr(
        model, "_get_compute_device", lambda: Device.TORCH
    )
    monkeypatch.setattr(
        model,
        "_to_array",
        lambda value, *args, **kwargs: torch.as_tensor(
            value, dtype=torch.float64
        ),
    )

    X = np.array([[-1.0], [1.0]], dtype=np.float64)
    prediction = model.predict(X)
    thresholded = model.predict_with_threshold(X, threshold=0.5)

    assert prediction.dtype == torch.int64
    assert thresholded.dtype == torch.int64
    assert prediction.shape == (2,)
    assert thresholded.shape == (2,)
''', encoding="utf-8")

for changelog, entry in [
    (
        Path("docs/en/changelog.md"),
        "- Aligned direct LogisticRegression prediction contracts across NumPy, CuPy, and Torch: hard labels are integer-valued, single-column responses score without broadcasting, and non-finite decision thresholds are rejected.\n\n",
    ),
    (
        Path("docs/cn/changelog.md"),
        "- 统一直接 LogisticRegression 在 NumPy、CuPy 与 Torch 下的预测契约：硬标签使用整数 dtype，单列响应评分不再发生广播，非有限决策阈值会被拒绝。\n\n",
    ),
]:
    current = changelog.read_text(encoding="utf-8")
    heading = "# Changelog\n\n"
    if not current.startswith(heading):
        raise RuntimeError(f"unexpected changelog heading: {changelog}")
    if entry not in current:
        current = heading + entry + current[len(heading):]
    changelog.write_text(current, encoding="utf-8")
