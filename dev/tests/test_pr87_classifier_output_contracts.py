from __future__ import annotations

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
        ValueError, match=r"finite real number in \[0, 1\]"
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
        ValueError, match=r"finite real number in \[0, 1\]"
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
