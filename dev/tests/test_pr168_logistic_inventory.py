"""Keep the binary-logistic inventory claim scoped to its public estimator."""

import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, ndtr

from statgpu.linear_model import LogisticRegression, OrderedProbitRegression

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("language", ["en", "cn"])
def test_inventory_distinguishes_logistic_from_ordered_probit(language):
    text = (ROOT / f"docs/{language}/guides/implemented-methods.md").read_text(
        encoding="utf-8"
    )
    rows = [line for line in text.splitlines()
            if line.startswith("| `LogisticRegression` |")]
    assert len(rows) == 1
    assert "logit" in rows[0].lower()
    assert "probit" not in rows[0].lower()
    assert "| `OrderedProbitRegression` |" in text
    assert "[`LogisticRegression`](../models/logistic-regression.md)" in text
    assert "[`OrderedProbitRegression`](../models/ordered.md)" in text
    assert "`n_categories=2`" in text
    assert callable(OrderedProbitRegression)


def test_standalone_logistic_uses_logit_without_a_link_selector():
    assert "link" not in inspect.signature(LogisticRegression).parameters
    with pytest.raises(TypeError, match="link"):
        LogisticRegression(link="probit", device="cpu")

    X = np.linspace(-2.0, 2.0, 12)[:, None]
    y = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1])
    model = LogisticRegression(
        C=0, device="cpu", compute_inference=False, max_iter=100, tol=1e-10,
    ).fit(X, y)
    assert model.converged_
    eta = X @ model.coef_ + model.intercept_
    probabilities = model.predict_proba(X)
    np.testing.assert_allclose(probabilities[:, 1], expit(eta), atol=1e-12)
    np.testing.assert_allclose(probabilities[:, 0], 1.0 - expit(eta), atol=1e-12)
    # Ensure this fixture would distinguish a normal-CDF prediction from logit.
    assert np.max(np.abs(expit(eta) - ndtr(eta))) > 0.05


def test_ordered_probit_supports_the_binary_normal_cdf_model():
    rng = np.random.default_rng(168)
    X = rng.normal(size=(300, 2))
    y = rng.binomial(1, ndtr(-0.3 + X @ np.array([0.7, -0.4])))
    model = OrderedProbitRegression(
        n_categories=2, device="cpu", compute_inference=False,
        max_iter=200, tol=1e-10,
    ).fit(X, y)
    assert model.n_iter_ < model.max_iter
    assert model.thresholds_.shape == (3,)
    eta = X @ model.coef_ - model.thresholds_[1]
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    np.testing.assert_allclose(probabilities[:, 1], ndtr(eta), atol=1e-12)
    np.testing.assert_allclose(probabilities[:, 0], ndtr(-eta), atol=1e-12)
    assert np.max(np.abs(expit(eta) - ndtr(eta))) > 0.05
