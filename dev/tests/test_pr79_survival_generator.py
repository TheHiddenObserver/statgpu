"""Regression tests for PR79 delayed-entry survival benchmark data."""

import numpy as np

from dev.benchmarks.pr79.generators.survival import generate_coxph_entry
from statgpu.survival import CoxPH


def test_delayed_entry_generator_produces_valid_aligned_observations():
    """Censoring must not leave entry times after observed follow-up times."""
    X, time, event, entry, beta = generate_coxph_entry(
        n_samples=200,
        n_features=4,
        seed=42,
    )

    n_observed = time.shape[0]
    assert 0 < n_observed <= 200
    assert X.shape == (n_observed, 4)
    assert event.shape == entry.shape == (n_observed,)
    assert beta.shape == (4,)
    assert np.all(entry < time)


def test_delayed_entry_full_case_fits_coxph():
    """The fixed full benchmark case is consumable by the CPU Cox path."""
    X, time, event, entry, _ = generate_coxph_entry(
        n_samples=200,
        n_features=4,
        seed=42,
    )

    model = CoxPH(
        ties="efron",
        penalty=0.0,
        device="cpu",
        max_iter=50,
        compute_inference=True,
    )
    model.fit(X, time, event, entry=entry)

    assert model.coef_.shape == (4,)
    assert np.all(np.isfinite(model.coef_))
    assert np.isfinite(model._log_likelihood)
