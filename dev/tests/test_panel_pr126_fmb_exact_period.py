"""Fresh regression coverage for exactly identified Fama-MacBeth periods."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.panel import FamaMacBeth


def _exact_period_fixture():
    # Automatic period intercept + two slopes => k=3. Period 0 has exactly
    # three observations and a full-rank square design. The other periods are
    # overidentified so dropping period 0 materially changes the FM target.
    x0 = np.asarray([[-1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    x1 = np.asarray([[-2.0, 0.5], [-0.5, -1.0], [0.4, 1.5], [1.3, -0.2], [2.2, 0.8]])
    x2 = np.asarray([[-1.5, -0.7], [-0.3, 0.9], [0.6, -1.2], [1.4, 0.4], [2.5, 1.1]])
    params = np.asarray([[0.2, 1.1, -0.4], [1.0, -0.6, 0.8], [-0.7, 0.3, 1.2]])
    blocks = [x0, x1, x2]
    ys = []
    for X_t, beta_t in zip(blocks, params):
        design = np.column_stack([np.ones(X_t.shape[0]), X_t])
        assert np.linalg.matrix_rank(design) == 3
        ys.append(design @ beta_t)
    X = np.vstack(blocks)
    y = np.concatenate(ys)
    time = np.concatenate([np.full(block.shape[0], i, dtype=np.int64) for i, block in enumerate(blocks)])
    return X, y, time, params


def test_fama_macbeth_retains_exactly_identified_full_rank_period_numpy():
    X, y, time, expected_betas = _exact_period_fixture()
    model = FamaMacBeth(cov_type="nonrobust", device="cpu").fit(X, y, time_ids=time)
    assert model.n_periods == 3
    assert model.df_resid == 2
    assert_allclose(np.asarray(model.betas_), expected_betas, rtol=2e-12, atol=2e-13)
    assert_allclose(np.asarray(model.coef_), expected_betas.mean(axis=0), rtol=2e-12, atol=2e-13)


def test_fama_macbeth_exact_period_matches_linearmodels_7():
    pd = pytest.importorskip("pandas")
    lm = pytest.importorskip("linearmodels.panel")
    X, y, time, _ = _exact_period_fixture()
    # Use a unique entity label per observation; Fama-MacBeth groups by time.
    entity = np.arange(len(y), dtype=np.int64)
    index = pd.MultiIndex.from_arrays([entity, time], names=["entity", "time"])
    dependent = pd.Series(y, index=index, name="y")
    exog = pd.DataFrame(
        {"const": 1.0, "x1": X[:, 0], "x2": X[:, 1]}, index=index
    )
    expected = lm.FamaMacBeth(dependent, exog).fit(cov_type="unadjusted", debiased=True)
    actual = FamaMacBeth(cov_type="nonrobust", device="cpu").fit(X, y, time_ids=time)
    assert actual.n_periods == 3
    assert expected.all_params.shape[0] == 3
    assert np.all(np.isfinite(expected.all_params.to_numpy()))
    assert_allclose(np.asarray(actual.betas_), expected.all_params.to_numpy(), rtol=2e-11, atol=2e-12)
    assert_allclose(np.asarray(actual.coef_), expected.params.to_numpy(), rtol=2e-11, atol=2e-12)
    assert_allclose(np.asarray(actual.cov_params_), expected.cov.to_numpy(), rtol=3e-10, atol=3e-12)


def test_fama_macbeth_exactly_identified_period_torch_cpu_matches_numpy():
    torch = pytest.importorskip("torch")
    X, y, time, _ = _exact_period_fixture()
    expected = FamaMacBeth(cov_type="newey-west", bandwidth=1, device="cpu").fit(X, y, time_ids=time)
    actual = FamaMacBeth(cov_type="newey-west", bandwidth=1).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time, dtype=torch.int64),
    )
    assert actual.n_periods == expected.n_periods == 3
    for name in ("betas_", "coef_", "cov_params_", "bse_", "pvalues_", "conf_int_"):
        av = getattr(actual, name)
        if hasattr(av, "detach"):
            av = av.detach().cpu().numpy()
        ev = getattr(expected, name)
        assert_allclose(np.asarray(av), np.asarray(ev), rtol=3e-10, atol=3e-12)


def test_fama_macbeth_square_rank_deficient_period_is_rejected_not_filtered():
    X, y, time, _ = _exact_period_fixture()
    # Make the exactly identified period rank deficient while leaving the two
    # overidentified periods untouched. It must now be retained by the count
    # rule and rejected by the shared SVD rank policy, rather than silently
    # disappearing from the coefficient average.
    mask = time == 0
    X[mask, 1] = 2.0 * X[mask, 0]
    with pytest.raises(ValueError, match=r"retained time period.*rank deficient"):
        FamaMacBeth(device="cpu").fit(X, y, time_ids=time)


def test_fama_macbeth_min_obs_per_period_still_filters_exact_period():
    X, y, time, expected_betas = _exact_period_fixture()
    model = FamaMacBeth(cov_type="nonrobust", min_obs_per_period=4, device="cpu").fit(
        X, y, time_ids=time
    )
    assert model.n_periods == 2
    assert_allclose(np.asarray(model.betas_), expected_betas[1:], rtol=2e-12, atol=2e-13)
