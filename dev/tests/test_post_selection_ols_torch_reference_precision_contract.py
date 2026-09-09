import numpy as np
import pytest

from statgpu.inference._reference_distribution import two_sided_reference_inference
from statgpu.linear_model import ElasticNet
from statgpu.linear_model.penalized._post_selection_ols import (
    compute_post_selection_ols_inference,
)


def test_torch_general_df_student_t_pvalue_matches_scipy_reference():
    torch = pytest.importorskip("torch")
    scipy_stats = pytest.importorskip("scipy.stats")

    # The first nonzero value is in the high-curvature portion of the
    # maintained 40k beta LUT for df=216.  Linear interpolation on Torch 2.0
    # misses the exact two-sided Student-t tail there by several parts in 1e-6.
    statistic = torch.tensor(
        [0.0, 1.4814782343390125, 2.0, 4.0],
        dtype=torch.float64,
    )
    pvalues, critical = two_sided_reference_inference(
        statistic,
        distribution="t",
        alpha=0.05,
        backend="torch",
        xp=torch,
        df=216,
        device="cpu",
    )
    expected = 2.0 * scipy_stats.t.sf(statistic.detach().cpu().numpy(), df=216)

    assert isinstance(pvalues, torch.Tensor)
    assert pvalues.device == statistic.device
    assert isinstance(critical, torch.Tensor)
    assert critical.device == statistic.device
    np.testing.assert_allclose(
        pvalues.detach().cpu().numpy(),
        expected,
        rtol=0.0,
        atol=2e-8,
    )


def test_torch_high_df_near_zero_student_t_pvalue_matches_scipy_reference():
    torch = pytest.importorskip("torch")
    scipy_stats = pytest.importorskip("scipy.stats")

    # At high df, df / (df + t**2) rounds to exactly one for sufficiently
    # small nonzero t even though the two-sided Student-t p-value is still
    # distinguishable from one.  The near-zero density integral must preserve
    # that tail instead of publishing an exact p-value of 1.0.
    statistic = torch.tensor(
        [0.0, 1e-8, 2.8e-6, 1e-4, 1e-3, 1e-2],
        dtype=torch.float64,
    )
    pvalues, _critical = two_sided_reference_inference(
        statistic,
        distribution="t",
        alpha=0.05,
        backend="torch",
        xp=torch,
        df=100_000,
        device="cpu",
    )
    expected = 2.0 * scipy_stats.t.sf(
        statistic.detach().cpu().numpy(),
        df=100_000,
    )

    np.testing.assert_allclose(
        pvalues.detach().cpu().numpy(),
        expected,
        rtol=0.0,
        atol=2e-8,
    )
    assert float(pvalues[2]) < 1.0


def test_rank_deficient_post_selection_torch_pvalues_match_numpy_contract():
    torch = pytest.importorskip("torch")

    # Match the schema-v7 physical rank-deficient closure fixture.  Fit only
    # once on CPU to establish the active set, then execute the auxiliary
    # post-selection refit independently with NumPy and Torch-CPU numerics.
    rng = np.random.default_rng(251)
    n = 220
    x = rng.normal(size=n)
    X = np.column_stack([x, x, rng.normal(size=n), rng.normal(size=n)])
    y = 0.25 + 2.0 * x + 0.6 * X[:, 2] + rng.normal(scale=0.35, size=n)
    weights = rng.uniform(0.4, 1.7, size=n)

    model = ElasticNet(
        alpha=0.02,
        l1_ratio=0.5,
        solver="fista",
        inference_method="post_selection_ols",
        compute_inference=False,
        device="cpu",
        max_iter=6000,
        tol=1e-9,
    ).fit(X, y, sample_weight=weights)
    selected = np.flatnonzero(np.abs(np.asarray(model.coef_)) > 1e-15)
    assert 0 in selected and 1 in selected

    model._selected_backend_name = "numpy"
    model._selected_backend_device = "cpu"
    compute_post_selection_ols_inference(model, X, y, sample_weight=weights)
    numpy_meta = dict(model._inference_result.metadata)
    numpy_outputs = {
        "params": np.asarray(model._params, dtype=np.float64).copy(),
        "bse": np.asarray(model._bse, dtype=np.float64).copy(),
        "statistic": np.asarray(model._tvalues, dtype=np.float64).copy(),
        "pvalue": np.asarray(model._pvalues, dtype=np.float64).copy(),
        "ci": np.asarray(model._conf_int, dtype=np.float64).copy(),
    }
    assert numpy_meta["refit_rank_deficient"] is True

    model._selected_backend_name = "torch"
    model._selected_backend_device = "cpu"
    compute_post_selection_ols_inference(
        model,
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        sample_weight=torch.as_tensor(weights, dtype=torch.float64),
    )
    torch_meta = dict(model._inference_result.metadata)

    assert torch_meta["numerical_backend"] == "torch"
    assert torch_meta["numerical_device"] == "cpu"
    assert torch_meta["refit_rank_deficient"] is True
    assert torch_meta["refit_rank"] == numpy_meta["refit_rank"]
    assert torch_meta["refit_df_resid"] == numpy_meta["refit_df_resid"]

    np.testing.assert_allclose(model._params, numpy_outputs["params"], rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(model._bse, numpy_outputs["bse"], rtol=0.0, atol=2e-7)
    np.testing.assert_allclose(
        model._tvalues,
        numpy_outputs["statistic"],
        rtol=0.0,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        model._pvalues,
        numpy_outputs["pvalue"],
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(model._conf_int, numpy_outputs["ci"], rtol=0.0, atol=5e-7)
