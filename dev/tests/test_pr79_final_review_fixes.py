"""Regression tests for the final PR #79 review-fix cycle."""

from pathlib import Path
import inspect
import subprocess

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from statgpu.linear_model import LinearRegression
from statgpu.panel import PooledOLS


def test_linear_regression_fit_preserves_backend_inputs_until_resolution():
    source = inspect.getsource(LinearRegression.fit)
    assert "X_arr = X" in source
    assert "y_arr = y" in source
    assert "from statgpu.backends._utils import _to_numpy" not in source


def test_linear_regression_predict_avoids_eager_numpy_conversion():
    source = inspect.getsource(LinearRegression.predict)
    assert "Preserve backend-native arrays" in source


def test_pooled_hac_time_index_makes_row_order_irrelevant():
    rng = np.random.default_rng(20260721)
    n = 80
    time_index = np.arange(n)
    X = rng.normal(size=(n, 3))
    y = 1.2 + X @ np.array([0.5, -0.8, 0.3]) + rng.normal(scale=0.4, size=n)
    perm = rng.permutation(n)
    ordered = PooledOLS(cov_type="hac", bandwidth=3).fit(X, y, time_index=time_index)
    shuffled = PooledOLS(cov_type="hac", bandwidth=3).fit(
        X[perm], y[perm], time_index=time_index[perm]
    )
    assert_allclose(shuffled.coef_, ordered.coef_, rtol=1e-11, atol=1e-11)
    assert_allclose(shuffled.bse_, ordered.bse_, rtol=1e-10, atol=1e-10)


def test_pooled_hac_time_index_validates_shape():
    X = np.arange(60.0).reshape(20, 3)
    y = np.arange(20.0)
    with pytest.raises(ValueError, match="time_index"):
        PooledOLS(cov_type="hac").fit(X, y, time_index=np.arange(19))


def test_pooled_rank_deficiency_uses_effective_rank_for_df():
    import statsmodels.api as sm

    rng = np.random.default_rng(79)
    x = np.arange(40.0)
    X = np.column_stack([x, 2.0 * x])
    y = 1.0 + 3.0 * x + rng.normal(scale=0.25, size=x.shape[0])
    model = PooledOLS().fit(X, y)
    design = np.column_stack([np.ones(X.shape[0]), X])
    reference = sm.OLS(y, design).fit()
    expected_rank = int(np.linalg.matrix_rank(design))

    assert model.rank_ == expected_rank
    assert model.df_resid == X.shape[0] - expected_rank
    assert model.df_resid == int(reference.df_resid)
    assert_allclose(model.bse_, reference.bse, rtol=1e-8, atol=1e-10)


def test_orchestrator_enforces_exact_clean_worktrees_and_pipefail():
    text = Path("dev/validation/pr79_gpu_orchestrator.py").read_text()
    assert "bash -o pipefail -c" in text
    assert 'for wt in ["head"]' in text
    assert 'for wt in ["base", "head"]' not in text
    assert 'self.upload_package()' not in text
    assert 'git status --porcelain' in text
    assert 'STATGPU_PR79_HEAD_SHA' in text
    assert 'reset --hard {sha}' in text


def test_linear_formula_intercept_semantics_do_not_mutate_public_parameter():
    x = np.linspace(-2.0, 2.0, 60)
    frame = pd.DataFrame({"x": x, "y": 1.75 + 2.5 * x})

    with_intercept = LinearRegression(fit_intercept=False).fit(
        formula="y ~ x", data=frame
    )
    assert with_intercept.fit_intercept is False
    assert np.isclose(with_intercept.intercept_, 1.75, atol=1e-10)
    assert_allclose(with_intercept.coef_, [2.5], atol=1e-10)

    without_intercept = LinearRegression(fit_intercept=True).fit(
        formula="y ~ x - 1", data=frame
    )
    assert without_intercept.fit_intercept is True
    assert without_intercept.intercept_ == 0.0
    expected = np.linalg.lstsq(x[:, None], frame["y"].to_numpy(), rcond=None)[0]
    assert_allclose(without_intercept.coef_, expected, atol=1e-10)


def test_weighted_formula_aligns_weights_after_patsy_drops_rows():
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(7906)
    n = 90
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.9 + 1.3 * x1 - 0.4 * x2 + rng.normal(scale=0.2, size=n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    frame.loc[[4, 17, 51], "x1"] = np.nan
    frame.loc[[9, 52], "y"] = np.nan
    weights = np.linspace(0.2, 2.5, n) ** 2

    model = LinearRegression().fit(
        formula="y ~ x1 + x2",
        data=frame,
        sample_weight=weights,
    )
    kept = frame[["y", "x1", "x2"]].notna().all(axis=1).to_numpy()
    reference = SkLinearRegression().fit(
        frame.loc[kept, ["x1", "x2"]].to_numpy(),
        frame.loc[kept, "y"].to_numpy(),
        sample_weight=weights[kept],
    )

    assert np.isclose(model.intercept_, reference.intercept_, rtol=1e-10, atol=1e-10)
    assert_allclose(model.coef_, reference.coef_, rtol=1e-10, atol=1e-10)
    assert model._sample_weight_fit.shape == (int(kept.sum()),)
    assert_allclose(model._sample_weight_fit, weights[kept])


def test_weighted_formula_rejects_unalignable_weight_length():
    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError, match="sample_weight"):
        LinearRegression().fit(
            formula="y ~ x", data=frame, sample_weight=np.ones(4)
        )


def test_weighted_linear_regression_matches_sklearn_and_statsmodels():
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(7903)
    X = rng.normal(size=(120, 4))
    y = 1.4 + X @ np.array([0.8, -1.1, 0.25, 0.6]) + rng.normal(scale=0.3, size=120)
    weights = np.linspace(0.2, 3.0, X.shape[0]) ** 2

    model = LinearRegression().fit(X, y, sample_weight=weights)
    sk = SkLinearRegression().fit(X, y, sample_weight=weights)
    reference = sm.WLS(y, sm.add_constant(X), weights=weights).fit()

    assert np.isclose(model.intercept_, sk.intercept_, rtol=1e-10, atol=1e-10)
    assert_allclose(model.coef_, sk.coef_, rtol=1e-10, atol=1e-10)
    assert_allclose(model._bse, reference.bse, rtol=1e-8, atol=1e-10)
    assert np.isclose(model.rsquared, sk.score(X, y, sample_weight=weights), atol=1e-12)


def test_weighted_linear_multioutput_broadcasts_weights_by_row():
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(7904)
    X = rng.normal(size=(70, 3))
    beta = np.array([[0.5, -0.2, 0.8], [-0.7, 1.2, 0.1]])
    y = X @ beta.T + np.array([1.0, -2.0]) + rng.normal(scale=0.1, size=(70, 2))
    weights = np.linspace(0.1, 2.0, X.shape[0])

    model = LinearRegression(compute_inference=False).fit(X, y, sample_weight=weights)
    reference = SkLinearRegression().fit(X, y, sample_weight=weights)
    assert_allclose(model.intercept_, reference.intercept_, rtol=1e-10, atol=1e-10)
    assert_allclose(model.coef_, reference.coef_, rtol=1e-10, atol=1e-10)


def test_weighted_linear_rejects_invalid_weights():
    X = np.arange(30.0).reshape(10, 3)
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="sample_weight"):
        LinearRegression().fit(X, y, sample_weight=np.ones(9))
    with pytest.raises(ValueError, match="sample_weight"):
        LinearRegression().fit(X, y, sample_weight=-np.ones(10))
    with pytest.raises(ValueError, match="sample_weight"):
        LinearRegression().fit(X, y, sample_weight=np.zeros(10))


def test_pipefail_propagates_the_failing_pytest_side_of_a_pipeline():
    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", "false | tee /dev/null"],
        check=False,
    )
    assert result.returncode != 0


@pytest.mark.parametrize("backend", ["cupy", "torch"])
def test_linear_regression_gpu_fit_does_not_use_backend_to_numpy(monkeypatch, backend):
    import statgpu.backends._utils as backend_utils
    rng = np.random.default_rng(7905)
    X_np = rng.normal(size=(40, 3))
    y_np = 0.7 + X_np @ np.array([0.5, -0.2, 0.1])
    weights_np = np.linspace(0.25, 2.0, X_np.shape[0])
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy CUDA device unavailable")
        X = cp.asarray(X_np)
        y = cp.asarray(y_np)
        weights = cp.asarray(weights_np)
        model = LinearRegression(device="cuda", compute_inference=False)
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA device unavailable")
        X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")
        y = torch.as_tensor(y_np, dtype=torch.float64, device="cuda")
        weights = torch.as_tensor(weights_np, dtype=torch.float64, device="cuda")
        model = LinearRegression(device="torch", compute_inference=False)

    def forbidden(value):
        raise AssertionError(f"unexpected backend-to-NumPy conversion: {type(value)!r}")

    monkeypatch.setattr(backend_utils, "_to_numpy", forbidden)
    model.fit(X, y, sample_weight=weights)
    pred = model.predict(X[:3])
    assert tuple(pred.shape) == (3,)
    cpu = LinearRegression(compute_inference=False).fit(
        X_np, y_np, sample_weight=weights_np
    )
    assert_allclose(model.coef_, cpu.coef_, rtol=1e-8, atol=1e-9)
    assert np.isclose(model.intercept_, cpu.intercept_, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("backend", ["cupy", "torch"])
def test_gpu_f_stat_degenerate_semantics(backend):
    """GPU helpers must match public CPU semantics on degenerate F tests."""
    y_np = np.array([-1.0, 0.0, 1.0, 2.0])
    design_np = np.column_stack([np.ones(y_np.size), y_np])
    intercept_only_np = np.ones((y_np.size, 1))

    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy CUDA device unavailable")
        from statgpu.backends._gpu_inference_cupy import compute_f_stat_gpu

        y = cp.asarray(y_np)
        perfect_f, perfect_p = compute_f_stat_gpu(
            y, cp.zeros_like(y), cp.asarray(design_np), df_resid=2
        )
        null_f, null_p = compute_f_stat_gpu(
            y,
            y - y.mean(),
            cp.asarray(intercept_only_np),
            df_resid=3,
        )
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA device unavailable")
        from statgpu.backends._gpu_inference_torch import compute_f_stat_torch

        y = torch.as_tensor(y_np, dtype=torch.float64, device="cuda")
        perfect_f, perfect_p = compute_f_stat_torch(
            y,
            torch.zeros_like(y),
            torch.as_tensor(design_np, dtype=torch.float64, device="cuda"),
            df_resid=2,
            device="cuda",
        )
        null_f, null_p = compute_f_stat_torch(
            y,
            y - y.mean(),
            torch.as_tensor(intercept_only_np, dtype=torch.float64, device="cuda"),
            df_resid=3,
            device="cuda",
        )

    assert np.isposinf(perfect_f)
    assert perfect_p == 0.0
    assert np.isnan(null_f)
    assert np.isnan(null_p)
