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


def test_pipefail_propagates_the_failing_pytest_side_of_a_pipeline():
    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", "false | tee /dev/null"],
        check=False,
    )
    assert result.returncode != 0


@pytest.mark.parametrize("backend", ["cupy", "torch"])
def test_linear_regression_gpu_fit_does_not_use_backend_to_numpy(monkeypatch, backend):
    import statgpu.backends._utils as backend_utils
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy CUDA device unavailable")
        X = cp.arange(60, dtype=cp.float64).reshape(20, 3)
        y = X @ cp.asarray([0.5, -0.2, 0.1])
        model = LinearRegression(device="cuda", compute_inference=False)
    else:
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA device unavailable")
        X = torch.arange(60, dtype=torch.float64, device="cuda").reshape(20, 3)
        y = X @ torch.tensor([0.5, -0.2, 0.1], dtype=torch.float64, device="cuda")
        model = LinearRegression(device="torch", compute_inference=False)

    def forbidden(value):
        raise AssertionError(f"unexpected backend-to-NumPy conversion: {type(value)!r}")

    monkeypatch.setattr(backend_utils, "_to_numpy", forbidden)
    model.fit(X, y)
    pred = model.predict(X[:3])
    assert tuple(pred.shape) == (3,)
