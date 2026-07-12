"""Regression tests for the post-Ridge ANOVA and kernel-method review."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from statgpu.anova import bonferroni, f_twoway, f_welch, tukey_hsd
from statgpu.nonparametric.kernel_methods import KernelPCA, KernelRidge, KernelRidgeCV, Nystroem, chi2_kernel
from statgpu.nonparametric.kernel_methods._kernels import _chi2_kernel_numpy_fallback


def _additive_factor_f(data, factor):
    """Reference partial F test from nested additive least-squares models."""
    rows = []
    y = []
    n_a = len(data)
    n_b = len(data[0])
    for i in range(n_a):
        for j in range(n_b):
            for value in np.asarray(data[i][j], dtype=float):
                rows.append((i, j))
                y.append(value)
    y = np.asarray(y)

    intercept = np.ones((len(y), 1))
    a_dummy = np.column_stack([
        np.fromiter((i == level for i, _ in rows), dtype=float)
        for level in range(1, n_a)
    ])
    b_dummy = np.column_stack([
        np.fromiter((j == level for _, j in rows), dtype=float)
        for level in range(1, n_b)
    ])
    full = np.column_stack([intercept, a_dummy, b_dummy])
    reduced = np.column_stack([intercept, b_dummy]) if factor == "a" else np.column_stack([intercept, a_dummy])

    resid_full = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
    resid_reduced = y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
    sse_full = float(resid_full @ resid_full)
    sse_reduced = float(resid_reduced @ resid_reduced)
    df_effect = full.shape[1] - reduced.shape[1]
    df_resid = len(y) - full.shape[1]
    return ((sse_reduced - sse_full) / df_effect) / (sse_full / df_resid), sse_full, df_resid


def test_twoway_additive_absorbs_interaction_into_residual():
    rng = np.random.RandomState(123)
    data = []
    for i in range(2):
        row = []
        for j in range(3):
            interaction = 2.5 if (i, j) == (1, 2) else 0.0
            row.append(rng.normal(loc=1.2 * i - 0.7 * j + interaction, scale=0.4, size=12))
        data.append(row)

    result = f_twoway(data, interaction=False)
    f_a, sse_additive, df_resid = _additive_factor_f(data, "a")
    f_b, _, _ = _additive_factor_f(data, "b")

    assert_allclose(result.factor_a_statistic, f_a, rtol=1e-10, atol=1e-12)
    assert_allclose(result.factor_b_statistic, f_b, rtol=1e-10, atol=1e-12)
    assert_allclose(result.ss_within, sse_additive, rtol=1e-10, atol=1e-12)
    assert result.df_within == df_resid


def test_twoway_rejects_unbalanced_design_until_ss_type_is_explicit():
    rng = np.random.RandomState(0)
    data = [
        [rng.normal(size=5), rng.normal(size=8)],
        [rng.normal(size=7), rng.normal(size=6)],
    ]
    with pytest.raises(ValueError, match="balanced"):
        f_twoway(data)


def test_twoway_requires_two_levels_per_factor():
    with pytest.raises(ValueError, match="at least 2 levels"):
        f_twoway([[np.arange(5.0), np.arange(5.0)]])


def test_posthoc_identical_constant_groups_do_not_reject():
    g1 = np.ones(8)
    g2 = np.ones(8)
    tukey = tukey_hsd(g1, g2).comparisons[0]
    bonf = bonferroni(g1, g2).comparisons[0]
    for comp in (tukey, bonf):
        assert comp.pvalue == pytest.approx(1.0)
        assert comp.reject is False
        assert comp.mean_diff == pytest.approx(0.0)
        assert comp.ci_lower == pytest.approx(0.0)
        assert comp.ci_upper == pytest.approx(0.0)


def test_welch_rejects_mixed_zero_variance_groups():
    with pytest.raises(ValueError, match="zero variance"):
        f_welch(np.ones(5), np.arange(5.0), np.arange(5.0) + 1)


def test_welch_preserves_fractional_denominator_df():
    result = f_welch(
        np.array([0.0, 1.0, 3.0, 8.0]),
        np.array([1.0, 2.0, 2.5, 4.0, 9.0]),
        np.array([-1.0, 0.0, 0.5, 1.0, 1.2, 7.0]),
    )
    assert isinstance(result.df_within, float)
    assert not float(result.df_within).is_integer()


def test_chi2_kernel_rejects_negative_input():
    X = np.array([[1.0, -0.1], [0.5, 0.2]])
    with pytest.raises(ValueError, match="non-negative"):
        chi2_kernel(X)


def test_chi2_numpy_fallback_matches_sklearn():
    from sklearn.metrics.pairwise import chi2_kernel as sklearn_chi2

    rng = np.random.RandomState(5)
    X = np.abs(rng.normal(size=(7, 9)))
    Y = np.abs(rng.normal(size=(4, 9)))
    expected = sklearn_chi2(X, Y, gamma=0.7)
    actual = _chi2_kernel_numpy_fallback(X, Y, gamma=0.7, max_elements=30)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_kernel_ridge_multioutput_score_matches_sklearn_r2():
    from sklearn.metrics import r2_score

    rng = np.random.RandomState(11)
    X = rng.normal(size=(50, 4))
    y = np.column_stack([
        X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.05, size=50),
        2 * X[:, 2] + rng.normal(scale=0.2, size=50),
    ])
    model = KernelRidge(alpha=0.2, kernel="rbf", gamma=0.4).fit(X, y)
    pred = np.asarray(model.predict(X))
    assert_allclose(model.score(X, y), r2_score(y, pred, multioutput="uniform_average"), rtol=1e-12)


def test_kernel_ridge_constant_target_force_finite_semantics():
    X = np.arange(12.0).reshape(-1, 1)
    model = KernelRidge(alpha=0.0, kernel="rbf", gamma=0.2).fit(X, np.ones(12))
    assert model.score(X, np.ones(12)) == pytest.approx(1.0)
    assert model.score(X, np.zeros(12)) == pytest.approx(0.0)


def test_kernel_ridge_cv_validates_cv_and_reports_fold_r2():
    X = np.arange(18.0).reshape(-1, 1)
    y = np.sin(X[:, 0])
    with pytest.raises(ValueError, match="cv"):
        KernelRidgeCV(cv=1).fit(X, y)
    with pytest.raises(ValueError, match="cv"):
        KernelRidgeCV(cv=19).fit(X, y)

    model = KernelRidgeCV(alphas=[0.01, 0.1, 1.0], cv=3, random_state=0).fit(X, y)
    best_idx = int(np.flatnonzero(np.asarray(model.cv_results_["alphas"]) == model.alpha_)[0])
    expected = np.asarray(model.cv_results_["mean_r2"])[best_idx].mean()
    assert model.best_score_ == pytest.approx(float(expected))


def test_kernel_pca_fit_transform_matches_training_transform():
    rng = np.random.RandomState(19)
    X = rng.normal(size=(35, 3))
    model = KernelPCA(n_components=4, kernel="rbf", gamma=0.6, alpha=1.0)
    fit_transformed = np.asarray(model.fit_transform(X))
    transformed = np.asarray(model.transform(X))
    assert_allclose(fit_transformed, transformed, rtol=1e-10, atol=1e-10)


def test_nystroem_sigmoid_uses_stable_svd_normalization():
    rng = np.random.RandomState(23)
    X = rng.normal(size=(40, 5))
    transformed = np.asarray(
        Nystroem(kernel="sigmoid", n_components=15, gamma=0.3, coef0=-0.4, random_state=1)
        .fit_transform(X)
    )
    assert np.all(np.isfinite(transformed))
    assert np.max(np.abs(transformed)) < 1e6
