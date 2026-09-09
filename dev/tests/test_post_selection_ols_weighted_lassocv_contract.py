import numpy as np
import pytest

from statgpu.linear_model import Lasso, LassoCV
from statgpu.linear_model import _weighted_lassocv_review_contract as weighted_cv


def _problem(seed=9138, n=96, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.asarray([1.2, -0.85, 0.5, 0.0, 0.0])
    y = 0.4 + X @ beta + rng.normal(scale=0.45, size=n)
    weights = rng.uniform(0.23, 1.71, size=n)
    return X, y, weights


def _folds(n):
    idx = np.arange(n, dtype=np.int64)
    blocks = np.array_split(idx, 3)
    return [
        (np.setdiff1d(idx, val, assume_unique=True), val)
        for val in blocks
    ]


def _weighted_mse(y_true, y_pred, weight):
    weight = np.asarray(weight, dtype=np.float64)
    return float(np.sum(weight * (y_true - y_pred) ** 2) / np.sum(weight))


def test_weighted_cv_working_problem_matches_declared_average_loss():
    X, y, weights = _problem(n=57)
    X_work, y_work, X_mean, y_mean = weighted_cv._weighted_working_numpy(
        X,
        y,
        weights,
        fit_intercept=True,
    )
    w_sum = float(np.sum(weights))
    X_centered = X - np.average(X, axis=0, weights=weights)
    y_centered = y - np.average(y, weights=weights)

    np.testing.assert_allclose(X_mean, np.average(X, axis=0, weights=weights))
    assert y_mean == pytest.approx(float(np.average(y, weights=weights)))
    np.testing.assert_allclose(
        X_work.T @ X_work / X.shape[0],
        X_centered.T @ (weights[:, None] * X_centered) / w_sum,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        X_work.T @ y_work / X.shape[0],
        X_centered.T @ (weights * y_centered) / w_sum,
        rtol=2e-14,
        atol=2e-14,
    )


def test_weighted_lassocv_selection_matches_direct_fold_refits():
    X, y, weights = _problem(seed=9139)
    folds = _folds(X.shape[0])
    alphas = np.asarray([0.22, 0.11, 0.055, 0.0275], dtype=np.float64)
    common = dict(
        max_iter=5000,
        tol=1e-9,
        solver="fista",
        device="cpu",
        compute_inference=False,
    )

    manual_mean_mse = []
    for alpha in alphas:
        fold_mse = []
        for train_idx, val_idx in folds:
            model = Lasso(alpha=float(alpha), **common).fit(
                X[train_idx],
                y[train_idx],
                sample_weight=weights[train_idx],
            )
            fold_mse.append(
                _weighted_mse(
                    y[val_idx],
                    model.predict(X[val_idx]),
                    weights[val_idx],
                )
            )
        manual_mean_mse.append(float(np.mean(fold_mse)))
    manual_mean_mse = np.asarray(manual_mean_mse)
    expected_alpha = float(alphas[int(np.argmin(manual_mean_mse))])

    cv = LassoCV(
        alphas=alphas,
        cv=3,
        cv_splits=folds,
        cv_solver="fista",
        random_state=0,
        **common,
    ).fit(X, y, sample_weight=weights)

    assert cv.alpha_ == pytest.approx(expected_alpha, rel=0, abs=0)
    np.testing.assert_allclose(
        cv.mean_mse_,
        manual_mean_mse,
        rtol=2e-8,
        atol=2e-10,
    )


def test_weighted_lassocv_and_default_grid_are_weight_scale_invariant():
    X, y, weights = _problem(seed=9140)
    common = dict(
        n_alphas=6,
        alpha_min_ratio=1e-2,
        cv=3,
        cv_splits=_folds(X.shape[0]),
        cv_solver="fista",
        solver="fista",
        compute_inference=False,
        device="cpu",
        max_iter=5000,
        tol=1e-9,
        random_state=0,
    )
    baseline = LassoCV(**common).fit(X, y, sample_weight=weights)
    scaled = LassoCV(**common).fit(X, y, sample_weight=17.3 * weights)

    np.testing.assert_allclose(scaled.alphas_, baseline.alphas_, rtol=0, atol=2e-14)
    np.testing.assert_allclose(scaled.mse_path_, baseline.mse_path_, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(scaled.mean_mse_, baseline.mean_mse_, rtol=2e-10, atol=2e-12)
    assert scaled.alpha_ == pytest.approx(baseline.alpha_, rel=0, abs=2e-14)
    np.testing.assert_allclose(scaled.coef_, baseline.coef_, rtol=0, atol=2e-11)
    assert scaled.intercept_ == pytest.approx(baseline.intercept_, rel=0, abs=2e-11)


def test_weighted_lassocv_rejects_zero_weight_training_fold():
    X, y, weights = _problem(seed=9141, n=12)
    folds = [
        (np.asarray([0, 1, 2, 3], dtype=np.int64), np.arange(4, 12, dtype=np.int64)),
        (np.arange(4, 12, dtype=np.int64), np.asarray([0, 1, 2, 3], dtype=np.int64)),
    ]
    weights[:4] = 0.0
    with pytest.raises(ValueError, match="training fold.*positive weight sum"):
        LassoCV(
            alphas=np.asarray([0.2, 0.1]),
            cv=2,
            cv_splits=folds,
            cv_solver="fista",
            solver="fista",
            compute_inference=False,
            device="cpu",
        ).fit(X, y, sample_weight=weights)
