"""Tests for top-level exports and Ridge/Lasso penalty behavior."""

import numpy as np
import pytest

import statgpu
import statgpu.inference as inference
from statgpu.linear_model import (
    Ridge,
    RidgeCV,
    Lasso,
    LassoCV,
    ElasticNetCV,
    LogisticRegressionCV,
)
from statgpu.linear_model._elasticnet_cv import _make_elasticnet_cv_auto_cache_key, _kfold_indices
from statgpu.survival import CoxPHCV
from statgpu._config import set_device, Device


def test_top_level_logistic_export():
    """Top-level package should expose LogisticRegression."""
    assert hasattr(statgpu, "LogisticRegression")


def test_top_level_knockoff_exports():
    """Top-level package should expose knockoff skeleton APIs."""
    assert hasattr(statgpu, "knockoff_filter")
    assert hasattr(statgpu, "fixed_x_knockoff_filter")
    assert hasattr(statgpu, "model_x_knockoff_filter")
    assert hasattr(statgpu, "KnockoffSelector")
    assert hasattr(statgpu, "FixedXKnockoffSelector")


def test_top_level_lasso_cv_export():
    """Top-level package should expose LassoCV."""
    assert hasattr(statgpu, "LassoCV")


@pytest.mark.parametrize(
    "name",
    ["PCA", "KMeans", "DBSCAN", "GaussianMixture", "NMF", "AgglomerativeClustering"],
)
def test_top_level_unsupervised_exports(name):
    """Top-level package should expose unsupervised estimators."""
    assert hasattr(statgpu, name)


def test_inference_r_style_distribution_quartets_exports():
    """Inference should expose full R-style d/p/q/r compatibility APIs by family."""
    expected_by_family = {
        "norm": ("dnorm_gpu", "pnorm_gpu", "qnorm_gpu", "rnorm_gpu"),
        "t": ("dt_gpu", "pt_gpu", "qt_gpu", "rt_gpu"),
        "chi2": ("dchisq_gpu", "pchisq_gpu", "qchisq_gpu", "rchisq_gpu"),
        "gamma": ("dgamma_gpu", "pgamma_gpu", "qgamma_gpu", "rgamma_gpu"),
        "beta": ("dbeta_gpu", "pbeta_gpu", "qbeta_gpu", "rbeta_gpu"),
        "f": ("df_gpu", "pf_gpu", "qf_gpu", "rf_gpu"),
        "poisson": ("dpois_gpu", "ppois_gpu", "qpois_gpu", "rpois_gpu"),
        "binom": ("dbinom_gpu", "pbinom_gpu", "qbinom_gpu", "rbinom_gpu"),
    }
    for family, names in expected_by_family.items():
        for name in names:
            assert hasattr(inference, name), f"Missing R-style API `{name}` in family `{family}`"


def test_inference_legacy_norm_isf_exported():
    """Inference should export legacy norm_isf_gpu alias for compatibility."""
    assert hasattr(inference, "norm_isf_gpu")
    assert "norm_isf_gpu" in inference.__all__


def test_inference_norm_distribution_matches_scipy_basics():
    """norm object should provide GPU outputs with basic scipy-compatible values."""
    cp = pytest.importorskip("cupy")
    sps = pytest.importorskip("scipy.stats")

    x = cp.asarray([-1.0, 0.0, 1.0], dtype=cp.float64)
    q = cp.asarray([0.1, 0.5, 0.9], dtype=cp.float64)

    cdf_gpu = inference.norm.cdf(x)
    ppf_gpu = inference.norm.ppf(q)
    roundtrip = inference.norm.cdf(ppf_gpu)

    assert isinstance(cdf_gpu, cp.ndarray)
    assert isinstance(ppf_gpu, cp.ndarray)
    assert np.allclose(cp.asnumpy(cdf_gpu), sps.norm.cdf(cp.asnumpy(x)), rtol=1e-6, atol=1e-8)
    assert np.allclose(cp.asnumpy(ppf_gpu), sps.norm.ppf(cp.asnumpy(q)), rtol=1e-6, atol=1e-8)
    assert cp.allclose(roundtrip, q, rtol=1e-6, atol=1e-8)


def test_legacy_non_r_distribution_names_emit_deprecation_warning():
    """Legacy non-R helper names should emit DeprecationWarning."""
    cp = pytest.importorskip("cupy")

    with pytest.warns(DeprecationWarning, match="norm_cdf_gpu"):
        out = inference.norm_cdf_gpu(cp.asarray([0.0], dtype=cp.float64))
    assert isinstance(out, cp.ndarray)

    with pytest.warns(DeprecationWarning, match="norm_isf_gpu"):
        out2 = inference.norm_isf_gpu(cp.asarray([0.2], dtype=cp.float64))
    assert isinstance(out2, cp.ndarray)


@pytest.mark.parametrize("name", ["RidgeCV", "LogisticRegressionCV", "CoxPHCV"])
def test_top_level_new_cv_exports(name):
    """Top-level package should expose new CV skeleton classes."""
    assert hasattr(statgpu, name)


def test_cv_skeleton_classes_instantiable():
    """CV skeleton classes should be constructible with sklearn-like signatures."""
    ridge_cv = RidgeCV(device="cpu", cv=3)
    logit_cv = LogisticRegressionCV(device="cpu", cv=3)
    cox_cv = CoxPHCV(device="cpu", cv=3)

    assert ridge_cv.cv == 3
    assert logit_cv.cv == 3
    assert cox_cv.cv == 3


@pytest.mark.parametrize("model_cls", [Ridge, Lasso])
def test_penalty_models_basic_predict_shape(model_cls):
    """Penalty models should predict 1D output matching sample size."""
    set_device("cpu")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 8))
    beta = rng.normal(size=8)
    y = X @ beta + rng.normal(scale=0.2, size=120)

    model = model_cls(device="cpu")
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (120,)
    assert np.all(np.isfinite(pred))


def test_lasso_cv_basic_interface_cpu():
    """LassoCV should expose sklearn-like fitted attributes and prediction API."""
    set_device("cpu")
    rng = np.random.default_rng(123)
    X = rng.normal(size=(180, 12))
    beta = np.zeros(12)
    beta[:4] = np.array([1.5, -1.2, 0.9, 0.6])
    y = X @ beta + rng.normal(scale=0.5, size=180)

    model = LassoCV(
        n_alphas=8,
        cv=4,
        random_state=7,
        device="cpu",
        compute_inference=False,
        max_iter=2000,
        tol=1e-4,
    )
    model.fit(X, y)

    assert np.isfinite(model.alpha_)
    assert model.alpha_ > 0.0
    assert model.alphas_.ndim == 1
    assert model.mse_path_.shape[0] == model.alphas_.shape[0]
    assert model.mse_path_.shape[1] >= 1
    assert model.coef_.shape == (X.shape[1],)

    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.isfinite(model.score(X, y))


def test_lasso_cv_glmnet_method_cpu():
    """LassoCV method='glmnet' should run and use coordinate-descent profile."""
    set_device("cpu")
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(220, 14))
    beta = np.zeros(14)
    beta[:5] = np.array([1.8, -1.4, 1.1, 0.7, -0.5])
    y = X @ beta + rng.normal(scale=0.4, size=220)

    model = LassoCV(
        n_alphas=10,
        cv=4,
        random_state=11,
        device="cpu",
        compute_inference=False,
        max_iter=2000,
        tol=1e-4,
        cpu_solver="fista",
        method="glmnet",
        cd_kkt_check_every=8,
    )
    model.fit(X, y)

    assert model.method == "glmnet"
    assert model.cd_kkt_check_every == 8
    assert model.estimator_.cpu_solver == "coordinate_descent"
    assert np.isfinite(model.alpha_)
    assert model.alpha_ > 0.0


def test_lasso_cv_fast_fold_stats_matches_weighted_fallback_cpu():
    """Fast fold-statistics CV path should match weighted fallback when weights are all ones."""
    set_device("cpu")
    rng = np.random.default_rng(20260410)
    X = rng.normal(size=(240, 16))
    beta = np.zeros(16)
    beta[:6] = np.array([1.7, -1.3, 1.0, 0.8, -0.6, 0.4])
    y = X @ beta + rng.normal(scale=0.35, size=240)

    idx = rng.permutation(X.shape[0])
    fold_edges = np.array_split(idx, 4)
    cv_splits = []
    for val_idx in fold_edges:
        train_idx = np.setdiff1d(idx, val_idx, assume_unique=False)
        cv_splits.append((train_idx, val_idx))

    base_kwargs = dict(
        n_alphas=12,
        cv=4,
        cv_splits=cv_splits,
        random_state=17,
        device="cpu",
        compute_inference=False,
        max_iter=2500,
        tol=1e-4,
        method="glmnet",
        cd_kkt_check_every=8,
    )

    fast_model = LassoCV(**base_kwargs)
    fast_model.fit(X, y)

    fallback_model = LassoCV(**base_kwargs)
    fallback_model.fit(X, y, sample_weight=np.ones(X.shape[0], dtype=np.float64))

    assert np.isclose(fast_model.alpha_, fallback_model.alpha_, rtol=1e-9, atol=1e-12)
    assert np.allclose(fast_model.mean_mse_, fallback_model.mean_mse_, rtol=1e-6, atol=1e-8)
    assert np.allclose(fast_model.coef_, fallback_model.coef_, rtol=1e-5, atol=1e-7)
    assert np.isclose(fast_model.intercept_, fallback_model.intercept_, rtol=1e-6, atol=1e-8)


def test_lasso_cv_cd_kkt_check_every_validation():
    """cd_kkt_check_every must be None or a positive integer."""
    with pytest.raises(ValueError, match="cd_kkt_check_every"):
        LassoCV(cd_kkt_check_every=0)


def test_elasticnet_cv_basic_interface_cpu():
    """ElasticNetCV should fit on CPU and expose selected hyperparameters."""
    set_device("cpu")
    rng = np.random.default_rng(20260420)
    X = rng.normal(size=(200, 15))
    beta = np.zeros(15)
    beta[:5] = np.array([1.6, -1.1, 0.9, 0.7, -0.4])
    y = X @ beta + rng.normal(scale=0.4, size=200)

    model = ElasticNetCV(
        l1_ratio=[0.2, 0.5, 0.8],
        n_alphas=10,
        cv=4,
        random_state=13,
        device="cpu",
        max_iter=2000,
        tol=1e-4,
    )
    model.fit(X, y)

    assert np.isfinite(model.alpha_)
    assert model.alpha_ > 0.0
    assert float(model.l1_ratio_) in {0.2, 0.5, 0.8}
    assert model.coef_.shape == (X.shape[1],)
    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.isfinite(model.score(X, y))


def test_elasticnet_cv_auto_cache_key_depends_on_folds():
    """Auto cache key should differ for different CV splits."""
    folds_a = _kfold_indices(n_samples=40, n_splits=4, random_state=1)
    folds_b = _kfold_indices(n_samples=40, n_splits=4, random_state=2)

    key_a = _make_elasticnet_cv_auto_cache_key(
        X_shape=(40, 6),
        y_shape=(40,),
        l1_ratios=(0.5, 0.9),
        alphas=None,
        n_alphas=8,
        alpha_min_ratio=1e-3,
        folds=folds_a,
        fit_intercept=True,
        use_gpu=False,
        max_iter=1000,
        tol=1e-4,
    )
    key_b = _make_elasticnet_cv_auto_cache_key(
        X_shape=(40, 6),
        y_shape=(40,),
        l1_ratios=(0.5, 0.9),
        alphas=None,
        n_alphas=8,
        alpha_min_ratio=1e-3,
        folds=folds_b,
        fit_intercept=True,
        use_gpu=False,
        max_iter=1000,
        tol=1e-4,
    )

    assert key_a != key_b


@pytest.mark.skipif(
    not Ridge(device="auto")._get_compute_device() == Device.CUDA,
    reason="CUDA not available",
)
@pytest.mark.parametrize("model_cls", [Ridge, Lasso])
def test_penalty_models_gpu_cpu_prediction_consistency(model_cls):
    """GPU and CPU predictions should be numerically close."""
    rng = np.random.default_rng(123)
    X = rng.normal(size=(256, 16)).astype(np.float64)
    beta = rng.normal(size=16)
    y = X @ beta + rng.normal(scale=0.3, size=256)

    cpu_model = model_cls(device="cpu")
    cpu_model.fit(X, y)
    cpu_pred = cpu_model.predict(X)

    gpu_model = model_cls(device="cuda")
    gpu_model.fit(X, y)
    gpu_pred = gpu_model.predict(X)
    # CuPy arrays require .get() to transfer data back to host memory.
    if hasattr(gpu_pred, "get"):
        gpu_pred = gpu_pred.get()
    gpu_pred = np.asarray(gpu_pred)

    assert np.allclose(cpu_pred, gpu_pred, rtol=1e-4, atol=1e-4)
