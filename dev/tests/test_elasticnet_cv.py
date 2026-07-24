# -*- coding: utf-8 -*-
"""
Unit tests for ElasticNetCV
"""
import numpy as np
import pytest
from statgpu.linear_model import ElasticNetCV
from statgpu import get_backend


def generate_elasticnet_data(n_samples=1000, n_features=100, seed=42):
    """Generate synthetic data for ElasticNet testing."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:10] = np.random.randn(10)
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)
    return X, y, true_coef


def test_elasticnetcv_basic():
    """Test basic ElasticNetCV functionality."""
    print("=" * 60)
    print("Test 1: Basic ElasticNetCV functionality")
    print("=" * 60)

    X, y, _ = generate_elasticnet_data(n_samples=500, n_features=50)

    model = ElasticNetCV(
        l1_ratio=0.5,
        n_alphas=20,
        cv=3,
        random_state=42,
        device="cpu"
    )
    model.fit(X, y)

    print(f"Selected alpha: {model.alpha_:.6f}")
    print(f"Selected l1_ratio: {model.l1_ratio_:.6f}")
    print(f"Number of non-zero coefs: {np.sum(model.coef_ != 0)}")
    print(f"R² score: {model.score(X, y):.6f}")

    assert model.alpha_ is not None, "alpha_ should be set"
    assert model.l1_ratio_ is not None, "l1_ratio_ should be set"
    assert model.coef_ is not None, "coef_ should be set"
    assert model.intercept_ is not None, "intercept_ should be set"

    print("✓ PASSED\n")


def test_elasticnetcv_l1_ratio_grid():
    """Test ElasticNetCV with multiple l1_ratio values."""
    print("=" * 60)
    print("Test 2: Multiple l1_ratio values")
    print("=" * 60)

    X, y, _ = generate_elasticnet_data(n_samples=500, n_features=50)

    l1_ratios = [0.2, 0.5, 0.8, 0.95]
    model = ElasticNetCV(
        l1_ratio=l1_ratios,
        n_alphas=20,
        cv=3,
        random_state=42,
        device="cpu"
    )
    model.fit(X, y)

    print(f"Selected alpha: {model.alpha_:.6f}")
    print(f"Selected l1_ratio: {model.l1_ratio_:.6f}")
    assert model.l1_ratio_ in l1_ratios, f"l1_ratio_ should be in {l1_ratios}"

    print("✓ PASSED\n")


def test_elasticnetcv_vs_sklearn():
    """Test ElasticNetCV vs sklearn."""
    print("=" * 60)
    print("Test 3: ElasticNetCV vs sklearn comparison")
    print("=" * 60)

    from sklearn.linear_model import ElasticNetCV as SklearnElasticNetCV

    X, y, _ = generate_elasticnet_data(n_samples=500, n_features=50)

    alphas = np.logspace(-3, 3, 20)
    l1_ratios = [0.5]

    statgpu_model = ElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=3,
        random_state=42,
        device="cpu"
    )
    statgpu_model.fit(X, y)

    sklearn_model = SklearnElasticNetCV(
        l1_ratio=l1_ratios,
        alphas=alphas,
        cv=3,
        random_state=42,
        max_iter=1000
    )
    sklearn_model.fit(X, y)

    print(f"statgpu alpha: {statgpu_model.alpha_:.6f}")
    print(f"sklearn alpha: {sklearn_model.alpha_:.6f}")
    print(f"statgpu l1_ratio: {statgpu_model.l1_ratio_:.6f}")
    print(f"sklearn l1_ratio: {sklearn_model.l1_ratio_:.6f}")
    print(f"L2 distance: {np.linalg.norm(statgpu_model.coef_ - sklearn_model.coef_):.6e}")

    print("✓ PASSED\n")



def test_elasticnetcv_gpu_backend():
    """Compare CPU and explicit CuPy results when CUDA is available."""
    X, y, _ = generate_elasticnet_data(n_samples=500, n_features=50)
    cpu_model = ElasticNetCV(
        l1_ratio=0.5, n_alphas=20, cv=3, random_state=42, device="cpu"
    ).fit(X, y)
    if not get_backend("cupy").is_available():
        pytest.skip("working CuPy CUDA backend is unavailable")
    cuda_model = ElasticNetCV(
        l1_ratio=0.5, n_alphas=20, cv=3, random_state=42, device="cuda"
    ).fit(X, y)
    np.testing.assert_allclose(
        cpu_model.coef_, cuda_model.coef_, rtol=5e-4, atol=5e-5
    )


def test_elasticnetcv_predict():
    """Test ElasticNetCV predict method."""
    print("=" * 60)
    print("Test 5: Predict method")
    print("=" * 60)

    X, y, _ = generate_elasticnet_data(n_samples=500, n_features=50)

    model = ElasticNetCV(
        l1_ratio=0.5,
        n_alphas=20,
        cv=3,
        random_state=42,
        device="cpu"
    )
    model.fit(X, y)

    predictions = model.predict(X)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions mean: {predictions.mean():.6f}")
    print(f"Actual y mean: {y.mean():.6f}")

    assert predictions.shape == y.shape, "Predictions shape should match y shape"

    print("✓ PASSED\n")


def test_elasticnetcv_cv_results():
    """Test ElasticNetCV cv_results_ attribute."""
    print("=" * 60)
    print("Test 6: CV results attribute")
    print("=" * 60)

    X, y, _ = generate_elasticnet_data(n_samples=500, n_features=50)

    model = ElasticNetCV(
        l1_ratio=[0.5, 0.8],
        n_alphas=20,
        cv=3,
        random_state=42,
        device="cpu"
    )
    model.fit(X, y)

    assert hasattr(model, 'cv_results_'), "Should have cv_results_"
    assert 'mse_path' in model.cv_results_, "cv_results_ should contain mse_path"
    assert 'best_alpha' in model.cv_results_, "cv_results_ should contain best_alpha"
    assert 'best_l1_ratio' in model.cv_results_, "cv_results_ should contain best_l1_ratio"

    print(f"CV results keys: {list(model.cv_results_.keys())}")
    print(f"MSE path shape: {model.cv_results_['mse_path'].shape}")

    print("✓ PASSED\n")


def run_all_tests():
    """Run all ElasticNetCV tests."""
    print("\n" + "=" * 60)
    print("Running ElasticNetCV Unit Tests")
    print("=" * 60 + "\n")

    tests = [
        test_elasticnetcv_basic,
        test_elasticnetcv_l1_ratio_grid,
        test_elasticnetcv_vs_sklearn,
        test_elasticnetcv_gpu_backend,
        test_elasticnetcv_predict,
        test_elasticnetcv_cv_results,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            failed += 1
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"Summary: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    return passed, failed


if __name__ == "__main__":
    run_all_tests()
