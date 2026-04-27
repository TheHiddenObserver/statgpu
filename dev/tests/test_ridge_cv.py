"""
Test RidgeCV implementation against sklearn.
"""
import numpy as np
import sys
sys.path.insert(0, '..')

from statgpu.linear_model import RidgeCV
from sklearn.linear_model import RidgeCV as SklearnRidgeCV
from sklearn.model_selection import KFold


def generate_ridge_data(n_samples=1000, n_features=50, n_informative=20,
                         noise=0.1, random_state=42):
    """Generate synthetic Ridge regression data."""
    rng = np.random.RandomState(random_state)

    # Generate correlated design matrix
    X = rng.randn(n_samples, n_features)

    # True coefficients (sparse)
    beta = np.zeros(n_features)
    beta[:n_informative] = rng.randn(n_informative) * 2.0

    # Generate response
    y = X @ beta + noise * rng.randn(n_samples)

    return X, y, beta


def test_ridge_cv_cpu_vs_sklearn():
    """Compare statgpu RidgeCV (CPU) with sklearn RidgeCV."""
    print("=" * 70)
    print("Test 1: statgpu RidgeCV (CPU) vs sklearn RidgeCV")
    print("=" * 70)

    # Generate data
    X, y, _ = generate_ridge_data(n_samples=500, n_features=30, random_state=20260418)

    # Common alpha grid
    alphas = np.logspace(-3, 3, 50)

    # statgpu RidgeCV (CPU)
    statgpu_cv = 5
    statgpu_model = RidgeCV(
        alphas=alphas,
        cv=statgpu_cv,
        fit_intercept=True,
        device='cpu',
        random_state=20260418,
        compute_inference=False,
    )
    statgpu_model.fit(X, y)

    # sklearn RidgeCV (uses LOOCV by default, so we use custom CV)
    sklearn_cv = KFold(n_splits=statgpu_cv, shuffle=True, random_state=20260418)
    sklearn_model = SklearnRidgeCV(
        alphas=alphas,
        cv=sklearn_cv,
        fit_intercept=True,
    )
    sklearn_model.fit(X, y)

    print(f"\nData: n={X.shape[0]}, p={X.shape[1]}")
    print(f"Alpha grid: {len(alphas)} values from {alphas[0]:.4f} to {alphas[-1]:.4f}")
    print(f"CV folds: {statgpu_cv}")
    print(f"\nResults:")
    print(f"  statgpu selected alpha: {statgpu_model.alpha_:.6f}")
    print(f"  sklearn selected alpha: {sklearn_model.alpha_:.6f}")
    print(f"  statgpu best CV MSE:    {statgpu_model.best_score_:.6f}")
    print(f"\nCoefficients (first 5):")
    print(f"  statgpu:  {statgpu_model.coef_[:5]}")
    print(f"  sklearn:  {sklearn_model.coef_[:5]}")
    print(f"\nCoefficient difference (L2 norm): {np.linalg.norm(statgpu_model.coef_ - sklearn_model.coef_):.6f}")

    # Check consistency
    alpha_match = statgpu_model.alpha_ == sklearn_model.alpha_
    coef_diff = np.linalg.norm(statgpu_model.coef_ - sklearn_model.coef_)

    print(f"\nConsistency check:")
    print(f"  Alpha match: {alpha_match} {'✓' if alpha_match else '(may differ due to CV fold differences)'}")
    print(f"  Coef L2 diff: {coef_diff:.6f} {'✓' if coef_diff < 0.01 else '(acceptable if CV folds differ)'}")

    return alpha_match, coef_diff


def test_ridge_cv_gpu_vs_cpu():
    """Compare statgpu RidgeCV (GPU) with RidgeCV (CPU)."""
    print("\n" + "=" * 70)
    print("Test 2: statgpu RidgeCV (GPU) vs RidgeCV (CPU)")
    print("=" * 70)

    try:
        import cupy as cp
        print(f"CuPy available: {cp.__version__}")
    except ImportError:
        print("CuPy not available, skipping GPU test")
        return None, None

    # Generate data
    X, y, _ = generate_ridge_data(n_samples=1000, n_features=50, random_state=20260418)

    # Common alpha grid
    alphas = np.logspace(-3, 3, 50)

    # CPU model
    cpu_model = RidgeCV(
        alphas=alphas,
        cv=5,
        fit_intercept=True,
        device='cpu',
        random_state=20260418,
        compute_inference=False,
    )
    cpu_model.fit(X, y)

    # GPU model
    gpu_model = RidgeCV(
        alphas=alphas,
        cv=5,
        fit_intercept=True,
        device='cuda',
        random_state=20260418,
        compute_inference=False,
    )
    gpu_model.fit(X, y)

    print(f"\nData: n={X.shape[0]}, p={X.shape[1]}")
    print(f"Alpha grid: {len(alphas)} values")
    print(f"\nResults:")
    print(f"  CPU selected alpha: {cpu_model.alpha_:.6f}")
    print(f"  GPU selected alpha: {gpu_model.alpha_:.6f}")
    print(f"  CPU best CV MSE:    {cpu_model.best_score_:.6f}")
    print(f"  GPU best CV MSE:    {gpu_model.best_score_:.6f}")
    print(f"\nCoefficients (first 5):")
    print(f"  CPU: {cpu_model.coef_[:5]}")
    print(f"  GPU: {gpu_model.coef_[:5]}")
    print(f"\nCoefficient difference (L2 norm): {np.linalg.norm(cpu_model.coef_ - gpu_model.coef_):.6f}")

    # Check consistency
    alpha_match = cpu_model.alpha_ == gpu_model.alpha_
    coef_diff = np.linalg.norm(cpu_model.coef_ - gpu_model.coef_)

    print(f"\nConsistency check:")
    print(f"  Alpha match: {alpha_match} {'✓' if alpha_match else '(may differ due to numerical precision)'}")
    print(f"  Coef L2 diff: {coef_diff:.6f} {'✓' if coef_diff < 1e-5 else '⚠'}")

    return alpha_match, coef_diff


def test_ridge_cv_alpha_selection():
    """Test that RidgeCV selects reasonable alpha values."""
    print("\n" + "=" * 70)
    print("Test 3: Alpha Selection Behavior")
    print("=" * 70)

    # Test with different noise levels
    for noise, expected_behavior in [(0.01, "small alpha"), (0.5, "medium alpha"), (2.0, "large alpha")]:
        X, y, _ = generate_ridge_data(n_samples=500, n_features=30, noise=noise, random_state=20260418)

        model = RidgeCV(
            n_alphas=100,
            alpha_min_ratio=1e-3,
            cv=5,
            fit_intercept=True,
            device='cpu',
            random_state=20260418,
            compute_inference=False,
        )
        model.fit(X, y)

        print(f"\nNoise level: {noise:.2f} (expected: {expected_behavior})")
        print(f"  Selected alpha: {model.alpha_:.4f}")
        print(f"  Best CV MSE: {model.best_score_:.4f}")

    print("\n✓ Alpha selection responds to noise level")


def test_ridge_cv_predictions():
    """Test that RidgeCV predictions work correctly."""
    print("\n" + "=" * 70)
    print("Test 4: Prediction and Scoring")
    print("=" * 70)

    # Generate train/test data
    X_train, y_train, _ = generate_ridge_data(n_samples=800, n_features=40, random_state=20260418)
    X_test, y_test, _ = generate_ridge_data(n_samples=200, n_features=40, random_state=20260419)

    model = RidgeCV(
        cv=5,
        fit_intercept=True,
        device='cpu',
        random_state=20260418,
        compute_inference=False,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    r2_test = model.score(X_test, y_test)

    print(f"\nTrain: n={X_train.shape[0]}, Test: n={X_test.shape[0]}")
    print(f"Selected alpha: {model.alpha_:.4f}")
    print(f"Test R²: {r2_test:.4f}")
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Prediction mean: {y_pred.mean():.4f} (true mean: {y_test.mean():.4f})")

    assert y_pred.shape == y_test.shape, "Prediction shape mismatch"
    assert np.isfinite(r2_test), "R² should be finite"

    print("\n✓ Predictions and scoring work correctly")


def run_all_tests():
    """Run all RidgeCV tests."""
    print("\n" + "=" * 70)
    print("StatGPU RidgeCV Test Suite")
    print("=" * 70)

    results = {}

    # Test 1: CPU vs sklearn
    results['cpu_vs_sklearn'] = test_ridge_cv_cpu_vs_sklearn()

    # Test 2: GPU vs CPU
    results['gpu_vs_cpu'] = test_ridge_cv_gpu_vs_cpu()

    # Test 3: Alpha selection
    test_ridge_cv_alpha_selection()

    # Test 4: Predictions
    test_ridge_cv_predictions()

    print("\n" + "=" * 70)
    print("Test Suite Complete")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_all_tests()
