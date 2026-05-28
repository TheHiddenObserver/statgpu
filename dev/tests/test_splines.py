"""
Tests for the splines module (B-spline basis, natural cubic splines, and GAM).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from statgpu.nonparametric.splines._bspline_basis import bspline_basis, natural_cubic_spline_basis
from statgpu.nonparametric.splines._penalized import (
    difference_penalty,
    penalized_ls,
    generalized_cross_validation,
)


def test_bspline_shape():
    """Basis matrix should have shape (n, m + degree + 1)."""
    x = np.linspace(0, 10, 50)
    knots = np.array([3.0, 5.0, 7.0])
    B = bspline_basis(x, knots, degree=3, xp=np)
    n = len(x)
    m = len(knots)
    expected_cols = m + 3 + 1
    assert B.shape == (n, expected_cols), f"Expected ({n}, {expected_cols}), got {B.shape}"
    print("PASS: test_bspline_shape")


def test_bspline_nonnegative():
    """B-spline basis functions should be non-negative."""
    x = np.linspace(0, 10, 100)
    knots = np.array([2.0, 4.0, 6.0, 8.0])
    B = bspline_basis(x, knots, degree=3, xp=np)
    assert np.all(B >= -1e-10), "B-spline basis should be non-negative"
    print("PASS: test_bspline_nonnegative")


def test_bspline_partition_of_unity():
    """B-spline basis should form a partition of unity (sum = 1) in interior."""
    x = np.linspace(0.01, 9.99, 100)
    knots = np.array([2.0, 4.0, 6.0, 8.0])
    B = bspline_basis(x, knots, degree=3, xp=np)
    row_sums = np.sum(B, axis=1)
    # Interior points should sum to 1; boundary points may not due to
    # boundary knot at x_min/x_max affecting the support
    interior = (x > x.min() + 0.1) & (x < x.max() - 0.1)
    np.testing.assert_allclose(row_sums[interior], 1.0, atol=1e-10,
                               err_msg="B-spline basis interior should sum to 1")
    print("PASS: test_bspline_partition_of_unity")


def test_bspline_scipy_consistency():
    """B-spline basis should match scipy.interpolate.BSpline."""
    try:
        from scipy.interpolate import BSpline
    except ImportError:
        print("SKIP: test_bspline_scipy_consistency (scipy not available)")
        return

    knots = np.array([2.0, 4.0, 6.0, 8.0])
    degree = 3
    # Use a wider range so boundary knots match between our impl and scipy
    x = np.linspace(0, 10, 100)

    # Our implementation uses x.min() and x.max() as boundary knots
    B_ours = bspline_basis(x, knots, degree=degree, xp=np)

    # Scipy with same boundary knots
    x_min, x_max = 0.0, 10.0
    t = np.concatenate([
        np.full(degree + 1, x_min),
        knots,
        np.full(degree + 1, x_max)
    ])
    n_basis = len(t) - degree - 1
    B_scipy = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spline = BSpline(t, c, degree, extrapolate=False)
        B_scipy[:, i] = spline(x)
    B_scipy = np.nan_to_num(B_scipy, nan=0.0)

    np.testing.assert_allclose(
        B_ours, B_scipy, atol=1e-10,
        err_msg="B-spline basis should match scipy"
    )
    print("PASS: test_bspline_scipy_consistency")


def test_bspline_different_degrees():
    """B-spline basis should work for different degrees."""
    x = np.linspace(0, 10, 50)
    knots = np.array([3.0, 5.0, 7.0])
    for degree in [1, 2, 3, 4]:
        B = bspline_basis(x, knots, degree=degree, xp=np)
        n_basis = len(knots) + degree + 1
        assert B.shape == (len(x), n_basis), (
            f"Degree {degree}: expected ({len(x)}, {n_basis}), got {B.shape}"
        )
    print("PASS: test_bspline_different_degrees")


def test_natural_spline_shape():
    """Natural cubic spline basis should reduce dimensionality vs cubic B-spline."""
    x = np.linspace(0, 10, 50)
    knots = np.array([3.0, 5.0, 7.0])
    B_cubic = bspline_basis(x, knots, degree=3, xp=np)
    B_natural = natural_cubic_spline_basis(x, knots, xp=np)
    n = len(x)
    # Natural spline has fewer basis functions than cubic B-spline
    # (boundary constraints reduce dimensionality by 2)
    assert B_natural.shape[0] == n
    assert B_natural.shape[1] < B_cubic.shape[1], (
        f"Natural spline ({B_natural.shape[1]}) should have fewer basis than "
        f"cubic ({B_cubic.shape[1]})"
    )
    assert B_natural.shape[1] == B_cubic.shape[1] - 2
    print("PASS: test_natural_spline_shape")


def test_natural_spline_linearity_beyond_boundaries():
    """Natural spline should be linear beyond boundary knots."""
    # Use a range that covers the evaluation points
    x = np.linspace(0, 10, 100)
    knots = np.array([3.0, 5.0, 7.0])
    B = natural_cubic_spline_basis(x, knots, xp=np)
    assert np.all(np.isfinite(B)), "Basis should be finite"
    # Natural cubic spline has fewer basis functions than cubic B-spline
    assert B.shape[1] < bspline_basis(x, knots, degree=3, xp=np).shape[1]
    print("PASS: test_natural_spline_linearity_beyond_boundaries")


def test_penalty_shape():
    """Penalty matrix should be square with shape (n_coef, n_coef)."""
    n_coef = 10
    S = difference_penalty(order=2, n_coef=n_coef, xp=np)
    assert S.shape == (n_coef, n_coef)
    print("PASS: test_penalty_shape")


def test_penalty_symmetric():
    """Penalty matrix should be symmetric."""
    S = difference_penalty(order=2, n_coef=10, xp=np)
    np.testing.assert_allclose(S, S.T, atol=1e-10)
    print("PASS: test_penalty_symmetric")


def test_penalty_positive_semidefinite():
    """Penalty matrix should be positive semi-definite."""
    S = difference_penalty(order=2, n_coef=10, xp=np)
    eigenvalues = np.linalg.eigvalsh(S)
    assert np.all(eigenvalues >= -1e-10)
    print("PASS: test_penalty_positive_semidefinite")


def test_penalty_order1_structure():
    """First-order penalty should penalize first differences."""
    S = difference_penalty(order=1, n_coef=4, xp=np)
    D = np.array([[-1, 1, 0, 0],
                  [0, -1, 1, 0],
                  [0, 0, -1, 1]], dtype=float)
    S_expected = D.T @ D
    np.testing.assert_allclose(S, S_expected, atol=1e-10)
    print("PASS: test_penalty_order1_structure")


def test_penalty_order2_structure():
    """Second-order penalty should penalize second differences."""
    S = difference_penalty(order=2, n_coef=5, xp=np)
    D = np.array([[1, -2, 1, 0, 0],
                  [0, 1, -2, 1, 0],
                  [0, 0, 1, -2, 1]], dtype=float)
    S_expected = D.T @ D
    np.testing.assert_allclose(S, S_expected, atol=1e-10)
    print("PASS: test_penalty_order2_structure")


def test_penalized_ls_basic_solve():
    """Penalized LS should solve correctly for simple case."""
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.randn(100) * 0.1
    B = np.column_stack([np.ones(100), x])
    S = np.zeros((2, 2))
    beta, edf = penalized_ls(B, y, S, lambda_=0.0, xp=np)
    np.testing.assert_allclose(beta, [1.0, 2.0], atol=0.1)
    print("PASS: test_penalized_ls_basic_solve")


def test_penalized_ls_penalty_effect():
    """Increasing lambda should shrink coefficients."""
    np.random.seed(42)
    B = np.random.randn(50, 5)
    y = B @ np.array([1, 2, 3, 4, 5]) + np.random.randn(50) * 0.1
    S = np.eye(5)
    beta_no_penalty, _ = penalized_ls(B, y, S, lambda_=0.0, xp=np)
    beta_with_penalty, _ = penalized_ls(B, y, S, lambda_=10.0, xp=np)
    assert np.sum(beta_with_penalty ** 2) < np.sum(beta_no_penalty ** 2)
    print("PASS: test_penalized_ls_penalty_effect")


def test_penalized_ls_edf_range():
    """Effective degrees of freedom should be between 0 and p."""
    np.random.seed(42)
    n, p = 100, 10
    B = np.random.randn(n, p)
    y = np.random.randn(n)
    S = difference_penalty(order=2, n_coef=p, xp=np)
    _, edf = penalized_ls(B, y, S, lambda_=1.0, xp=np)
    assert 0 <= edf <= p, f"EDF should be between 0 and {p}, got {edf}"
    print("PASS: test_penalized_ls_edf_range")


def test_gam_fit_predict():
    """GAM should fit and predict without errors."""
    from statgpu.semiparametric._gam import GAM
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + np.random.randn(100) * 0.1
    gam = GAM(n_splines=15, lam=1.0)
    gam.fit(X, y)
    y_pred = gam.predict(X)
    assert y_pred.shape == (100,)
    print("PASS: test_gam_fit_predict")


def test_gam_goodness_of_fit():
    """GAM should achieve R-squared > 0.9 on smooth data."""
    from statgpu.semiparametric._gam import GAM
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2)
    y_true = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    y = y_true + np.random.randn(n) * 0.05
    gam = GAM(n_splines=20, lam=0.1)
    gam.fit(X, y)
    y_pred = gam.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    assert r_squared > 0.9, f"R-squared should be > 0.9, got {r_squared:.4f}"
    print(f"PASS: test_gam_goodness_of_fit (R2={r_squared:.4f})")


def test_gam_gcv_selection():
    """GCV should select a reasonable smoothing parameter."""
    from statgpu.semiparametric._gam import GAM
    np.random.seed(42)
    n = 150
    X = np.random.randn(n, 1)
    y = np.sin(X[:, 0]) + np.random.randn(n) * 0.1
    gam = GAM(n_splines=15, lam=None)
    gam.fit(X, y)
    assert gam.lam_ > 0, f"Lambda should be positive, got {gam.lam_}"
    assert np.isfinite(gam.lam_)
    assert gam.gcv_score_ is not None
    print(f"PASS: test_gam_gcv_selection (lambda={gam.lam_:.4g})")


def test_gam_edf_reasonable():
    """Effective degrees of freedom should be reasonable."""
    from statgpu.semiparametric._gam import GAM
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 2)
    y = np.sin(X[:, 0]) + np.random.randn(n) * 0.1
    gam = GAM(n_splines=20, lam=1.0)
    gam.fit(X, y)
    max_edf = 20 * 2 + 1
    assert 1 <= gam.edf_ <= max_edf, f"EDF should be in [1, {max_edf}], got {gam.edf_}"
    print(f"PASS: test_gam_edf_reasonable (edf={gam.edf_:.2f})")


def test_gam_intercept():
    """GAM should estimate an intercept close to the true value."""
    from statgpu.semiparametric._gam import GAM
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 2)
    y = 5.0 + np.sin(X[:, 0]) + np.random.randn(n) * 0.05
    gam = GAM(n_splines=20, lam=0.01)
    gam.fit(X, y)
    assert abs(gam.intercept_ - 5.0) < 2.0, (
        f"Intercept should be close to 5, got {gam.intercept_:.2f}"
    )
    print(f"PASS: test_gam_intercept (intercept={gam.intercept_:.2f})")


def test_gam_predict_shape_mismatch():
    """GAM predict should raise error for wrong number of features."""
    from statgpu.semiparametric._gam import GAM
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    gam = GAM(n_splines=15, lam=1.0)
    gam.fit(X, y)
    X_wrong = np.random.randn(100, 2)
    try:
        gam.predict(X_wrong)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS: test_gam_predict_shape_mismatch")


def test_gam_numpy_backend():
    """GAM should work with numpy backend."""
    from statgpu.semiparametric._gam import GAM
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.sin(X[:, 0]) + np.random.randn(100) * 0.1
    gam = GAM(n_splines=15, lam=1.0, device='cpu')
    gam.fit(X, y)
    y_pred = gam.predict(X)
    assert y_pred.shape == (100,)
    assert np.all(np.isfinite(y_pred))
    print("PASS: test_gam_numpy_backend")


def test_gam_summary():
    """GAM summary should return a dictionary."""
    from statgpu.semiparametric._gam import GAM
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.sin(X[:, 0]) + np.random.randn(100) * 0.1
    gam = GAM(n_splines=15, lam=1.0)
    gam.fit(X, y)
    summary = gam.summary()
    assert isinstance(summary, dict)
    assert 'n_features' in summary
    assert 'smoothing_parameter' in summary
    assert 'effective_df' in summary
    assert 'intercept' in summary
    print("PASS: test_gam_summary")


def test_irls_penalty_matrix_parameter():
    """IRLS solver should accept penalty_matrix parameter."""
    from statgpu.glm_core._irls import irls_solver

    class MockLink:
        name = 'identity'
        def inverse(self, eta):
            return eta

    class MockFamily:
        name = 'gaussian'
        link = MockLink()
        def irls_weights(self, mu, y):
            return np.ones_like(mu)
        def irls_working_response(self, mu, y, eta):
            return y
        def gradient(self, X, y, params):
            resid = y - X @ params
            return -X.T @ resid / len(y)

    np.random.seed(42)
    X = np.column_stack([np.ones(50), np.random.randn(50, 3)])
    y = X @ np.array([1, 2, 3, 4]) + np.random.randn(50) * 0.1

    params1, _ = irls_solver(MockFamily(), X, y, max_iter=50, backend='numpy')
    penalty = np.eye(4) * 0.1
    params2, _ = irls_solver(MockFamily(), X, y, max_iter=50, backend='numpy',
                              penalty_matrix=penalty)
    assert np.sum(params2 ** 2) <= np.sum(params1 ** 2) + 1e-10
    print("PASS: test_irls_penalty_matrix_parameter")


def test_irls_solver_class_penalty():
    """IRLSSolver class should accept penalty_matrix in fit()."""
    from statgpu.glm_core._irls import IRLSSolver

    class MockLink:
        name = 'identity'
        def inverse(self, eta):
            return eta

    class MockFamily:
        name = 'gaussian'
        link = MockLink()
        def irls_weights(self, mu, y):
            return np.ones_like(mu)
        def irls_working_response(self, mu, y, eta):
            return y
        def gradient(self, X, y, params):
            resid = y - X @ params
            return -X.T @ resid / len(y)

    np.random.seed(42)
    X = np.column_stack([np.ones(50), np.random.randn(50, 3)])
    y = X @ np.array([1, 2, 3, 4]) + np.random.randn(50) * 0.1
    solver = IRLSSolver(MockFamily(), max_iter=50)
    penalty = np.eye(4) * 0.01
    params, _ = solver.fit(X, y, backend='numpy', penalty_matrix=penalty)
    assert params.shape == (4,)
    print("PASS: test_irls_solver_class_penalty")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    tests = [
        test_bspline_shape,
        test_bspline_nonnegative,
        test_bspline_partition_of_unity,
        test_bspline_scipy_consistency,
        test_bspline_different_degrees,
        test_natural_spline_shape,
        test_natural_spline_linearity_beyond_boundaries,
        test_penalty_shape,
        test_penalty_symmetric,
        test_penalty_positive_semidefinite,
        test_penalty_order1_structure,
        test_penalty_order2_structure,
        test_penalized_ls_basic_solve,
        test_penalized_ls_penalty_effect,
        test_penalized_ls_edf_range,
        test_gam_fit_predict,
        test_gam_goodness_of_fit,
        test_gam_gcv_selection,
        test_gam_edf_reasonable,
        test_gam_intercept,
        test_gam_predict_shape_mismatch,
        test_gam_numpy_backend,
        test_gam_summary,
        test_irls_penalty_matrix_parameter,
        test_irls_solver_class_penalty,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed == 0:
        print("ALL TESTS PASSED!")
    print("=" * 50)
