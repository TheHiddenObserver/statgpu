"""Temporary final patch script for the repository-wide review."""
from pathlib import Path
from textwrap import dedent
import re


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:80]!r}")
    p.write_text(text.replace(old, new), encoding="utf-8")


# Ridge exact CPU path should implement sklearn's un-normalized objective:
# ||y - Xb||^2 + alpha ||b||^2, including weighted fits.
replace_once(
    "statgpu/linear_model/wrappers/_ridge.py",
    '''            if self.fit_intercept:
                XtX -= w_sum * np.outer(X_wmean, X_wmean)
                Xty -= w_sum * X_wmean * y_wmean
                n_eff = w_sum
            else:
                n_eff = float(sw.sum())
''',
    '''            if self.fit_intercept:
                XtX -= w_sum * np.outer(X_wmean, X_wmean)
                Xty -= w_sum * X_wmean * y_wmean
''',
)
replace_once(
    "statgpu/linear_model/wrappers/_ridge.py",
    '''            n_eff = float(n_samples)

        if Xty.ndim == 0:
''',
    '''
        if Xty.ndim == 0:
''',
)
replace_once(
    "statgpu/linear_model/wrappers/_ridge.py",
    '''        # Solve (XtX + n_eff*alpha*I) @ coef = Xty
        # n_eff scaling matches PenalizedGeneralizedLinearModel exact ridge
        # and sklearn Ridge convention.
        A = XtX + float(self.alpha) * n_eff * np.eye(n_features, dtype=np.float64)
''',
    '''        # sklearn Ridge minimizes ||y - Xb||^2 + alpha * ||b||^2.
        # The penalty is therefore not multiplied by n_samples or weight sum.
        A = XtX + float(self.alpha) * np.eye(n_features, dtype=np.float64)
''',
)

# Cox tie-specific kernels historically expose opposite Hessian orientations.
# Normalize to the positive-semidefinite observed information at the inference
# boundary instead of clipping negative covariance diagonals to zero.
cox_path = Path("statgpu/survival/_cox.py")
cox = cox_path.read_text(encoding="utf-8")
marker = '''    def _compute_inference_cpu(self, X, time, event, cluster=None):
'''
helper = dedent('''
    @staticmethod
    def _observed_information(hess):
        """Return a symmetric positive-oriented observed information matrix.

        Breslow kernels return the log-likelihood Hessian, while legacy Efron
        kernels return its negation.  Select the orientation with the larger
        positive spectral mass and keep this compatibility normalization at the
        inference boundary.
        """
        sym = 0.5 * (np.asarray(hess, dtype=np.float64) + np.asarray(hess, dtype=np.float64).T)
        eigvals = np.linalg.eigvalsh(sym)
        positive_mass = float(np.sum(np.clip(eigvals, 0.0, None)))
        negative_mass = float(np.sum(np.clip(-eigvals, 0.0, None)))
        return sym if positive_mass >= negative_mass else -sym

    def _compute_inference_cpu(self, X, time, event, cluster=None):
''')
if cox.count(marker) != 1:
    raise RuntimeError(f"Cox inference marker count={cox.count(marker)}")
cox = cox.replace(marker, helper, 1)
old = '''        # Bread matrix from observed information.
        try:
            bread = np.linalg.solve(-hess, np.eye(n_features))
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(-hess)
'''
new = '''        # Bread matrix from observed information.
        information = self._observed_information(hess)
        try:
            bread = np.linalg.solve(information, np.eye(n_features))
        except np.linalg.LinAlgError:
            bread = np.linalg.pinv(information)
'''
if cox.count(old) != 1:
    raise RuntimeError(f"Cox bread block count={cox.count(old)}")
cox = cox.replace(old, new, 1)
old = '''            _, hess_0 = self._compute_gradient_hessian(np.zeros(n_features), X, time, event, ep, entry=getattr(self, "_entry", None))
            info_0 = -hess_0
            info_0_inv = np.linalg.solve(info_0, np.eye(n_features))
'''
new = '''            _, hess_0 = self._compute_gradient_hessian(np.zeros(n_features), X, time, event, ep, entry=getattr(self, "_entry", None))
            info_0 = self._observed_information(hess_0)
            info_0_inv = np.linalg.solve(info_0, np.eye(n_features))
'''
if cox.count(old) != 1:
    raise RuntimeError(f"Cox score information block count={cox.count(old)}")
cox_path.write_text(cox.replace(old, new, 1), encoding="utf-8")

# Optional Torch test must skip cleanly on CPU-only validation environments.
replace_once(
    "dev/tests/test_dbscan_edge_cases.py",
    '''        if backend == "torch":
            import torch
            X = torch.tensor(X_np, dtype=torch.float64)
            y = torch.tensor(y_np, dtype=torch.float64)
''',
    '''        if backend == "torch":
            torch = pytest.importorskip("torch")
            X = torch.tensor(X_np, dtype=torch.float64)
            y = torch.tensor(y_np, dtype=torch.float64)
''',
)

# V10 smoke tests were stale relative to the documented public loss namespace
# and benchmark-backed solver dispatch table.
v10_path = Path("dev/tests/test_v10_import_smoke.py")
v10 = v10_path.read_text(encoding="utf-8")
old = '''def test_old_losses_namespace_is_not_a_compatibility_entrypoint():
    import importlib.util

    assert importlib.util.find_spec("statgpu.losses") is None
'''
new = '''def test_losses_namespace_remains_a_public_compatibility_entrypoint():
    from statgpu.losses import LossBase, get_loss

    assert LossBase is not None
    assert get_loss("huber").name == "huber"
'''
if v10.count(old) != 1:
    raise RuntimeError(f"loss namespace test count={v10.count(old)}")
v10 = v10.replace(old, new, 1)
v10 = v10.replace(
    '''    # Smooth L2 GLMs dispatch to IRLS on all backends
    assert logit._select_solver(logit_loss, backend_name="numpy") == "irls"
    assert logit._select_solver(logit_loss, backend_name="cupy") == "irls"
    assert logit._select_solver(logit_loss, backend_name="torch") == "irls"
''',
    '''    # Smooth L2 GLMs dispatch to Newton on all backends.
    assert logit._select_solver(logit_loss, backend_name="numpy") == "newton"
    assert logit._select_solver(logit_loss, backend_name="cupy") == "newton"
    assert logit._select_solver(logit_loss, backend_name="torch") == "newton"
''',
    1,
)
v10 = v10.replace(
    '''    # Smooth L2 GLMs dispatch to IRLS on all backends
    assert poisson._select_solver(poisson_loss, backend_name="numpy") == "irls"
    assert poisson._select_solver(poisson_loss, backend_name="cupy") == "irls"
    assert poisson._select_solver(poisson_loss, backend_name="torch") == "irls"
''',
    '''    # Smooth L2 GLMs dispatch to Newton on all backends.
    assert poisson._select_solver(poisson_loss, backend_name="numpy") == "newton"
    assert poisson._select_solver(poisson_loss, backend_name="cupy") == "newton"
    assert poisson._select_solver(poisson_loss, backend_name="torch") == "newton"
''',
    1,
)
v10 = v10.replace(
    '''    assert ridge._select_solver(ridge_loss, backend_name="numpy") == "exact"
    assert ridge._select_solver(ridge_loss, backend_name="cupy") == "exact"
''',
    '''    assert ridge._select_solver(ridge_loss, backend_name="numpy") == "exact"
    assert ridge._select_solver(ridge_loss, backend_name="cupy") == "newton"
    assert ridge._select_solver(ridge_loss, backend_name="torch") == "newton"
''',
    1,
)
v10_path.write_text(v10, encoding="utf-8")

# Convert helper-style RidgeCV tests into real pytest assertions/skips.
ridge_cv_path = Path("dev/tests/test_ridge_cv.py")
ridge_cv = ridge_cv_path.read_text(encoding="utf-8")
ridge_cv = ridge_cv.replace(
    "import numpy as np\nimport sys\n",
    "import numpy as np\nimport pytest\nimport sys\n",
    1,
)
ridge_cv = ridge_cv.replace(
    "from statgpu.linear_model import RidgeCV\n",
    "from statgpu.linear_model import RidgeCV\nfrom statgpu.backends import get_backend\n",
    1,
)
ridge_cv = ridge_cv.replace(
    '''    return alpha_match, coef_diff


def test_ridge_cv_gpu_vs_cpu():
''',
    '''    assert coef_diff < 0.01


def test_ridge_cv_gpu_vs_cpu():
''',
    1,
)
old = '''    try:
        import cupy as cp
        print(f"CuPy available: {cp.__version__}")
    except ImportError:
        print("CuPy not available, skipping GPU test")
        return None, None
'''
new = '''    cp = pytest.importorskip("cupy")
    if not get_backend("cupy").is_available():
        pytest.skip("working CuPy CUDA backend is unavailable")
    print(f"CuPy available: {cp.__version__}")
'''
if ridge_cv.count(old) != 1:
    raise RuntimeError(f"RidgeCV GPU import block count={ridge_cv.count(old)}")
ridge_cv = ridge_cv.replace(old, new, 1)
ridge_cv = ridge_cv.replace(
    '''    return alpha_match, coef_diff


def test_ridge_cv_alpha_selection():
''',
    '''    assert alpha_match
    assert coef_diff < 1e-5


def test_ridge_cv_alpha_selection():
''',
    1,
)
ridge_cv_path.write_text(ridge_cv, encoding="utf-8")

# Focused regression coverage for the final numerical fixes.
Path("dev/tests/test_repository_review_final.py").write_text(
    dedent(
        '''
        import numpy as np
        import pytest

        from statgpu import Ridge
        from statgpu.survival import CoxPH


        sklearn = pytest.importorskip("sklearn")
        statsmodels = pytest.importorskip("statsmodels.duration.api")
        from sklearn.linear_model import Ridge as SklearnRidge


        def test_ridge_exact_matches_sklearn_alpha_convention():
            rng = np.random.default_rng(987)
            X = rng.normal(size=(600, 12))
            y = X @ rng.normal(size=12) + 1.7 + rng.normal(scale=0.2, size=600)
            ours = Ridge(alpha=1.0, fit_intercept=True, device="cpu").fit(X, y)
            reference = SklearnRidge(alpha=1.0, fit_intercept=True).fit(X, y)
            np.testing.assert_allclose(ours.coef_, reference.coef_, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(ours.intercept_, reference.intercept_, rtol=1e-9, atol=1e-9)


        @pytest.mark.parametrize("ties", ["breslow", "efron"])
        def test_cox_information_orientation_matches_statsmodels(ties):
            rng = np.random.default_rng(654)
            n, p = 700, 5
            X = rng.normal(size=(n, p))
            beta = rng.normal(scale=0.25, size=p)
            u = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
            true_time = -np.log(u) / (0.04 * np.exp(X @ beta))
            censor = rng.exponential(scale=np.median(true_time), size=n)
            event = (true_time <= censor).astype(int)
            time = np.minimum(true_time, censor)

            ours = CoxPH(ties=ties, device="cpu", max_iter=80, tol=1e-8).fit(
                X, time, event
            )
            reference = statsmodels.PHReg(time, X, status=event, ties=ties).fit()
            assert np.all(ours._bse > 0)
            np.testing.assert_allclose(ours.coef_, reference.params, rtol=2e-2, atol=2e-3)
            np.testing.assert_allclose(ours._bse, reference.bse, rtol=2e-1, atol=2e-3)
        '''
    ),
    encoding="utf-8",
)
