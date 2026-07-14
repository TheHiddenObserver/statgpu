"""Regression tests for the third review/fix cycle of PR #79."""

from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def panel_data():
    rng = np.random.default_rng(20260714)
    n_entities, n_times = 12, 6
    entity = np.repeat(np.array([f"entity-{i}" for i in range(n_entities)]), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    effects = np.repeat(rng.normal(scale=0.8, size=n_entities), n_times)
    y = 1.3 * X[:, 0] - 0.6 * X[:, 1] + effects + rng.normal(scale=0.1, size=entity.size)
    return X, y, entity, time


def _torch_backend_patch():
    torch = pytest.importorskip("torch")
    from statgpu._base import BaseEstimator
    from statgpu.backends import TorchBackend

    backend = TorchBackend(device="cpu")
    return torch, patch.object(
        BaseEstimator, "_get_backend", lambda self, backend="auto", _resolved=backend: _resolved
    )


class TestTorchLinearAlgebraAndPanel:
    def test_cholesky_solve_accepts_vector_and_matrix_rhs(self):
        torch = pytest.importorskip("torch")
        from statgpu.backends import xp_cholesky_solve

        A = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
        b = torch.tensor([1.0, 2.0], dtype=torch.float64)
        B = torch.column_stack([b, 2.0 * b])
        np.testing.assert_allclose(
            xp_cholesky_solve(A, b, torch).numpy(),
            np.linalg.solve(A.numpy(), b.numpy()),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            xp_cholesky_solve(A, B, torch).numpy(),
            np.linalg.solve(A.numpy(), B.numpy()),
            rtol=1e-12,
        )

    @pytest.mark.parametrize(
        "model_factory,extra",
        [
            (lambda: __import__("statgpu.panel", fromlist=["PanelOLS"]).PanelOLS(entity_effects=True), {}),
            (lambda: __import__("statgpu.panel", fromlist=["RandomEffects"]).RandomEffects(), {}),
            (lambda: __import__("statgpu.panel", fromlist=["BetweenOLS"]).BetweenOLS(), {}),
            (lambda: __import__("statgpu.panel", fromlist=["FirstDifferenceOLS"]).FirstDifferenceOLS(), {"use_time": True}),
        ],
    )
    def test_panel_estimators_accept_string_labels_on_torch(self, panel_data, model_factory, extra):
        X, y, entity, time = panel_data
        expected = model_factory().fit(
            X, y, entity_ids=entity,
            **({"time_ids": time} if extra.get("use_time") else {}),
        )
        _, backend_patch = _torch_backend_patch()
        with backend_patch:
            actual = model_factory().fit(
                X, y, entity_ids=entity,
                **({"time_ids": time} if extra.get("use_time") else {}),
            )
        np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=1e-9, atol=1e-9)
        assert np.all(np.isfinite(actual.bse_))

    def test_pooled_ols_preserves_torch_array_input_through_formula_helper(self, panel_data):
        torch, backend_patch = _torch_backend_patch()
        from statgpu.panel import PooledOLS
        from statgpu.panel._formula import _prepare_formula_fit

        X, y, _, _ = panel_data
        X_t = torch.tensor(X, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64)
        y_out, X_out, *_ = _prepare_formula_fit(
            None, None, X_t, y_t, model_has_intercept=True
        )
        assert X_out is X_t
        assert y_out is y_t
        with backend_patch:
            model = PooledOLS().fit(X_t, y_t)
        assert np.all(np.isfinite(model.coef_))

    def test_panel_effect_predictions_preserve_original_string_keys(self, panel_data):
        from statgpu.panel import PanelOLS

        X, y, entity, _ = panel_data
        model = PanelOLS(entity_effects=True).fit(X, y, entity_ids=entity)
        with_effects = model.predict(X, entity_ids=entity)
        without_effects = model.predict(X)
        assert np.max(np.abs(with_effects - without_effects)) > 0.01
        assert set(model._entity_effects_map) == set(np.unique(entity))

    def test_random_effects_formula_aligns_explicit_entity_ids(self, panel_data):
        pd = pytest.importorskip("pandas")
        from statgpu.panel import RandomEffects

        X, y, entity, _ = panel_data
        data = pd.DataFrame({"y": y, "x1": X[:, 0], "x2": X[:, 1]})
        data.loc[3, "x1"] = np.nan
        model = RandomEffects().fit(
            formula="y ~ x1 + x2", data=data, entity_ids=entity
        )
        assert model.nobs == len(data) - 1

    def test_first_difference_keeps_numeric_design_on_backend(self):
        from pathlib import Path
        import statgpu.panel._first_diff as module

        text = Path(module.__file__).read_text()
        function = text[text.index("def _first_diff_transform"):]
        assert "_to_numpy(X)" not in function
        assert "_to_numpy(y)" not in function
        assert "sort_idx = xp_asarray" in function


class TestKernelAndSplineTorchPaths:
    def test_kernel_pca_torch_matches_numpy_and_rejects_nonfinite(self):
        from statgpu.nonparametric.kernel_methods import KernelPCA

        rng = np.random.default_rng(14)
        X = rng.normal(size=(35, 4))
        expected = KernelPCA(n_components=3, alpha=0.1).fit_transform(X)
        torch, backend_patch = _torch_backend_patch()
        with backend_patch:
            actual = KernelPCA(n_components=3, alpha=0.1).fit_transform(
                torch.tensor(X, dtype=torch.float64)
            )
        # Eigenvector signs are arbitrary; compare Gram matrices of embeddings.
        np.testing.assert_allclose(
            actual.detach().numpy() @ actual.detach().numpy().T,
            expected @ expected.T,
            rtol=1e-8,
            atol=1e-8,
        )
        with pytest.raises(ValueError, match="finite"):
            KernelPCA().fit(np.array([[0.0, np.nan], [1.0, 2.0]]))

    def test_ridge_gram_eigen_solver_accepts_torch_rank_deficiency(self):
        torch = pytest.importorskip("torch")
        from statgpu.backends import TorchBackend
        from statgpu.linear_model.cv._ridge_cv import _solve_ridge_path_gpu_from_gram_eig

        gram = np.array([[[1.0, 1.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 0.0]]])
        cross = np.array([[1.0, 1.0], [2.0, 0.0]])
        alphas = np.array([0.1, 1.0])
        sizes = np.array([10.0, 8.0])
        actual = _solve_ridge_path_gpu_from_gram_eig(
            torch.tensor(gram, dtype=torch.float64),
            torch.tensor(cross, dtype=torch.float64),
            alphas,
            TorchBackend(device="cpu"),
            n_samples_vec=sizes,
        ).numpy()
        expected = np.empty_like(actual)
        for a, alpha in enumerate(alphas):
            for fold in range(gram.shape[0]):
                expected[a, fold] = np.linalg.solve(
                    gram[fold] + sizes[fold] * alpha * np.eye(2), cross[fold]
                )
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    def test_thin_plate_spline_torch_matches_numpy(self):
        torch = pytest.importorskip("torch")
        from statgpu.nonparametric.splines import thin_plate_spline_basis

        X = np.column_stack([np.linspace(0.0, 1.0, 15), np.linspace(1.0, 0.0, 15)])
        knots = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        expected = thin_plate_spline_basis(X, knots, xp=np)
        actual = thin_plate_spline_basis(
            torch.tensor(X, dtype=torch.float64),
            torch.tensor(knots, dtype=torch.float64),
            xp=torch,
        )
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
        assert actual.device.type == "cpu"
        with pytest.raises(ValueError, match="finite"):
            thin_plate_spline_basis(np.array([0.0, np.nan]), np.array([0.0, 1.0]))


class TestFiniteInputContracts:
    @pytest.mark.parametrize("estimator", [
        pytest.param("KMeans"),
        pytest.param("PCA"),
        pytest.param("GaussianMixture"),
        pytest.param("NMF"),
    ])
    def test_unsupervised_estimators_reject_nonfinite(self, estimator):
        import statgpu.unsupervised as unsupervised

        constructors = {
            "KMeans": lambda cls: cls(n_clusters=2),
            "PCA": lambda cls: cls(n_components=1),
            "GaussianMixture": lambda cls: cls(n_components=2),
            "NMF": lambda cls: cls(n_components=1),
        }
        model = constructors[estimator](getattr(unsupervised, estimator))
        with pytest.raises(ValueError, match="finite"):
            model.fit(np.array([[1.0, np.nan], [2.0, 3.0]]))

    @pytest.mark.parametrize(
        "estimator_name",
        ["EmpiricalCovariance", "LedoitWolf", "OAS", "ShrunkCovariance"],
    )
    def test_covariance_estimators_reject_nonfinite_and_empty_features(self, estimator_name):
        import statgpu.covariance as covariance

        cls = getattr(covariance, estimator_name)
        with pytest.raises(ValueError, match="finite"):
            cls().fit(np.array([[1.0, np.inf], [2.0, 3.0]]))
        with pytest.raises(ValueError, match="feature"):
            cls().fit(np.empty((3, 0)))

    def test_nystroem_rejects_nonfinite_in_fit_and_transform(self):
        from statgpu.nonparametric.kernel_methods import Nystroem

        with pytest.raises(ValueError, match="finite"):
            Nystroem(n_components=2).fit(np.array([[0.0, np.nan], [1.0, 2.0]]))
        model = Nystroem(n_components=2).fit(np.array([[0.0, 1.0], [1.0, 2.0]]))
        with pytest.raises(ValueError, match="finite"):
            model.transform(np.array([[np.inf, 1.0]]))
