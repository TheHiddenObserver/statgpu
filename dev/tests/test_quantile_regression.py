"""Tests for standalone QuantileRegression with compute_inference."""
import numpy as np
import pytest
from statgpu.linear_model import QuantileRegression


# ---- GPU availability checks ----
try:
    import cupy as cp
    _HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUPY = False

try:
    import torch
    _HAS_TORCH_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_TORCH_CUDA = False


class TestQuantileRegression:
    """Basic fit + inference for the standalone QuantileRegression class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.X = np.random.randn(200, 3)
        self.y = 1.0 + self.X @ [0.5, -0.3, 0.8] + 0.5 * np.random.randn(200)

    def test_fit_without_inference(self):
        m = QuantileRegression(quantile=0.5)
        m.fit(self.X, self.y)
        assert m.coef_ is not None
        assert len(m.coef_) == 3
        assert m.n_iter_ > 0
        assert not hasattr(m, '_bse') or m._bse is None

    def test_extreme_quantile_kernel_inference_fails_closed(self):
        model = QuantileRegression(
            quantile=0.01,
            compute_inference=True,
            inference_method="kernel",
            bandwidth="hsheather",
        )
        with pytest.raises(ValueError, match="q ± h leaves the probability interval"):
            model.fit(self.X, self.y)
        assert model._fitted is False
        assert model.coef_ is None
        assert model._inference_result is None

    def test_degenerate_kernel_bandwidth_fails_closed(self):
        X = self.X.copy()
        y = np.full(self.X.shape[0], 2.0, dtype=np.float64)
        model = QuantileRegression(
            quantile=0.5,
            compute_inference=True,
            inference_method="kernel",
        )
        with pytest.raises(ValueError, match="bandwidth must be finite and positive"):
            model.fit(X, y)
        assert model._fitted is False
        assert model._inference_result is None

    def test_fit_with_kernel_inference(self):
        m = QuantileRegression(quantile=0.5, compute_inference=True,
                                inference_method='kernel')
        m.fit(self.X, self.y)
        assert m._bse is not None
        assert len(m._bse) == 4  # 3 coef + intercept
        assert m._pvalues is not None
        assert m._conf_int.shape == (4, 2)
        # SE should be positive
        assert np.all(m._bse > 0)

    def test_nonmedian_pinball_eta_gradient_matches_requested_quantile(self):
        from statgpu.linear_model.wrappers._quantile import _pinball_eta_gradient_values

        g_nonnegative, g_negative = _pinball_eta_gradient_values(0.2)
        assert g_nonnegative == pytest.approx(-0.2)
        assert g_negative == pytest.approx(0.8)

    def test_nonmedian_batched_bootstrap_targets_requested_quantile(self):
        tau = 0.2
        n = 80
        B = 8
        X = np.ones((n, 1), dtype=np.float64)
        y = np.linspace(-4.0, 4.0, n, dtype=np.float64)
        model = QuantileRegression(
            quantile=tau,
            fit_intercept=False,
            max_iter=300,
            tol=1e-7,
            n_bootstrap=B,
            random_state=17,
        )
        model.coef_ = np.zeros(1, dtype=np.float64)
        model.intercept_ = 0.0

        boot_params, _, _ = model._compute_bootstrap_batched(X, y)

        rng = np.random.default_rng(model.random_state)
        y_batch = np.array([
            y[rng.integers(0, n, size=n)]
            for _ in range(B)
        ])
        target_q = np.quantile(y_batch, tau, axis=1)
        wrong_q = np.quantile(y_batch, 1.0 - tau, axis=1)
        estimated = np.asarray(boot_params[:, 0], dtype=np.float64)

        target_error = np.mean(np.abs(estimated - target_q))
        wrong_error = np.mean(np.abs(estimated - wrong_q))
        assert target_error < wrong_error
        assert float(np.median(estimated)) < 0.0

    @pytest.mark.parametrize("inference_method", ["kernel", "bootstrap"])
    def test_nonuniform_weighted_inference_fails_before_solver(
        self, monkeypatch, inference_method
    ):
        import statgpu.linear_model.wrappers._quantile as quantile_mod

        model = QuantileRegression(
            quantile=0.3,
            compute_inference=True,
            inference_method=inference_method,
            n_bootstrap=4,
        )
        weights = np.linspace(0.5, 1.5, self.X.shape[0])

        def forbidden(*args, **kwargs):
            raise AssertionError("unsupported weighted inference must fail before FISTA")

        monkeypatch.setattr(quantile_mod, "fista_solver", forbidden)
        with pytest.raises(NotImplementedError, match="non-uniform sample_weight"):
            model.fit(self.X, self.y, sample_weight=weights)
        assert model._fitted is False
        assert model.coef_ is None
        assert model._inference_result is None

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"quantile": "0.5"}, "quantile must be a finite real number"),
            ({"quantile": True}, "quantile must be a finite real number"),
            ({"fit_intercept": "False"}, "fit_intercept must be boolean"),
            ({"max_iter": True}, "max_iter must be a positive integer"),
            ({"max_iter": 0}, "max_iter must be a positive integer"),
            ({"tol": "1e-4"}, "tol must be a finite positive number"),
            ({"tol": 0.0}, "tol must be a finite positive number"),
            ({"compute_inference": "False"}, "compute_inference must be boolean"),
            ({"gpu_memory_cleanup": "False"}, "gpu_memory_cleanup must be boolean"),
        ],
    )
    def test_invalid_public_controls_fail_before_backend(self, monkeypatch, kwargs, message):
        model = QuantileRegression(**kwargs)

        def forbidden(*args, **kwargs):
            raise AssertionError("invalid public controls must fail before backend work")

        monkeypatch.setattr(model, "_get_backend", forbidden)
        with pytest.raises(ValueError, match=message):
            model.fit(self.X, self.y)
        assert model._fitted is False
        assert model.coef_ is None

    def test_mutated_quantile_control_fails_before_backend(self, monkeypatch):
        model = QuantileRegression(quantile=0.5)
        model.set_params(quantile="0.5")

        def forbidden(*args, **kwargs):
            raise AssertionError("invalid mutated quantile must fail before backend work")

        monkeypatch.setattr(model, "_get_backend", forbidden)
        with pytest.raises(ValueError, match="quantile must be a finite real number"):
            model.fit(self.X, self.y)
        assert model._fitted is False
        assert model.coef_ is None

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"kernel": ["epa"], "compute_inference": True}, "kernel must be one of"),
            (
                {"bandwidth": ["hsheather"], "compute_inference": True},
                "bandwidth must be",
            ),
        ],
    )
    def test_unhashable_inference_controls_raise_public_value_error(
        self, kwargs, message
    ):
        model = QuantileRegression(**kwargs)
        with pytest.raises(ValueError, match=message):
            model.fit(self.X, self.y)
        assert model._fitted is False
        assert model.coef_ is None

    def test_cleanup_backend_routes_cupy_name_to_cuda_cleanup(self, monkeypatch):
        model = QuantileRegression(gpu_memory_cleanup=True)
        calls = []
        monkeypatch.setattr(model, "_cleanup_cuda_memory", lambda: calls.append("cupy"))
        monkeypatch.setattr(model, "_cleanup_torch_memory", lambda: calls.append("torch"))

        model._cleanup_backend_memory("cupy")
        model._cleanup_backend_memory("torch")

        assert calls == ["cupy", "torch"]

    def test_failed_cupy_fit_runs_cleanup_before_state_reset(self, monkeypatch):
        import statgpu.linear_model.wrappers._quantile as quantile_mod

        class FakeBackend:
            name = "cupy"

        model = QuantileRegression(
            fit_intercept=False,
            gpu_memory_cleanup=True,
            compute_inference=False,
        )
        cleanup_calls = []
        monkeypatch.setattr(model, "_get_backend", lambda backend="auto": FakeBackend())
        monkeypatch.setattr(model, "_to_array", lambda value, backend=None: np.asarray(value))
        monkeypatch.setattr(
            model, "_cleanup_cuda_memory", lambda: cleanup_calls.append("cupy")
        )

        def failing_solver(*args, **kwargs):
            raise RuntimeError("synthetic solver failure")

        monkeypatch.setattr(quantile_mod, "fista_solver", failing_solver)
        with pytest.raises(RuntimeError, match="synthetic solver failure"):
            model.fit(self.X, self.y)

        assert cleanup_calls == ["cupy"]
        assert model._fitted is False
        assert model.coef_ is None
        assert model._selected_backend_name is None

    @pytest.mark.parametrize("n_bootstrap", [0, 1, True, 2.5])
    def test_invalid_bootstrap_count_fails_before_solver(self, monkeypatch, n_bootstrap):
        import statgpu.linear_model.wrappers._quantile as quantile_mod

        model = QuantileRegression(
            quantile=0.3,
            compute_inference=True,
            inference_method="bootstrap",
            n_bootstrap=n_bootstrap,
        )

        def forbidden(*args, **kwargs):
            raise AssertionError("invalid bootstrap count must fail before FISTA")

        monkeypatch.setattr(quantile_mod, "fista_solver", forbidden)
        with pytest.raises(ValueError, match="n_bootstrap must be an integer"):
            model.fit(self.X, self.y)
        assert model._fitted is False
        assert model.coef_ is None

    def test_failed_inference_fit_clears_partial_state(self):
        model = QuantileRegression(
            quantile=0.4,
            compute_inference=True,
            inference_method="invalid",
        )
        with pytest.raises(ValueError, match="Unknown inference_method"):
            model.fit(self.X, self.y)
        assert model._fitted is False
        assert model.coef_ is None
        assert model.intercept_ == 0.0
        assert model._params is None
        assert model._inference_result is None

    def test_predict_rejects_non_2d_and_wrong_feature_width(self):
        model = QuantileRegression(quantile=0.5).fit(self.X, self.y)
        with pytest.raises(ValueError, match="two-dimensional design matrix"):
            model.predict(self.X[0])
        with pytest.raises(ValueError, match="same number of features"):
            model.predict(self.X[:, :2])

    def test_fit_with_bootstrap_inference(self):
        m = QuantileRegression(quantile=0.5, compute_inference=True,
                                inference_method='bootstrap', n_bootstrap=50)
        m.fit(self.X, self.y)
        assert m._bse is not None
        assert len(m._bse) == 4
        assert np.all(m._bse > 0)

    def test_predict(self):
        m = QuantileRegression(quantile=0.5)
        m.fit(self.X, self.y)
        pred = m.predict(self.X)
        assert pred.shape == (200,)
        assert isinstance(pred, np.ndarray)

    def test_invalid_kernel_raises(self):
        m = QuantileRegression(kernel='invalid', compute_inference=True,
                                inference_method='kernel')
        with pytest.raises(ValueError, match="kernel must be one of"):
            m.fit(self.X, self.y)

    def test_singular_design_raises(self):
        X_bad = np.column_stack([self.X[:, 0], self.X[:, 0]])
        m = QuantileRegression(compute_inference=True, inference_method='kernel')
        try:
            m.fit(X_bad, self.y)
            # If fit succeeds, BSE should be NaN for the collinear columns
            assert m._bse is not None
            # At least one BSE should be invalid (NaN or Inf)
            assert np.any(np.isnan(m._bse)) or np.any(np.isinf(m._bse)), \
                f"Expected NaN/Inf BSE for singular design, got {m._bse}"
        except (np.linalg.LinAlgError, RuntimeError):
            pass  # raising is also acceptable

    def test_n_categories_validator_is_not_quantile(self):
        """QuantileRegression does not have n_categories."""
        m = QuantileRegression()
        assert not hasattr(m, 'n_categories')

    # ---- GPU backend tests ----

    @pytest.mark.skipif(not _HAS_CUPY, reason="CuPy GPU not available")
    def test_kernel_inference_cupy(self):
        """Kernel-based inference on CuPy GPU — BSE must match CPU."""
        import cupy as cp
        X_cp = cp.asarray(self.X)
        y_cp = cp.asarray(self.y)
        # CPU reference
        m_cpu = QuantileRegression(quantile=0.5, compute_inference=True,
                                    inference_method='kernel')
        m_cpu.fit(self.X, self.y)
        # GPU
        m_gpu = QuantileRegression(quantile=0.5, compute_inference=True,
                                    inference_method='kernel', device='cuda')
        m_gpu.fit(X_cp, y_cp)
        assert m_gpu._bse is not None
        assert len(m_gpu._bse) == 4
        assert np.all(m_gpu._bse > 0)
        # BSE must match CPU within floating-point tolerance
        assert np.allclose(m_cpu._bse, m_gpu._bse, rtol=1e-8)

    @pytest.mark.skipif(not _HAS_CUPY, reason="CuPy GPU not available")
    def test_bootstrap_inference_cupy(self):
        """Bootstrap inference on CuPy GPU — BSE should be similar to CPU."""
        import cupy as cp
        X_cp = cp.asarray(self.X)
        y_cp = cp.asarray(self.y)
        m = QuantileRegression(quantile=0.5, compute_inference=True,
                                inference_method='bootstrap', n_bootstrap=50,
                                device='cuda')
        m.fit(X_cp, y_cp)
        assert m._bse is not None
        assert len(m._bse) == 4
        assert np.all(m._bse > 0)

    @pytest.mark.skipif(not _HAS_TORCH_CUDA, reason="Torch CUDA not available")
    def test_kernel_inference_torch(self):
        """Kernel-based inference on Torch GPU — BSE must match CPU."""
        import torch
        X_t = torch.tensor(self.X, dtype=torch.float64, device='cuda')
        y_t = torch.tensor(self.y, dtype=torch.float64, device='cuda')
        # CPU reference
        m_cpu = QuantileRegression(quantile=0.5, compute_inference=True,
                                    inference_method='kernel')
        m_cpu.fit(self.X, self.y)
        # GPU
        m_gpu = QuantileRegression(quantile=0.5, compute_inference=True,
                                    inference_method='kernel', device='torch')
        m_gpu.fit(X_t, y_t)
        assert m_gpu._bse is not None
        assert len(m_gpu._bse) == 4
        assert np.all(m_gpu._bse > 0)
        # BSE must match CPU within floating-point tolerance
        assert np.allclose(m_cpu._bse, m_gpu._bse, rtol=1e-8)

    @pytest.mark.skipif(not _HAS_TORCH_CUDA, reason="Torch CUDA not available")
    def test_bootstrap_inference_torch(self):
        """Bootstrap inference on Torch GPU — BSE should be similar to CPU."""
        import torch
        X_t = torch.tensor(self.X, dtype=torch.float64, device='cuda')
        y_t = torch.tensor(self.y, dtype=torch.float64, device='cuda')
        m = QuantileRegression(quantile=0.5, compute_inference=True,
                                inference_method='bootstrap', n_bootstrap=50,
                                device='torch')
        m.fit(X_t, y_t)
        assert m._bse is not None
        assert len(m._bse) == 4
        assert np.all(m._bse > 0)
