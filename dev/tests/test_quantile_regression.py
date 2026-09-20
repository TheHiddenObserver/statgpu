"""Tests for standalone QuantileRegression with compute_inference."""
import sys
import types
import warnings

import numpy as np
import pytest

from statgpu._config import Device
from statgpu.linear_model import QuantileRegression
import statgpu.backends._utils as _backend_utils
from statgpu.solvers._convergence import ConvergenceWarning
from statgpu.linear_model.wrappers._quantile import (
    _align_quantile_fit_inputs,
    _batched_quantile_irls,
    _bootstrap_schedule_to_backend,
    _center_quantile_bootstrap_residuals,
)


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

    def test_predict_integer_design_preserves_float_coefficients(self):
        model = QuantileRegression(quantile=0.5, device="cpu")
        model._fitted = True
        model.coef_ = np.asarray([0.5, -0.25], dtype=np.float64)
        model.intercept_ = 0.2
        model._selected_backend_name = "numpy"
        model._selected_backend_device = "cpu"

        X = np.asarray([[2, 4], [-1, 3]], dtype=np.int64)
        pred = model.predict(X)

        assert pred.dtype == np.float64
        np.testing.assert_allclose(
            pred,
            X.astype(np.float64) @ model.coef_ + model.intercept_,
            rtol=0.0,
            atol=0.0,
        )

    def test_prediction_design_reuses_recorded_cupy_device(self, monkeypatch):
        model = QuantileRegression(quantile=0.5, device="cpu")
        model._selected_backend_device = "cuda:3"
        captured = {"entered": [], "convert": []}

        class FakeDevice:
            def __init__(self, device_id):
                self.device_id = int(device_id)

            def __enter__(self):
                captured["entered"].append(self.device_id)
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        fake_cupy = types.SimpleNamespace(
            float64=np.float64,
            cuda=types.SimpleNamespace(Device=FakeDevice),
            asarray=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("ambient cp.asarray must not own placement")
            ),
        )
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

        def fake_align(value, target_device, dtype=None):
            captured["target_device"] = int(target_device)
            captured["dtype"] = dtype
            return np.asarray(value, dtype=dtype)

        monkeypatch.setattr(
            _backend_utils,
            "_cupy_asarray_on_device",
            fake_align,
        )
        monkeypatch.setattr(
            model,
            "_to_array",
            lambda value, device, backend=None: (
                captured["convert"].append((device, backend))
                or np.asarray(value)
            ),
        )

        X = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
        result = model._prediction_array_on_fit_device(X, "cupy")

        assert captured["entered"] == [3]
        assert captured["convert"] == [(Device.CUDA, "cupy")]
        assert captured["target_device"] == 3
        assert captured["dtype"] is np.float64
        assert result.dtype == np.float64

    def test_prediction_design_reuses_recorded_torch_device(self, monkeypatch):
        torch = pytest.importorskip("torch")
        model = QuantileRegression(quantile=0.5, device="cpu")
        model._selected_backend_device = "cuda:2"
        calls = []

        class FakeTensor:
            def to(self, *, device, dtype):
                calls.append(("to", str(device), dtype))
                return self

        def fake_to_torch(value, device=None):
            calls.append(("convert", str(device)))
            return FakeTensor()

        monkeypatch.setattr(model, "_to_torch", fake_to_torch)

        result = model._prediction_array_on_fit_device(
            np.asarray([[1.0]], dtype=np.float64),
            "torch",
        )

        assert isinstance(result, FakeTensor)
        assert calls == [
            ("convert", "cuda:2"),
            ("to", "cuda:2", torch.float64),
        ]

    def test_fractional_weights_are_preserved_for_integer_design(self):
        X_int = np.ones((4, 1), dtype=np.int64)
        X_float = X_int.astype(np.float64)
        y = np.array([0.0, 1.0, 2.0, 10.0], dtype=np.float64)
        weights = np.array([0.6, 0.6, 0.6, 1.1], dtype=np.float64)

        common = dict(
            quantile=0.5,
            fit_intercept=False,
            max_iter=2000,
            tol=1e-8,
            compute_inference=False,
            device="cpu",
        )
        integer_fit = QuantileRegression(**common).fit(
            X_int,
            y,
            sample_weight=weights,
        )
        float_fit = QuantileRegression(**common).fit(
            X_float,
            y,
            sample_weight=weights,
        )

        np.testing.assert_allclose(
            integer_fit.coef_,
            float_fit.coef_,
            rtol=0.0,
            atol=1e-7,
        )
        # Correct fractional weighting places the median near y=2; truncating
        # the first three 0.6 weights to zero would instead target y=10.
        assert float(integer_fit.coef_[0]) < 5.0

    def test_fit_without_inference(self):
        m = QuantileRegression(quantile=0.5)
        m.fit(self.X, self.y)
        assert m.coef_ is not None
        assert len(m.coef_) == 3
        assert m.n_iter_ > 0
        assert not hasattr(m, '_bse') or m._bse is None

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("bse", np.asarray([1.0, np.nan])),
            ("statistic", np.asarray([0.0, np.inf])),
            ("conf_int", np.asarray([[0.0, 1.0], [np.nan, 2.0]])),
        ],
    )
    def test_inference_output_validator_rejects_nonfinite_arrays(self, name, value):
        with pytest.raises(ValueError, match=f"non-finite {name}"):
            QuantileRegression._validate_inference_outputs(**{name: value})

    def test_inference_output_validator_rejects_invalid_pvalues(self):
        with pytest.raises(ValueError, match="p-values outside"):
            QuantileRegression._validate_inference_outputs(
                pvalues=np.asarray([0.2, 1.1])
            )

    def test_bootstrap_nonfinite_snapshot_is_not_published(self, monkeypatch):
        model = QuantileRegression(
            quantile=0.5,
            compute_inference=False,
            n_bootstrap=4,
        ).fit(self.X, self.y)

        bad = np.zeros((4, len(model._params)), dtype=np.float64)
        bad[0, 0] = np.nan
        monkeypatch.setattr(
            model,
            "_compute_bootstrap_batched",
            lambda X, y: (bad, None, None),
        )

        with pytest.raises(ValueError, match="non-finite boot_params"):
            model._compute_inference_bootstrap(self.X, self.y)

        assert model._inference_result is None
        assert model._bse is None
        assert model._pvalues is None
        assert model._conf_int is None

    def test_zero_kernel_density_estimate_fails_closed(self, monkeypatch):
        model = QuantileRegression(quantile=0.5).fit(self.X, self.y)

        monkeypatch.setattr(
            model,
            "_get_kernel_fn",
            lambda name, xp=None: (
                lambda u: np.zeros_like(np.asarray(u), dtype=np.float64)
            ),
        )

        with pytest.raises(ValueError, match="finite positive residual density estimate"):
            model._compute_inference_kernel(self.X, self.y)

        assert model._inference_result is None
        assert model._bse is None

    @pytest.mark.parametrize("fhat", [0.0, -1.0, np.nan, np.inf])
    def test_kernel_density_estimate_validator_rejects_invalid_values(self, fhat):
        with pytest.raises(ValueError, match="finite positive residual density estimate"):
            QuantileRegression._validate_kernel_density_estimate(fhat)

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

    def test_kernel_inference_records_concrete_backend_device(self):
        model = QuantileRegression(
            quantile=0.5,
            compute_inference=True,
            inference_method="kernel",
        ).fit(self.X, self.y)

        assert model._selected_backend_name == "numpy"
        assert model._selected_backend_device == "cpu"
        metadata = model._inference_result.metadata
        assert metadata["numerical_backend"] == "numpy"
        assert metadata["numerical_device"] == "cpu"
        assert metadata["reporting_backend"] == "numpy"

    def test_kernel_inference_records_cpu_provenance(self):
        model = QuantileRegression(
            quantile=0.5,
            compute_inference=True,
            inference_method="kernel",
        ).fit(self.X, self.y)

        metadata = model._inference_result.metadata
        assert metadata["numerical_backend"] == "numpy"
        assert metadata["numerical_device"] == "cpu"
        assert metadata["reporting_backend"] == "numpy"

    def test_kernel_gpu_singular_design_matches_cpu_failure_semantics(self):
        torch = pytest.importorskip("torch")

        x = torch.linspace(-1.0, 1.0, 48, dtype=torch.float64)
        X = torch.column_stack([x, x])
        y = 0.4 * x + torch.linspace(-0.25, 0.25, 48, dtype=torch.float64)
        model = QuantileRegression(
            quantile=0.5,
            fit_intercept=False,
            kernel="gau",
            bandwidth="hsheather",
        )
        model.coef_ = np.zeros(2, dtype=np.float64)
        model.intercept_ = 0.0
        model._selected_backend_name = "torch"
        model._selected_backend_device = "cpu"

        with pytest.raises(
            np.linalg.LinAlgError,
            match="design matrix is singular.*cannot compute kernel standard errors",
        ):
            model._compute_inference_kernel_gpu(X, y)

    def test_kernel_torch_cpu_uses_numpy_population_std_bandwidth(self):
        torch = pytest.importorskip("torch")

        X_np = np.linspace(-1.0, 1.0, 48, dtype=np.float64).reshape(-1, 1)
        y_np = (
            0.35 * X_np[:, 0]
            + np.linspace(-0.4, 0.6, 48, dtype=np.float64)
        )

        cpu = QuantileRegression(
            quantile=0.4,
            fit_intercept=False,
            kernel="gau",
            bandwidth="hsheather",
        )
        cpu.coef_ = np.array([0.3], dtype=np.float64)
        cpu.intercept_ = 0.0
        cpu._selected_backend_name = "numpy"
        cpu._selected_backend_device = "cpu"
        cpu._compute_inference_kernel(X_np, y_np)

        X_t = torch.as_tensor(X_np, dtype=torch.float64)
        y_t = torch.as_tensor(y_np, dtype=torch.float64)
        torch_model = QuantileRegression(
            quantile=0.4,
            fit_intercept=False,
            kernel="gau",
            bandwidth="hsheather",
        )
        torch_model.coef_ = np.array([0.3], dtype=np.float64)
        torch_model.intercept_ = 0.0
        torch_model._selected_backend_name = "torch"
        torch_model._selected_backend_device = "cpu"
        torch_model._compute_inference_kernel_gpu(X_t, y_t)

        assert torch_model._inference_result.metadata["bandwidth"] == pytest.approx(
            cpu._inference_result.metadata["bandwidth"],
            rel=1e-12,
            abs=1e-12,
        )
        np.testing.assert_allclose(
            torch_model._bse,
            cpu._bse,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_kernel_gpu_reference_distribution_follows_torch_device(
        self, monkeypatch
    ):
        torch = pytest.importorskip("torch")
        import statgpu.inference._distributions_backend as dist_mod

        X = torch.linspace(-1.0, 1.0, 32, dtype=torch.float64).reshape(-1, 1)
        y = 0.4 * X[:, 0] + torch.linspace(
            -0.3, 0.3, 32, dtype=torch.float64
        )
        model = QuantileRegression(
            quantile=0.5,
            fit_intercept=False,
            kernel="gau",
            bandwidth="hsheather",
        )
        model.coef_ = np.array([0.35], dtype=np.float64)
        model.intercept_ = 0.0
        model._selected_backend_name = "torch"
        model._selected_backend_device = "cpu"

        observed = {}
        host_shapes = []
        import statgpu.backends as backends

        original_to_numpy = backends._to_numpy

        def recording_to_numpy(value):
            shape = getattr(value, "shape", None)
            host_shapes.append(None if shape is None else tuple(shape))
            return original_to_numpy(value)

        monkeypatch.setattr(backends, "_to_numpy", recording_to_numpy)

        class FakeNorm:
            def sf(self, value):
                observed["sf_device"] = str(value.device)
                return torch.full_like(value, 0.25)

            def ppf(self, value):
                observed["ppf_device"] = str(value.device)
                observed["ppf_dtype"] = value.dtype
                return torch.as_tensor(
                    1.959963984540054,
                    dtype=value.dtype,
                    device=value.device,
                )

        original_get_distribution = dist_mod.get_distribution

        def recording_get_distribution(name, backend="auto", device=None, **kwargs):
            if name == "norm" and backend == "torch":
                observed["backend"] = backend
                observed["device"] = device
                return FakeNorm()
            return original_get_distribution(
                name,
                backend=backend,
                device=device,
                **kwargs,
            )

        monkeypatch.setattr(
            dist_mod,
            "get_distribution",
            recording_get_distribution,
        )

        model._compute_inference_kernel_gpu(X, y)

        assert observed["backend"] == "torch"
        assert observed["device"] == "cpu"
        assert observed["sf_device"] == "cpu"
        assert observed["ppf_device"] == "cpu"
        assert observed["ppf_dtype"] == torch.float64
        assert (X.shape[0],) not in host_shapes

    def test_refit_without_inference_clears_all_statistic_aliases(self):
        model = QuantileRegression(
            quantile=0.5,
            compute_inference=True,
            inference_method="kernel",
        ).fit(self.X, self.y)

        assert model._zvalues is not None
        assert model._tvalues is not None

        model.set_params(compute_inference=False)
        model.fit(self.X, self.y)

        assert model._inference_result is None
        assert model._bse is None
        assert model._zvalues is None
        assert model._tvalues is None
        assert model._statistic is None
        assert model._pvalues is None
        assert model._conf_int is None

    def test_inference_fit_rejects_unconverged_point_estimate(self):
        model = QuantileRegression(
            quantile=0.5,
            max_iter=1,
            tol=1e-16,
            compute_inference=True,
            inference_method="kernel",
        )

        with pytest.raises(RuntimeError, match="point-estimation FISTA solve did not converge"):
            model.fit(self.X, self.y)

        assert model._fitted is False
        assert model.coef_ is None
        assert model._inference_result is None
        assert model._bse is None
        assert model._selected_backend_name is None
        assert model._selected_backend_device is None

    def test_estimation_fit_reemits_nonconvergence_independent_warnings(
        self, monkeypatch
    ):
        import statgpu.linear_model.wrappers._quantile as quantile_mod

        def warning_solver(loss, penalty, X, y, **kwargs):
            warnings.warn(
                "synthetic numerical warning",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros(X.shape[1], dtype=np.float64), 1

        monkeypatch.setattr(quantile_mod, "fista_solver", warning_solver)
        model = QuantileRegression(
            quantile=0.5,
            fit_intercept=False,
            compute_inference=False,
        )

        with pytest.warns(RuntimeWarning, match="synthetic numerical warning"):
            model.fit(self.X, self.y)

        assert model._fitted is True
        assert model.coef_ is not None

    def test_estimation_only_preserves_fista_nonconvergence_warning(self):
        model = QuantileRegression(
            quantile=0.5,
            max_iter=1,
            tol=1e-16,
            compute_inference=False,
        )

        with pytest.warns(ConvergenceWarning, match="did not converge within 1 iterations"):
            model.fit(self.X, self.y)

        assert model._fitted is True
        assert model.coef_ is not None
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

    def test_fit_inputs_follow_exact_cupy_design_device(self, monkeypatch):
        import statgpu.backends._utils as backend_utils

        events = []

        class Device:
            id = 5

        class FakeX:
            device = Device()

        y_source = object()
        weight_source = object()

        def fake_cupy_align(value, device_id, dtype=None):
            events.append((value, int(device_id), dtype))
            return ("aligned", value, int(device_id))

        monkeypatch.setattr(
            backend_utils,
            "_cupy_asarray_on_device",
            fake_cupy_align,
        )

        y_aligned, w_aligned, device = _align_quantile_fit_inputs(
            FakeX(),
            y_source,
            weight_source,
            "cupy",
        )

        assert y_aligned == ("aligned", y_source, 5)
        assert w_aligned == ("aligned", weight_source, 5)
        assert device == "cuda:5"
        assert events == [
            (y_source, 5, None),
            (weight_source, 5, None),
        ]

    def test_bootstrap_schedule_follows_exact_cupy_device(self):
        events = []

        class FakeDeviceContext:
            def __init__(self, device_id):
                self.device_id = int(device_id)

            def __enter__(self):
                events.append(("enter", self.device_id))
                return self

            def __exit__(self, exc_type, exc, tb):
                events.append(("exit", self.device_id))

        class FakeCuda:
            Device = FakeDeviceContext

        class FakeXP:
            cuda = FakeCuda()
            int64 = np.int64

            @staticmethod
            def asarray(value, dtype=None):
                events.append(("asarray", dtype))
                return np.asarray(value, dtype=dtype)

        class FakeResid:
            class Device:
                id = 3

            device = Device()

        schedule = np.array([[0, 1], [1, 0]], dtype=np.int64)
        converted = _bootstrap_schedule_to_backend(
            schedule,
            FakeResid(),
            "cupy",
            FakeXP(),
        )

        np.testing.assert_array_equal(converted, schedule)
        assert events[0] == ("enter", 3)
        assert events[-1] == ("exit", 3)
        assert any(event[0] == "asarray" for event in events)

    @pytest.mark.parametrize("tau", [0.2, 0.5, 0.8])
    def test_bootstrap_residual_centering_sets_empirical_tau_quantile_to_zero(
        self, tau
    ):
        residual = np.asarray(
            [-3.0, -1.5, -0.4, 0.2, 0.9, 1.7, 4.0],
            dtype=np.float64,
        )
        centered = _center_quantile_bootstrap_residuals(
            residual,
            tau,
            np,
        )
        index = min(
            max(int(np.ceil(tau * residual.size)) - 1, 0),
            residual.size - 1,
        )
        assert np.sort(centered)[index] == pytest.approx(0.0, abs=0.0)

    def test_bootstrap_residual_centering_preserves_torch_backend(self):
        torch = pytest.importorskip("torch")
        residual = torch.tensor(
            [-3.0, -1.5, -0.4, 0.2, 0.9, 1.7, 4.0],
            dtype=torch.float64,
        )
        centered = _center_quantile_bootstrap_residuals(
            residual,
            0.2,
            torch,
        )
        assert torch.is_tensor(centered)
        assert centered.device == residual.device
        assert centered.dtype == residual.dtype
        index = int(np.ceil(0.2 * residual.numel())) - 1
        assert float(torch.sort(centered).values[index]) == 0.0

    def test_batched_bootstrap_torch_keeps_response_construction_native(
        self, monkeypatch
    ):
        torch = pytest.importorskip("torch")
        import statgpu.backends as backends

        X = torch.zeros((12, 1), dtype=torch.float64)
        y = torch.zeros(12, dtype=torch.float64)
        model = QuantileRegression(
            quantile=0.5,
            fit_intercept=False,
            max_iter=5,
            tol=1e-8,
            n_bootstrap=3,
            random_state=11,
        )
        model.coef_ = np.zeros(1, dtype=np.float64)
        model.intercept_ = 0.0

        calls = []
        original_to_numpy = backends._to_numpy

        def recording_to_numpy(value):
            calls.append(value)
            return original_to_numpy(value)

        monkeypatch.setattr(backends, "_to_numpy", recording_to_numpy)
        boot_params, params_native, design_native = model._compute_bootstrap_batched(X, y)

        assert torch.is_tensor(params_native)
        assert torch.is_tensor(design_native)
        np.testing.assert_allclose(boot_params, 0.0, rtol=0.0, atol=0.0)
        # Only the completed parameter matrix crosses the reporting boundary.
        assert len(calls) == 1
        assert torch.is_tensor(calls[0])

    def test_nonmedian_batched_bootstrap_torch_cpu_runs_irls(self):
        torch = pytest.importorskip("torch")

        tau = 0.2
        n = 64
        B = 6
        X_np = np.ones((n, 1), dtype=np.float64)
        y_np = np.linspace(-4.0, 4.0, n, dtype=np.float64)
        X = torch.as_tensor(X_np, dtype=torch.float64)
        y = torch.as_tensor(y_np, dtype=torch.float64)

        model = QuantileRegression(
            quantile=tau,
            fit_intercept=False,
            max_iter=300,
            tol=1e-7,
            n_bootstrap=B,
            random_state=29,
        )
        model.coef_ = np.zeros(1, dtype=np.float64)
        model.intercept_ = 0.0

        boot_params, params_native, design_native = model._compute_bootstrap_batched(X, y)
        assert torch.is_tensor(params_native)
        assert torch.is_tensor(design_native)
        assert 1 <= model._bootstrap_n_iter_ <= model.max_iter

        center_index = min(max(int(np.ceil(tau * n)) - 1, 0), n - 1)
        residual_center = np.sort(y_np)[center_index]
        centered_residual = y_np - residual_center

        rng = np.random.default_rng(model.random_state)
        y_batch = np.array([
            centered_residual[rng.integers(0, n, size=n)]
            for _ in range(B)
        ])
        sorted_batch = np.sort(y_batch, axis=1)
        target_q = sorted_batch[:, center_index]
        wrong_index = min(
            max(int(np.ceil((1.0 - tau) * n)) - 1, 0),
            n - 1,
        )
        wrong_q = sorted_batch[:, wrong_index]
        estimated = np.asarray(boot_params[:, 0], dtype=np.float64)

        assert np.mean(np.abs(estimated - target_q)) < np.mean(
            np.abs(estimated - wrong_q)
        )
        assert abs(float(np.median(estimated))) < abs(float(np.median(wrong_q)))

    def test_bootstrap_schedule_hash_is_deterministic_and_seed_sensitive(self):
        X = np.zeros((16, 1), dtype=np.float64)
        y = np.zeros(16, dtype=np.float64)

        def run(seed):
            model = QuantileRegression(
                quantile=0.5,
                fit_intercept=False,
                max_iter=5,
                tol=1e-8,
                n_bootstrap=4,
                random_state=seed,
            )
            model.coef_ = np.zeros(1, dtype=np.float64)
            model.intercept_ = 0.0
            model._compute_bootstrap_batched(X, y)
            return model._bootstrap_schedule_sha256_

        hash_a = run(31)
        hash_b = run(31)
        hash_c = run(32)

        assert isinstance(hash_a, str) and len(hash_a) == 64
        int(hash_a, 16)
        assert hash_a == hash_b
        assert hash_a != hash_c

    def test_batched_bootstrap_zero_gradient_returns_current_point(self):
        X = np.zeros((12, 1), dtype=np.float64)
        y = np.zeros(12, dtype=np.float64)
        model = QuantileRegression(
            quantile=0.5,
            fit_intercept=False,
            max_iter=5,
            tol=1e-8,
            n_bootstrap=3,
            random_state=11,
        )
        model.coef_ = np.zeros(1, dtype=np.float64)
        model.intercept_ = 0.0

        boot_params, _, _ = model._compute_bootstrap_batched(X, y)

        np.testing.assert_allclose(boot_params, 0.0, rtol=0.0, atol=0.0)
        assert model._bootstrap_n_iter_ == 1

    def test_batched_quantile_irls_returns_backend_matrix_and_chunk_metadata(self):
        X = np.asarray(
            [
                [1.0, -1.0],
                [1.0, -0.2],
                [1.0, 0.4],
                [1.0, 1.1],
                [1.0, 1.8],
            ],
            dtype=np.float64,
        )
        y_matrix = np.column_stack(
            [
                np.asarray([-1.0, -0.3, 0.2, 0.9, 1.6]),
                np.asarray([-0.8, -0.1, 0.5, 1.0, 1.9]),
            ]
        )
        init = np.zeros(X.shape[1], dtype=np.float64)

        params, n_iter, chunk_size = _batched_quantile_irls(
            X,
            y_matrix,
            init,
            0.3,
            max_iter=500,
            tol=1e-8,
            xp=np,
        )

        assert params.shape == (X.shape[1], y_matrix.shape[1])
        assert np.all(np.isfinite(params))
        assert 1 <= n_iter <= 500
        assert 1 <= chunk_size <= y_matrix.shape[1]

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

        center_index = min(max(int(np.ceil(tau * n)) - 1, 0), n - 1)
        residual_center = np.sort(y)[center_index]
        centered_residual = y - residual_center

        rng = np.random.default_rng(model.random_state)
        y_batch = np.array([
            centered_residual[rng.integers(0, n, size=n)]
            for _ in range(B)
        ])
        sorted_batch = np.sort(y_batch, axis=1)
        target_q = sorted_batch[:, center_index]
        wrong_index = min(
            max(int(np.ceil((1.0 - tau) * n)) - 1, 0),
            n - 1,
        )
        wrong_q = sorted_batch[:, wrong_index]
        estimated = np.asarray(boot_params[:, 0], dtype=np.float64)

        target_error = np.mean(np.abs(estimated - target_q))
        wrong_error = np.mean(np.abs(estimated - wrong_q))
        assert target_error < wrong_error
        assert abs(float(np.median(estimated))) < abs(float(np.median(wrong_q)))
        assert model._bootstrap_residual_centering_ == "empirical_tau_quantile"

    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_nonmedian_multifeature_bootstrap_matches_lp_objective(
        self,
        fit_intercept,
    ):
        sklearn_linear = pytest.importorskip("sklearn.linear_model")
        tau = 0.3
        n = 48
        B = 4
        seed = 16695
        rng = np.random.default_rng(16694)
        X = rng.normal(size=(n, 2)).astype(np.float64)
        coef0 = np.asarray([0.7, -0.35], dtype=np.float64)
        intercept0 = 0.25 if fit_intercept else 0.0
        residual = (
            np.linspace(-1.2, 1.1, n, dtype=np.float64)
            + 0.03 * rng.normal(size=n)
        )
        y = intercept0 + X @ coef0 + residual

        model = QuantileRegression(
            quantile=tau,
            fit_intercept=fit_intercept,
            max_iter=1200,
            tol=1e-8,
            n_bootstrap=B,
            random_state=seed,
        )
        model.coef_ = coef0.copy()
        model.intercept_ = float(intercept0)

        boot_params, _, _ = model._compute_bootstrap_batched(X, y)

        center_index = min(max(int(np.ceil(tau * n)) - 1, 0), n - 1)
        residual_center = np.sort(residual)[center_index]
        centered_residual = residual - residual_center
        eta = intercept0 + X @ coef0
        schedule_rng = np.random.default_rng(seed)
        schedule = np.stack(
            [schedule_rng.integers(0, n, size=n, dtype=np.int64) for _ in range(B)]
        )

        for draw in range(B):
            y_draw = eta + centered_residual[schedule[draw]]
            reference = sklearn_linear.QuantileRegressor(
                quantile=tau,
                alpha=0.0,
                fit_intercept=fit_intercept,
                solver="highs",
            ).fit(X, y_draw)

            if fit_intercept:
                actual_intercept = float(boot_params[draw, 0])
                actual_coef = np.asarray(boot_params[draw, 1:], dtype=np.float64)
            else:
                actual_intercept = 0.0
                actual_coef = np.asarray(boot_params[draw], dtype=np.float64)

            actual_residual = y_draw - (X @ actual_coef + actual_intercept)
            reference_residual = y_draw - reference.predict(X)
            actual_loss = float(
                np.mean(
                    np.where(
                        actual_residual >= 0.0,
                        tau * actual_residual,
                        (tau - 1.0) * actual_residual,
                    )
                )
            )
            reference_loss = float(
                np.mean(
                    np.where(
                        reference_residual >= 0.0,
                        tau * reference_residual,
                        (tau - 1.0) * reference_residual,
                    )
                )
            )
            assert actual_loss <= reference_loss + 2e-6

    def test_batched_bootstrap_budget_exhaustion_fails_closed(self):
        n = 48
        X = np.ones((n, 1), dtype=np.float64)
        y = np.linspace(-3.0, 3.0, n, dtype=np.float64)
        model = QuantileRegression(
            quantile=0.2,
            fit_intercept=False,
            max_iter=1,
            tol=1e-12,
            n_bootstrap=4,
            random_state=23,
        )
        model.coef_ = np.zeros(1, dtype=np.float64)
        model.intercept_ = 0.0

        with pytest.raises(RuntimeError, match="did not converge within 1 iterations"):
            model._compute_bootstrap_batched(X, y)
        assert model._bootstrap_n_iter_ == 1
        assert model._bootstrap_draw_chunk_size_ == 4

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
            raise AssertionError(
                "unsupported weighted inference must fail before numerical work"
            )

        monkeypatch.setattr(quantile_mod, "fista_solver", forbidden)
        monkeypatch.setattr(quantile_mod, "_batched_quantile_irls", forbidden)
        with pytest.raises(NotImplementedError, match="non-uniform sample_weight"):
            model.fit(self.X, self.y, sample_weight=weights)
        assert model._fitted is False
        assert model.coef_ is None
        assert model._inference_result is None

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
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

    def test_direct_public_control_replacement_drives_next_fit(
        self, monkeypatch
    ):
        import statgpu.linear_model.wrappers._quantile as quantile_mod

        model = QuantileRegression(
            quantile=0.5,
            fit_intercept=True,
            max_iter=100,
            tol=1e-4,
            device="cpu",
        )
        model.quantile = 0.2
        model.fit_intercept = False
        model.max_iter = 7
        model.tol = 2e-7

        captured = {}

        def fake_fista(loss, penalty, X, y, **kwargs):
            captured["quantile"] = float(loss.quantile)
            captured["shape"] = tuple(X.shape)
            captured["max_iter"] = kwargs["max_iter"]
            captured["tol"] = kwargs["tol"]
            return np.zeros(X.shape[1], dtype=np.float64), 1

        monkeypatch.setattr(quantile_mod, "fista_solver", fake_fista)
        model.fit(self.X, self.y)

        assert captured == {
            "quantile": pytest.approx(0.2),
            "shape": self.X.shape,
            "max_iter": 7,
            "tol": pytest.approx(2e-7),
        }
        assert model._quantile == pytest.approx(0.2)
        assert model._fit_intercept is False
        assert model._max_iter == 7
        assert model._tol == pytest.approx(2e-7)

    @pytest.mark.parametrize("bad_seed", [True, -1, "42"])
    def test_invalid_bootstrap_random_state_fails_before_backend(
        self, monkeypatch, bad_seed
    ):
        model = QuantileRegression(
            quantile=0.5,
            compute_inference=True,
            inference_method="bootstrap",
            n_bootstrap=4,
            random_state=42,
            device="cpu",
        )
        model.random_state = bad_seed

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "invalid bootstrap random_state must fail before backend work"
            )

        monkeypatch.setattr(model, "_get_backend", forbidden)
        with pytest.raises(ValueError, match="random_state must be None or"):
            model.fit(self.X, self.y)

        assert model._fitted is False
        assert model.coef_ is None

    def test_direct_invalid_device_replacement_fails_before_backend(
        self, monkeypatch
    ):
        model = QuantileRegression(quantile=0.5, device="cpu")
        model.device = "not-a-device"

        def forbidden(*args, **kwargs):
            raise AssertionError("invalid device must fail before backend work")

        monkeypatch.setattr(model, "_get_backend", forbidden)
        with pytest.raises(ValueError, match="device must be one of"):
            model.fit(self.X, self.y)

        assert model._fitted is False
        assert model.coef_ is None
        assert model._selected_backend_name is None

    @pytest.mark.parametrize("bad_quantile", ["0.5", True])
    def test_invalid_quantile_constructor_fails_closed(self, bad_quantile):
        with pytest.raises(ValueError, match="quantile must be a finite real number"):
            QuantileRegression(quantile=bad_quantile)

    def test_invalid_quantile_set_params_is_atomic(self):
        model = QuantileRegression(quantile=0.5)
        with pytest.raises(ValueError, match="quantile must be a finite real number"):
            model.set_params(quantile="0.5")
        assert model.quantile == pytest.approx(0.5)
        assert model._quantile == pytest.approx(0.5)
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

    def test_early_gpu_validation_failure_cleans_input_device(self, monkeypatch):
        import sys
        import types

        class FakeFinite:
            def all(self):
                return self

            def item(self):
                return True

        fake_cupy = types.SimpleNamespace(
            isfinite=lambda value: FakeFinite(),
        )
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

        class FakeCuPyArray:
            pass

        FakeCuPyArray.__module__ = "cupy._core.core"
        X = FakeCuPyArray()
        X.device = types.SimpleNamespace(id=3)

        model = QuantileRegression(gpu_memory_cleanup=True)
        cleanup_calls = []

        def failing_fit_impl(*args, **kwargs):
            raise RuntimeError("synthetic early validation failure")

        monkeypatch.setattr(model, "_fit_impl", failing_fit_impl)
        monkeypatch.setattr(
            model,
            "_cleanup_backend_memory",
            lambda backend, device=None: cleanup_calls.append((backend, device)),
        )

        with pytest.raises(RuntimeError, match="synthetic early validation failure"):
            model.fit(X, np.zeros(2, dtype=np.float64))

        assert cleanup_calls == [("cupy", "cuda:3")]
        assert model._selected_backend_name is None
        assert model._selected_backend_device is None

    def test_cupy_cleanup_uses_recorded_concrete_device(self, monkeypatch):
        import sys
        import types

        state = {"current": 0, "events": []}

        class FakeDevice:
            def __init__(self, device_id):
                self.device_id = int(device_id)
                self.previous = None

            def __enter__(self):
                self.previous = state["current"]
                state["current"] = self.device_id
                state["events"].append(("enter", self.device_id))
                return self

            def __exit__(self, exc_type, exc, tb):
                state["events"].append(("exit", self.device_id))
                state["current"] = self.previous

        class FakePool:
            def __init__(self, name):
                self.name = name

            def free_all_blocks(self):
                state["events"].append(
                    (self.name, state["current"])
                )

        fake_cupy = types.SimpleNamespace(
            cuda=types.SimpleNamespace(Device=FakeDevice),
            get_default_memory_pool=lambda: FakePool("device_pool"),
            get_default_pinned_memory_pool=lambda: FakePool("pinned_pool"),
        )
        monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

        model = QuantileRegression(gpu_memory_cleanup=True)
        model._selected_backend_name = "cupy"
        model._selected_backend_device = "cuda:4"

        model._cleanup_backend_memory("cupy", "cuda:4")

        assert ("enter", 4) in state["events"]
        assert ("device_pool", 4) in state["events"]
        assert ("pinned_pool", 4) in state["events"]
        assert ("exit", 4) in state["events"]
        assert state["current"] == 0

    def test_cleanup_backend_routes_cupy_name_to_cuda_cleanup(self, monkeypatch):
        model = QuantileRegression(gpu_memory_cleanup=True)
        calls = []
        monkeypatch.setattr(
            model,
            "_cleanup_cuda_memory",
            lambda device_label=None: calls.append("cupy"),
        )
        monkeypatch.setattr(
            model,
            "_cleanup_torch_memory",
            lambda device_label=None: calls.append("torch"),
        )

        model._cleanup_backend_memory("cupy")
        model._cleanup_backend_memory("torch")

        assert calls == ["cupy", "torch"]

    def test_failed_cupy_fit_runs_cleanup_before_state_reset(self, monkeypatch):
        import statgpu.backends._utils as backend_utils
        import statgpu.linear_model.wrappers._quantile as quantile_mod

        monkeypatch.setattr(
            backend_utils,
            "_cupy_asarray_on_device",
            lambda value, device_id, dtype=None: value,
        )

        class FakeBackend:
            name = "cupy"

        model = QuantileRegression(
            fit_intercept=False,
            gpu_memory_cleanup=True,
            compute_inference=False,
        )
        cleanup_calls = []

        class FakeCuPyArray(np.ndarray):
            @property
            def device(self):
                import types

                return types.SimpleNamespace(id=0)

        def fake_to_array(value, backend=None):
            return np.asarray(value).view(FakeCuPyArray)

        monkeypatch.setattr(model, "_get_backend", lambda backend="auto": FakeBackend())
        monkeypatch.setattr(model, "_to_array", fake_to_array)
        monkeypatch.setattr(
            model,
            "_cleanup_cuda_memory",
            lambda device_label=None: cleanup_calls.append("cupy"),
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

    def test_refit_without_inference_clears_prior_statistic_aliases(self):
        model = QuantileRegression(
            quantile=0.5,
            compute_inference=True,
            inference_method="kernel",
        ).fit(self.X, self.y)
        assert model._zvalues is not None
        assert model._tvalues is not None

        model.set_params(compute_inference=False)
        model.fit(self.X, self.y)

        assert model._inference_result is None
        assert model._bse is None
        assert model._zvalues is None
        assert model._tvalues is None
        assert model._statistic is None
        assert model._pvalues is None
        assert model._conf_int is None

    def test_failed_refit_clears_prior_statistic_aliases(self):
        model = QuantileRegression(
            quantile=0.5,
            compute_inference=True,
            inference_method="kernel",
        ).fit(self.X, self.y)
        assert model._tvalues is not None

        model.set_params(inference_method="bootstrap", n_bootstrap=1)
        with pytest.raises(ValueError, match="n_bootstrap must be an integer"):
            model.fit(self.X, self.y)

        assert model._fitted is False
        assert model._inference_result is None
        assert model._zvalues is None
        assert model._tvalues is None
        assert model._statistic is None

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

    def test_bootstrap_inference_records_concrete_backend_device(self):
        model = QuantileRegression(
            quantile=0.5,
            compute_inference=True,
            inference_method="bootstrap",
            n_bootstrap=8,
            random_state=41,
            max_iter=400,
            tol=1e-6,
        ).fit(self.X, self.y)

        metadata = model._inference_result.metadata
        assert metadata["numerical_backend"] == "numpy"
        assert metadata["numerical_device"] == "cpu"
        assert metadata["reporting_backend"] == "numpy"
        assert metadata["bootstrap_type"] == "iid_residual"
        assert metadata["residual_centering"] == "empirical_tau_quantile"
        assert metadata["heteroscedastic_robust"] is False

    def test_bootstrap_inference_metadata_records_schedule_identity(self):
        model = QuantileRegression(
            quantile=0.5,
            compute_inference=True,
            inference_method="bootstrap",
            n_bootstrap=8,
            random_state=37,
            max_iter=400,
            tol=1e-6,
        ).fit(self.X, self.y)

        metadata = model._inference_result.metadata
        assert metadata["random_state"] == 37
        assert metadata["resampling_schedule"] == "numpy_generator_control_plane"
        assert metadata["resampling_schedule_sha256"] == model._bootstrap_schedule_sha256_
        assert len(metadata["resampling_schedule_sha256"]) == 64

    def test_fit_with_bootstrap_inference(self):
        m = QuantileRegression(quantile=0.5, compute_inference=True,
                                inference_method='bootstrap', n_bootstrap=50)
        m.fit(self.X, self.y)
        assert m._bse is not None
        assert len(m._bse) == 4
        assert np.all(m._bse > 0)
        assert 1 <= m._bootstrap_n_iter_ <= m.max_iter
        assert (
            m._inference_result.metadata["solver_n_iter"]
            == m._bootstrap_n_iter_
        )
        assert (
            m._inference_result.statistic_name
            == "estimate_over_bootstrap_se"
        )
        assert (
            m._inference_result.metadata["pvalue_method"]
            == "bootstrap_sign_test"
        )
        assert (
            m._inference_result.metadata["statistic_method"]
            == "estimate_over_bootstrap_se"
        )
        np.testing.assert_allclose(
            m._zvalues,
            m._inference_result.statistic,
            rtol=0.0,
            atol=0.0,
        )
        table = m._inference_result.to_dataframe()
        assert "estimate_over_bootstrap_se" in table.columns
        assert "z" not in table.columns

    def test_score_is_negative_pinball_loss(self):
        model = QuantileRegression(quantile=0.25).fit(self.X, self.y)
        pred = model.predict(self.X)
        residual = self.y - pred
        expected = -float(np.mean(np.where(
            residual >= 0.0,
            0.25 * residual,
            (0.25 - 1.0) * residual,
        )))
        assert model.score(self.X, self.y) == pytest.approx(
            expected, rel=0.0, abs=1e-12
        )

    def test_score_supports_analytic_weights(self):
        model = QuantileRegression(quantile=0.35).fit(self.X, self.y)
        weights = np.linspace(0.5, 1.5, self.X.shape[0])
        pred = model.predict(self.X)
        residual = self.y - pred
        per_sample = np.where(
            residual >= 0.0,
            0.35 * residual,
            (0.35 - 1.0) * residual,
        )
        expected = -float(np.average(per_sample, weights=weights))
        assert model.score(self.X, self.y, sample_weight=weights) == pytest.approx(
            expected, rel=0.0, abs=1e-12
        )

    def test_score_rejects_complex_response_before_prediction(self, monkeypatch):
        model = QuantileRegression(quantile=0.5)
        model._fitted = True
        model.coef_ = np.zeros(self.X.shape[1], dtype=np.float64)

        def forbidden_predict(*args, **kwargs):
            raise AssertionError("non-real score response must fail before prediction")

        monkeypatch.setattr(model, "predict", forbidden_predict)
        with pytest.raises(ValueError, match="real numeric values"):
            model.score(
                self.X,
                self.y.astype(np.complex128) + 1j,
            )


    def test_score_rejects_response_shape_and_length_mismatch(self):
        model = QuantileRegression(quantile=0.5).fit(self.X, self.y)
        with pytest.raises(ValueError, match="one-dimensional"):
            model.score(self.X, self.y[:, None])
        with pytest.raises(ValueError, match="same number of observations as X"):
            model.score(self.X, self.y[:1])

    def test_sklearn_clone_preserves_public_controls(self):
        sklearn_base = pytest.importorskip("sklearn.base")
        model = QuantileRegression(
            quantile=0.3,
            fit_intercept=False,
            max_iter=321,
            tol=2e-5,
            compute_inference=True,
            inference_method="bootstrap",
            n_bootstrap=7,
            random_state=19,
        )
        cloned = sklearn_base.clone(model)
        assert cloned.get_params(deep=False) == model.get_params(deep=False)
        assert cloned.quantile == pytest.approx(0.3)
        assert cloned._quantile == pytest.approx(0.3)
        assert cloned._max_iter == 321
        assert cloned._n_bootstrap == 7

    def test_summary_without_inference_is_method_neutral(self):
        model = QuantileRegression(
            quantile=0.5,
            compute_inference=False,
            inference_method="kernel",
        ).fit(self.X, self.y)
        text = model.summary()
        assert "inference not computed" in text
        assert "bootstrap inference not computed" not in text

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

    def test_singular_design_fails_instead_of_publishing_nonfinite_inference(self):
        X_bad = np.column_stack([self.X[:, 0], self.X[:, 0]])
        model = QuantileRegression(
            compute_inference=True,
            inference_method="kernel",
        )

        with pytest.raises((np.linalg.LinAlgError, RuntimeError)):
            model.fit(X_bad, self.y)

        assert model._fitted is False
        assert model._inference_result is None
        assert model._bse is None
        assert model._pvalues is None
        assert model._conf_int is None

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
