"""Tests for unified inference resampling engine."""

import numpy as np
import pytest

from statgpu._config import set_device
from statgpu._config import Device
from statgpu.inference import bootstrap_statistic, permutation_test
from statgpu.linear_model import LinearRegression


class TestBootstrapEngine:
    """Function-level bootstrap tests."""

    def test_bootstrap_iid_mean_statistic(self):
        rng = np.random.default_rng(123)
        x = rng.normal(loc=1.5, scale=0.3, size=300)

        result = bootstrap_statistic(
            lambda arr: np.mean(arr),
            x,
            n_resamples=200,
            strategy="iid",
            random_state=123,
            statistic_name="mean",
        )

        assert result.statistic_name == "mean"
        assert result.samples.shape == (200,)
        assert abs(result.observed - np.mean(x)) < 1e-12
        lo, hi = result.confidence_interval
        assert lo <= result.observed <= hi

    def test_bootstrap_stratified_runs(self):
        rng = np.random.default_rng(42)
        y = rng.normal(size=240)
        strata = np.repeat(np.array([0, 1, 2]), 80)

        result = bootstrap_statistic(
            lambda arr: np.mean(arr),
            y,
            n_resamples=100,
            strategy="stratified",
            strata=strata,
            random_state=42,
        )

        assert result.samples.shape == (100,)
        assert np.isfinite(result.samples).all()

    def test_bootstrap_cluster_runs(self):
        rng = np.random.default_rng(5)
        y = rng.normal(size=180)
        clusters = np.repeat(np.arange(30), 6)

        result = bootstrap_statistic(
            lambda arr: np.std(arr),
            y,
            n_resamples=80,
            strategy="cluster",
            clusters=clusters,
            random_state=5,
            statistic_name="std",
        )

        assert result.samples.shape == (80,)
        assert result.statistic_name == "std"

    def test_bootstrap_block_runs(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=101)

        result = bootstrap_statistic(
            lambda arr: np.mean(arr),
            x,
            n_resamples=90,
            strategy="block",
            block_size=7,
            random_state=17,
            statistic_name="mean",
        )

        assert result.strategy == "block"
        assert result.samples.shape == (90,)
        assert result.metadata["n_samples"] == 101

    def test_bootstrap_multi_array_and_serialization(self):
        rng = np.random.default_rng(1234)
        X = rng.normal(size=(220, 2))
        y = 1.4 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.6, size=220)

        def corr_stat(X_in, y_in):
            return np.corrcoef(X_in[:, 0], y_in)[0, 1]

        result = bootstrap_statistic(
            corr_stat,
            X,
            y,
            n_resamples=70,
            strategy="iid",
            random_state=1234,
            statistic_name="corr",
        )

        payload = result.to_dict()
        assert payload["statistic_name"] == "corr"
        assert payload["n_resamples"] == 70
        assert len(payload["samples"]) == 70

        try:
            import pandas as pd
        except ImportError:
            with pytest.raises(ImportError):
                result.to_dataframe()
        else:
            df = result.to_dataframe()
            assert list(df.columns) == ["sample_index", "statistic"]
            assert df.shape[0] == 70

    def test_bootstrap_force_vectorized_numpy_and_hint(self):
        rng = np.random.default_rng(314)
        x = rng.normal(size=400)

        result_vec = bootstrap_statistic(
            lambda arr: np.mean(arr, axis=-1),
            x,
            n_resamples=120,
            strategy="iid",
            random_state=314,
            backend="numpy",
            force_vectorized=True,
        )
        assert result_vec.samples.shape == (120,)

        result_hint = bootstrap_statistic(
            lambda arr: np.mean(arr),
            x,
            n_resamples=120,
            strategy="iid",
            random_state=314,
            backend="numpy",
            force_vectorized=True,
            statistic_hint="mean",
        )
        assert result_hint.samples.shape == (120,)

        with pytest.raises(ValueError):
            bootstrap_statistic(
                lambda arr: np.mean(arr),
                x,
                n_resamples=40,
                strategy="iid",
                random_state=314,
                backend="numpy",
                force_vectorized=True,
            )

    def test_bootstrap_stratified_hint_mean_batched_numpy(self):
        rng = np.random.default_rng(1201)
        x = rng.normal(size=360)
        strata = np.repeat(np.arange(6), 60)
        calls = {"n": 0}

        def mean_scalar(arr):
            calls["n"] += 1
            arr_np = np.asarray(arr)
            if arr_np.ndim != 1:
                raise RuntimeError("unexpected batched callback")
            return np.mean(arr_np)

        result = bootstrap_statistic(
            mean_scalar,
            x,
            n_resamples=90,
            strategy="stratified",
            strata=strata,
            random_state=1201,
            backend="numpy",
            statistic_hint="mean",
        )

        assert result.samples.shape == (90,)
        assert calls["n"] == 1

    def test_bootstrap_block_hint_mean_batched_numpy(self):
        rng = np.random.default_rng(1202)
        x = rng.normal(size=420)
        calls = {"n": 0}

        def mean_scalar(arr):
            calls["n"] += 1
            arr_np = np.asarray(arr)
            if arr_np.ndim != 1:
                raise RuntimeError("unexpected batched callback")
            return np.mean(arr_np)

        result = bootstrap_statistic(
            mean_scalar,
            x,
            n_resamples=80,
            strategy="block",
            block_size=7,
            random_state=1202,
            backend="numpy",
            statistic_hint="mean",
        )

        assert result.samples.shape == (80,)
        assert calls["n"] == 1

    def test_bootstrap_cluster_hint_mean_batched_numpy(self):
        rng = np.random.default_rng(1203)
        x = rng.normal(size=360)
        clusters = np.repeat(np.arange(45), 8)
        calls = {"n": 0}

        def mean_scalar(arr):
            calls["n"] += 1
            arr_np = np.asarray(arr)
            if arr_np.ndim != 1:
                raise RuntimeError("unexpected batched callback")
            return np.mean(arr_np)

        result = bootstrap_statistic(
            mean_scalar,
            x,
            n_resamples=90,
            strategy="cluster",
            clusters=clusters,
            random_state=1203,
            backend="numpy",
            statistic_hint="mean",
        )

        assert result.samples.shape == (90,)
        assert calls["n"] == 1

    def test_bootstrap_invalid_configs_raise(self):
        x = np.arange(12.0)

        with pytest.raises(ValueError):
            bootstrap_statistic(lambda arr: np.mean(arr), n_resamples=10)

        with pytest.raises(ValueError):
            bootstrap_statistic(lambda arr: np.mean(arr), x, n_resamples=0)

        with pytest.raises(ValueError):
            bootstrap_statistic(lambda arr: np.mean(arr), x, confidence_level=1.0)

        with pytest.raises(ValueError):
            bootstrap_statistic(lambda arr: np.mean(arr), x, strategy="stratified")

        with pytest.raises(ValueError):
            bootstrap_statistic(lambda arr: np.mean(arr), x, strategy="cluster")

        with pytest.raises(ValueError):
            bootstrap_statistic(lambda arr: np.mean(arr), x, strategy="block", block_size=0)

        with pytest.raises(ValueError):
            bootstrap_statistic(lambda a, b: np.mean(a), x, np.arange(10.0), strategy="iid")

        with pytest.raises(ValueError):
            bootstrap_statistic(
                lambda arr: np.mean(arr),
                x,
                strategy="stratified",
                strata=np.arange(10),
            )

        with pytest.raises(ValueError):
            bootstrap_statistic(
                lambda arr: np.mean(arr),
                x,
                strategy="cluster",
                clusters=np.arange(10),
            )


class TestPermutationEngine:
    """Function-level permutation tests."""

    def test_permutation_detects_signal(self):
        rng = np.random.default_rng(21)
        X = rng.normal(size=(280, 1))
        y = 2.5 * X[:, 0] + rng.normal(scale=0.5, size=280)

        def corr_stat(X_in, y_in):
            return np.corrcoef(X_in[:, 0], y_in)[0, 1]

        result = permutation_test(
            corr_stat,
            X,
            y,
            n_resamples=300,
            strategy="iid",
            alternative="two-sided",
            random_state=21,
            statistic_name="corr",
        )

        assert result.statistic_name == "corr"
        assert result.samples.shape == (300,)
        assert result.pvalue < 0.01

    def test_permutation_grouped_runs(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(160, 2))
        y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.8, size=160)
        groups = np.repeat(np.arange(20), 8)

        def mse_zero_model(X_in, y_in):
            return np.mean((y_in - 0.0) ** 2)

        result = permutation_test(
            mse_zero_model,
            X,
            y,
            n_resamples=120,
            strategy="grouped",
            groups=groups,
            alternative="greater",
            random_state=9,
            statistic_name="mse",
        )

        assert result.samples.shape == (120,)
        assert 0.0 <= result.pvalue <= 1.0

    def test_permutation_stratified_runs(self):
        rng = np.random.default_rng(31)
        X = rng.normal(size=(240, 2))
        strata = np.repeat(np.array([0, 1, 2, 3]), 60)
        y = 1.6 * X[:, 0] + rng.normal(scale=0.9, size=240)

        result = permutation_test(
            lambda X_in, y_in: np.corrcoef(X_in[:, 0], y_in)[0, 1],
            X,
            y,
            n_resamples=100,
            strategy="stratified",
            strata=strata,
            alternative="two-sided",
            random_state=31,
            statistic_name="corr",
        )

        assert result.strategy == "stratified"
        assert result.samples.shape == (100,)
        assert 0.0 <= result.pvalue <= 1.0

    def test_permutation_less_alternative_detects_negative_signal(self):
        rng = np.random.default_rng(88)
        X = rng.normal(size=(220, 1))
        y = -2.0 * X[:, 0] + rng.normal(scale=0.6, size=220)

        result = permutation_test(
            lambda X_in, y_in: np.corrcoef(X_in[:, 0], y_in)[0, 1],
            X,
            y,
            n_resamples=250,
            strategy="iid",
            alternative="less",
            random_state=88,
            statistic_name="corr",
        )

        assert result.alternative == "less"
        assert result.pvalue < 0.05

    def test_permutation_serialization(self):
        rng = np.random.default_rng(202)
        X = rng.normal(size=(140, 2))
        y = 0.9 * X[:, 0] + rng.normal(scale=0.7, size=140)

        result = permutation_test(
            lambda X_in, y_in: np.corrcoef(X_in[:, 0], y_in)[0, 1],
            X,
            y,
            n_resamples=80,
            strategy="iid",
            alternative="greater",
            random_state=202,
            statistic_name="corr",
        )

        payload = result.to_dict()
        assert payload["statistic_name"] == "corr"
        assert payload["n_resamples"] == 80
        assert len(payload["samples"]) == 80

        try:
            import pandas as pd
        except ImportError:
            with pytest.raises(ImportError):
                result.to_dataframe()
        else:
            df = result.to_dataframe()
            assert list(df.columns) == ["sample_index", "statistic"]
            assert df.shape[0] == 80

    def test_permutation_force_vectorized_numpy_and_hint(self):
        rng = np.random.default_rng(808)
        X = rng.normal(size=(260, 1))
        y = 1.6 * X[:, 0] + rng.normal(scale=0.5, size=260)

        def corr_vec(X_in, y_in):
            x = X_in[:, 0]
            y_arr = np.asarray(y_in)
            if y_arr.ndim == 1:
                return np.corrcoef(x, y_arr)[0, 1]
            x_centered = x - np.mean(x)
            y_centered = y_arr - np.mean(y_arr, axis=1, keepdims=True)
            x_norm_sq = np.sum(x_centered * x_centered)
            y_norm_sq = np.sum(y_centered * y_centered, axis=1)
            denom = np.sqrt(x_norm_sq * y_norm_sq)
            denom = np.where(denom > 0.0, denom, np.inf)
            numer = np.sum(y_centered * x_centered.reshape(1, -1), axis=1)
            return numer / denom

        result_vec = permutation_test(
            corr_vec,
            X,
            y,
            n_resamples=120,
            strategy="iid",
            alternative="two-sided",
            random_state=808,
            backend="numpy",
            force_vectorized=True,
        )
        assert result_vec.samples.shape == (120,)

        result_hint = permutation_test(
            corr_vec,
            X,
            y,
            n_resamples=120,
            strategy="iid",
            alternative="two-sided",
            random_state=808,
            backend="numpy",
            force_vectorized=True,
            statistic_hint="pearson_corr",
        )
        assert result_hint.samples.shape == (120,)

        with pytest.raises(ValueError):
            permutation_test(
                lambda X_in, y_in: np.corrcoef(X_in[:, 0], y_in)[0, 1],
                X,
                y,
                n_resamples=40,
                strategy="iid",
                alternative="two-sided",
                random_state=808,
                backend="numpy",
                force_vectorized=True,
            )

    def test_permutation_stratified_hint_pearson_batched_numpy(self):
        rng = np.random.default_rng(2201)
        X = rng.normal(size=(300, 1))
        y = 1.4 * X[:, 0] + rng.normal(scale=0.6, size=300)
        strata = np.repeat(np.arange(6), 50)
        calls = {"n": 0}

        def corr_scalar(X_in, y_in):
            calls["n"] += 1
            y_arr = np.asarray(y_in)
            if y_arr.ndim != 1:
                raise RuntimeError("unexpected batched callback")
            return np.corrcoef(X_in[:, 0], y_arr)[0, 1]

        result = permutation_test(
            corr_scalar,
            X,
            y,
            n_resamples=120,
            strategy="stratified",
            strata=strata,
            alternative="two-sided",
            random_state=2201,
            backend="numpy",
            statistic_hint="pearson_corr",
        )

        assert result.samples.shape == (120,)
        assert 0.0 <= result.pvalue <= 1.0
        assert calls["n"] == 1

    def test_permutation_grouped_hint_pearson_batched_numpy(self):
        rng = np.random.default_rng(2202)
        X = rng.normal(size=(300, 1))
        y = 1.3 * X[:, 0] + rng.normal(scale=0.7, size=300)
        groups = np.repeat(np.arange(60), 5)
        calls = {"n": 0}

        def corr_scalar(X_in, y_in):
            calls["n"] += 1
            y_arr = np.asarray(y_in)
            if y_arr.ndim != 1:
                raise RuntimeError("unexpected batched callback")
            return np.corrcoef(X_in[:, 0], y_arr)[0, 1]

        result = permutation_test(
            corr_scalar,
            X,
            y,
            n_resamples=100,
            strategy="grouped",
            groups=groups,
            alternative="two-sided",
            random_state=2202,
            backend="numpy",
            statistic_hint="pearson_corr",
        )

        assert result.samples.shape == (100,)
        assert 0.0 <= result.pvalue <= 1.0
        assert calls["n"] == 1

    def test_permutation_invalid_inputs_raise(self):
        X = np.ones((20, 2))
        y = np.arange(20, dtype=float)

        with pytest.raises(ValueError):
            permutation_test(lambda X_in, y_in: 0.0, X, y[:-1])

        with pytest.raises(ValueError):
            permutation_test(lambda X_in, y_in: 0.0, X, y, n_resamples=0)

        with pytest.raises(ValueError):
            permutation_test(lambda X_in, y_in: 0.0, X, y, alternative="invalid")

        with pytest.raises(ValueError):
            permutation_test(lambda X_in, y_in: 0.0, X, y, strategy="invalid")

        with pytest.raises(ValueError):
            permutation_test(lambda X_in, y_in: 0.0, X, y, strategy="stratified")

        with pytest.raises(ValueError):
            permutation_test(lambda X_in, y_in: 0.0, X, y, strategy="grouped")

        with pytest.raises(ValueError):
            permutation_test(
                lambda X_in, y_in: 0.0,
                X,
                y,
                strategy="stratified",
                strata=np.arange(19),
            )

        with pytest.raises(ValueError):
            permutation_test(
                lambda X_in, y_in: 0.0,
                X,
                y,
                strategy="grouped",
                groups=np.arange(19),
            )


class TestModelWrappers:
    """BaseEstimator wrapper integration tests."""

    def test_model_bootstrap_wrapper(self):
        set_device("cpu")
        rng = np.random.default_rng(77)

        X = rng.normal(size=(260, 3))
        y = X @ np.array([1.1, -0.9, 0.6]) + rng.normal(scale=0.3, size=260)

        model = LinearRegression(device="cpu", compute_inference=True)
        model.fit(X, y)

        result = model.bootstrap_statistic(
            lambda X_s, y_s: np.mean(y_s - np.mean(y_s)),
            X,
            y,
            n_resamples=60,
            strategy="iid",
            random_state=77,
            statistic_name="centered_mean",
        )

        assert result.samples.shape == (60,)
        assert result.statistic_name == "centered_mean"

    def test_model_permutation_wrapper(self):
        set_device("cpu")
        rng = np.random.default_rng(91)

        X = rng.normal(size=(220, 2))
        y = 1.8 * X[:, 0] + rng.normal(scale=0.7, size=220)

        model = LinearRegression(device="cpu", compute_inference=False)
        model.fit(X, y)

        result = model.permutation_test(
            lambda X_in, y_in: np.corrcoef(X_in[:, 0], y_in)[0, 1],
            X,
            y,
            n_resamples=120,
            strategy="iid",
            alternative="two-sided",
            random_state=91,
            statistic_name="corr",
        )

        assert result.samples.shape == (120,)
        assert result.pvalue < 0.05

    def test_bootstrap_wrapper_without_arrays_and_no_cache_raises(self):
        set_device("cpu")
        model = LinearRegression(device="cpu", compute_inference=False)

        with pytest.raises(RuntimeError):
            model.bootstrap_statistic(lambda a, b: 0.0)

    def test_bootstrap_wrapper_uses_cached_training_arrays(self):
        set_device("cpu")
        rng = np.random.default_rng(404)

        X = rng.normal(size=(210, 3))
        y = X @ np.array([0.7, -1.2, 1.0]) + rng.normal(scale=0.4, size=210)

        model = LinearRegression(device="cpu", compute_inference=True)
        model.fit(X, y)

        result = model.bootstrap_statistic(
            lambda X_s, y_s: np.mean(y_s),
            n_resamples=40,
            strategy="iid",
            random_state=404,
            statistic_name="mean_y",
        )

        assert result.samples.shape == (40,)
        assert result.metadata["n_samples"] == 210


class TestGPUBackends:
    """CuPy backend tests for unified resampling engine."""

    @pytest.mark.skipif(
        not LinearRegression(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    def test_bootstrap_cupy_backend_returns_cupy_samples(self):
        import cupy as cp

        rng = np.random.default_rng(123)
        x = cp.asarray(rng.normal(size=240))

        result = bootstrap_statistic(
            lambda arr: cp.mean(arr),
            x,
            n_resamples=80,
            strategy="iid",
            random_state=123,
            statistic_name="mean",
            backend="cupy",
        )

        assert isinstance(result.samples, cp.ndarray)
        assert result.metadata["backend"] == "cupy"
        assert result.samples.shape == (80,)

        payload = result.to_dict()
        assert len(payload["samples"]) == 80

    @pytest.mark.skipif(
        not LinearRegression(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    def test_permutation_cupy_backend_returns_cupy_samples(self):
        import cupy as cp

        rng = np.random.default_rng(456)
        X = cp.asarray(rng.normal(size=(220, 1)))
        y = 1.8 * X[:, 0] + cp.asarray(rng.normal(scale=0.6, size=220))

        result = permutation_test(
            lambda X_in, y_in: cp.corrcoef(X_in[:, 0], y_in)[0, 1],
            X,
            y,
            n_resamples=120,
            strategy="iid",
            alternative="two-sided",
            random_state=456,
            statistic_name="corr",
            backend="cupy",
        )

        assert isinstance(result.samples, cp.ndarray)
        assert result.metadata["backend"] == "cupy"
        assert result.samples.shape == (120,)
        assert 0.0 <= result.pvalue <= 1.0

    @pytest.mark.skipif(
        not LinearRegression(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    def test_model_wrapper_auto_uses_cupy_on_cuda(self):
        import cupy as cp

        set_device("cuda")
        rng = np.random.default_rng(789)
        X = rng.normal(size=(200, 2))
        y = 1.2 * X[:, 0] - 0.7 * X[:, 1] + rng.normal(scale=0.5, size=200)

        model = LinearRegression(device="cuda", compute_inference=False)
        model.fit(X, y)

        boot_auto = model.bootstrap_statistic(
            lambda X_s, y_s: cp.mean(y_s),
            X,
            y,
            n_resamples=50,
            strategy="iid",
            random_state=789,
            statistic_name="mean",
            backend="auto",
        )
        assert isinstance(boot_auto.samples, cp.ndarray)
        assert boot_auto.metadata["backend"] == "cupy"

        perm_auto = model.permutation_test(
            lambda X_in, y_in: cp.corrcoef(X_in[:, 0], y_in)[0, 1],
            X,
            y,
            n_resamples=80,
            strategy="iid",
            alternative="two-sided",
            random_state=789,
            statistic_name="corr",
            backend="auto",
        )
        assert isinstance(perm_auto.samples, cp.ndarray)
        assert perm_auto.metadata["backend"] == "cupy"

        boot_numpy = model.bootstrap_statistic(
            lambda X_s, y_s: np.mean(y_s),
            X,
            y,
            n_resamples=30,
            strategy="iid",
            random_state=790,
            statistic_name="mean",
            backend="numpy",
        )
        assert isinstance(boot_numpy.samples, np.ndarray)
        assert boot_numpy.metadata["backend"] == "numpy"
