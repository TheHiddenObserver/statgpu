"""Tests for multiple-testing inference utilities."""

import numpy as np
import pytest

from statgpu._config import set_device
from statgpu._config import Device
from statgpu.inference import adjust_pvalues, combine_pvalues, multipletests
from statgpu.linear_model import LinearRegression


class TestMultipleTestingUtilities:
    """Unit tests for p-value adjustment methods."""

    def test_bh_known_values(self):
        p = np.array([0.001, 0.010, 0.020, 0.200])
        reject, p_adj = adjust_pvalues(p, method="bh", alpha=0.05)

        expected = np.array([0.004, 0.020, 0.02666666666666667, 0.200])
        assert np.allclose(p_adj, expected)
        assert np.array_equal(reject, np.array([True, True, True, False]))

    def test_alias_matches_primary_name(self):
        p = np.array([0.003, 0.020, 0.040, 0.400])

        reject_a, adj_a = adjust_pvalues(p, method="bh")
        reject_b, adj_b = adjust_pvalues(p, method="fdr_bh")
        reject_c, adj_c = multipletests(p, method="benjamini-hochberg")

        assert np.array_equal(reject_a, reject_b)
        assert np.array_equal(reject_a, reject_c)
        assert np.allclose(adj_a, adj_b)
        assert np.allclose(adj_a, adj_c)

    def test_axis_adjustment_for_matrix(self):
        p = np.array(
            [
                [0.001, 0.200],
                [0.010, 0.030],
                [0.020, 0.040],
                [0.900, 0.800],
            ]
        )

        reject, p_adj = adjust_pvalues(p, method="bh", axis=0)

        for col in range(p.shape[1]):
            reject_col, p_adj_col = adjust_pvalues(p[:, col], method="bh")
            assert np.array_equal(reject[:, col], reject_col)
            assert np.allclose(p_adj[:, col], p_adj_col)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            adjust_pvalues(np.array([0.1, -0.2]))

        with pytest.raises(ValueError):
            adjust_pvalues(np.array([0.1, 0.2]), alpha=1.2)

        with pytest.raises(ValueError):
            adjust_pvalues(np.array([0.1, 0.2]), method="unknown")

    def test_all_supported_methods_behave_as_expected(self):
        p = np.array([0.001, 0.010, 0.020, 0.200, 0.800])

        _, p_bh = adjust_pvalues(p, method="bh")
        _, p_by = adjust_pvalues(p, method="by")
        _, p_holm = adjust_pvalues(p, method="holm")
        _, p_bonf = adjust_pvalues(p, method="bonferroni")

        assert np.all(p_bh >= p - 1e-12)
        assert np.all(p_by >= p_bh - 1e-12)
        assert np.all(p_holm <= p_bonf + 1e-12)
        assert np.all((p_bonf >= 0.0) & (p_bonf <= 1.0))

    def test_axis_none_flattens_and_negative_axis_supported(self):
        p = np.array(
            [
                [0.010, 0.200],
                [0.030, 0.040],
            ]
        )

        reject_flat, p_flat = adjust_pvalues(p, method="bonferroni", axis=None)
        expected_flat = np.minimum(p.reshape(-1) * p.size, 1.0).reshape(p.shape)
        assert reject_flat.shape == p.shape
        assert np.allclose(p_flat, expected_flat)

        reject_neg, p_neg = adjust_pvalues(p, method="bh", axis=-1)
        for row in range(p.shape[0]):
            reject_row, p_row = adjust_pvalues(p[row], method="bh")
            assert np.array_equal(reject_neg[row], reject_row)
            assert np.allclose(p_neg[row], p_row)

    def test_aliases_for_by_holm_bonferroni(self):
        p = np.array([0.004, 0.030, 0.400, 0.900])

        _, by_a = adjust_pvalues(p, method="by")
        _, by_b = adjust_pvalues(p, method="fdr_by")
        assert np.allclose(by_a, by_b)

        _, holm_a = adjust_pvalues(p, method="holm")
        _, holm_b = adjust_pvalues(p, method="holm-bonferroni")
        assert np.allclose(holm_a, holm_b)

        _, bonf_a = adjust_pvalues(p, method="bonferroni")
        _, bonf_b = multipletests(p, method="bonf")
        assert np.allclose(bonf_a, bonf_b)

    def test_scalar_with_axis_raises(self):
        with pytest.raises(ValueError):
            adjust_pvalues(0.2, axis=0)

    def test_empty_input_returns_empty_arrays(self):
        reject, p_adj = adjust_pvalues(np.array([]), method="bh")
        assert reject.shape == (0,)
        assert p_adj.shape == (0,)

    def test_combine_pvalues_fisher_matches_scipy(self):
        scipy_stats = pytest.importorskip("scipy.stats")

        p = np.array([0.010, 0.030, 0.200, 0.400], dtype=float)
        stat, p_comb = combine_pvalues(p, method="fisher", backend="numpy")
        stat_ref, p_ref = scipy_stats.combine_pvalues(p, method="fisher")

        assert np.isclose(float(stat), float(stat_ref), rtol=1e-12, atol=1e-12)
        assert np.isclose(float(p_comb), float(p_ref), rtol=1e-12, atol=1e-12)

    def test_combine_pvalues_cauchy_alias_and_weights(self):
        p = np.array([0.001, 0.020, 0.050, 0.500], dtype=float)
        w = np.array([1.0, 2.0, 1.0, 1.0], dtype=float)

        stat_a, p_a = combine_pvalues(p, method="cauchy", weights=w, backend="numpy")
        stat_b, p_b = combine_pvalues(p, method="acat", weights=w, backend="numpy")

        assert np.isclose(float(stat_a), float(stat_b), rtol=1e-12, atol=1e-12)
        assert np.isclose(float(p_a), float(p_b), rtol=1e-12, atol=1e-12)
        assert 0.0 <= float(p_a) <= 1.0

    def test_combine_pvalues_axis_behavior(self):
        p = np.array(
            [
                [0.010, 0.040, 0.200],
                [0.002, 0.030, 0.400],
            ],
            dtype=float,
        )

        stat_axis, p_axis = combine_pvalues(p, method="fisher", axis=1, backend="numpy")
        assert stat_axis.shape == (2,)
        assert p_axis.shape == (2,)

        for i in range(p.shape[0]):
            stat_i, p_i = combine_pvalues(p[i], method="fisher", backend="numpy")
            assert np.isclose(float(stat_axis[i]), float(stat_i), rtol=1e-12, atol=1e-12)
            assert np.isclose(float(p_axis[i]), float(p_i), rtol=1e-12, atol=1e-12)

    def test_combine_pvalues_invalid_inputs_raise(self):
        p = np.array([0.010, 0.050, 0.200], dtype=float)

        with pytest.raises(ValueError):
            combine_pvalues(p, method="unknown")

        with pytest.raises(ValueError):
            combine_pvalues(p, method="fisher", weights=np.ones_like(p))

        with pytest.raises(ValueError):
            combine_pvalues(p, method="cauchy", weights=np.array([1.0, -1.0, 1.0]))

        with pytest.raises(ValueError):
            combine_pvalues(p, method="cauchy", weights=np.array([1.0, 2.0]))


class TestModelWrapper:
    """Integration tests for BaseEstimator-level thin wrapper."""

    def test_model_adjust_pvalues_uses_internal_pvalues(self):
        set_device("cpu")
        rng = np.random.default_rng(42)

        X = rng.normal(size=(400, 4))
        y = X @ np.array([0.9, -1.4, 0.7, 1.2]) + rng.normal(scale=0.3, size=400)

        model = LinearRegression(device="cpu", compute_inference=True, cov_type="hc1")
        model.fit(X, y)

        out = model.adjust_pvalues(method="bh", alpha=0.05)

        assert out["method"] == "bh"
        assert np.isclose(out["alpha"], 0.05)
        assert out["pvalues"].shape == model._pvalues.shape
        assert out["pvalues_adjusted"].shape == model._pvalues.shape
        assert out["reject"].shape == model._pvalues.shape
        assert np.all(out["pvalues_adjusted"] >= out["pvalues"] - 1e-12)

    def test_model_adjust_pvalues_without_internal_pvalues_raises(self):
        set_device("cpu")
        rng = np.random.default_rng(7)

        X = rng.normal(size=(200, 3))
        y = X @ np.array([0.6, 1.0, -0.8]) + rng.normal(scale=0.2, size=200)

        model = LinearRegression(device="cpu", compute_inference=False)
        model.fit(X, y)

        with pytest.raises(RuntimeError):
            model.adjust_pvalues()

    def test_model_adjust_pvalues_with_explicit_values(self):
        set_device("cpu")
        model = LinearRegression(device="cpu", compute_inference=False)

        p = np.array([0.002, 0.020, 0.200, 0.800])
        out = model.adjust_pvalues(pvalues=p, method="holm", alpha=0.1)

        reject_ref, p_ref = adjust_pvalues(p, method="holm", alpha=0.1)
        assert np.array_equal(out["reject"], reject_ref)
        assert np.allclose(out["pvalues_adjusted"], p_ref)
        assert np.isclose(out["alpha"], 0.1)

    def test_model_combine_pvalues_uses_internal_pvalues(self):
        set_device("cpu")
        rng = np.random.default_rng(11)

        X = rng.normal(size=(280, 4))
        y = X @ np.array([0.8, -1.0, 0.6, 1.1]) + rng.normal(scale=0.3, size=280)

        model = LinearRegression(device="cpu", compute_inference=True, cov_type="hc1")
        model.fit(X, y)

        out = model.combine_pvalues(method="fisher", backend="numpy")
        stat_ref, p_ref = combine_pvalues(model._pvalues, method="fisher", backend="numpy")

        assert out["method"] == "fisher"
        assert out["backend"] == "numpy"
        assert np.isclose(float(out["statistic"]), float(stat_ref), rtol=1e-12, atol=1e-12)
        assert np.isclose(float(out["pvalue"]), float(p_ref), rtol=1e-12, atol=1e-12)

    def test_model_combine_pvalues_without_internal_values_raises(self):
        set_device("cpu")
        rng = np.random.default_rng(12)

        X = rng.normal(size=(200, 3))
        y = X @ np.array([0.7, -0.8, 0.9]) + rng.normal(scale=0.2, size=200)

        model = LinearRegression(device="cpu", compute_inference=False)
        model.fit(X, y)

        with pytest.raises(RuntimeError):
            model.combine_pvalues()


class TestGPUBackends:
    """CuPy backend tests for multiple-testing utilities."""

    @pytest.mark.skipif(
        not LinearRegression(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    def test_adjust_pvalues_cupy_matches_numpy(self):
        import cupy as cp

        p_np = np.array([0.001, 0.010, 0.020, 0.200, 0.800], dtype=float)
        p_cp = cp.asarray(p_np)

        reject_np, adj_np = adjust_pvalues(p_np, method="bh", alpha=0.05, backend="numpy")
        reject_cp, adj_cp = adjust_pvalues(p_cp, method="bh", alpha=0.05, backend="cupy")

        assert isinstance(reject_cp, cp.ndarray)
        assert isinstance(adj_cp, cp.ndarray)
        assert np.array_equal(reject_np, cp.asnumpy(reject_cp))
        assert np.allclose(adj_np, cp.asnumpy(adj_cp))

    @pytest.mark.skipif(
        not LinearRegression(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    def test_multipletests_cupy_alias_matches_numpy(self):
        import cupy as cp

        p_np = np.array([0.003, 0.020, 0.040, 0.400], dtype=float)
        p_cp = cp.asarray(p_np)

        reject_np, adj_np = multipletests(p_np, method="benjamini-hochberg", backend="numpy")
        reject_cp, adj_cp = multipletests(p_cp, method="benjamini-hochberg", backend="cupy")

        assert isinstance(reject_cp, cp.ndarray)
        assert isinstance(adj_cp, cp.ndarray)
        assert np.array_equal(reject_np, cp.asnumpy(reject_cp))
        assert np.allclose(adj_np, cp.asnumpy(adj_cp))

    @pytest.mark.skipif(
        not LinearRegression(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    def test_model_wrapper_auto_uses_cupy_on_cuda(self):
        import cupy as cp

        set_device("cuda")
        rng = np.random.default_rng(123)
        X = rng.normal(size=(260, 4))
        y = X @ np.array([0.8, -1.1, 0.5, 1.3]) + rng.normal(scale=0.3, size=260)

        model = LinearRegression(device="cuda", compute_inference=True, cov_type="hc1")
        model.fit(X, y)

        out_auto = model.adjust_pvalues(method="bh", alpha=0.05, backend="auto")
        assert out_auto["backend"] == "cupy"
        assert isinstance(out_auto["pvalues"], cp.ndarray)
        assert isinstance(out_auto["pvalues_adjusted"], cp.ndarray)
        assert isinstance(out_auto["reject"], cp.ndarray)

        out_np = model.adjust_pvalues(method="bh", alpha=0.05, backend="numpy")
        assert out_np["backend"] == "numpy"
        assert isinstance(out_np["pvalues"], np.ndarray)
        assert isinstance(out_np["pvalues_adjusted"], np.ndarray)
        assert isinstance(out_np["reject"], np.ndarray)

    @pytest.mark.skipif(
        not LinearRegression(device="auto")._get_compute_device() == Device.CUDA,
        reason="CUDA not available",
    )
    def test_combine_pvalues_cupy_matches_numpy(self):
        import cupy as cp

        p_np = np.array([0.001, 0.020, 0.040, 0.500], dtype=float)
        p_cp = cp.asarray(p_np)

        stat_np_f, p_np_f = combine_pvalues(p_np, method="fisher", backend="numpy")
        stat_cp_f, p_cp_f = combine_pvalues(p_cp, method="fisher", backend="cupy")
        assert np.isclose(float(stat_np_f), float(cp.asnumpy(stat_cp_f)), rtol=1e-12, atol=1e-12)
        assert np.isclose(float(p_np_f), float(cp.asnumpy(p_cp_f)), rtol=1e-12, atol=1e-12)

        w_np = np.array([1.0, 2.0, 1.0, 1.0], dtype=float)
        w_cp = cp.asarray(w_np)
        stat_np_c, p_np_c = combine_pvalues(p_np, method="cauchy", weights=w_np, backend="numpy")
        stat_cp_c, p_cp_c = combine_pvalues(p_cp, method="cauchy", weights=w_cp, backend="cupy")
        assert np.isclose(float(stat_np_c), float(cp.asnumpy(stat_cp_c)), rtol=1e-10, atol=1e-10)
        assert np.isclose(float(p_np_c), float(cp.asnumpy(p_cp_c)), rtol=1e-10, atol=1e-10)
