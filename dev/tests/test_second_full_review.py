"""Regression tests for the second repository-wide review of PR #79."""

import numpy as np
import pytest


class TestStepwiseSelectorContracts:
    def test_forward_uses_same_feature_order_for_fit_and_predict(self):
        from statgpu.feature_selection import StepwiseSelector
        from statgpu.linear_model import LinearRegression

        rng = np.random.default_rng(123)
        X = rng.normal(size=(300, 4))
        y = 10.0 * X[:, 2] + 2.0 * X[:, 0] + rng.normal(scale=0.1, size=300)
        selector = StepwiseSelector(
            LinearRegression,
            direction="forward",
            max_features=2,
            compute_inference=False,
        ).fit(X, y)

        assert selector.selected_features_ == [0, 2]
        assert np.mean((selector.predict(X) - y) ** 2) < 0.05
        assert selector.selection_history_[-1]["features"] == (0, 2)

    def test_backward_obeys_hard_feature_cap(self):
        from statgpu.feature_selection import StepwiseSelector
        from statgpu.linear_model import LinearRegression

        rng = np.random.default_rng(321)
        X = rng.normal(size=(250, 5))
        y = 4.0 * X[:, 1] - 3.0 * X[:, 4] + rng.normal(scale=0.1, size=250)
        selector = StepwiseSelector(
            LinearRegression,
            direction="backward",
            max_features=2,
            compute_inference=False,
        ).fit(X, y)

        assert selector.selected_features_ == [1, 4]
        assert len(selector.selection_history_) >= 4
        assert np.mean((selector.predict(X) - y) ** 2) < 0.05

    def test_null_model_can_win_and_repeated_fit_resets_state(self):
        from statgpu.feature_selection import StepwiseSelector
        from statgpu.linear_model import LinearRegression

        X = np.zeros((40, 3))
        y = np.ones(40)
        selector = StepwiseSelector(
            LinearRegression,
            direction="forward",
            compute_inference=False,
        ).fit(X, y)
        assert selector.selected_features_ == []
        first_history_length = len(selector.aic_history_)
        selector.fit(X, y)
        assert selector.selected_features_ == []
        assert len(selector.aic_history_) == first_history_length == 1
        np.testing.assert_allclose(selector.predict(X), y)

    @pytest.mark.parametrize("direction", ["invalid", "", None])
    def test_invalid_direction_rejected(self, direction):
        from statgpu.feature_selection import StepwiseSelector
        from statgpu.linear_model import LinearRegression

        with pytest.raises(ValueError, match="direction"):
            StepwiseSelector(LinearRegression, direction=direction)

    def test_constructor_parameters_are_not_mutated_by_fit(self):
        from statgpu.feature_selection import StepwiseSelector
        from statgpu.linear_model import LinearRegression

        X = np.arange(30.0).reshape(10, 3)
        y = np.arange(10.0)
        selector = StepwiseSelector(
            LinearRegression,
            max_features=None,
            compute_inference=False,
        ).fit(X, y)
        assert selector.max_features is None
        assert selector.get_params()["compute_inference"] is False


class TestWelchBackendAndReference:
    def test_numpy_matches_scipy_welch_anova(self):
        from scipy import stats
        from statgpu.anova import f_welch

        rng = np.random.default_rng(12)
        groups = (
            rng.normal(0.0, 1.0, 80),
            rng.normal(0.5, 3.0, 120),
            rng.normal(-0.2, 0.5, 60),
        )
        actual = f_welch(*groups)
        expected = stats.f_oneway(*groups, equal_var=False)
        np.testing.assert_allclose(actual.statistic, expected.statistic, rtol=1e-12)
        np.testing.assert_allclose(actual.pvalue, expected.pvalue, rtol=1e-10)
        assert isinstance(actual.df_within, float)

    def test_torch_cpu_matches_numpy_without_full_numpy_fallback(self):
        torch = pytest.importorskip("torch")
        from statgpu.anova import f_welch

        groups = [
            np.array([0.2, 0.4, 1.1, 1.4]),
            np.array([1.0, 1.2, 2.4, 3.0, 3.1]),
            np.array([-0.4, 0.1, 0.2, 0.3]),
        ]
        expected = f_welch(*groups, backend="numpy")
        actual = f_welch(
            *(torch.tensor(group, dtype=torch.float64) for group in groups),
            backend="torch",
        )
        np.testing.assert_allclose(actual.statistic, expected.statistic, rtol=1e-12)
        np.testing.assert_allclose(actual.df_within, expected.df_within, rtol=1e-12)
        np.testing.assert_allclose(actual.pvalue, expected.pvalue, rtol=5e-5)

    def test_partial_eta_rejects_invalid_sum_of_squares(self):
        from statgpu.anova import partial_eta_squared

        with pytest.raises(ValueError, match="non-negative"):
            partial_eta_squared(-1.0, 2.0)
        with pytest.raises(ValueError, match="finite"):
            partial_eta_squared(np.inf, 2.0)


class TestGenericCrossValidationContracts:
    def test_cache_key_is_framed_and_dtype_sensitive(self):
        from statgpu.cross_validation import CVCache

        assert CVCache.make_key("ab", "c") != CVCache.make_key("a", "bc")
        assert CVCache.make_key(np.array([1], dtype=np.int32)) != CVCache.make_key(
            np.array([1], dtype=np.int64)
        )
        assert CVCache.make_key(["a", "bc"]) != CVCache.make_key(["ab", "c"])

    def test_incomplete_alpha_is_excluded_from_selection(self):
        from statgpu.cross_validation import run_cv

        X = np.arange(24.0).reshape(12, 2)
        y = np.arange(12.0)

        def evaluate(X_train, y_train, X_val, y_val, alpha, **kwargs):
            if alpha == 0.1 and np.min(y_val) < 4:
                raise ValueError("intentional fold failure")
            return alpha

        best, means, scores = run_cv(
            X,
            y,
            np.array([0.1, 0.2]),
            evaluate,
            n_folds=3,
            random_state=None,
        )
        assert best == pytest.approx(0.2)
        assert np.isnan(means[0])
        assert np.isfinite(means[1])
        assert np.sum(np.isfinite(scores[:, 0])) < 3

    def test_all_incomplete_alphas_raise(self):
        from statgpu.cross_validation import run_cv

        X = np.arange(16.0).reshape(8, 2)
        y = np.arange(8.0)

        def fail(*args, **kwargs):
            raise RuntimeError("no convergence")

        with pytest.raises(ValueError, match="No alpha completed every CV fold"):
            run_cv(X, y, np.array([0.1, 1.0]), fail, n_folds=2)

    def test_sample_weight_validation_preserves_torch_backend(self):
        torch = pytest.importorskip("torch")
        from statgpu.cross_validation import validate_cv_sample_weight

        weights = torch.tensor([1.0, 2.0, 3.0])
        result = validate_cv_sample_weight(weights, 3)
        assert isinstance(result, torch.Tensor)
        assert result.device == weights.device
        assert result.dtype == torch.float64

    def test_kfold_argument_and_empty_completeness_validation(self):
        from statgpu.cross_validation import folds_are_complete, kfold_indices

        with pytest.raises(TypeError, match="n_samples"):
            kfold_indices(10.5, 2)
        with pytest.raises(ValueError, match="positive"):
            kfold_indices(0, 2)
        assert folds_are_complete([], 0) is False


class TestRegressionDiagnosticsReference:
    def test_influence_measures_match_statsmodels(self):
        sm = pytest.importorskip("statsmodels.api")
        from statgpu.diagnostics import RegressionDiagnostics
        from statgpu.linear_model import LinearRegression

        rng = np.random.default_rng(7)
        X = rng.normal(size=(120, 3))
        y = 1.0 + X @ np.array([1.5, -0.7, 0.3]) + rng.normal(size=120)
        model = LinearRegression().fit(X, y)
        diagnostics = RegressionDiagnostics(model)
        reference = sm.OLS(y, sm.add_constant(X)).fit().get_influence()

        np.testing.assert_allclose(
            diagnostics.leverage, reference.hat_matrix_diag, atol=1e-12
        )
        np.testing.assert_allclose(
            diagnostics.studentized_residuals,
            reference.resid_studentized_internal,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            diagnostics.externally_studentized_residuals,
            reference.resid_studentized_external,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            diagnostics.cooks_distance, reference.cooks_distance[0], atol=1e-10
        )

    def test_rank_deficient_design_produces_finite_leverage(self):
        from statgpu.diagnostics import RegressionDiagnostics
        from statgpu.linear_model import LinearRegression

        x = np.linspace(-1.0, 1.0, 50)
        X = np.column_stack([x, x])
        y = 2.0 * x + 0.1
        diagnostics = RegressionDiagnostics(LinearRegression().fit(X, y))
        assert np.all(np.isfinite(diagnostics.leverage))
        assert np.all((0.0 <= diagnostics.leverage) & (diagnostics.leverage <= 1.0))
        assert np.all(np.isinf(diagnostics.vif()))


class TestCompositePenaltyValidation:
    def test_empty_or_invalid_components_rejected(self):
        from statgpu.penalties import CompositePenalty

        with pytest.raises(ValueError, match="at least one"):
            CompositePenalty([])
        with pytest.raises(TypeError, match="Penalty"):
            CompositePenalty([object()])

    def test_invalid_weights_rejected(self):
        from statgpu.penalties import CompositePenalty, L1Penalty, L2Penalty

        penalties = [L1Penalty(alpha=0.1), L2Penalty(alpha=0.2)]
        with pytest.raises(ValueError, match="non-negative"):
            CompositePenalty(penalties, weights=[1.0, -1.0])
        with pytest.raises(ValueError, match="finite"):
            CompositePenalty(penalties, weights=[1.0, np.nan])
        with pytest.raises(ValueError, match="positive"):
            CompositePenalty(penalties, weights=[0.0, 0.0])

    def test_constructor_inputs_are_copied(self):
        from statgpu.penalties import CompositePenalty, L1Penalty, L2Penalty

        penalties = [L1Penalty(alpha=0.1), L2Penalty(alpha=0.2)]
        weights = [0.25, 0.75]
        composite = CompositePenalty(penalties, weights=weights)
        penalties.clear()
        weights[0] = 99.0
        assert composite.n_penalties == 2
        assert composite.weights == (0.25, 0.75)

class TestEstimatorCloneAndFeatureSelectionBackend:
    def test_all_default_public_estimators_clone(self):
        import inspect
        import statgpu
        from sklearn.base import clone

        failures = []
        for name in statgpu.__all__:
            estimator_class = getattr(statgpu, name, None)
            if not inspect.isclass(estimator_class) or not hasattr(estimator_class, "fit"):
                continue
            if inspect.isabstract(estimator_class):
                continue
            signature = inspect.signature(estimator_class)
            required = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.default is inspect._empty
                and parameter.kind
                not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            ]
            if required:
                continue
            try:
                cloned = clone(estimator_class())
                assert type(cloned) is estimator_class
            except Exception as exc:  # aggregate all public contract failures
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        assert failures == []

    def test_stepwise_selector_clones_without_fit_state(self):
        from sklearn.base import clone
        from statgpu.feature_selection import StepwiseSelector
        from statgpu.linear_model import LinearRegression

        selector = StepwiseSelector(
            LinearRegression,
            direction="forward",
            max_features=2,
            compute_inference=False,
        )
        cloned = clone(selector)
        assert cloned.get_params() == selector.get_params()
        assert cloned.selected_features_ is None

    def test_knockoff_selectors_follow_sklearn_parameter_contract(self):
        from sklearn.base import clone
        from statgpu.feature_selection import FixedXKnockoffSelector, KnockoffSelector

        for selector in (KnockoffSelector(q=0.2), FixedXKnockoffSelector(q=0.2)):
            cloned = clone(selector)
            assert cloned.q == pytest.approx(0.2)
            cloned.set_params(q=0.15)
            assert cloned.q == pytest.approx(0.15)
            with pytest.raises(ValueError, match="Invalid parameter"):
                cloned.set_params(not_a_parameter=1)

    def test_knockoff_transform_preserves_torch_backend(self):
        torch = pytest.importorskip("torch")
        from statgpu.feature_selection import KnockoffSelector
        from statgpu.feature_selection._knockoff import KnockoffResult

        selector = KnockoffSelector()
        selector.selected_features_ = np.array([0, 2], dtype=int)
        selector.result_ = KnockoffResult(
            knockoff_type="fixed_x",
            selected_features=selector.selected_features_,
            W=np.array([2.0, 0.0, 1.0]),
            threshold=1.0,
            q=0.1,
            estimated_fdr=0.0,
            q_trajectory=[],
            method="corr_diff",
            fdr_control="knockoff_plus",
            random_state=0,
            backend="torch",
        )
        X = torch.arange(15.0).reshape(5, 3)
        transformed = selector.transform(X)
        assert isinstance(transformed, torch.Tensor)
        assert transformed.device == X.device
        torch.testing.assert_close(transformed, X[:, [0, 2]])


class TestGaussianModelSummaryStatistics:
    def test_perfect_fit_has_infinite_f_and_zero_tail_probability(self):
        from statgpu.linear_model import LinearRegression

        X = np.arange(1.0, 9.0).reshape(-1, 1)
        y = 2.0 * X[:, 0] + 3.0
        model = LinearRegression().fit(X, y)
        assert np.isposinf(model.fvalue)
        assert model.f_pvalue == 0.0

        # Explicitly exercise the zero-variance likelihood boundary without
        # relying on platform-specific least-squares roundoff.
        model._resid = np.zeros_like(y)
        assert np.isposinf(model.llf)
        assert np.isneginf(model.aic)
        assert np.isneginf(model.bic)

    def test_intercept_only_overall_f_test_is_undefined(self):
        from statgpu.linear_model import LinearRegression

        y = np.arange(8.0)
        model = LinearRegression().fit(np.empty((y.size, 0)), y)
        assert np.isnan(model.fvalue)
        assert np.isnan(model.f_pvalue)

    def test_nonpositive_residual_df_makes_adjusted_r2_undefined(self):
        from statgpu.linear_model import LinearRegression

        X = np.eye(4)
        y = np.arange(4.0)
        model = LinearRegression(fit_intercept=False).fit(X, y)
        assert model._df_resid == 0
        assert np.isnan(model.rsquared_adj)
        assert np.isnan(model.fvalue)
        assert np.isnan(model.f_pvalue)

    def test_legacy_result_conf_int_remains_callable(self):
        from statgpu.linear_model._stats import RegressionResults

        class Model:
            _X_design = np.column_stack([np.ones(6), np.arange(6.0)])
            _y = 1.0 + 2.0 * np.arange(6.0)
            _feature_names = ["(Intercept)", "x"]

        result = RegressionResults(
            Model(),
            params=np.array([1.0, 2.0]),
            resid=np.zeros(6),
            scale=0.0,
            nobs=6,
            df_resid=4,
        )
        assert callable(result.conf_int)
        assert result.conf_int().shape == (2, 2)
        assert np.isposinf(result.fvalue)
        assert result.f_pvalue == 0.0
        assert np.isposinf(result.llf)


class TestKernelBackendContracts:
    def test_rbf_kernel_supports_torch_and_integer_inputs(self):
        torch = pytest.importorskip("torch")
        from statgpu.nonparametric.kernel_methods import rbf_kernel

        X_int = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        K_torch = rbf_kernel(X_int, gamma=0.5, xp=torch)
        expected = np.exp(
            -0.5
            * np.array(
                [[0.0, 8.0], [8.0, 0.0]],
                dtype=float,
            )
        )
        assert K_torch.dtype == torch.float64
        np.testing.assert_allclose(K_torch.numpy(), expected, rtol=1e-12, atol=1e-12)

        K_numpy = rbf_kernel(X_int.numpy(), gamma=0.5)
        assert K_numpy.dtype == np.float64
        np.testing.assert_allclose(K_numpy, expected, rtol=1e-12, atol=1e-12)

    def test_large_numpy_rbf_preserves_float64_dtype(self):
        from statgpu.nonparametric.kernel_methods import rbf_kernel

        # Cross the previous 4M-element branch without allocating an excessive
        # test matrix. The old implementation silently returned float32 here.
        X = np.linspace(0.0, 1.0, 2001, dtype=np.float64).reshape(-1, 1)
        Y = np.linspace(0.0, 1.0, 2000, dtype=np.float64).reshape(-1, 1)
        K = rbf_kernel(X, Y, gamma=0.5)
        assert K.dtype == np.float64
        np.testing.assert_allclose(K[[0, -1], :][:, [0, -1]],
                                   np.exp(-0.5 * (X[[0, -1]] - Y[[0, -1]].T) ** 2))

    def test_user_kernel_internal_typeerror_is_not_retried(self):
        from statgpu.nonparametric.kernel_methods import pairwise_kernels

        calls = []

        def broken_kernel(X, Y=None, xp=None):
            calls.append(1)
            raise TypeError("internal kernel failure")

        with pytest.raises(TypeError, match="internal kernel failure"):
            pairwise_kernels(np.ones((2, 1)), metric=broken_kernel, xp=np)
        assert len(calls) == 1


class TestKDECompactSupportNumerics:
    def test_zero_density_rows_do_not_emit_runtime_warnings(self):
        from statgpu.nonparametric import KernelDensityEstimator

        model = KernelDensityEstimator(kernel="uniform", bandwidth=0.1).fit(
            np.array([0.0, 0.05, 0.1])
        )
        import warnings
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            density = model.pdf(np.array([10.0]))
        assert record == []
        np.testing.assert_array_equal(np.asarray(density), np.array([0.0]))


class TestResamplingPublicContracts:
    def test_resampling_rejects_fractional_counts_and_empty_samples(self):
        from statgpu.inference import bootstrap_statistic, permutation_test

        with pytest.raises(ValueError, match="positive integer"):
            bootstrap_statistic(np.mean, np.arange(5.0), n_resamples=2.5)
        with pytest.raises(TypeError, match="positive integer"):
            permutation_test(lambda X, y: np.mean(y), np.ones((5, 1)), np.arange(5.0), n_resamples=True)
        with pytest.raises(ValueError, match="at least one observation"):
            bootstrap_statistic(np.mean, np.array([]), n_resamples=5)
        with pytest.raises(ValueError, match="at least one observation"):
            permutation_test(lambda X, y: np.mean(y), np.empty((0, 1)), np.array([]), n_resamples=5)

    def test_resampling_validates_confidence_block_size_and_statistic(self):
        from statgpu.inference import bootstrap_statistic

        x = np.arange(6.0)
        with pytest.raises(ValueError, match="finite"):
            bootstrap_statistic(np.mean, x, confidence_level=np.nan)
        with pytest.raises(ValueError, match="block_size"):
            bootstrap_statistic(np.mean, x, strategy="block", block_size=1.5)
        with pytest.raises(TypeError, match="callable"):
            bootstrap_statistic(3.0, x)
        with pytest.raises(ValueError, match="finite scalar"):
            bootstrap_statistic(lambda values: np.nan, x, n_resamples=5)

    def test_result_uses_normalized_strategy_name(self):
        from statgpu.inference import bootstrap_statistic, permutation_test

        x = np.arange(8.0)
        boot = bootstrap_statistic(np.mean, x, n_resamples=5, strategy=" IID ", random_state=0)
        assert boot.strategy == "iid"
        perm = permutation_test(
            lambda X, y: float(np.mean(y)),
            np.ones((8, 1)),
            x,
            n_resamples=5,
            strategy=" IID ",
            random_state=0,
        )
        assert perm.strategy == "iid"


class TestTorchBackendFactoryContract:
    def test_explicit_torch_backend_can_run_on_cpu_without_cuda(self):
        torch = pytest.importorskip("torch")
        from statgpu.backends import get_backend
        from statgpu.inference import adjust_pvalues, bootstrap_statistic

        backend = get_backend("torch", device="cpu")
        values = backend.asarray([1.0, 2.0])
        assert isinstance(values, torch.Tensor)
        assert values.device.type == "cpu"

        p = torch.tensor([0.01, 0.2], dtype=torch.float64)
        _, adjusted = adjust_pvalues(p, method="bh", backend="torch")
        assert isinstance(adjusted, torch.Tensor)
        assert adjusted.device.type == ("cuda" if torch.cuda.is_available() else "cpu")

        result = bootstrap_statistic(
            lambda x: x.mean(),
            torch.arange(6.0, dtype=torch.float64),
            n_resamples=5,
            backend="torch",
            random_state=0,
        )
        assert isinstance(result.samples, torch.Tensor)
        assert result.samples.device.type == ("cuda" if torch.cuda.is_available() else "cpu")


class TestQuadraticLLARouting:
    def test_weighted_squared_scad_uses_fista_without_newton_failure(self):
        import warnings
        from statgpu.linear_model import PenalizedLinearRegression

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 5))
        y = X @ np.array([3.0, -2.0, 0.0, 0.0, 0.0]) + rng.normal(scale=0.1, size=100)
        weights = np.ones(100)
        weights[:50] = 10.0

        model = PenalizedLinearRegression(
            penalty="scad",
            alpha=0.1,
            max_iter=200,
            tol=1e-8,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X, y, sample_weight=weights)
        assert not any("proximal_newton line search failed" in str(w.message) for w in caught)
        assert np.all(np.isfinite(model.coef_))
        assert model.n_iter_ > 0

        weighted_mean_X = np.average(X, axis=0, weights=weights)
        weighted_mean_y = np.average(y, weights=weights)
        expected_intercept = weighted_mean_y - weighted_mean_X @ model.coef_
        assert model.intercept_ == pytest.approx(expected_intercept, rel=1e-10, abs=1e-10)

class TestPublicFeatureSelectionAndDiagnosticsExports:
    @pytest.mark.parametrize("invalid", [0, -1, 1.5, True, "3"])
    def test_modelx_draws_must_be_strict_positive_integer(self, invalid):
        from statgpu.feature_selection import model_x_knockoff_filter

        X = np.arange(24.0).reshape(8, 3)
        y = np.arange(8.0)
        expected = TypeError if isinstance(invalid, (bool, float, str)) else ValueError
        with pytest.raises(expected, match="modelx_draws"):
            model_x_knockoff_filter(X, y, modelx_draws=invalid)

    def test_root_exports_stepwise_and_diagnostics(self):
        import statgpu
        from statgpu.diagnostics import RegressionDiagnostics, diagnose_model
        from statgpu.feature_selection import StepwiseSelector, stepwise_selection

        assert statgpu.StepwiseSelector is StepwiseSelector
        assert statgpu.stepwise_selection is stepwise_selection
        assert statgpu.RegressionDiagnostics is RegressionDiagnostics
        assert statgpu.diagnose_model is diagnose_model
        for name in (
            "StepwiseSelector",
            "stepwise_selection",
            "RegressionDiagnostics",
            "diagnose_model",
        ):
            assert name in statgpu.__all__

class TestCoxInferenceComputation:
    def test_score_test_reuses_single_gradient_hessian_call(self):
        from statgpu.survival import CoxPH

        model = CoxPH(compute_inference=False, compute_cindex=False)
        model.coef_ = np.array([0.2, -0.1])
        model._log_likelihood = -10.0
        model._log_likelihood_null = -11.0
        calls = []

        def fake_gradient_hessian(beta, X, time, event, ep, entry=None):
            calls.append(np.asarray(beta).copy())
            if np.allclose(beta, 0.0):
                return np.array([1.0, 2.0]), -np.eye(2)
            return np.zeros(2), -2.0 * np.eye(2)

        model._compute_gradient_hessian = fake_gradient_hessian
        X = np.ones((5, 2))
        model._compute_inference_cpu(X, np.arange(5.0), np.ones(5))

        assert len(calls) == 2
        assert model._score_test_stat == pytest.approx(5.0)
        assert model._score_test_pvalue > 0.0
        assert np.all(model._pvalues > 0.0)
