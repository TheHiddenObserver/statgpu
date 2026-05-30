"""Tests for the statgpu agent pipeline."""

import numpy as np
import pytest

from statgpu.agent import (
    StatGPUAnalysisAgent,
    AgentConfig,
    MethodRegistry,
    PruningRuleRegistry,
    DataProfile,
    AnalysisPlan,
    ModelResult,
    AnalysisResult,
    CVResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def regression_data(rng):
    X = rng.normal(size=(200, 5))
    y = X[:, 0] - 2 * X[:, 1] + rng.normal(size=200) * 0.1
    return X, y


@pytest.fixture
def classification_data(rng):
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + rng.normal(size=200) * 0.5 > 0).astype(float)
    return X, y


@pytest.fixture
def highdim_data(rng):
    X = rng.normal(size=(50, 100))
    y = X[:, 0] + rng.normal(size=50) * 0.1
    return X, y


@pytest.fixture
def table_data():
    return [
        {"age": 30 + i, "sex": "M" if i % 2 == 0 else "F", "outcome": i % 2}
        for i in range(100)
    ]


# ---------------------------------------------------------------------------
# Profiler tests
# ---------------------------------------------------------------------------

class TestProfiler:
    def test_numeric_array(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert result.profile.n_samples == 200
        assert result.profile.n_features == 5
        assert result.profile.task_type == "regression"

    def test_table_with_categoricals(self, table_data):
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(
            data=table_data, target="outcome"
        )
        assert result.profile.n_samples == 100
        assert "sex" in result.profile.encoded_features
        assert result.profile.encoded_features["sex"][0] == "M"  # reference level

    def test_missing_values_imputation(self, rng):
        X = rng.normal(size=(100, 3))
        X[rng.random(X.shape) < 0.1] = np.nan
        y = X[:, 0] + rng.normal(size=100) * 0.1
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert result.profile.imputed_values > 0
        assert result.models[0].error is None


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------

class TestPlanner:
    def test_regression_task(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert result.profile.task_type == "regression"

    def test_binary_classification_task(self, classification_data):
        X, y = classification_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert result.profile.task_type == "binary_classification"

    def test_unsupervised_task(self, rng):
        X = rng.normal(size=(100, 5))
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X)
        assert result.profile.task_type == "unsupervised"


# ---------------------------------------------------------------------------
# Pruning tests
# ---------------------------------------------------------------------------

class TestPruning:
    def test_highdim_prunes_ols(self, highdim_data):
        X, y = highdim_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        model_names = [m.name for m in result.models]
        assert "LinearRegression" not in model_names
        assert any("Ridge" in n for n in model_names)

    def test_normal_does_not_prune(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        model_names = [m.name for m in result.models]
        assert "LinearRegression" in model_names


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------

class TestRunner:
    def test_regression_models(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert len(result.models) >= 2
        for model in result.models[:2]:
            assert model.error is None
            assert "estimate" in model.coefficients[0]

    def test_classification_models(self, classification_data):
        X, y = classification_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert any("Logistic" in m.name for m in result.models)
        logistic = [m for m in result.models if "Logistic" in m.name][0]
        assert "accuracy" in logistic.metrics or "roc_auc" in logistic.metrics

    def test_unsupervised_models(self, rng):
        X = rng.normal(size=(100, 5))
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X)
        model_names = [m.name for m in result.models]
        assert any("PCA" in n for n in model_names)


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------

class TestValidator:
    def test_small_sample_warning(self, rng):
        X = rng.normal(size=(20, 3))
        y = X[:, 0] + rng.normal(size=20) * 0.1
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert any("below" in w.lower() for w in result.warnings)

    def test_imbalanced_warning(self, rng):
        X = rng.normal(size=(200, 5))
        y = np.zeros(200)
        y[:5] = 1.0  # 2.5% positive rate
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert any("imbalanced" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Reporter tests
# ---------------------------------------------------------------------------

class TestReporter:
    def test_markdown_output(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        md = result.to_markdown()
        assert "statgpu Automatic Analysis Report" in md
        assert "## Data Profile" in md
        assert "## Results" in md

    def test_json_output(self, regression_data, tmp_path):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        path = str(tmp_path / "result.json")
        result.save_json(path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert "version" in data
        assert "result" in data
        assert "provenance" in data

    def test_dict_output(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        d = result.to_dict()
        assert "profile" in d
        assert "plan" in d
        assert "models" in d
        assert "warnings" in d


# ---------------------------------------------------------------------------
# Multiple testing tests
# ---------------------------------------------------------------------------

class TestMultipleTesting:
    def test_no_correction_by_default(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        coef = result.models[0].coefficients
        assert all(c.get("adj_p_value") is None for c in coef)

    def test_bh_correction(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(
            device="cpu", cv_folds=0, multiple_testing_method="bh"
        ).analyze(X=X, y=y)
        coef = result.models[0].coefficients
        non_intercept = [c for c in coef if c["term"] != "Intercept" and c.get("p_value") is not None]
        if len(non_intercept) >= 2:
            assert any(c.get("adj_p_value") is not None for c in non_intercept)
            assert any(c.get("rejected") is not None for c in non_intercept)


# ---------------------------------------------------------------------------
# Model comparison tests
# ---------------------------------------------------------------------------

class TestModelComparison:
    def test_comparison_present(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert result.comparison is not None
        assert result.comparison.best_model is not None
        assert len(result.comparison.ranking) >= 2

    def test_comparison_dict(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        d = result.comparison.to_dict()
        assert "ranking_metric" in d
        assert "best_model" in d
        assert "ranking" in d


# ---------------------------------------------------------------------------
# Method registry tests
# ---------------------------------------------------------------------------

class TestMethodRegistry:
    def test_default_methods_registered(self):
        methods = MethodRegistry.get_method_names("regression")
        assert "LinearRegression" in methods
        assert "Ridge(alpha=1.0)" in methods

    def test_register_custom_method(self):
        from statgpu.linear_model import ElasticNet
        MethodRegistry.register(
            "regression", "TestElasticNet",
            factory=lambda: ElasticNet(alpha=0.5),
            priority=5,
        )
        methods = MethodRegistry.get_method_names("regression")
        assert "TestElasticNet" in methods
        # Cleanup
        MethodRegistry._registry["regression"] = [
            m for m in MethodRegistry._registry["regression"]
            if m["name"] != "TestElasticNet"
        ]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline_csv_format(self, regression_data, tmp_path):
        X, y = regression_data
        import csv
        path = str(tmp_path / "data.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x1", "x2", "x3", "x4", "x5", "y"])
            for i in range(len(y)):
                writer.writerow(list(X[i]) + [y[i]])

        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze_csv(
            path, target="y"
        )
        assert result.profile.task_type == "regression"
        assert len(result.models) >= 1

    def test_backward_compatible_api(self, regression_data):
        X, y = regression_data
        agent = StatGPUAnalysisAgent(device="auto")
        result = agent.analyze(X=X, y=y, target="outcome", feature_names=["a", "b", "c", "d", "e"])
        assert isinstance(result, AnalysisResult)
        assert isinstance(result.profile, DataProfile)
        assert isinstance(result.plan, AnalysisPlan)
        assert all(isinstance(m, ModelResult) for m in result.models)


# ---------------------------------------------------------------------------
# Survival tests
# ---------------------------------------------------------------------------

class TestSurvival:
    def test_coxph_basic(self, rng):
        X = rng.normal(size=(100, 5))
        time = rng.exponential(size=100)
        event = rng.integers(0, 2, size=100).astype(float)
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(
            X=X, time=time, event=event, task="survival"
        )
        assert result.profile.task_type == "survival"
        assert any("Cox" in m.name for m in result.models)
        cox = [m for m in result.models if "Cox" in m.name][0]
        assert cox.error is None
        assert "c_index" in cox.metrics

    def test_coxph_low_events_pruned(self, rng):
        """Low events should trigger pruning to penalized CoxPH."""
        X = rng.normal(size=(50, 20))
        time = rng.exponential(size=50)
        event = np.zeros(50)
        event[:3] = 1.0  # Only 3 events, p=20
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(
            X=X, time=time, event=event, task="survival"
        )
        model_names = [m.name for m in result.models]
        assert any("penalized" in n for n in model_names)


# ---------------------------------------------------------------------------
# Poisson tests
# ---------------------------------------------------------------------------

class TestPoisson:
    def test_poisson_basic(self, rng):
        X = rng.normal(size=(200, 3))
        eta = X[:, 0] * 0.5
        y = rng.poisson(lam=np.exp(np.clip(eta, -5, 5)))
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert result.profile.task_type == "poisson"
        poisson = [m for m in result.models if "Poisson" in m.name]
        assert len(poisson) >= 1
        assert poisson[0].error is None


# ---------------------------------------------------------------------------
# Cross-validation tests
# ---------------------------------------------------------------------------

class TestCrossValidation:
    def test_cv_regression(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=5).analyze(X=X, y=y)
        m = result.models[0]
        assert m.cv_results is not None
        assert isinstance(m.cv_results, CVResult)
        assert m.cv_results.n_folds == 5
        assert m.cv_results.metric_name == "r2"
        assert not np.isnan(m.cv_results.mean)
        assert len(m.cv_results.fold_scores) == 5

    def test_cv_classification(self, rng):
        X = rng.normal(size=(200, 5))
        y = (X[:, 0] > 0).astype(float)
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=3).analyze(X=X, y=y)
        logistic = [m for m in result.models if "Logistic" in m.name][0]
        assert logistic.cv_results is not None
        assert logistic.cv_results.n_folds == 3

    def test_cv_disabled(self, regression_data):
        X, y = regression_data
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert result.models[0].cv_results is None


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------

class TestMemory:
    def test_memory_store_record_and_recall(self, tmp_path):
        from statgpu.agent import MemoryStore
        path = str(tmp_path / "memory.json")
        store = MemoryStore(path)
        store.record_analysis(
            n_samples=100, n_features=5, task_type="regression",
            successful=["LinearRegression", "Ridge(alpha=1.0)"],
            failed=[],
        )
        assert len(store.memories) == 1
        mem = store.recall("n=100_p=5_task=regression")
        assert mem is not None
        assert "LinearRegression" in mem.successful_methods

    def test_memory_store_recall_similar(self, tmp_path):
        from statgpu.agent import MemoryStore
        path = str(tmp_path / "memory.json")
        store = MemoryStore(path)
        store.record_analysis(
            n_samples=200, n_features=10, task_type="regression",
            successful=["LinearRegression"], failed=[],
        )
        mem = store.recall_similar(150, 8, "regression")
        assert mem is not None

    def test_memory_store_clear(self, tmp_path):
        from statgpu.agent import MemoryStore
        path = str(tmp_path / "memory.json")
        store = MemoryStore(path)
        store.record_analysis(n_samples=100, n_features=5, task_type="regression",
                              successful=["LinearRegression"], failed=[])
        store.clear()
        assert len(store.memories) == 0


# ---------------------------------------------------------------------------
# Pipeline serialization tests
# ---------------------------------------------------------------------------

class TestPipelineSerialization:
    def test_save_load_pipeline(self, regression_data, tmp_path):
        X, y = regression_data
        agent = StatGPUAnalysisAgent(device="cpu", cv_folds=0)
        result = agent.analyze(X=X, y=y)
        path = str(tmp_path / "pipeline.pkl")
        agent.save_pipeline(path, result)
        state = StatGPUAnalysisAgent.load_pipeline(path)
        assert "config" in state
        assert "models" in state
        assert len(state["models"]) >= 1
        assert state["config"]["device"] == "cpu"


# ---------------------------------------------------------------------------
# Self-correction tests
# ---------------------------------------------------------------------------

class TestSelfCorrection:
    def test_rank_deficient_correction(self, rng):
        """Rank-deficient data should trigger Ridge correction."""
        X = rng.normal(size=(50, 5))
        X = np.column_stack([X, X[:, 0]])  # Duplicate column
        y = rng.normal(size=50)
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        # Should still produce results (via Ridge correction)
        assert any(m.error is None for m in result.models)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_feature(self, rng):
        X = rng.normal(size=(100, 1))
        y = X[:, 0] + rng.normal(size=100) * 0.1
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert result.profile.n_features == 1
        assert any(m.error is None for m in result.models)

    def test_small_sample(self, rng):
        X = rng.normal(size=(15, 3))
        y = X[:, 0] + rng.normal(size=15) * 0.1
        result = StatGPUAnalysisAgent(device="cpu", cv_folds=0).analyze(X=X, y=y)
        assert any("below" in w.lower() for w in result.warnings)
