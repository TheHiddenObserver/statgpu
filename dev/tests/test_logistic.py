"""Tests for logistic regression."""

import numpy as np
import pytest

from statgpu.linear_model import LogisticRegression
from statgpu._config import set_device, Device
from statgpu.evaluation import (
    binary_average_precision_score,
    binary_precision_recall_curve,
    evaluate_binary_classification,
)


class TestLogisticRegression:
    """Test LogisticRegression class."""
    
    def test_basic_fit_cpu(self):
        """Test basic fitting on CPU."""
        set_device('cpu')
        
        # Generate simple binary classification data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        # Check that coefficients are reasonable
        assert model.coef_ is not None
        assert len(model.coef_) == 2
        assert model.intercept_ is not None
        assert model.n_iter_ <= 100

    def test_invalid_hac_maxlags_raises(self):
        with pytest.raises(ValueError):
            LogisticRegression(device="cpu", cov_type="hac", hac_maxlags=-1)

    @pytest.mark.parametrize("cov_type", ["hc2", "hc3", "hac"])
    def test_extended_cov_types_cpu(self, cov_type):
        """Extended robust covariance types should run and produce finite inference."""
        set_device('cpu')

        rng = np.random.default_rng(123)
        X = rng.normal(size=(1200, 6))
        beta = rng.normal(scale=0.7, size=6)
        logits = X @ beta + 0.25
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.random(1200) < p).astype(int)

        kwargs = {"hac_maxlags": 4} if cov_type == "hac" else {}
        model = LogisticRegression(
            device='cpu',
            C=1e10,
            max_iter=200,
            cov_type=cov_type,
            compute_inference=True,
            **kwargs,
        )
        model.fit(X, y)

        assert model._bse is not None
        assert np.all(np.isfinite(model._bse))
        assert np.all(model._bse > 0)
        assert np.all(np.isfinite(model._pvalues))
        assert np.all((model._pvalues >= 0) & (model._pvalues <= 1))
    
    def test_fit_with_intercept(self):
        """Test fitting with intercept."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(200, 3)
        # True model with intercept
        true_coef = np.array([1.5, -2.0, 3.0])
        true_intercept = 0.5
        z = X @ true_coef + true_intercept
        y = (z > 0).astype(int)
        
        model = LogisticRegression(fit_intercept=True, device='cpu', max_iter=100)
        model.fit(X, y)
        
        # Check coefficients have correct signs
        assert np.sign(model.coef_[0]) == np.sign(true_coef[0])
        assert np.sign(model.coef_[1]) == np.sign(true_coef[1])
        assert np.sign(model.coef_[2]) == np.sign(true_coef[2])
    
    def test_fit_without_intercept(self):
        """Test fitting without intercept."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(fit_intercept=False, device='cpu', max_iter=100)
        model.fit(X, y)
        
        assert model.coef_ is not None
        assert model.intercept_ == 0.0
    
    def test_predict_proba(self):
        """Test probability predictions."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        proba = model.predict_proba(X)
        assert proba.shape == (50, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)
    
    def test_predict(self):
        """Test class predictions.""" 
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert y_pred.shape == (50,)
        assert np.all(np.isin(y_pred, [0, 1]))

    def test_predict_with_threshold(self):
        """Test class predictions with a custom threshold."""
        set_device('cpu')

        np.random.seed(42)
        X = np.random.randn(120, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)

        y_pred_lo = model.predict_with_threshold(X, threshold=0.2)
        y_pred_hi = model.predict_with_threshold(X, threshold=0.8)

        assert y_pred_lo.shape == (120,)
        assert y_pred_hi.shape == (120,)
        # Lower threshold should not reduce the number of positives.
        assert np.sum(y_pred_lo == 1) >= np.sum(y_pred_hi == 1)

        with pytest.raises(ValueError):
            model.predict_with_threshold(X, threshold=1.5)
    
    def test_score(self):
        """Test accuracy score."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        score = model.score(X, y)
        assert 0 <= score <= 1
        assert score > 0.7  # Should be reasonably accurate

    def test_confusion_matrix_and_classification_table(self):
        """Test confusion matrix and derived classification table."""
        set_device('cpu')

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)

        cm = model.confusion_matrix(X, y)
        table = model.classification_table(X, y)

        assert cm.shape == (2, 2)
        assert int(np.sum(cm)) == len(y)
        assert table['tn'] == int(cm[0, 0])
        assert table['fp'] == int(cm[0, 1])
        assert table['fn'] == int(cm[1, 0])
        assert table['tp'] == int(cm[1, 1])
        assert 0 <= table['accuracy'] <= 1
        assert 0 <= table['precision'] <= 1
        assert 0 <= table['recall'] <= 1
        assert 0 <= table['f1'] <= 1

    def test_roc_curve_and_auc(self):
        """Test ROC curve arrays and ROC-AUC."""
        set_device('cpu')

        np.random.seed(42)
        X = np.random.randn(150, 3)
        y = (1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.6 * X[:, 2] > 0).astype(int)

        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)

        fpr, tpr, thresholds = model.roc_curve(X, y)
        auc = model.roc_auc_score(X, y)

        assert fpr.ndim == 1
        assert tpr.ndim == 1
        assert thresholds.ndim == 1
        assert len(fpr) == len(tpr) == len(thresholds)
        assert fpr[0] == 0.0
        assert tpr[0] == 0.0
        assert np.isinf(thresholds[0])
        assert np.all(np.diff(fpr) >= 0)
        assert np.all(np.diff(tpr) >= 0)
        assert 0.0 <= auc <= 1.0
        assert auc > 0.8
        assert model.auc is not None

    def test_precision_recall_curve_and_average_precision(self):
        """Test precision-recall arrays and average precision."""
        set_device('cpu')

        np.random.seed(42)
        X = np.random.randn(180, 3)
        y = (1.1 * X[:, 0] - 0.9 * X[:, 1] + 0.7 * X[:, 2] > 0).astype(int)

        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)

        precision, recall, thresholds = model.precision_recall_curve(X, y)
        ap = model.average_precision_score(X, y)

        assert precision.ndim == 1
        assert recall.ndim == 1
        assert thresholds.ndim == 1
        assert len(precision) == len(recall) == len(thresholds)
        assert np.isinf(thresholds[0])
        assert np.all(precision >= 0) and np.all(precision <= 1)
        assert np.all(recall >= 0) and np.all(recall <= 1)
        assert np.all(np.diff(recall) >= 0)
        assert 0.0 <= ap <= 1.0
        assert ap > 0.8
        assert model.average_precision is not None

    def test_evaluate_classification_batch_cpu(self):
        """Test one-shot batch evaluation API on CPU."""
        set_device('cpu')

        np.random.seed(42)
        X = np.random.randn(220, 4)
        y = (1.0 * X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2] > 0).astype(int)

        model = LogisticRegression(device='cpu', max_iter=120)
        model.fit(X, y)

        out = model.evaluate_classification(X, y, threshold=0.5, include_curves=True)

        assert out['threshold'] == 0.5
        assert np.array_equal(out['confusion_matrix'], model.confusion_matrix(X, y))
        assert np.isclose(out['roc_auc'], model.roc_auc_score(X, y))
        assert np.isclose(out['average_precision'], model.average_precision_score(X, y))
        assert 'roc_curve' in out
        assert 'precision_recall_curve' in out
        assert out['roc_curve']['fpr'].ndim == 1
        assert out['roc_curve']['tpr'].ndim == 1
        assert out['precision_recall_curve']['precision'].ndim == 1
        assert out['precision_recall_curve']['recall'].ndim == 1

        out_no_curves = model.evaluate_classification(X, y, include_curves=False)
        assert 'roc_curve' not in out_no_curves
        assert 'precision_recall_curve' not in out_no_curves

    def test_external_probability_evaluation_module_cpu(self):
        """Test standalone evaluation module with external probabilities on CPU."""
        set_device('cpu')

        np.random.seed(42)
        X = np.random.randn(240, 4)
        y = (1.1 * X[:, 0] - 0.9 * X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)

        model = LogisticRegression(device='cpu', max_iter=120)
        model.fit(X, y)

        y_score = model.predict_proba(X)[:, 1]
        out_module = evaluate_binary_classification(
            y,
            y_score,
            threshold=0.5,
            include_curves=True,
            backend='numpy',
        )
        out_model = model.evaluate_classification(X, y, threshold=0.5, include_curves=True)

        assert np.array_equal(out_module['confusion_matrix'], out_model['confusion_matrix'])
        assert np.isclose(out_module['roc_auc'], out_model['roc_auc'])
        assert np.isclose(out_module['average_precision'], out_model['average_precision'])
        assert np.allclose(out_module['roc_curve']['fpr'], out_model['roc_curve']['fpr'])
        assert np.allclose(out_module['roc_curve']['tpr'], out_model['roc_curve']['tpr'])
        assert np.allclose(
            out_module['precision_recall_curve']['precision'],
            out_model['precision_recall_curve']['precision'],
        )
        assert np.allclose(
            out_module['precision_recall_curve']['recall'],
            out_model['precision_recall_curve']['recall'],
        )

    def test_plot_roc_curve_returns_axes(self):
        """Test ROC plotting API returns a matplotlib Axes when available."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        set_device('cpu')
        np.random.seed(42)
        X = np.random.randn(120, 3)
        y = (1.0 * X[:, 0] - 0.7 * X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)

        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)

        ax = model.plot_roc_curve(X, y)
        assert ax is not None
        assert len(ax.lines) >= 1
        plt.close(ax.figure)

    def test_plot_precision_recall_curve_returns_axes(self):
        """Test precision-recall plotting API returns a matplotlib Axes when available."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        set_device('cpu')
        np.random.seed(42)
        X = np.random.randn(120, 3)
        y = (1.0 * X[:, 0] - 0.7 * X[:, 1] + 0.5 * X[:, 2] > 0).astype(int)

        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)

        ax = model.plot_precision_recall_curve(X, y)
        assert ax is not None
        assert len(ax.lines) >= 1
        plt.close(ax.figure)
    
    def test_regularization(self):
        """Test L2 regularization."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Strong regularization
        model_strong = LogisticRegression(C=0.01, device='cpu', max_iter=100)
        model_strong.fit(X, y)
        
        # Weak regularization
        model_weak = LogisticRegression(C=1000, device='cpu', max_iter=100)
        model_weak.fit(X, y)
        
        # Strong regularization should produce smaller coefficients
        assert np.linalg.norm(model_strong.coef_) < np.linalg.norm(model_weak.coef_)
    
    def test_stats(self):
        """Test statistical outputs."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        # Check that stats are computed
        assert model.loglikelihood is not None
        assert model.loglikelihood_null is not None
        assert model.aic is not None
        assert model.bic is not None
        assert model.pseudo_rsquared is not None
        assert model.accuracy is not None
        assert model.precision is not None
        assert model.recall is not None
        assert model.f1 is not None
        assert model.average_precision is not None
        
        # Check ranges
        assert model.pseudo_rsquared >= 0 and model.pseudo_rsquared <= 1
        assert model.aic < model.bic  # BIC penalizes more
    
    def test_summary(self):
        """Test summary output."""
        set_device('cpu')
        
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cpu', max_iter=100)
        model.fit(X, y)
        
        # Just make sure it doesn't raise
        model.summary()
    
    def test_not_fitted_error(self):
        """Test error when predicting before fitting."""
        model = LogisticRegression(device='cpu')
        
        with pytest.raises(RuntimeError):
            model.predict(np.array([[1, 2]]))
        
        with pytest.raises(RuntimeError):
            model.predict_proba(np.array([[1, 2]]))


class TestGPU:
    """GPU-specific tests (only run if CUDA available)."""
    
    @pytest.mark.skipif(
        not LogisticRegression(device='auto')._get_compute_device() == Device.CUDA,
        reason="CUDA not available"
    )
    def test_gpu_fit(self):
        """Test fitting on GPU."""
        set_device('cuda')
        
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(device='cuda', max_iter=100)
        model.fit(X, y)
        
        assert model.coef_ is not None
        assert len(model.coef_) == 5
    
    @pytest.mark.skipif(
        not LogisticRegression(device='auto')._get_compute_device() == Device.CUDA,
        reason="CUDA not available"
    )
    def test_gpu_matches_cpu(self):
        """Test GPU and CPU produce same results."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float64)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # CPU model
        model_cpu = LogisticRegression(device='cpu', max_iter=100)
        model_cpu.fit(X, y)
        
        # GPU model
        model_gpu = LogisticRegression(device='cuda', max_iter=100)
        model_gpu.fit(X, y)
        
        # Compare coefficients
        assert np.allclose(model_cpu.coef_, model_gpu.coef_, rtol=1e-3)
        assert np.allclose(model_cpu.intercept_, model_gpu.intercept_, rtol=1e-3)

    @pytest.mark.skipif(
        not LogisticRegression(device='auto')._get_compute_device() == Device.CUDA,
        reason="CUDA not available"
    )
    def test_gpu_evaluation_outputs_are_gpu_backed(self):
        """Test evaluation APIs return GPU-backed values on CUDA device."""
        cp = pytest.importorskip("cupy")

        set_device('cuda')
        np.random.seed(42)
        X = np.random.randn(300, 6).astype(np.float32)
        y = (X[:, 0] - 0.8 * X[:, 1] + 0.6 * X[:, 2] > 0).astype(int)

        model = LogisticRegression(device='cuda', max_iter=120)
        model.fit(X, y)

        cm = model.confusion_matrix(X, y)
        table = model.classification_table(X, y)
        fpr, tpr, thresholds = model.roc_curve(X, y)
        precision, recall, pr_thresholds = model.precision_recall_curve(X, y)
        auc = model.roc_auc_score(X, y)
        ap = model.average_precision_score(X, y)

        assert isinstance(cm, cp.ndarray)
        assert isinstance(fpr, cp.ndarray)
        assert isinstance(tpr, cp.ndarray)
        assert isinstance(thresholds, cp.ndarray)
        assert isinstance(precision, cp.ndarray)
        assert isinstance(recall, cp.ndarray)
        assert isinstance(pr_thresholds, cp.ndarray)
        assert cp.asarray(auc).shape == ()
        assert cp.asarray(ap).shape == ()
        assert cp.asarray(table['accuracy']).shape == ()
        assert int(cp.asnumpy(cm.sum())) == len(y)

    @pytest.mark.skipif(
        not LogisticRegression(device='auto')._get_compute_device() == Device.CUDA,
        reason="CUDA not available"
    )
    def test_gpu_batch_evaluation_outputs_are_gpu_backed(self):
        """Test one-shot batch evaluation API returns GPU-backed outputs on CUDA."""
        cp = pytest.importorskip("cupy")

        set_device('cuda')
        np.random.seed(42)
        X = np.random.randn(320, 6).astype(np.float32)
        y = (X[:, 0] - 0.8 * X[:, 1] + 0.6 * X[:, 2] > 0).astype(int)

        model = LogisticRegression(device='cuda', max_iter=120)
        model.fit(X, y)

        out = model.evaluate_classification(X, y, include_curves=True)
        assert out['threshold'] == 0.5
        assert isinstance(out['confusion_matrix'], cp.ndarray)
        assert cp.asarray(out['roc_auc']).shape == ()
        assert cp.asarray(out['average_precision']).shape == ()
        assert isinstance(out['roc_curve']['fpr'], cp.ndarray)
        assert isinstance(out['roc_curve']['tpr'], cp.ndarray)
        assert isinstance(out['precision_recall_curve']['precision'], cp.ndarray)
        assert isinstance(out['precision_recall_curve']['recall'], cp.ndarray)

        cm_single = model.confusion_matrix(X, y)
        assert bool(cp.all(out['confusion_matrix'] == cm_single).item())

        out_no_curves = model.evaluate_classification(X, y, include_curves=False)
        assert 'roc_curve' not in out_no_curves
        assert 'precision_recall_curve' not in out_no_curves

    @pytest.mark.skipif(
        not LogisticRegression(device='auto')._get_compute_device() == Device.CUDA,
        reason="CUDA not available"
    )
    def test_gpu_external_probability_evaluation_module(self):
        """Test standalone evaluation module with external probabilities on GPU."""
        cp = pytest.importorskip("cupy")

        set_device('cuda')
        np.random.seed(42)
        X = np.random.randn(280, 5).astype(np.float32)
        y = (X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2] > 0).astype(int)

        model = LogisticRegression(device='cuda', max_iter=120)
        model.fit(X, y)

        y_true_gpu = cp.asarray(y)
        y_score_gpu = model.predict_proba(X)[:, 1]
        out_module = evaluate_binary_classification(
            y_true_gpu,
            y_score_gpu,
            threshold=0.5,
            include_curves=True,
            backend='cupy',
        )
        out_model = model.evaluate_classification(X, y, threshold=0.5, include_curves=True)

        assert isinstance(out_module['confusion_matrix'], cp.ndarray)
        assert bool(cp.all(out_module['confusion_matrix'] == out_model['confusion_matrix']).item())
        assert bool(cp.isclose(out_module['roc_auc'], out_model['roc_auc']).item())
        assert bool(cp.isclose(out_module['average_precision'], out_model['average_precision']).item())
        assert isinstance(out_module['roc_curve']['fpr'], cp.ndarray)
        assert isinstance(out_module['precision_recall_curve']['precision'], cp.ndarray)


class TestEvaluationModuleBackends:
    """Standalone evaluation module backend tests."""

    def test_torch_precision_recall_curve_matches_numpy(self):
        """Torch PR curve/AP should align with NumPy semantics."""
        torch = pytest.importorskip("torch")

        y_true_np = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
        y_score_np = np.array([0.9, 0.9, 0.7, 0.7, 0.2, 0.2, 0.2, 0.1], dtype=np.float64)

        precision_np, recall_np, thresholds_np = binary_precision_recall_curve(
            y_true_np, y_score_np, backend="numpy"
        )
        ap_np = binary_average_precision_score(y_true_np, y_score_np, backend="numpy")

        y_true_t = torch.as_tensor(y_true_np, dtype=torch.int64)
        y_score_t = torch.as_tensor(y_score_np, dtype=torch.float64)
        precision_t, recall_t, thresholds_t = binary_precision_recall_curve(
            y_true_t, y_score_t, backend="torch"
        )
        ap_t = binary_average_precision_score(y_true_t, y_score_t, backend="torch")

        assert np.allclose(precision_t.cpu().numpy(), precision_np)
        assert np.allclose(recall_t.cpu().numpy(), recall_np)
        assert np.allclose(thresholds_t.cpu().numpy(), thresholds_np)
        assert np.isclose(float(ap_t.item()), ap_np)

    @pytest.mark.skipif(
        not LogisticRegression(device='auto')._get_compute_device() == Device.CUDA,
        reason="CUDA not available"
    )
    def test_cupy_precision_recall_curve_matches_numpy(self):
        """CuPy PR curve/AP should align with NumPy semantics."""
        cp = pytest.importorskip("cupy")

        y_true_np = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
        y_score_np = np.array([0.9, 0.9, 0.7, 0.7, 0.2, 0.2, 0.2, 0.1], dtype=np.float64)

        precision_np, recall_np, thresholds_np = binary_precision_recall_curve(
            y_true_np, y_score_np, backend="numpy"
        )
        ap_np = binary_average_precision_score(y_true_np, y_score_np, backend="numpy")

        y_true_cp = cp.asarray(y_true_np)
        y_score_cp = cp.asarray(y_score_np)
        precision_cp, recall_cp, thresholds_cp = binary_precision_recall_curve(
            y_true_cp, y_score_cp, backend="cupy"
        )
        ap_cp = binary_average_precision_score(y_true_cp, y_score_cp, backend="cupy")

        assert np.allclose(cp.asnumpy(precision_cp), precision_np)
        assert np.allclose(cp.asnumpy(recall_cp), recall_np)
        assert np.allclose(cp.asnumpy(thresholds_cp), thresholds_np)
        assert np.isclose(float(cp.asnumpy(ap_cp)), ap_np)

    def test_torch_external_probability_evaluation(self):
        """Test torch backend for external probability one-shot evaluation."""
        torch = pytest.importorskip("torch")

        y_true = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1], dtype=torch.int64)
        y_score = torch.tensor([0.05, 0.92, 0.20, 0.81, 0.73, 0.11, 0.40, 0.88], dtype=torch.float64)

        out = evaluate_binary_classification(
            y_true,
            y_score,
            threshold=0.5,
            include_curves=True,
            backend='torch',
        )

        assert isinstance(out['confusion_matrix'], torch.Tensor)
        assert out['confusion_matrix'].shape == (2, 2)
        assert isinstance(out['roc_auc'], torch.Tensor)
        assert isinstance(out['average_precision'], torch.Tensor)
        assert 0.0 <= float(out['roc_auc'].item()) <= 1.0
        assert 0.0 <= float(out['average_precision'].item()) <= 1.0
        assert isinstance(out['roc_curve']['fpr'], torch.Tensor)
        assert isinstance(out['precision_recall_curve']['precision'], torch.Tensor)
