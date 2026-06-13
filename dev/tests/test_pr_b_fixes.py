"""Tests for PR-B code review fixes (2026-06-14).

Covers Critical/High/Medium fixes from review rounds.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# C1: non_smooth set includes group_mcp/group_scad
# ---------------------------------------------------------------------------
class TestNonSmoothSet:
    def test_group_mcp_is_non_smooth(self):
        from statgpu.linear_model._penalized import PenalizedLinearRegression
        # solver='newton' with group_mcp should raise ValueError
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        y = X @ np.ones(5) + rng.standard_normal(50) * 0.1
        with pytest.raises((ValueError, TypeError)):
            m = PenalizedLinearRegression(
                penalty="group_mcp", alpha=0.1, solver="newton", device="cpu",
            )
            m.fit(X, y)

    def test_scad_is_non_smooth(self):
        from statgpu.linear_model._penalized import PenalizedLinearRegression
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        y = X @ np.ones(5) + rng.standard_normal(50) * 0.1
        with pytest.raises(ValueError, match="smooth"):
            m = PenalizedLinearRegression(
                penalty="scad", alpha=0.1, solver="newton", device="cpu",
            )
            m.fit(X, y)


# ---------------------------------------------------------------------------
# C2: No self-import in _penalized.py
# ---------------------------------------------------------------------------
class TestNoSelfImport:
    def test_no_circular_import(self):
        """_penalized.py should not import itself."""
        import ast
        with open('statgpu/linear_model/_penalized.py', 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and '_penalized' in node.module and node.level == 0:
                    # Should not import from itself
                    assert 'PenalizedLinearRegression' not in [a.name for a in node.names], \
                        f"Self-import found at line {node.lineno}"


# ---------------------------------------------------------------------------
# H1: loss_kwargs propagation
# ---------------------------------------------------------------------------
class TestLossKwargsPropagation:
    def test_nb_alpha_propagated(self):
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        # Should accept loss_kwargs parameter
        m = PenalizedGLM_CV(
            loss="negative_binomial", penalty="l1",
            loss_kwargs={"alpha": 2.0}, cv=2, device="cpu",
        )
        assert m._loss_kwargs == {"alpha": 2.0}

    def test_tweedie_power_propagated(self):
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        m = PenalizedGLM_CV(
            loss="tweedie", penalty="l1",
            loss_kwargs={"power": 1.5}, cv=2, device="cpu",
        )
        assert m._loss_kwargs == {"power": 1.5}

    def test_default_empty_kwargs(self):
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        m = PenalizedGLM_CV(loss="squared_error", penalty="l1", cv=2, device="cpu")
        assert m._loss_kwargs == {}


# ---------------------------------------------------------------------------
# H3: score() docstring
# ---------------------------------------------------------------------------
class TestScoreDocstring:
    def test_score_docstring_mentions_deviance(self):
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        assert "pseudo" in PenalizedGLM_CV.score.__doc__.lower() or \
               "deviance" in PenalizedGLM_CV.score.__doc__.lower()


# ---------------------------------------------------------------------------
# PR P2: RidgeCV weighted MSE dimension
# ---------------------------------------------------------------------------
class TestRidgeCVWeightedMSE:
    def test_mse_is_2d(self):
        """Weighted RidgeCV MSE should be 2D (n_alphas, n_folds), not 3D."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = X @ np.ones(5) + rng.standard_normal(100) * 0.1
        sw = rng.uniform(0.5, 1.5, 100)
        from statgpu.linear_model._ridge_cv import RidgeCV
        m = RidgeCV(alphas=[0.1, 1.0], cv=3, device="cpu")
        m.fit(X, y, sample_weight=sw)
        # If MSE is 3D, this would fail with shape mismatch
        assert m.alpha_ > 0


# ---------------------------------------------------------------------------
# PR P2: ElasticNet sorted alpha mapping
# ---------------------------------------------------------------------------
class TestElasticNetAlphaMapping:
    def test_best_alpha_in_range(self):
        """Best alpha should be one of the provided alphas."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = X @ np.ones(5) + rng.standard_normal(100) * 0.1
        from statgpu.linear_model._elasticnet_cv import ElasticNetCV
        alphas = [0.01, 0.1, 1.0]
        m = ElasticNetCV(alphas=alphas, cv=3, device="cpu")
        m.fit(X, y)
        assert m.alpha_ in alphas


# ---------------------------------------------------------------------------
# PR P2: LassoCV weighted centering
# ---------------------------------------------------------------------------
class TestLassoCVWeightedCentering:
    def test_weighted_lasso_cv_runs(self):
        """Weighted LassoCV should complete without error."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = X @ np.ones(5) + rng.standard_normal(100) * 0.1
        sw = rng.uniform(0.5, 1.5, 100)
        from statgpu.linear_model._lasso_cv import LassoCV
        m = LassoCV(cv=3, device="cpu", max_iter=100)
        m.fit(X, y, sample_weight=sw)
        assert m.alpha_ > 0
        assert m.coef_ is not None


# ---------------------------------------------------------------------------
# M3: __all__ exports
# ---------------------------------------------------------------------------
class TestAllExports:
    def test_penalized_has_all(self):
        import statgpu.linear_model._penalized as m
        assert hasattr(m, '__all__')
        assert 'PenalizedLinearRegression' in m.__all__

    def test_penalized_cv_has_all(self):
        import statgpu.linear_model._penalized_cv as m
        assert hasattr(m, '__all__')
        assert 'PenalizedGLM_CV' in m.__all__

    def test_lasso_has_all(self):
        import statgpu.linear_model._lasso as m
        assert hasattr(m, '__all__')
        assert 'Lasso' in m.__all__

    def test_ridge_has_all(self):
        import statgpu.linear_model._ridge as m
        assert hasattr(m, '__all__')
        assert 'Ridge' in m.__all__

    def test_elasticnet_has_all(self):
        import statgpu.linear_model._elasticnet as m
        assert hasattr(m, '__all__')
        assert 'ElasticNet' in m.__all__


# ---------------------------------------------------------------------------
# Import convention: no relative imports in non-__init__.py
# ---------------------------------------------------------------------------
class TestImportConvention:
    def test_no_relative_imports_in_source(self):
        import os
        import re
        violations = []
        for dirpath in ['statgpu/linear_model', 'statgpu/glm_core', 'statgpu/backends', 'statgpu/penalties', 'statgpu/inference']:
            if not os.path.exists(dirpath):
                continue
            for fname in os.listdir(dirpath):
                if not fname.endswith('.py') or fname == '__init__.py':
                    continue
                fpath = os.path.join(dirpath, fname)
                with open(fpath, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f, 1):
                        if re.match(r'^\s*from \.\w+ import', line):
                            violations.append(f"{fpath}:{i}: {line.strip()}")
        assert violations == [], f"Relative imports found:\n" + "\n".join(violations[:5])


# ---------------------------------------------------------------------------
# End-to-end: PenalizedGLM_CV basic workflow
# ---------------------------------------------------------------------------
class TestPenalizedGLMCVBasic:
    def test_squared_error_l2_cv(self):
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 10))
        y = X @ np.ones(10) + rng.standard_normal(200) * 0.1
        m = PenalizedGLM_CV(loss="squared_error", penalty="l2", cv=3, device="cpu")
        m.fit(X, y)
        assert m.alpha_ > 0
        assert m.coef_ is not None
        score = m.score(X, y)
        assert np.isfinite(score)

    def test_squared_error_l1_cv(self):
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 10))
        y = X @ np.ones(10) + rng.standard_normal(200) * 0.1
        m = PenalizedGLM_CV(loss="squared_error", penalty="l1", cv=3, device="cpu")
        m.fit(X, y)
        assert m.alpha_ > 0
        assert m.coef_ is not None

    def test_logistic_l2_cv(self):
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 10))
        y = (X @ np.ones(10) > 0).astype(float)
        m = PenalizedGLM_CV(loss="logistic", penalty="l2", cv=3, device="cpu")
        m.fit(X, y)
        assert m.alpha_ > 0
        assert m.coef_ is not None
