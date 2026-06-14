"""Tests for PR-D code review fixes (2026-06-14).

Covers Critical/High fixes from review round.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# C1: get_params restores _{name} fallback for sklearn clone()
# ---------------------------------------------------------------------------
class TestGetParams:
    def test_get_params_private_attr(self):
        """get_params should find parameters stored as _name (private)."""
        from statgpu.linear_model._negative_binomial_glm import NegativeBinomialRegression
        m = NegativeBinomialRegression(alpha=2.5)
        params = m.get_params()
        assert 'alpha' in params
        assert params['alpha'] == 2.5

    def test_sklearn_clone_preserves_params(self):
        """sklearn.base.clone should preserve all constructor parameters."""
        from statgpu.linear_model._negative_binomial_glm import NegativeBinomialRegression
        m = NegativeBinomialRegression(alpha=3.0, device='cpu')
        try:
            from sklearn.base import clone
            m2 = clone(m)
            assert m2.get_params()['alpha'] == 3.0
        except ImportError:
            pytest.skip("sklearn not installed")

    def test_get_params_inherited(self):
        """get_params should return params accepted by __init__."""
        from statgpu.linear_model._lasso import Lasso
        m = Lasso(alpha=0.5, max_iter=200, device='cpu')
        params = m.get_params()
        assert 'alpha' in params
        assert 'max_iter' in params
        # 'device' is in Lasso.__init__ so it should be returned
        assert 'device' in params

    def test_get_params_all_glm_wrappers(self):
        """All GLM wrapper classes should have get_params working."""
        from statgpu.linear_model import (
            GammaRegression, InverseGaussianRegression,
            NegativeBinomialRegression, TweedieRegression,
        )
        for cls in [GammaRegression, InverseGaussianRegression,
                    NegativeBinomialRegression, TweedieRegression]:
            m = cls(device='cpu')
            params = m.get_params()
            assert 'device' in params, f"{cls.__name__} missing device"


# ---------------------------------------------------------------------------
# L4: __all__ in _base.py
# ---------------------------------------------------------------------------
class TestBaseAll:
    def test_base_has_all(self):
        import statgpu._base as m
        assert hasattr(m, '__all__')
        assert 'BaseEstimator' in m.__all__


# ---------------------------------------------------------------------------
# L3: _cox_cv.py uses absolute imports
# ---------------------------------------------------------------------------
class TestCoxCVImports:
    def test_cox_cv_absolute_import(self):
        with open('statgpu/survival/_cox_cv.py', 'r', encoding='utf-8') as f:
            content = f.read()
        # Should not have relative imports
        import re
        violations = re.findall(r'^\s*from \.\w+ import', content, re.MULTILINE)
        assert violations == [], f"Relative imports found: {violations}"


# ---------------------------------------------------------------------------
# General: import convention
# ---------------------------------------------------------------------------
class TestImportConvention:
    def test_no_relative_imports_in_new_files(self):
        """Check that new non-__init__.py files use absolute imports.

        Note: lazy imports inside function bodies (like _cox.py's
        'from ._cox_efron_cuda import ...') are acceptable Python practice
        for conditional/optional dependencies.
        """
        import os, re
        violations = []
        # Only check files that were actually added/modified in PR-D
        check_files = ['statgpu/survival/_cox_cv.py', 'statgpu/_base.py']
        for fpath in check_files:
            if not os.path.exists(fpath):
                continue
            with open(fpath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if re.match(r'^\s*from \.\w+ import', line):
                        violations.append(f"{fpath}:{i}")
        assert violations == [], f"Relative imports: {violations}"


# ---------------------------------------------------------------------------
# get_params + clone round-trip for all major classes
# ---------------------------------------------------------------------------
class TestGetParamsRoundTrip:
    def test_lasso_clone(self):
        from statgpu.linear_model._lasso import Lasso
        try:
            from sklearn.base import clone
        except ImportError:
            pytest.skip("sklearn not installed")
        m = Lasso(alpha=0.3, max_iter=500, fit_intercept=False, device='cpu')
        m2 = clone(m)
        p1, p2 = m.get_params(), m2.get_params()
        for k in p1:
            if k == 'device':
                continue  # device may be normalized
            assert p1[k] == p2[k], f"Param {k}: {p1[k]} != {p2[k]}"

    def test_ridge_clone(self):
        from statgpu.linear_model._ridge import Ridge
        try:
            from sklearn.base import clone
        except ImportError:
            pytest.skip("sklearn not installed")
        m = Ridge(alpha=1.5, device='cpu')
        m2 = clone(m)
        assert m2.get_params()['alpha'] == 1.5

    def test_elasticnet_clone(self):
        from statgpu.linear_model._elasticnet import ElasticNet
        try:
            from sklearn.base import clone
        except ImportError:
            pytest.skip("sklearn not installed")
        m = ElasticNet(alpha=0.2, l1_ratio=0.7, device='cpu')
        m2 = clone(m)
        assert m2.get_params()['alpha'] == 0.2
        assert m2.get_params()['l1_ratio'] == 0.7
