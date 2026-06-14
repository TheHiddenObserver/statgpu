"""Tests for PR-C code review fixes (2026-06-14).

Covers Critical/High fixes from 5 review rounds.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# C1: Panel RE between RSS uses unique group means
# ---------------------------------------------------------------------------
class TestRandomEffectsGroupMeans:
    def test_re_between_estimation_correct(self):
        """Between estimation should use unique group means, not per-obs means."""
        from statgpu.panel._random_effects import RandomEffects
        rng = np.random.default_rng(42)
        n, k = 60, 3
        entity_ids = np.repeat(np.arange(10), 6)  # 10 entities, 6 obs each
        X = rng.standard_normal((n, k))
        beta = np.array([1.0, -1.0, 0.5])
        y = X @ beta + rng.standard_normal(n) * 0.1

        m = RandomEffects(device='cpu')
        m.fit(y, X, entity_ids=entity_ids)
        # Coefficients should be close to true beta
        assert np.allclose(m.coef_, beta, atol=0.3)


# ---------------------------------------------------------------------------
# C2: Panel FE two-way predict subtracts grand mean
# ---------------------------------------------------------------------------
class TestTwoWayFEPredict:
    def test_two_way_fe_no_double_count(self):
        """Two-way FE predict should not double-count grand mean."""
        from statgpu.panel._fixed_effects import PanelOLS
        rng = np.random.default_rng(42)
        n = 100
        entity_ids = np.repeat(np.arange(10), 10)
        time_ids = np.tile(np.arange(10), 10)
        X = rng.standard_normal((n, 3))
        beta = np.array([1.0, -1.0, 0.5])
        y = X @ beta + rng.standard_normal(n) * 0.1

        m = PanelOLS(device='cpu', entity_effects=True, time_effects=True)
        m.fit(y, X, entity_ids=entity_ids, time_ids=time_ids)
        y_pred = m.predict(X, entity_ids=entity_ids, time_ids=time_ids)
        # Predictions should be reasonable (not NaN/Inf)
        assert np.all(np.isfinite(y_pred))
        # R² should be positive
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5


# ---------------------------------------------------------------------------
# C3: Covariance Device enum uses .value
# ---------------------------------------------------------------------------
class TestCovarianceDevice:
    def test_device_enum_value(self):
        """Device enum .value should work for torch detection."""
        from statgpu._config import Device
        assert Device.TORCH.value == "torch"
        assert Device.CUDA.value == "cuda"
        assert Device.CPU.value == "cpu"


# ---------------------------------------------------------------------------
# C5: KernelRidge score() ravels y_arr and y_pred
# ---------------------------------------------------------------------------
class TestKernelRidgeScore:
    def test_score_2d_input(self):
        """score() should work correctly when y is 2D (n,1)."""
        from statgpu.nonparametric.kernel_methods._krr import KernelRidge
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = X @ np.ones(5) + rng.standard_normal(100) * 0.1

        m = KernelRidge(alpha=1.0, kernel='rbf', device='cpu')
        m.fit(X, y)
        score_1d = m.score(X, y)
        score_2d = m.score(X, y.reshape(-1, 1))
        # Both should give the same result
        assert abs(score_1d - score_2d) < 1e-10
        assert score_1d > 0.5  # reasonable fit


# ---------------------------------------------------------------------------
# C6: GAM uses self.device (not self._device)
# ---------------------------------------------------------------------------
class TestGAMDevice:
    def test_gam_cpu_fit(self):
        """GAM should fit on CPU without errors."""
        from statgpu.semiparametric._gam import GAM
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        y = np.sin(X[:, 0]) + X[:, 1] + rng.standard_normal(200) * 0.1

        m = GAM(n_splines=10, device='cpu')
        m.fit(X, y)
        assert m.coef_ is not None
        assert m.edf_ is not None
        assert isinstance(m.edf_, float)  # not a torch tensor


# ---------------------------------------------------------------------------
# C7: AgglomerativeClustering uses float() instead of backend.item()
# ---------------------------------------------------------------------------
class TestAgglomerativeClustering:
    def test_agglomerative_cpu(self):
        """AgglomerativeClustering should work on CPU."""
        from statgpu.unsupervised import AgglomerativeClustering
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))

        m = AgglomerativeClustering(n_clusters=3)
        m.fit(X)
        assert m.labels_ is not None
        assert len(m.labels_) == 50


# ---------------------------------------------------------------------------
# C8: MiniBatchNMF uses xp.linalg.norm()
# ---------------------------------------------------------------------------
class TestMiniBatchNMF:
    def test_minibatch_nmf_cpu(self):
        """MiniBatchNMF should work on CPU."""
        from statgpu.unsupervised import MiniBatchNMF
        rng = np.random.default_rng(42)
        X = np.abs(rng.standard_normal((100, 5)))

        m = MiniBatchNMF(n_components=2, max_iter=50, device='cpu')
        m.fit(X)
        assert m.components_ is not None
        assert m.components_.shape == (2, 5)


# ---------------------------------------------------------------------------
# C9: _solver_legacy.py is dead code (should not crash import)
# ---------------------------------------------------------------------------
class TestSolverLegacy:
    def test_import_does_not_crash(self):
        """statgpu should import without errors even with legacy files."""
        import statgpu
        assert statgpu.__version__ is not None


# ---------------------------------------------------------------------------
# H2: Knockpy placeholders raise NotImplementedError
# ---------------------------------------------------------------------------
class TestKnockoffPlaceholders:
    def test_placeholder_raises(self):
        """Knockpy sampler placeholders should raise NotImplementedError."""
        from statgpu.feature_selection._knockoff import knockpy_gaussian_mvr_sampler
        with pytest.raises(NotImplementedError):
            knockpy_gaussian_mvr_sampler(np.random.randn(100, 5))


# ---------------------------------------------------------------------------
# H17: LedoitWolf/OAS torch device placement
# ---------------------------------------------------------------------------
class TestShrinkageDevice:
    def test_ledoitwolf_cpu(self):
        """LedoitWolf should work on CPU."""
        from statgpu.covariance import LedoitWolf
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))

        m = LedoitWolf(device='cpu')
        m.fit(X)
        assert m.covariance_ is not None
        assert m.shrinkage_ >= 0


# ---------------------------------------------------------------------------
# Import convention: no relative imports in non-__init__.py
# ---------------------------------------------------------------------------
class TestImportConvention:
    def test_no_relative_imports(self):
        import os
        import re
        violations = []
        for dirpath in ['statgpu/anova', 'statgpu/covariance', 'statgpu/panel',
                        'statgpu/semiparametric', 'statgpu/nonparametric/kernel_methods',
                        'statgpu/nonparametric/kernel_smoothing', 'statgpu/nonparametric/splines',
                        'statgpu/unsupervised']:
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
# __all__ exports
# ---------------------------------------------------------------------------
class TestAllExports:
    def test_anova_has_all(self):
        import statgpu.anova._oneway as m
        assert hasattr(m, '__all__')

    def test_covariance_has_all(self):
        import statgpu.covariance._empirical as m
        assert hasattr(m, '__all__')

    def test_panel_has_all(self):
        import statgpu.panel._fixed_effects as m
        assert hasattr(m, '__all__')

    def test_gam_has_all(self):
        import statgpu.semiparametric._gam as m
        assert hasattr(m, '__all__')
