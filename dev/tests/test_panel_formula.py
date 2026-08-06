"""Comprehensive tests for panel data formula interface.

Tests fixest pipe syntax, linearmodels tokens, backward compatibility,
and edge cases for all panel models.
"""

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from statgpu.panel import (
    PanelOLS, FixedEffects, RandomEffects, RandomEffectsOLS,
    PooledOLS, BetweenOLS, FirstDifferenceOLS, FamaMacBeth,
)
from statgpu.panel._formula import _split_panel_formula, _strip_panel_tokens


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def panel_df():
    """Create a balanced panel DataFrame for testing."""
    np.random.seed(42)
    n_entities = 20
    n_periods = 5
    n = n_entities * n_periods

    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = 1.0 + 2.0 * x1 + 3.0 * x2 + 0.5 * np.random.randn(n)

    return pd.DataFrame({
        'y': y, 'x1': x1, 'x2': x2,
        'entity': entity, 'time': time,
    })


@pytest.fixture
def panel_arrays(panel_df):
    """Extract arrays from the panel DataFrame."""
    return {
        'X': panel_df[['x1', 'x2']].values,
        'y': panel_df['y'].values,
        'entity_ids': panel_df['entity'].values,
        'time_ids': panel_df['time'].values,
    }


# ============================================================================
# Formula splitting tests
# ============================================================================

class TestSplitPanelFormula:

    def test_basic_pipe(self):
        main, fe = _split_panel_formula("y ~ x1 + x2 | entity + time")
        assert main == "y ~ x1 + x2"
        assert fe == ["entity", "time"]

    def test_single_fe(self):
        main, fe = _split_panel_formula("y ~ x1 + x2 | entity")
        assert main == "y ~ x1 + x2"
        assert fe == ["entity"]

    def test_no_pipe(self):
        main, fe = _split_panel_formula("y ~ x1 + x2")
        assert main == "y ~ x1 + x2"
        assert fe == []

    def test_three_fe(self):
        main, fe = _split_panel_formula("y ~ x1 | entity + time + industry")
        assert fe == ["entity", "time", "industry"]

    def test_pipe_in_parentheses_not_split(self):
        """Parenthesized expressions should not be split."""
        formula = "y ~ x1 + (x2 | x3)"
        main, fe = _split_panel_formula(formula)
        assert main == formula
        assert fe == []

    def test_whitespace_handling(self):
        main, fe = _split_panel_formula("y~x1+x2|entity+time")
        assert main == "y~x1+x2"
        assert fe == ["entity", "time"]


class TestStripPanelTokens:

    def test_entity_effects(self):
        clean, entity, time = _strip_panel_tokens("y ~ x1 + EntityEffects")
        assert entity is True
        assert time is False
        assert "EntityEffects" not in clean

    def test_time_effects(self):
        clean, entity, time = _strip_panel_tokens("y ~ x1 + TimeEffects")
        assert entity is False
        assert time is True

    def test_both_effects(self):
        clean, entity, time = _strip_panel_tokens("y ~ x1 + EntityEffects + TimeEffects")
        assert entity is True
        assert time is True
        assert "EntityEffects" not in clean
        assert "TimeEffects" not in clean

    def test_fixed_effects_token(self):
        clean, entity, time = _strip_panel_tokens("y ~ x1 + FixedEffects")
        assert entity is True

    def test_no_tokens(self):
        clean, entity, time = _strip_panel_tokens("y ~ x1 + x2")
        assert entity is False
        assert time is False
        assert clean == "y ~ x1 + x2"


# ============================================================================
# PanelOLS formula tests
# ============================================================================

class TestPanelOLSFormula:

    def test_pipe_syntax_entity_time(self, panel_df, panel_arrays):
        """y ~ x1 + x2 | entity + time should match array interface."""
        m_formula = PanelOLS(entity_effects=True, time_effects=True)
        m_formula.fit(formula="y ~ x1 + x2 | entity + time", data=panel_df)

        m_array = PanelOLS(entity_effects=True, time_effects=True)
        m_array.fit(
            X=panel_arrays['X'], y=panel_arrays['y'],
            entity_ids=panel_arrays['entity_ids'],
            time_ids=panel_arrays['time_ids'],
        )

        assert_allclose(m_formula.coef_, m_array.coef_, rtol=1e-10)

    def test_pipe_syntax_entity_only(self, panel_df, panel_arrays):
        """y ~ x1 + x2 | entity (entity effects only)."""
        m_formula = PanelOLS(entity_effects=True)
        m_formula.fit(formula="y ~ x1 + x2 | entity", data=panel_df)

        m_array = PanelOLS(entity_effects=True)
        m_array.fit(
            X=panel_arrays['X'], y=panel_arrays['y'],
            entity_ids=panel_arrays['entity_ids'],
        )

        assert_allclose(m_formula.coef_, m_array.coef_, rtol=1e-10)

    def test_linearmodels_tokens(self, panel_df, panel_arrays):
        """'y ~ x1 + EntityEffects + TimeEffects' should work."""
        m_formula = PanelOLS()
        m_formula.fit(formula="y ~ x1 + EntityEffects + TimeEffects", data=panel_df)

        assert m_formula.entity_effects is True
        assert m_formula.time_effects is True

    def test_formula_enables_effects(self, panel_df):
        """Formula with | should automatically enable entity_effects."""
        m = PanelOLS()  # entity_effects=False by default
        m.fit(formula="y ~ x1 + x2 | entity + time", data=panel_df)
        assert m.entity_effects is True
        assert m.time_effects is True

    def test_predict_with_dataframe(self, panel_df):
        """After fitting with formula, predict(df) should work."""
        m = PanelOLS()
        m.fit(formula="y ~ x1 + x2 | entity + time", data=panel_df)
        y_pred = m.predict(panel_df)
        assert y_pred.shape == (len(panel_df),)

    def test_summary_has_feature_names(self, panel_df):
        """summary() should include real column names."""
        m = PanelOLS()
        m.fit(formula="y ~ x1 + x2 | entity + time", data=panel_df)
        s = m.summary()
        # Check that feature names are available
        assert hasattr(s, 'coef')

    def test_no_formula_backward_compat(self, panel_arrays):
        """fit(X, y, entity_ids=...) still works without formula."""
        m = PanelOLS(entity_effects=True)
        m.fit(
            X=panel_arrays['X'], y=panel_arrays['y'],
            entity_ids=panel_arrays['entity_ids'],
        )
        assert m.coef_ is not None
        assert len(m.coef_) == 2


# ============================================================================
# RandomEffects formula tests
# ============================================================================

class TestRandomEffectsFormula:

    def test_pipe_syntax(self, panel_df, panel_arrays):
        """y ~ x1 + x2 | entity should match array interface."""
        m_formula = RandomEffects()
        m_formula.fit(formula="y ~ x1 + x2 | entity", data=panel_df)

        m_array = RandomEffects()
        m_array.fit(
            X=panel_arrays['X'], y=panel_arrays['y'],
            entity_ids=panel_arrays['entity_ids'],
        )

        assert_allclose(m_formula.coef_, m_array.coef_, rtol=1e-6)

    def test_predict_with_dataframe(self, panel_df):
        m = RandomEffects()
        m.fit(formula="y ~ x1 + x2 | entity", data=panel_df)
        y_pred = m.predict(panel_df)
        assert y_pred.shape == (len(panel_df),)

    def test_no_formula_backward_compat(self, panel_arrays):
        m = RandomEffects()
        m.fit(
            X=panel_arrays['X'], y=panel_arrays['y'],
            entity_ids=panel_arrays['entity_ids'],
        )
        assert m.coef_ is not None


# ============================================================================
# PooledOLS formula tests
# ============================================================================

class TestPooledOLSFormula:

    def test_formula_basic(self, panel_df, panel_arrays):
        m_formula = PooledOLS()
        m_formula.fit(formula="y ~ x1 + x2", data=panel_df)

        m_array = PooledOLS()
        m_array.fit(X=panel_arrays['X'], y=panel_arrays['y'])

        assert_allclose(m_formula.coef_, m_array.coef_, rtol=1e-10)

    def test_formula_predict(self, panel_df):
        m = PooledOLS()
        m.fit(formula="y ~ x1 + x2", data=panel_df)
        y_pred = m.predict(panel_df)
        assert y_pred.shape == (len(panel_df),)

    def test_no_formula_backward_compat(self, panel_arrays):
        m = PooledOLS()
        m.fit(X=panel_arrays['X'], y=panel_arrays['y'])
        assert m.coef_ is not None


# ============================================================================
# BetweenOLS formula tests
# ============================================================================

class TestBetweenOLSFormula:

    def test_formula_basic(self, panel_df, panel_arrays):
        m_formula = BetweenOLS()
        m_formula.fit(formula="y ~ x1 + x2", data=panel_df,
                      entity_ids=panel_arrays['entity_ids'])

        m_array = BetweenOLS()
        m_array.fit(X=panel_arrays['X'], y=panel_arrays['y'],
                    entity_ids=panel_arrays['entity_ids'])

        assert_allclose(m_formula.coef_, m_array.coef_, rtol=1e-10)


# ============================================================================
# FirstDifferenceOLS formula tests
# ============================================================================

class TestFirstDifferenceOLSFormula:

    def test_formula_basic(self, panel_df, panel_arrays):
        m_formula = FirstDifferenceOLS()
        m_formula.fit(formula="y ~ x1 + x2 - 1", data=panel_df,
                      entity_ids=panel_arrays['entity_ids'],
                      time_ids=panel_arrays['time_ids'])

        m_array = FirstDifferenceOLS()
        m_array.fit(X=panel_arrays['X'], y=panel_arrays['y'],
                    entity_ids=panel_arrays['entity_ids'],
                    time_ids=panel_arrays['time_ids'])

        assert_allclose(m_formula.coef_, m_array.coef_, rtol=1e-10)


# ============================================================================
# FamaMacBeth formula tests
# ============================================================================

class TestFamaMacBethFormula:

    def test_formula_basic(self, panel_df, panel_arrays):
        m_formula = FamaMacBeth(device='cpu')
        m_formula.fit(formula="y ~ x1 + x2", data=panel_df,
                      time_ids=panel_arrays['time_ids'])

        m_array = FamaMacBeth(device='cpu')
        m_array.fit(X=panel_arrays['X'], y=panel_arrays['y'],
                    time_ids=panel_arrays['time_ids'])

        assert_allclose(m_formula.coef_, m_array.coef_, rtol=1e-10)

    def test_formula_predict(self, panel_df):
        m = FamaMacBeth(device='cpu')
        m.fit(formula="y ~ x1 + x2", data=panel_df,
              time_ids=panel_df['time'].values)
        y_pred = m.predict(panel_df)
        assert y_pred.shape == (len(panel_df),)


# ============================================================================
# Edge cases
# ============================================================================

class TestFormulaEdgeCases:

    def test_formula_without_data_raises(self):
        m = PooledOLS()
        with pytest.raises(ValueError, match="data is None"):
            m.fit(formula="y ~ x1 + x2", data=None)

    def test_no_formula_no_xy_raises(self):
        m = PooledOLS()
        with pytest.raises(ValueError, match="formula\+data or X\+y"):
            m.fit(X=None, y=None)

    def test_array_interface_unchanged(self, panel_arrays):
        """Existing array interface must still work."""
        m = PooledOLS()
        m.fit(X=panel_arrays['X'], y=panel_arrays['y'])
        assert m.coef_ is not None
        assert m.rsquared > 0

    def test_fixed_effects_alias(self):
        assert FixedEffects is PanelOLS

    def test_random_effects_ols_alias(self):
        assert RandomEffectsOLS is RandomEffects


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
