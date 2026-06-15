"""
Tests for statgpu.core.formula module.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """Standard test DataFrame."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "y": np.random.randn(n),
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "cat": pd.Categorical(np.random.choice(["A", "B", "C"], n)),
    })


class TestFormulaParserBasic:
    """Test basic formula parsing."""

    def test_simple_formula(self, sample_df):
        """Test simple y ~ x1 + x2 formula."""
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser("y ~ x1 + x2")
        y, X, info = parser.eval(sample_df)

        assert y.shape == (200,)
        assert X.shape == (200, 3)  # intercept + x1 + x2
        assert parser.column_names == ["Intercept", "x1", "x2"]

    def test_no_intercept(self, sample_df):
        """Test y ~ x1 + x2 - 1 (no intercept)."""
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser("y ~ x1 + x2 - 1")
        y, X, info = parser.eval(sample_df)

        assert X.shape == (200, 2)
        assert parser.column_names == ["x1", "x2"]

    def test_categorical_encoding(self, sample_df):
        """Test C() for categorical variables."""
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser("y ~ x1 + C(cat)")
        y, X, info = parser.eval(sample_df)

        # Intercept + x1 + cat[T.B] + cat[T.C] = 4 columns
        assert X.shape[1] == 4
        assert "x1" in parser.column_names
        assert any("cat" in name for name in parser.column_names)

    def test_interaction(self, sample_df):
        """Test x1:x2 interaction."""
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser("y ~ x1 + x2 + x1:x2")
        y, X, info = parser.eval(sample_df)

        assert X.shape[1] == 4  # intercept + x1 + x2 + x1:x2

    def test_star_operator(self, sample_df):
        """Test x1*x2 (main effects + interaction)."""
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser("y ~ x1 * x2")
        y, X, info = parser.eval(sample_df)

        assert X.shape[1] == 4  # intercept + x1 + x2 + x1:x2

    def test_transform(self, sample_df):
        """Test np() transformations."""
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser("y ~ np.log(np.abs(x1)) + x2")
        y, X, info = parser.eval(sample_df)

        assert y.shape == (200,)
        assert X.shape[1] == 3  # intercept + transformed_x1 + x2


class TestFormulaParserTransform:
    """Test transform (predict-time) functionality."""

    def test_transform_new_data(self, sample_df):
        """Test transform on new data with same structure."""
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser("y ~ x1 + x2")
        parser.eval(sample_df)

        new_data = pd.DataFrame({
            "x1": [0.5, -0.3],
            "x2": [1.2, 0.8],
        })
        X_new = parser.transform(new_data)

        assert X_new.shape == (2, 3)  # 2 rows, intercept + 2 cols

    def test_transform_with_categorical(self, sample_df):
        """Test transform handles categorical encoding from training."""
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser("y ~ x1 + C(cat)")
        parser.eval(sample_df)

        new_data = pd.DataFrame({
            "x1": [0.5],
            "cat": pd.Categorical(["A"]),
        })
        X_new = parser.transform(new_data)

        assert X_new.shape == (1, 4)  # intercept + x1 + cat[B] + cat[C]

    def test_transform_no_design_info(self):
        """Test transform raises when not yet evaluated."""
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser("y ~ x1")
        new_data = pd.DataFrame({"x1": [1.0]})

        with pytest.raises(RuntimeError, match="no design_info available"):
            parser.transform(new_data)


class TestParseFormulaSafe:
    """Test parse_formula_safe fallback logic."""

    def test_formula_path(self, sample_df):
        """Test formula path works."""
        from statgpu.core.formula import parse_formula_safe

        y, X, info = parse_formula_safe("y ~ x1", data=sample_df)
        assert y.shape == (200,)
        assert info is not None

    def test_array_path(self, sample_df):
        """Test array path when formula is None."""
        from statgpu.core.formula import parse_formula_safe

        X = sample_df[["x1", "x2"]].values
        y = sample_df["y"].values
        y_out, X_out, info = parse_formula_safe(None, None, X=X, y=y)

        assert info is None
        np.testing.assert_array_equal(y_out, y)
        np.testing.assert_array_equal(X_out, X)

    def test_formula_without_data_raises(self):
        """Test that formula without data raises."""
        from statgpu.core.formula import parse_formula_safe

        with pytest.raises(ValueError, match="data"):
            parse_formula_safe("y ~ x1", None)

    def test_no_input_raises(self):
        """Test that no input raises."""
        from statgpu.core.formula import parse_formula_safe

        with pytest.raises(ValueError, match="Either formula"):
            parse_formula_safe(None, None)


class TestFormulaParserSummary:
    """Test FormulaParser.summary() output."""

    def test_summary_before_eval(self, sample_df):
        """Test summary shows pending state."""
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser("y ~ x1 + x2")
        s = parser.summary()

        assert "y ~ x1 + x2" in s
        assert "pending" in s.lower() or "Not yet evaluated" in s

    def test_summary_after_eval(self, sample_df):
        """Test summary shows parsed info."""
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser("y ~ x1 + x2")
        parser.eval(sample_df)
        s = parser.summary()

        assert "y ~ x1 + x2" in s
        assert "x1" in s
        assert "x2" in s
        assert "Predictors (3)" in s
