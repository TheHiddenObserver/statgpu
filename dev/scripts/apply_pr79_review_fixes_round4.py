#!/usr/bin/env python3
"""Align LinearRegression formula sample weights after Patsy row filtering."""

from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one match, found {count}: {old[:120]!r}"
        )
    p.write_text(text.replace(old, new, 1))


def patch_formula_weight_alignment():
    path = "statgpu/linear_model/wrappers/_linear.py"
    replace_once(
        path,
        '''def _parse_formula_if_provided(formula, data, X, y):
    """Parse formula+data or fall back to raw arrays. Returns (y, X, info)."""
    if formula is not None:
        from statgpu.core.formula import parse_formula
        return parse_formula(formula, data)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    return y, np.asarray(X), None
''',
        '''def _parse_formula_if_provided(formula, data, X, y):
    """Parse formula data and return retained source-row positions."""
    if formula is not None:
        from statgpu.core.formula import FormulaParser

        parser = FormulaParser(formula)
        y_arr, X_arr, info = parser.eval(data)
        return y_arr, X_arr, info, parser.row_positions
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    return y, np.asarray(X), None, None
''',
    )
    replace_once(
        path,
        '''            y_arr, X_arr, design_info = _parse_formula_if_provided(
                formula, data, None, None
            )
''',
        '''            y_arr, X_arr, design_info, retained_rows = _parse_formula_if_provided(
                formula, data, None, None
            )
''',
    )
    replace_once(
        path,
        '''            self._feature_names = [name for name in formula_column_names if name != "Intercept"]
            if self._formula_has_intercept:
''',
        '''            self._feature_names = [name for name in formula_column_names if name != "Intercept"]

            if sample_weight is not None:
                from statgpu.backends import _to_numpy

                weights = np.asarray(_to_numpy(sample_weight), dtype=float)
                if weights.ndim != 1:
                    raise ValueError("sample_weight must be one-dimensional")
                retained_rows = np.asarray(retained_rows, dtype=np.int64)
                if weights.shape[0] == len(data):
                    sample_weight = weights[retained_rows]
                elif weights.shape[0] == len(y_arr):
                    # Already aligned weights are accepted for programmatic use.
                    sample_weight = weights
                else:
                    raise ValueError(
                        "sample_weight must match the original data length or "
                        "the number of formula rows retained after missing-value filtering"
                    )

            if self._formula_has_intercept:
''',
    )


def add_tests():
    path = Path("dev/tests/test_pr79_final_review_fixes.py")
    text = path.read_text()
    insertion = '''

def test_weighted_formula_aligns_weights_after_patsy_drops_rows():
    from sklearn.linear_model import LinearRegression as SkLinearRegression

    rng = np.random.default_rng(7906)
    n = 90
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.9 + 1.3 * x1 - 0.4 * x2 + rng.normal(scale=0.2, size=n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    frame.loc[[4, 17, 51], "x1"] = np.nan
    frame.loc[[9, 52], "y"] = np.nan
    weights = np.linspace(0.2, 2.5, n) ** 2

    model = LinearRegression().fit(
        formula="y ~ x1 + x2",
        data=frame,
        sample_weight=weights,
    )
    kept = frame[["y", "x1", "x2"]].notna().all(axis=1).to_numpy()
    reference = SkLinearRegression().fit(
        frame.loc[kept, ["x1", "x2"]].to_numpy(),
        frame.loc[kept, "y"].to_numpy(),
        sample_weight=weights[kept],
    )

    assert np.isclose(model.intercept_, reference.intercept_, rtol=1e-10, atol=1e-10)
    assert_allclose(model.coef_, reference.coef_, rtol=1e-10, atol=1e-10)
    assert model._sample_weight_fit.shape == (int(kept.sum()),)
    assert_allclose(model._sample_weight_fit, weights[kept])


def test_weighted_formula_rejects_unalignable_weight_length():
    frame = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError, match="sample_weight"):
        LinearRegression().fit(
            formula="y ~ x", data=frame, sample_weight=np.ones(4)
        )
'''
    anchor = '''

def test_weighted_linear_regression_matches_sklearn_and_statsmodels():
'''
    if text.count(anchor) != 1:
        raise RuntimeError("formula-weight test insertion anchor mismatch")
    path.write_text(text.replace(anchor, insertion + anchor, 1))


def main():
    patch_formula_weight_alignment()
    add_tests()


if __name__ == "__main__":
    main()
