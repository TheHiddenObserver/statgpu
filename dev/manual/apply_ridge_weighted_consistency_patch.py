"""Temporary follow-up patch for formula row and RidgeCV weight consistency."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


parser_path = ROOT / "statgpu/core/formula/_parser.py"
parser = parser_path.read_text()
parser = replace_once(
    parser,
    """        self._design_info = None
        self._y_names: Optional[List[str]] = None
""",
    """        self._design_info = None
        self._y_names: Optional[List[str]] = None
        self._row_positions: Optional[np.ndarray] = None
        self._row_index = None
""",
    "parser state",
)
parser = replace_once(
    parser,
    """    @property
    def column_names(self) -> Optional[List[str]]:
""",
    """    @property
    def row_positions(self) -> Optional[np.ndarray]:
        \"\"\"Zero-based positions retained after Patsy missing-value filtering.\"\"\"
        if self._row_positions is None:
            return None
        return self._row_positions.copy()

    @property
    def row_index(self):
        \"\"\"Original DataFrame index retained after formula evaluation.\"\"\"
        if self._row_index is None:
            return None
        return self._row_index.copy()

    @property
    def column_names(self) -> Optional[List[str]]:
""",
    "parser row properties",
)
parser = replace_once(
    parser,
    """        patsy = self._require_patsy()
        data = data.copy()

        y, X = patsy.dmatrices(
            self.formula,
            data,
            eval_env=eval_env + 1,
            return_type="matrix",
        )

        self._y_names = list(y.design_info.column_names)
        self._design_info = X.design_info
""",
    """        patsy = self._require_patsy()
        data = data.copy()
        original_index = data.index.copy()
        # A positional index lets callers align side arrays (sample weights,
        # clusters, offsets) after Patsy drops rows containing missing values.
        data.index = pd.RangeIndex(len(data))

        y, X = patsy.dmatrices(
            self.formula,
            data,
            eval_env=eval_env + 1,
            return_type="dataframe",
        )

        self._y_names = list(y.design_info.column_names)
        self._design_info = X.design_info
        self._row_positions = np.asarray(X.index, dtype=np.int64)
        self._row_index = original_index.take(self._row_positions)
""",
    "parser positional retention",
)
parser_path.write_text(parser)

fit_path = ROOT / "statgpu/linear_model/penalized/_fit_mixin.py"
fit = fit_path.read_text()
fit = replace_once(
    fit,
    """            parser = FormulaParser(formula)
            y, X, design_info = parser.eval(data)
            formula_column_names = list(design_info.column_names)
""",
    """            parser = FormulaParser(formula)
            y, X, design_info = parser.eval(data)
            if sample_weight is not None:
                sw_formula = np.asarray(_to_numpy(sample_weight), dtype=np.float64).reshape(-1)
                row_positions = parser.row_positions
                if sw_formula.shape[0] == len(data):
                    sample_weight = sw_formula[row_positions]
                elif sw_formula.shape[0] == X.shape[0]:
                    sample_weight = sw_formula
                else:
                    raise ValueError(
                        "For formula fitting, sample_weight must have length "
                        "len(data) or the number of rows retained by the formula."
                    )
            formula_column_names = list(design_info.column_names)
""",
    "formula sample-weight alignment",
)
fit_path.write_text(fit)

# Add the formula parser to static validation through the permanent workflow later.

test_path = ROOT / "dev/tests/test_ridge_weighted_consistency.py"
test = test_path.read_text()
test += '''


def test_formula_missing_rows_aligns_full_length_sample_weights():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1206)
    n = 150
    X = rng.normal(size=(n, 3))
    y = 0.4 + X @ np.array([0.8, -0.6, 0.3]) + rng.normal(scale=0.25, size=n)
    w = rng.uniform(0.2, 3.0, size=n)
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    frame["y"] = y
    frame.loc[[4, 31, 92], "x2"] = np.nan
    frame.loc[[17, 108], "y"] = np.nan
    keep = frame[["y", "x1", "x2", "x3"]].notna().all(axis=1).to_numpy()

    formula = Ridge(alpha=0.13, compute_inference=False, device="cpu").fit(
        formula="y ~ x1 + x2 + x3", data=frame, sample_weight=w
    )
    direct = Ridge(alpha=0.13, compute_inference=False, device="cpu").fit(
        X[keep], y[keep], sample_weight=w[keep]
    )

    np.testing.assert_allclose(formula.coef_, direct.coef_, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(formula.intercept_, direct.intercept_, rtol=1e-11, atol=1e-11)


def test_ridgecv_is_invariant_to_global_weight_rescaling():
    from statgpu.linear_model import RidgeCV

    rng = np.random.default_rng(1207)
    X = rng.normal(size=(180, 6))
    y = 0.3 + X @ rng.normal(size=6) + rng.normal(scale=0.5, size=180)
    w = rng.uniform(0.1, 2.5, size=180)
    alphas = np.array([0.01, 0.04, 0.12, 0.4])

    first = RidgeCV(
        alphas=alphas, cv=4, random_state=9, device="cpu",
        compute_inference=False,
    ).fit(X, y, sample_weight=w)
    second = RidgeCV(
        alphas=alphas, cv=4, random_state=9, device="cpu",
        compute_inference=False,
    ).fit(X, y, sample_weight=11.0 * w)

    assert first.alpha_ == second.alpha_
    np.testing.assert_allclose(first.mean_mse_, second.mean_mse_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(first.coef_, second.coef_, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(first.intercept_, second.intercept_, rtol=1e-11, atol=1e-11)
'''
test_path.write_text(test)

print("Formula weight-alignment follow-up patch applied")
