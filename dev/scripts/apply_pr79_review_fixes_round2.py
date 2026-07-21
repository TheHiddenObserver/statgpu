#!/usr/bin/env python3
"""Apply second-round fixes found while reviewing the first PR79 repair."""

from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one match, found {count}: {old[:100]!r}"
        )
    p.write_text(text.replace(old, new, 1))


def patch_formula_intercept_contract():
    path = "statgpu/linear_model/wrappers/_linear.py"
    replace_once(
        path,
        '''        self._formula_has_intercept = None

    def _clear_inference_result(self):
''',
        '''        self._formula_has_intercept = None
        self._effective_fit_intercept = bool(fit_intercept)

    def _clear_inference_result(self):
''',
    )
    replace_once(
        path,
        '''        # Handle formula interface
        _orig_fit_intercept = self.fit_intercept
''',
        '''        # Formula syntax controls the fitted design without mutating the
        # public constructor parameter required by sklearn-style cloning.
        effective_fit_intercept = bool(self.fit_intercept)
''',
    )
    replace_once(
        path,
        '''                X_arr = np.delete(X_arr, intercept_idx, axis=1)
                self.fit_intercept = True
''',
        '''                X_arr = np.delete(X_arr, intercept_idx, axis=1)
                effective_fit_intercept = True
''',
    )
    replace_once(
        path,
        '''                # Formula syntax owns intercept semantics, matching statsmodels/R.
                self.fit_intercept = False
''',
        '''                # Formula syntax owns intercept semantics, matching statsmodels/R.
                effective_fit_intercept = False
''',
    )
    replace_once(
        path,
        '''        self.fit_intercept = _orig_fit_intercept

        # Resolve the backend before converting raw arrays so CuPy/Torch inputs
''',
        '''        self._effective_fit_intercept = effective_fit_intercept

        # Resolve the backend before converting raw arrays so CuPy/Torch inputs
''',
    )

    p = Path(path)
    text = p.read_text()
    marker = "    def _fit_cpu(self, X, y, sample_weight=None):\n"
    if text.count(marker) != 1:
        raise RuntimeError(f"{path}: _fit_cpu marker mismatch")
    prefix, tail = text.split(marker, 1)
    if "self.fit_intercept" not in tail:
        raise RuntimeError(f"{path}: no internal fit_intercept usages found")
    tail = tail.replace("self.fit_intercept", "self._effective_fit_intercept")
    p.write_text(prefix + marker + tail)


def strengthen_tests():
    path = Path("dev/tests/test_pr79_final_review_fixes.py")
    text = path.read_text()
    text = text.replace(
        '''from pathlib import Path
import inspect

import numpy as np
''',
        '''from pathlib import Path
import inspect
import subprocess

import numpy as np
import pandas as pd
''',
        1,
    )
    old_rank = '''def test_pooled_rank_deficiency_uses_effective_rank_for_df():
    x = np.arange(20.0)
    X = np.column_stack([x, 2.0 * x])
    y = 1.0 + 3.0 * x
    model = PooledOLS().fit(X, y)
    design = np.column_stack([np.ones(X.shape[0]), X])
    expected_rank = int(np.linalg.matrix_rank(design))
    assert model.rank_ == expected_rank
    assert model.df_resid == X.shape[0] - expected_rank
    assert np.all(np.isfinite(model.bse_))
'''
    new_rank = '''def test_pooled_rank_deficiency_uses_effective_rank_for_df():
    import statsmodels.api as sm

    rng = np.random.default_rng(79)
    x = np.arange(40.0)
    X = np.column_stack([x, 2.0 * x])
    y = 1.0 + 3.0 * x + rng.normal(scale=0.25, size=x.shape[0])
    model = PooledOLS().fit(X, y)
    design = np.column_stack([np.ones(X.shape[0]), X])
    reference = sm.OLS(y, design).fit()
    expected_rank = int(np.linalg.matrix_rank(design))

    assert model.rank_ == expected_rank
    assert model.df_resid == X.shape[0] - expected_rank
    assert model.df_resid == int(reference.df_resid)
    assert_allclose(model.bse_, reference.bse, rtol=1e-8, atol=1e-10)
'''
    if text.count(old_rank) != 1:
        raise RuntimeError("rank-deficiency test block mismatch")
    text = text.replace(old_rank, new_rank, 1)

    insertion = '''

def test_linear_formula_intercept_semantics_do_not_mutate_public_parameter():
    x = np.linspace(-2.0, 2.0, 60)
    frame = pd.DataFrame({"x": x, "y": 1.75 + 2.5 * x})

    with_intercept = LinearRegression(fit_intercept=False).fit(
        formula="y ~ x", data=frame
    )
    assert with_intercept.fit_intercept is False
    assert np.isclose(with_intercept.intercept_, 1.75, atol=1e-10)
    assert_allclose(with_intercept.coef_, [2.5], atol=1e-10)

    without_intercept = LinearRegression(fit_intercept=True).fit(
        formula="y ~ x - 1", data=frame
    )
    assert without_intercept.fit_intercept is True
    assert without_intercept.intercept_ == 0.0
    expected = np.linalg.lstsq(x[:, None], frame["y"].to_numpy(), rcond=None)[0]
    assert_allclose(without_intercept.coef_, expected, atol=1e-10)


def test_pipefail_propagates_the_failing_pytest_side_of_a_pipeline():
    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", "false | tee /dev/null"],
        check=False,
    )
    assert result.returncode != 0
'''
    anchor = '''

@pytest.mark.parametrize("backend", ["cupy", "torch"])
'''
    if text.count(anchor) != 1:
        raise RuntimeError("GPU test anchor mismatch")
    text = text.replace(anchor, insertion + anchor, 1)
    path.write_text(text)


def main():
    patch_formula_intercept_contract()
    strengthen_tests()


if __name__ == "__main__":
    main()
