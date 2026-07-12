from pathlib import Path
import runpy

core = Path('dev/manual/apply_anova_kernel_review_core.py')
runpy.run_path(str(core), run_name='__main__')

# Migrate the old unbalanced two-way ANOVA expectation.  The previous test
# accepted a number computed from balanced-design sums of squares; the public
# API now rejects that ambiguous case until an SS type is selected explicitly.
path = Path('dev/tests/test_anova_p2.py')
text = path.read_text()
old = '''    def test_unbalanced(self):
        np.random.seed(42)
        data = [
            [np.random.randn(5), np.random.randn(10), np.random.randn(8)],
            [np.random.randn(12), np.random.randn(6), np.random.randn(9)],
        ]
        r = f_twoway(data, interaction=True)
        assert r.factor_a_statistic > 0
'''
new = '''    def test_unbalanced_requires_explicit_ss_type(self):
        np.random.seed(42)
        data = [
            [np.random.randn(5), np.random.randn(10), np.random.randn(8)],
            [np.random.randn(12), np.random.randn(6), np.random.randn(9)],
        ]
        with pytest.raises(ValueError, match="balanced"):
            f_twoway(data, interaction=True)
'''
if text.count(old) != 1:
    raise RuntimeError('unexpected legacy unbalanced ANOVA test content')
path.write_text(text.replace(old, new, 1))

# A chi-square kernel is only defined for non-negative histogram features.
# Preserve registry coverage while respecting the documented domain.
path = Path('dev/tests/test_kernel_methods_p2.py')
text = path.read_text()
old = '''        for metric in ["rbf", "linear", "poly", "laplacian", "sigmoid", "cosine", "chi2"]:
            K = pairwise_kernels(X, metric=metric)
            assert K.shape == (10, 10)
'''
new = '''        for metric in ["rbf", "linear", "poly", "laplacian", "sigmoid", "cosine", "chi2"]:
            X_metric = np.abs(X) if metric == "chi2" else X
            K = pairwise_kernels(X_metric, metric=metric)
            assert K.shape == (10, 10)
'''
if text.count(old) != 1:
    raise RuntimeError('unexpected kernel registry test content')
path.write_text(text.replace(old, new, 1))

# Handle the exactly-degenerate Welch pair directly.  Passing df=inf through
# the backend t-distribution produced NaN even though the correct p-value and
# confidence interval are 1 and [0, 0] for identical constant groups.
path = Path('statgpu/anova/_posthoc.py')
text = path.read_text()
old = '''            # Welch's t-test
            se = np.sqrt(var_i / ni + var_j / nj)
            if se > 0:
                t_stat = mean_diff / se
            else:
                t_stat = 0.0 if mean_diff == 0.0 else np.copysign(float("inf"), mean_diff)

            # Welch-Satterthwaite df
            num = (var_i / ni + var_j / nj) ** 2
            den = (var_i / ni) ** 2 / (ni - 1) + (var_j / nj) ** 2 / (nj - 1)
            df = num / den if den > 0 else float("inf")

            # Two-sided p-value
            pvalue_raw = _to_float_scalar(t_dist.sf(abs(t_stat), df)) * 2
            pvalue = min(pvalue_raw, 1.0)

            # Bonferroni-corrected CI
            t_crit = _to_float_scalar(t_dist.isf(alpha_bonf / 2, df))
            margin = t_crit * se
'''
new = '''            # Welch's t-test
            se = np.sqrt(var_i / ni + var_j / nj)
            if se == 0.0:
                df = float("inf")
                if mean_diff == 0.0:
                    t_stat = 0.0
                    pvalue = 1.0
                else:
                    t_stat = np.copysign(float("inf"), mean_diff)
                    pvalue = 0.0
                margin = 0.0
            else:
                t_stat = mean_diff / se

                # Welch-Satterthwaite df
                num = (var_i / ni + var_j / nj) ** 2
                den = (var_i / ni) ** 2 / (ni - 1) + (var_j / nj) ** 2 / (nj - 1)
                df = num / den if den > 0 else float("inf")

                # Two-sided p-value
                pvalue_raw = _to_float_scalar(t_dist.sf(abs(t_stat), df)) * 2
                pvalue = min(pvalue_raw, 1.0)

                # Bonferroni-corrected CI
                t_crit = _to_float_scalar(t_dist.isf(alpha_bonf / 2, df))
                margin = t_crit * se
'''
if text.count(old) != 1:
    raise RuntimeError('unexpected Bonferroni Welch block content')
path.write_text(text.replace(old, new, 1))

core.unlink()
Path('dev/manual/apply_anova_kernel_review_wrapper.tmp').unlink(missing_ok=True)
