from pathlib import Path

path = Path("dev/tests/test_second_full_review.py")
text = path.read_text(encoding="utf-8")
old = '''    def test_numpy_matches_scipy_welch_anova(self):
        from scipy import stats
        from statgpu.anova import f_welch

        rng = np.random.default_rng(12)
        groups = (
            rng.normal(0.0, 1.0, 80),
            rng.normal(0.5, 3.0, 120),
            rng.normal(-0.2, 0.5, 60),
        )
        actual = f_welch(*groups)
        expected = stats.f_oneway(*groups, equal_var=False)
        np.testing.assert_allclose(actual.statistic, expected.statistic, rtol=1e-12)
        np.testing.assert_allclose(actual.pvalue, expected.pvalue, rtol=1e-10)
        assert isinstance(actual.df_within, float)
'''
new = '''    def test_numpy_matches_statsmodels_welch_anova(self):
        from statsmodels.stats.oneway import anova_oneway
        from statgpu.anova import f_welch

        rng = np.random.default_rng(12)
        groups = (
            rng.normal(0.0, 1.0, 80),
            rng.normal(0.5, 3.0, 120),
            rng.normal(-0.2, 0.5, 60),
        )
        actual = f_welch(*groups)
        expected = anova_oneway(groups, use_var="unequal", welch_correction=True)
        np.testing.assert_allclose(actual.statistic, expected.statistic, rtol=1e-12)
        np.testing.assert_allclose(actual.pvalue, expected.pvalue, rtol=1e-10)
        np.testing.assert_allclose(actual.df_between, expected.df[0], rtol=0, atol=0)
        np.testing.assert_allclose(actual.df_within, expected.df[1], rtol=1e-12)
        assert isinstance(actual.df_within, float)
'''
if old not in text:
    raise RuntimeError("Welch SciPy reference block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
