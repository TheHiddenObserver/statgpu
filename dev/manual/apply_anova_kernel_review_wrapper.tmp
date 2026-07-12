from pathlib import Path
import runpy

core = Path('dev/manual/apply_anova_kernel_review_core.py')
runpy.run_path(str(core), run_name='__main__')

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
core.unlink()
Path('dev/manual/apply_anova_kernel_review_wrapper.tmp').unlink(missing_ok=True)
