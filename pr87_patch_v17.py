from pathlib import Path
import runpy

runpy.run_path("pr87_patch_v16.py", run_name="__main__")

path = Path("dev/tests/test_maintenance_024_025.py")
text = path.read_text(encoding="utf-8")
old = 'with pytest.raises(ValueError, match="sample_weight must have length"):'
new = 'with pytest.raises(ValueError, match="sample_weight must (?:have length|match)"):'
if old not in text:
    raise RuntimeError("legacy GLM error-message assertion anchor missing")
text = text.replace(old, new, 1)

old = '''        factory = lambda: PenalizedLinearRegression(
            loss="squared_error",
            penalty="l1",
'''
new = '''        factory = lambda: PenalizedLinearRegression(
            penalty="l1",
'''
if old not in text:
    raise RuntimeError("PenalizedLinearRegression test factory anchor missing")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
