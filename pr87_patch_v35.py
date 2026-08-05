from pathlib import Path
import runpy

runpy.run_path("pr87_patch_v34.py", run_name="__main__")


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:180]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Direct IRLS must validate the original response before any float cast can
# discard an imaginary component or coerce an unsupported dtype.
replace_once(
    "statgpu/glm_core/_irls.py",
    '''    y_work = _to_backend(y, backend, X)
    family_name = getattr(family, "name", "")
    objective_loss = _objective_loss_for_family(family)
    y_work = objective_loss.validate_response(y_work)
    if int(y_work.shape[0]) != int(X.shape[0]):
''',
    '''    family_name = getattr(family, "name", "")
    objective_loss = _objective_loss_for_family(family)
    y_validated = objective_loss.validate_response(y)
    y_work = _to_backend(y_validated, backend, X)
    if int(y_work.shape[0]) != int(X.shape[0]):
''',
)

# Migrate the pre-v34 public-error assertion to the stricter real-numeric text.
tests = Path("dev/tests/test_maintenance_024_025.py")
text = tests.read_text(encoding="utf-8")
old = '''    with pytest.raises(ValueError, match="numeric finite values"):
        GeneralizedLinearModel(
'''
new = '''    with pytest.raises(ValueError, match="real numeric values"):
        GeneralizedLinearModel(
'''
if old not in text:
    raise RuntimeError("legacy nonnumeric response assertion anchor missing")
tests.write_text(text.replace(old, new, 1), encoding="utf-8")
