from pathlib import Path
import compileall
import runpy

script = Path(".github/review_fix_constructor_contract.py")
text = script.read_text(encoding="utf-8")
old = '''                if name in normalized_names:
                    if hasattr(self, name):
                        runtime_value = getattr(self, name)
                    elif hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    else:
                        runtime_value = raw_value
'''
new = '''                if name in normalized_names:
                    # Constructor wrappers are nested across the inheritance
                    # chain. An inner wrapper may already have restored the
                    # public raw value, so the private runtime value is the
                    # authoritative source when it exists.
                    if hasattr(self, private_name):
                        runtime_value = getattr(self, private_name)
                    elif hasattr(self, name):
                        runtime_value = getattr(self, name)
                    else:
                        runtime_value = raw_value
'''
if text.count(old) != 1:
    raise SystemExit(f"nested runtime anchor count={text.count(old)}")
script.write_text(text.replace(old, new, 1), encoding="utf-8")

runpy.run_path(".github/review_fix_constructor_contract_v3.py", run_name="__main__")

tests = Path("dev/tests/test_maintenance_024_025.py")
text = tests.read_text(encoding="utf-8")
old = '''    cloned = clone(estimator)
    assert type(cloned) is CopyingEstimator
    assert cloned.solver == "auto"
'''
new = '''    cloned = clone(estimator)
    assert type(cloned) is CopyingEstimator
    assert cloned.solver == "AUTO"
    assert cloned._solver == "auto"
'''
if text.count(old) != 1:
    raise SystemExit(f"legacy solver expectation count={text.count(old)}")
text = text.replace(old, new, 1)
old = '''    assert model.get_params(deep=False)["cov_type"] == "HAC"
    assert model.cov_type == "hac"
    assert model._fitted is False
'''
new = '''    assert model.get_params(deep=False)["cov_type"] == "HAC"
    assert model.cov_type == "HAC"
    assert model._cov_type == "hac"
    assert model._fitted is False
'''
if text.count(old) != 1:
    raise SystemExit(f"panel normalized expectation count={text.count(old)}")
tests.write_text(text.replace(old, new, 1), encoding="utf-8")

if not compileall.compile_file(str(tests), quiet=1):
    raise SystemExit("maintenance tests failed to compile")
