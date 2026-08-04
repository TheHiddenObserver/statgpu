from pathlib import Path

p = Path('statgpu/_base.py')
text = p.read_text(encoding='utf-8')
old = '''        # Constructor validation and normalization occur before mutating self.
        fresh = type(self)(**direct)
        for root, sub_params in nested.items():
            nested_estimator = getattr(fresh, root, None)
            if nested_estimator is None:
                nested_estimator = getattr(fresh, f"_{root}", None)
            if not hasattr(nested_estimator, "set_params"):
                raise ValueError(
                    f"Parameter {root!r} of {type(self).__name__} does not "
                    "support nested parameters."
                )
            nested_estimator.set_params(**sub_params)

        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        return self
'''
new = '''        # Valid constructor values rebuild normalized runtime state. Some
        # estimators intentionally defer selected validation to fit(); preserve
        # that established boundary when the constructor rejects a set_params
        # value, while retaining the raw constructor ledger for sklearn clone.
        try:
            fresh = type(self)(**direct)
        except (TypeError, ValueError):
            for key, value in params.items():
                root, delimiter, _ = key.partition("__")
                if delimiter:
                    continue
                raw_value = value
                if root == "device" and isinstance(value, str):
                    value = Device(value)
                if hasattr(self, root):
                    setattr(self, root, value)
                else:
                    setattr(self, f"_{root}", value)
                raw_params = getattr(self, "_constructor_params_raw", None)
                if raw_params is None:
                    raw_params = {}
                    self._constructor_params_raw = raw_params
                raw_params[root] = raw_value

            for root, sub_params in nested.items():
                nested_estimator = getattr(self, root, None)
                if nested_estimator is None:
                    nested_estimator = getattr(self, f"_{root}", None)
                if not hasattr(nested_estimator, "set_params"):
                    raise ValueError(
                        f"Parameter {root!r} of {type(self).__name__} does not "
                        "support nested parameters."
                    )
                nested_estimator.set_params(**sub_params)
            return self

        for root, sub_params in nested.items():
            nested_estimator = getattr(fresh, root, None)
            if nested_estimator is None:
                nested_estimator = getattr(fresh, f"_{root}", None)
            if not hasattr(nested_estimator, "set_params"):
                raise ValueError(
                    f"Parameter {root!r} of {type(self).__name__} does not "
                    "support nested parameters."
                )
            nested_estimator.set_params(**sub_params)

        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        return self
'''
if text.count(old) != 1:
    raise SystemExit(f'set_params follow-up anchor count={text.count(old)}')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# Add an explicit regression beside the normalized-state test.
p = Path('dev/tests/test_maintenance_024_025.py')
text = p.read_text(encoding='utf-8')
marker = 'def test_set_params_preserves_estimator_fit_validation_boundary'
if marker not in text:
    text += '''\n\n
def test_set_params_preserves_estimator_fit_validation_boundary():
    from statgpu.survival import CoxPH

    model = CoxPH()
    model.set_params(compute_inference="False")
    assert model.get_params(deep=False)["compute_inference"] == "False"
'''
p.write_text(text, encoding='utf-8')

import compileall
if not compileall.compile_file('statgpu/_base.py', quiet=1):
    raise SystemExit('base compile failed')
if not compileall.compile_file('dev/tests/test_maintenance_024_025.py', quiet=1):
    raise SystemExit('test compile failed')
