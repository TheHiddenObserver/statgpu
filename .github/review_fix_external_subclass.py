from pathlib import Path

p = Path('statgpu/_base.py')
text = p.read_text(encoding='utf-8')
start = text.index('    def __init_subclass__(cls, **kwargs):')
end = text.index('    @classmethod\n    def _install_constructor_capture', start)
new_block = '''    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__.get("_estimator_type", ...)
        if declared is ...:
            inherited_type = next(
                (
                    base.__dict__.get("_estimator_type")
                    for base in cls.__mro__[1:]
                    if base.__dict__.get("_estimator_type")
                    in {"classifier", "regressor"}
                ),
                None,
            )
            name = cls.__name__.lower()
            module = cls.__module__
            internal_module = module.startswith("statgpu.")
            nonpredictive_module = module.startswith(
                (
                    "statgpu.covariance",
                    "statgpu.unsupervised",
                    "statgpu.preprocessing",
                    "statgpu.feature_selection",
                )
            )
            classifier_module = module.startswith("statgpu.linear_model")
            regression_module = module.startswith(
                (
                    "statgpu.linear_model",
                    "statgpu.panel",
                    "statgpu.survival",
                    "statgpu.semiparametric",
                )
            )
            if not internal_module:
                inferred_type = inherited_type
            elif nonpredictive_module:
                inferred_type = None
            elif ("classifier" in name or "logistic" in name) and classifier_module:
                inferred_type = "classifier"
            elif (
                "regressor" in name
                or "kernelridge" in name
                or (
                    regression_module
                    and any(
                        token in name
                        for token in (
                            "regression",
                            "ridge",
                            "lasso",
                            "elasticnet",
                            "quantile",
                            "cox",
                            "panel",
                            "ols",
                            "effects",
                            "fama",
                            "gam",
                        )
                    )
                )
            ):
                inferred_type = "regressor"
            else:
                inferred_type = inherited_type
            cls._estimator_type = inferred_type
        elif declared not in {None, "classifier", "regressor"}:
            raise ValueError(
                "_estimator_type must be None, 'classifier', or 'regressor'"
            )
        cls._install_constructor_capture()
        cls._install_public_finite_validation()

'''
text = text[:start] + new_block + text[end:]
p.write_text(text, encoding='utf-8')

p = Path('dev/tests/test_maintenance_024_025.py')
text = p.read_text(encoding='utf-8')
needle = '''    assert not is_regressor(GraphicalLasso())
    assert not is_classifier(GraphicalLasso())
'''
replacement = '''    assert not is_regressor(GraphicalLasso())
    assert not is_classifier(GraphicalLasso())

    class ExternalRidge(Ridge):
        pass

    assert is_regressor(ExternalRidge(compute_inference=False))
'''
if text.count(needle) != 1:
    raise SystemExit(f'external subclass test anchor count={text.count(needle)}')
p.write_text(text.replace(needle, replacement, 1), encoding='utf-8')

import compileall
for path in ('statgpu/_base.py', 'dev/tests/test_maintenance_024_025.py'):
    if not compileall.compile_file(path, quiet=1):
        raise SystemExit(f'compile failed: {path}')
