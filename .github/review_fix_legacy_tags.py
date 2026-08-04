from pathlib import Path

p = Path('statgpu/_base.py')
text = p.read_text(encoding='utf-8')
old = '''    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._install_constructor_capture()
        cls._install_public_finite_validation()
'''
new = '''    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "_estimator_type" not in cls.__dict__:
            name = cls.__name__.lower()
            if "classifier" in name or "logistic" in name:
                cls._estimator_type = "classifier"
            elif any(
                token in name
                for token in (
                    "regression",
                    "regressor",
                    "ridge",
                    "lasso",
                    "elasticnet",
                    "quantile",
                    "cox",
                    "panel",
                    "ols",
                    "effects",
                    "fama",
                    "kernelridge",
                    "gam",
                )
            ):
                cls._estimator_type = "regressor"
        cls._install_constructor_capture()
        cls._install_public_finite_validation()
'''
if text.count(old) != 1:
    raise SystemExit(f'init_subclass anchor count={text.count(old)}')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

import compileall
if not compileall.compile_file('statgpu/_base.py', quiet=1):
    raise SystemExit('base compile failed')
