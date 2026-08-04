from pathlib import Path

p = Path('statgpu/_base.py')
text = p.read_text(encoding='utf-8')

start = text.index('    def __init_subclass__(cls, **kwargs):')
end = text.index('    @classmethod\n    def _install_constructor_capture', start)
new_init_subclass = '''    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        declared_type = cls.__dict__.get("_estimator_type")
        if declared_type not in {"classifier", "regressor"}:
            name = cls.__name__.lower()
            module = cls.__module__
            classifier_module = module.startswith("statgpu.linear_model")
            regression_module = module.startswith(
                (
                    "statgpu.linear_model",
                    "statgpu.panel",
                    "statgpu.survival",
                    "statgpu.semiparametric",
                )
            )
            inferred_type = None
            if ("classifier" in name or "logistic" in name) and classifier_module:
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
            # Override inherited classifications for covariance, unsupervised,
            # transformers, and other non-predictive estimator families.
            cls._estimator_type = inferred_type
        cls._install_constructor_capture()
        cls._install_public_finite_validation()

'''
text = text[:start] + new_init_subclass + text[end:]

start = text.index('    def _statgpu_estimator_type(self):')
end = text.index('    def __sklearn_tags__(self):', start)
new_instance_type = '''    def _statgpu_estimator_type(self):
        """Return the class-level sklearn estimator classification."""
        estimator_type = getattr(type(self), "_estimator_type", None)
        if estimator_type in {"classifier", "regressor"}:
            return estimator_type
        return None

'''
text = text[:start] + new_instance_type + text[end:]
p.write_text(text, encoding='utf-8')

import compileall
if not compileall.compile_file('statgpu/_base.py', quiet=1):
    raise SystemExit('base compile failed')
