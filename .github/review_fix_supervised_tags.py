from pathlib import Path
import compileall


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
old = '''            classifier_module = module.startswith("statgpu.linear_model")
            regression_module = module.startswith(
                (
                    "statgpu.linear_model",
                    "statgpu.panel",
                    "statgpu.survival",
                    "statgpu.semiparametric",
                )
            )
'''
new = '''            classifier_module = module.startswith("statgpu.linear_model")
            regression_module = module.startswith(
                (
                    "statgpu.linear_model",
                    "statgpu.nonparametric",
                    "statgpu.panel",
                    "statgpu.survival",
                    "statgpu.semiparametric",
                )
            )
'''
text = replace_once(text, old, new, "supervised module inventory")
old = '''            elif (
                "classifier" in name
                or "logistic" in name
                or "logit" in name
                or "probit" in name
            ) and classifier_module:
'''
new = '''            elif (
                "classifier" in name
                or "logistic" in name
                or "logit" in name
                or "probit" in name
                or "orderedgeneralizedlinearmodel" in name
            ) and classifier_module:
'''
text = replace_once(text, old, new, "ordered classifier inference")
old = '''                            "regression",
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
'''
new = '''                            "regression",
                            "generalizedlinearmodel",
                            "glm",
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
'''
text = replace_once(text, old, new, "GLM regressor inference")
p.write_text(text, encoding="utf-8")

p = Path("dev/tests/test_maintenance_024_025.py")
text = p.read_text(encoding="utf-8")
text += r'''


def test_supervised_generic_estimators_have_sklearn_types():
    from sklearn.base import is_classifier, is_regressor
    from statgpu.linear_model import (
        GeneralizedLinearModel,
        OrderedGeneralizedLinearModel,
        PenalizedGLM_CV,
        PenalizedGeneralizedLinearModel,
    )
    from statgpu.nonparametric import KernelRegression

    assert is_regressor(GeneralizedLinearModel())
    assert is_classifier(OrderedGeneralizedLinearModel())
    assert is_regressor(PenalizedGeneralizedLinearModel())
    assert is_regressor(PenalizedGLM_CV())
    assert is_regressor(KernelRegression())
'''
p.write_text(text, encoding="utf-8")

for path in ("statgpu/_base.py", "dev/tests/test_maintenance_024_025.py"):
    if not compileall.compile_file(path, quiet=1):
        raise SystemExit(f"compile failed: {path}")
