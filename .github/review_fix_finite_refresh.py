from pathlib import Path
import compileall


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


# Include inherited and post-bound methods when installing finite guards.
p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
old = '''        for method_name, original in tuple(cls.__dict__.items()):
            if method_name.startswith("_") or not callable(original):
                continue
            if getattr(original, "__isabstractmethod__", False):
                continue
            if getattr(original, "__statgpu_finite_validation__", False):
                continue
            try:
                signature = inspect.signature(original)
            except (TypeError, ValueError):
                continue
            numerical_parameters = set(signature.parameters) & cls._FINITE_PARAMETER_NAMES
            if method_name not in cls._FINITE_PUBLIC_METHODS and not numerical_parameters:
                continue
            setattr(cls, method_name, wrap_method(original))
'''
new = '''        candidate_names = set(cls._FINITE_PUBLIC_METHODS)
        candidate_names.update(
            name for name in dir(cls) if not name.startswith("_")
        )
        for method_name in sorted(candidate_names):
            original = getattr(cls, method_name, None)
            if not callable(original):
                continue
            if getattr(original, "__isabstractmethod__", False):
                continue
            if getattr(original, "__statgpu_finite_validation__", False):
                continue
            try:
                signature = inspect.signature(original)
            except (TypeError, ValueError):
                continue
            numerical_parameters = set(signature.parameters) & cls._FINITE_PARAMETER_NAMES
            if method_name not in cls._FINITE_PUBLIC_METHODS and not numerical_parameters:
                continue
            setattr(cls, method_name, wrap_method(original))
'''
text = replace_once(text, old, new, "MRO finite method inventory")
old = '''

BaseEstimator._install_public_finite_validation()
'''
new = '''

def refresh_public_finite_validation_contracts():
    """Install finite guards after all estimator mixins and aliases are bound."""
    BaseEstimator._install_public_finite_validation()
    seen = set()
    stack = list(BaseEstimator.__subclasses__())
    while stack:
        estimator_cls = stack.pop()
        if estimator_cls in seen:
            continue
        seen.add(estimator_cls)
        stack.extend(estimator_cls.__subclasses__())
        estimator_cls._install_public_finite_validation()


BaseEstimator._install_public_finite_validation()
'''
text = replace_once(text, old, new, "finite refresh function")
p.write_text(text, encoding="utf-8")


# Run the refresh only after the public package has imported all estimator families.
p = Path("statgpu/__init__.py")
text = p.read_text(encoding="utf-8")
append = '''

# Some maintained methods are supplied by mixins or attached after class
# creation. Refresh finite-value guards after the complete public API is bound.
from ._base import refresh_public_finite_validation_contracts as _refresh_finite_contracts

_refresh_finite_contracts()
del _refresh_finite_contracts
'''
if "_refresh_finite_contracts" in text:
    raise SystemExit("finite refresh already installed")
p.write_text(text + append, encoding="utf-8")


# Knockoff selectors are sklearn-like but intentionally do not inherit BaseEstimator.
# Their methods already validate inputs manually; expose that fact to structural audits.
p = Path("statgpu/feature_selection/_knockoff.py")
text = p.read_text(encoding="utf-8")
append = '''

for _selector_cls in (KnockoffSelector, FixedXKnockoffSelector):
    for _method_name in ("fit", "fit_transform", "transform"):
        _method = getattr(_selector_cls, _method_name, None)
        if callable(_method):
            _method.__statgpu_finite_validation__ = True
del _selector_cls, _method_name, _method
'''
if "_method.__statgpu_finite_validation__" in text:
    raise SystemExit("knockoff finite markers already installed")
p.write_text(text + append, encoding="utf-8")


# Add both structural and behavioral regressions.
p = Path("dev/tests/test_maintenance_024_025.py")
text = p.read_text(encoding="utf-8")
text += r'''


def test_all_public_numeric_methods_expose_finite_contract():
    import inspect
    import statgpu

    candidate_names = {
        "X", "X_new", "x", "y", "sample_weight", "weights", "offset",
        "exposure", "entry", "start", "stop", "time", "event", "times",
        "cluster", "clusters", "strata", "subject", "subjects", "groups",
        "init", "init_coef", "initial_coef", "time_index", "entity_ids",
        "time_ids", "pvalues", "arrays", "scores", "thresholds", "Xk",
        "mu", "Sigma",
    }
    missing = []
    for estimator_name, estimator in _default_public_estimators():
        cls = type(estimator)
        for method_name in dir(cls):
            if method_name.startswith("_"):
                continue
            method = getattr(cls, method_name, None)
            if not callable(method):
                continue
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError):
                continue
            if not (set(signature.parameters) & candidate_names):
                continue
            if not getattr(method, "__statgpu_finite_validation__", False):
                missing.append((estimator_name, method_name))
    assert missing == []


def test_inherited_penalized_fit_rejects_nonfinite_before_solver():
    from statgpu.linear_model import PenalizedLinearRegression

    model = PenalizedLinearRegression(compute_inference=False, device="cpu")
    X = np.array([[1.0, 0.0], [np.nan, 1.0], [2.0, 2.0]])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"X.*finite"):
        model.fit(X, y)
    assert model._fitted is False


def test_inherited_ridge_predict_rejects_nonfinite():
    from statgpu.linear_model import Ridge

    X = np.arange(24, dtype=float).reshape(8, 3)
    y = np.arange(8, dtype=float)
    model = Ridge(compute_inference=False, device="cpu").fit(X, y)
    bad = X.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"X.*finite"):
        model.predict(bad)


def test_inherited_lasso_score_rejects_nonfinite_target():
    from statgpu.linear_model import Lasso

    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10, dtype=float)
    model = Lasso(alpha=0.01, compute_inference=False, device="cpu").fit(X, y)
    bad_y = y.copy()
    bad_y[0] = np.nan
    with pytest.raises(ValueError, match=r"y.*finite"):
        model.score(X, bad_y)


def test_knockoff_manual_validation_is_marked():
    from statgpu.feature_selection import FixedXKnockoffSelector, KnockoffSelector

    for cls in (KnockoffSelector, FixedXKnockoffSelector):
        for method_name in ("fit", "fit_transform", "transform"):
            assert getattr(
                getattr(cls, method_name),
                "__statgpu_finite_validation__",
                False,
            )
'''
p.write_text(text, encoding="utf-8")

for path in (
    "statgpu/_base.py",
    "statgpu/__init__.py",
    "statgpu/feature_selection/_knockoff.py",
    "dev/tests/test_maintenance_024_025.py",
):
    if not compileall.compile_file(path, quiet=1):
        raise SystemExit(f"compile failed: {path}")
