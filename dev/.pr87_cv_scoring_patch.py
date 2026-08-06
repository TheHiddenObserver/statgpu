from pathlib import Path

source_path = Path("statgpu/linear_model/penalized/_penalized_cv.py")
source = source_path.read_text(encoding="utf-8")
old = '''        (
            NotImplementedError,
            ValueError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ),
'''
new = '''        (
            NotImplementedError,
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
        ),
'''
if source.count(old) != 1:
    raise RuntimeError(f"recoverable loss tuple count={source.count(old)}")
source = source.replace(old, new, 1)
source_path.write_text(source, encoding="utf-8")

review_path = Path("dev/tests/test_pr87_code_review_fix_cycle.py")
review = review_path.read_text(encoding="utf-8")
old_test = '''def test_cv_valueerror_retry_is_visible_but_typeerror_stays_fatal(monkeypatch):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    class Model:
        coef_ = np.array([0.0])
        intercept_ = 0.0
        fit_intercept = True
        def predict(self, X):
            return np.zeros(len(X))

    class Loss:
        def value(self, *args, **kwargs):
            return 1.25

    owner = object.__new__(cv_mod.PenalizedGLM_CV)
    owner.loss = "poisson"
    monkeypatch.setattr(
        cv_mod, '_evaluate_loss_numpy',
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('registered evaluator unavailable')),
    )
    with pytest.warns(RuntimeWarning, match='generic loss interface'):
        assert owner._evaluate_single(
            Model(), np.ones((2, 1)), np.ones(2), loss_fn=Loss()
        ) == pytest.approx(1.25)
    monkeypatch.setattr(
        cv_mod, '_evaluate_loss_numpy',
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError('programming bug')),
    )
    with pytest.raises(TypeError, match='programming bug'):
        owner._evaluate_single(
            Model(), np.ones((2, 1)), np.ones(2), loss_fn=Loss()
        )
'''
new_test = '''@pytest.mark.parametrize(
    "exc",
    [
        ValueError("programming shape bug"),
        TypeError("programming signature bug"),
    ],
)
def test_cv_scoring_programming_errors_stay_fatal(monkeypatch, exc):
    import statgpu.linear_model.penalized._penalized_cv as cv_mod

    class Model:
        coef_ = np.array([0.0])
        intercept_ = 0.0
        fit_intercept = True

        def predict(self, X):
            return np.zeros(len(X))

    class Loss:
        def value(self, *args, **kwargs):
            pytest.fail("programming errors must not retry generic scoring")

    owner = object.__new__(cv_mod.PenalizedGLM_CV)
    owner.loss = "poisson"
    monkeypatch.setattr(
        cv_mod,
        "_evaluate_loss_numpy",
        lambda *args, **kwargs: (_ for _ in ()).throw(exc),
    )
    with pytest.raises(type(exc), match="programming"):
        owner._evaluate_single(
            Model(), np.ones((2, 1)), np.ones(2), loss_fn=Loss()
        )
'''
if review.count(old_test) != 1:
    raise RuntimeError(f"review scoring test count={review.count(old_test)}")
review = review.replace(old_test, new_test, 1)
review_path.write_text(review, encoding="utf-8")

maintenance_path = Path("dev/tests/test_maintenance_024_025.py")
maintenance = maintenance_path.read_text(encoding="utf-8")
replacements = [
    (
        'raise ValueError("generic poisson evaluation failed")',
        'raise FloatingPointError("generic poisson evaluation failed")',
    ),
    (
        'ValueError("registered poisson evaluation failed")',
        'FloatingPointError("registered poisson evaluation failed")',
    ),
    (
        'raise ValueError("generic squared evaluation failed")',
        'raise FloatingPointError("generic squared evaluation failed")',
    ),
    (
        'ValueError("registered squared evaluation failed")',
        'FloatingPointError("registered squared evaluation failed")',
    ),
]
for before, after in replacements:
    if maintenance.count(before) != 1:
        raise RuntimeError(f"maintenance anchor {before!r} count={maintenance.count(before)}")
    maintenance = maintenance.replace(before, after, 1)
maintenance_path.write_text(maintenance, encoding="utf-8")
