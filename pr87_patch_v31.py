from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:180]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Accept array-like public responses while preserving native GPU arrays.
replace_once(
    "statgpu/glm_core/_base.py",
    '''        xp = _xp(y)
        invalid = xp.any(~xp.isfinite(y))
        y_type = str(getattr(self, "y_type", "continuous")).lower()
        if y_type == "binary":
            invalid = invalid | xp.any(y < 0) | xp.any(y > 1)
''',
    '''        xp = _xp(y)
        if xp.__name__ == "torch":
            import torch

            values = y if torch.is_tensor(y) else torch.as_tensor(y)
        else:
            values = xp.asarray(y)
        invalid = xp.any(~xp.isfinite(values))
        y_type = str(getattr(self, "y_type", "continuous")).lower()
        if y_type == "binary":
            invalid = invalid | xp.any(values < 0) | xp.any(values > 1)
''',
)
replace_once(
    "statgpu/glm_core/_base.py",
    '''        elif y_type in ("count", "nonnegative"):
            invalid = invalid | xp.any(y < 0)
            requirement = "non-negative values"
        elif y_type == "positive":
            invalid = invalid | xp.any(y <= 0)
''',
    '''        elif y_type in ("count", "nonnegative"):
            invalid = invalid | xp.any(values < 0)
            requirement = "non-negative values"
        elif y_type == "positive":
            invalid = invalid | xp.any(values <= 0)
''',
)

# Penalized estimators resolve one loss object; validate with that exact object
# before any solver/backend branch or robust-loss preprocessing can run.
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''        self._penalty = self._resolve_penalty()
        self._loss = self._resolve_loss()
        self._validate_solver_penalty()
''',
    '''        self._penalty = self._resolve_penalty()
        self._loss = self._resolve_loss()
        if hasattr(self._loss, "validate_response"):
            self._loss.validate_response(y)
        self._validate_solver_penalty()
''',
)

# CV has specialized fold-batched and manual refit paths that can bypass
# model.fit().  Reject invalid scalar GLM responses transactionally before any
# fold is constructed; Cox retains its dedicated two-column validation.
replace_once(
    "statgpu/linear_model/penalized/_penalized_cv.py",
    '''            if str(self.loss).lower() == "cox_ph":
                from ._penalized_cox_cv import fit_penalized_cox_cv

                return fit_penalized_cox_cv(
                    self, X, y, sample_weight=sample_weight
                )
            return self._fit_standard(X, y, sample_weight=sample_weight)
''',
    '''            if str(self.loss).lower() == "cox_ph":
                from ._penalized_cox_cv import fit_penalized_cox_cv

                return fit_penalized_cox_cv(
                    self, X, y, sample_weight=sample_weight
                )
            from statgpu.linear_model.penalized._fit_mixin import _resolve_loss_name

            resolved_loss = _resolve_loss_name(
                self.loss,
                loss_kwargs=getattr(self, "_loss_kwargs", None),
            )
            if hasattr(resolved_loss, "validate_response"):
                resolved_loss.validate_response(y)
            return self._fit_standard(X, y, sample_weight=sample_weight)
''',
)

# Tests: direct penalized, formula row ownership, CV transactional boundary,
# array-like input, and physical Torch/CuPy device purity.
tests = Path("dev/tests/test_maintenance_024_025.py")
text = tests.read_text(encoding="utf-8")
marker = "# PR87_PENALIZED_GLM_RESPONSE_DOMAIN_TESTS"
if marker not in text:
    text += '''

# PR87_PENALIZED_GLM_RESPONSE_DOMAIN_TESTS
@pytest.mark.parametrize(
    "loss,bad_y,message",
    [
        ("logistic", [0.0, 1.0, 1.2, 0.5], r"logistic response.*\\[0, 1\\]"),
        ("poisson", [0.0, 1.0, -1.0, 2.0], "poisson response.*non-negative"),
        ("gamma", [1.0, 2.0, 0.0, 3.0], "gamma response.*strictly positive"),
        ("inverse_gaussian", [1.0, 2.0, -0.1, 3.0], "inverse_gaussian response.*strictly positive"),
        ("negative_binomial", [0.0, 1.0, -1.0, 2.0], "negative_binomial response.*non-negative"),
        ("tweedie", [0.0, 1.0, -0.1, 2.0], "tweedie response.*non-negative"),
    ],
)
def test_penalized_glm_validates_array_like_response_before_solver(loss, bad_y, message):
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = np.arange(8.0, dtype=np.float64).reshape(4, 2)
    with pytest.raises(ValueError, match=message):
        PenalizedGeneralizedLinearModel(
            loss=loss,
            penalty="l2",
            alpha=0.1,
            solver="fista",
            device="cpu",
            compute_inference=False,
        ).fit(X, bad_y)


def test_penalized_formula_response_validation_uses_retained_rows():
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    data = pd.DataFrame(
        {"y": [0.0, 1.0, -3.0, 2.0, 4.0], "x": [0.0, 1.0, np.nan, 3.0, 4.0]}
    )
    model = PenalizedGeneralizedLinearModel(
        loss="poisson", penalty="l2", alpha=0.1,
        solver="fista", max_iter=100, device="cpu",
        compute_inference=False,
    ).fit(formula="y ~ x", data=data)
    assert np.isfinite(model.coef_).all()

    data.loc[1, "y"] = -1.0
    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        ).fit(formula="y ~ x", data=data)


def test_penalized_glm_cv_rejects_invalid_response_before_folds_and_resets_state(monkeypatch):
    from statgpu.linear_model.penalized import PenalizedGLM_CV

    X = np.arange(12.0, dtype=np.float64).reshape(6, 2)
    y = np.array([0.0, 1.0, 2.0, -1.0, 3.0, 4.0])
    model = PenalizedGLM_CV(
        loss="poisson", penalty="l2", alpha_grid=[0.1, 1.0],
        cv=2, device="cpu", max_iter=20,
    )
    fold_called = False

    def forbidden(*args, **kwargs):
        nonlocal fold_called
        fold_called = True
        raise AssertionError("CV folds must not run for an invalid response")

    monkeypatch.setattr(model, "_fit_standard", forbidden)
    model._fitted = True
    model.alpha_ = 99.0
    model.coef_ = np.ones(2)
    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        model.fit(X, y)
    assert not fold_called
    assert model._fitted is False
    assert model.alpha_ is None
    assert model.coef_ is None


def test_torch_penalized_glm_response_validation_stays_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = torch.arange(8.0, dtype=torch.float64, device="cuda").reshape(4, 2)
    y = torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="poisson response.*non-negative"):
        PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.1,
            solver="fista", device="torch", compute_inference=False,
        ).fit(X, y)
    assert X.is_cuda and y.is_cuda


def test_cupy_penalized_glm_response_validation_stays_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = cp.arange(8.0, dtype=cp.float64).reshape(4, 2)
    y = cp.asarray([1.0, 2.0, 0.0, 3.0], dtype=cp.float64)
    with pytest.raises(ValueError, match="gamma response.*strictly positive"):
        PenalizedGeneralizedLinearModel(
            loss="gamma", penalty="l2", alpha=0.1,
            solver="fista", device="cuda", compute_inference=False,
        ).fit(X, y)
    assert isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)
'''
    tests.write_text(text, encoding="utf-8")

# Clarify that the response-domain contract includes penalized and CV entrypoints.
replace_once(
    "CHANGELOG.md",
    '''  backend-native response-domain validation for every supported GLM family.
''',
    '''  backend-native response-domain validation for every supported GLM family,
  including penalized estimators and cross-validation entrypoints.
''',
)
replace_once(
    "docs/en/changelog.md",
    '''- Every supported GLM family now enforces its response domain before any solver
  dispatch, using NumPy, Torch, or CuPy reductions on the selected backend.
''',
    '''- Every supported GLM family, including penalized and CV estimators, now
  enforces its response domain before any solver or fold dispatch, using NumPy,
  Torch, or CuPy reductions on the selected backend.
''',
)
replace_once(
    "docs/cn/changelog.md",
    '''- 所有支持的 GLM family 都在 solver dispatch 之前执行 backend-native response-domain
  validation；active IRLS/FISTA 编译统一走 centralized compile policy，且不再把无关的
''',
    '''- 所有支持的 GLM family（包括 penalized 与 CV estimator）都在 solver 或 fold
  dispatch 之前执行 backend-native response-domain validation；active IRLS/FISTA 编译
  统一走 centralized compile policy，且不再把无关的
''',
)
