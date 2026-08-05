from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:180]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# GLM response validation owns array-like conversion and scalar-response shape.
replace_once(
    "statgpu/glm_core/_base.py",
    '''        xp = _xp(y)
        if xp.__name__ == "torch":
            import torch

            values = y if torch.is_tensor(y) else torch.as_tensor(y)
        else:
            values = xp.asarray(y)
        invalid = xp.any(~xp.isfinite(values))
''',
    '''        xp = _xp(y)
        if xp.__name__ == "torch":
            import torch

            values = y if torch.is_tensor(y) else torch.as_tensor(y)
        else:
            try:
                values = xp.asarray(y)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{self.name} response must be a numeric array-like."
                ) from exc

        ndim = int(values.ndim)
        if ndim == 2 and int(values.shape[1]) == 1:
            values = values.reshape(-1)
        elif ndim != 1:
            raise ValueError(
                f"{self.name} response must be one-dimensional; "
                "a single-column (n_samples, 1) response is also accepted."
            )

        try:
            invalid = xp.any(~xp.isfinite(values))
        except TypeError as exc:
            raise ValueError(
                f"{self.name} response must contain numeric finite values."
            ) from exc
''',
)
replace_once(
    "statgpu/glm_core/_base.py",
    '''        return y
''',
    '''        return values
''',
)

# Non-penalized GLM: assign normalized response and check length before solver.
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        fit_loss = self._resolve_loss_for_inference()
        fit_loss.validate_response(y_arr)
        _solver_lower = self._solver.lower() if isinstance(self._solver, str) else self._solver
''',
    '''        fit_loss = self._resolve_loss_for_inference()
        y_arr = fit_loss.validate_response(y_arr)
        if int(y_arr.shape[0]) != int(X_arr.shape[0]):
            raise ValueError("Response length must match X.shape[0].")
        _solver_lower = self._solver.lower() if isinstance(self._solver, str) else self._solver
''',
)

# Direct IRLS callers receive the same normalization and length contract.
replace_once(
    "statgpu/glm_core/_irls.py",
    '''    objective_loss.validate_response(y_work)
''',
    '''    y_work = objective_loss.validate_response(y_work)
    if int(y_work.shape[0]) != int(X.shape[0]):
        raise ValueError("Response length must match X.shape[0].")
''',
)

# Penalized model assigns the normalized response before initialization/solver.
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''        if hasattr(self._loss, "validate_response"):
            self._loss.validate_response(y)
        self._validate_solver_penalty()
''',
    '''        if hasattr(self._loss, "validate_response"):
            y = self._loss.validate_response(y)
            if int(y.shape[0]) != int(X.shape[0]):
                raise ValueError("Response length must match X.shape[0].")
        self._validate_solver_penalty()
''',
)

# CV normalizes once before any fold slicing, preserving transactional reset.
replace_once(
    "statgpu/linear_model/penalized/_penalized_cv.py",
    '''            if hasattr(resolved_loss, "validate_response"):
                resolved_loss.validate_response(y)
            return self._fit_standard(X, y, sample_weight=sample_weight)
''',
    '''            if hasattr(resolved_loss, "validate_response"):
                y = resolved_loss.validate_response(y)
                if int(y.shape[0]) != int(X.shape[0]):
                    raise ValueError("Response length must match X.shape[0].")
            return self._fit_standard(X, y, sample_weight=sample_weight)
''',
)

# Regression matrix for shape, length, single-column compatibility, CV boundary,
# and native GPU shape rejection.
tests = Path("dev/tests/test_maintenance_024_025.py")
text = tests.read_text(encoding="utf-8")
marker = "# PR87_GLM_RESPONSE_SHAPE_CONTRACT_TESTS"
if marker not in text:
    text += '''

# PR87_GLM_RESPONSE_SHAPE_CONTRACT_TESTS
@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
def test_scalar_glm_rejects_multicolumn_response_before_solver(kind, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    X = np.arange(12.0, dtype=np.float64).reshape(6, 2)
    y = np.ones((6, 2), dtype=np.float64)
    if kind == "glm":
        model = GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        )
    elif kind == "penalized":
        model = PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        )
    else:
        model = PenalizedGLM_CV(
            loss="poisson", penalty="l2", alpha_grid=[0.1, 1.0],
            cv=2, device="cpu", max_iter=20,
        )
        monkeypatch.setattr(
            model,
            "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for multicolumn y")
            ),
        )
    with pytest.raises(ValueError, match="response must be one-dimensional"):
        model.fit(X, y)


@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
def test_scalar_glm_rejects_response_length_mismatch_before_solver(kind, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    X = np.arange(12.0, dtype=np.float64).reshape(6, 2)
    y = np.ones(5, dtype=np.float64)
    if kind == "glm":
        model = GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        )
    elif kind == "penalized":
        model = PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        )
    else:
        model = PenalizedGLM_CV(
            loss="poisson", penalty="l2", alpha_grid=[0.1, 1.0],
            cv=2, device="cpu", max_iter=20,
        )
        monkeypatch.setattr(
            model,
            "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for length mismatch")
            ),
        )
    with pytest.raises(ValueError, match=r"Response length must match X\.shape\[0\]"):
        model.fit(X, y)


def test_scalar_glm_accepts_single_column_response_consistently():
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = np.array([[-1.0], [0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([[0.0], [1.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    glm = GeneralizedLinearModel(
        family="poisson", solver="irls", C=0.0,
        device="cpu", compute_inference=False,
    ).fit(X, y)
    penalized = PenalizedGeneralizedLinearModel(
        loss="poisson", penalty="l2", alpha=0.1,
        solver="fista", max_iter=100, device="cpu",
        compute_inference=False,
    ).fit(X, y)
    assert np.isfinite(glm.coef_).all()
    assert np.isfinite(penalized.coef_).all()


def test_direct_irls_rejects_multicolumn_and_length_mismatch():
    from statgpu.glm_core._family import Poisson
    from statgpu.glm_core._irls import IRLSSolver

    X = np.ones((4, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="response must be one-dimensional"):
        IRLSSolver(Poisson()).fit(X, np.ones((4, 2)), backend="numpy")
    with pytest.raises(ValueError, match=r"Response length must match X\.shape\[0\]"):
        IRLSSolver(Poisson()).fit(X, np.ones(3), backend="numpy")


def test_glm_rejects_nonnumeric_response_with_public_value_error():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.ones((3, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="numeric finite values"):
        GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        ).fit(X, np.array(["0", "one", "2"], dtype=object))


def test_torch_glm_multicolumn_response_rejected_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model import GeneralizedLinearModel

    X = torch.ones((4, 2), dtype=torch.float64, device="cuda")
    y = torch.ones((4, 2), dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="response must be one-dimensional"):
        GeneralizedLinearModel(
            family="poisson", solver="irls", C=0.0,
            device="torch", compute_inference=False,
        ).fit(X, y)
    assert X.is_cuda and y.is_cuda


def test_cupy_penalized_glm_multicolumn_response_rejected_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = cp.ones((4, 2), dtype=cp.float64)
    y = cp.ones((4, 2), dtype=cp.float64)
    with pytest.raises(ValueError, match="response must be one-dimensional"):
        PenalizedGeneralizedLinearModel(
            loss="poisson", penalty="l2", alpha=0.1,
            solver="fista", device="cuda", compute_inference=False,
        ).fit(X, y)
    assert isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)
'''
    tests.write_text(text, encoding="utf-8")

replace_once(
    "CHANGELOG.md",
    '''  including penalized estimators and cross-validation entrypoints.
''',
    '''  including penalized estimators and cross-validation entrypoints; scalar
  GLMs now normalize single-column responses and reject multicolumn or
  length-mismatched responses before solver/fold dispatch.
''',
)
replace_once(
    "docs/en/changelog.md",
    '''  Torch, or CuPy reductions on the selected backend.
''',
    '''  Torch, or CuPy reductions on the selected backend. Scalar GLM responses
  accept one-dimensional or single-column input and reject multicolumn or
  length-mismatched data before solver/fold dispatch.
''',
)
replace_once(
    "docs/cn/changelog.md",
    '''  dispatch 之前执行 backend-native response-domain validation；active IRLS/FISTA 编译
''',
    '''  dispatch 之前执行 backend-native response-domain validation；scalar GLM response
  支持一维或单列输入，并在 solver/fold dispatch 前拒绝多列或长度不匹配；active IRLS/FISTA 编译
''',
)
