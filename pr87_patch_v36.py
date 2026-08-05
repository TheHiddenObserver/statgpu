from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:180]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


validation = '''"""Backend-native validation for scalar GLM design and weight inputs."""

from __future__ import annotations

import numpy as np


def _as_native_array(value, *, name):
    """Return an array while preserving existing Torch/CuPy device residency."""
    module = type(value).__module__
    if module.startswith("pandas"):
        value = value.to_numpy()
        module = type(value).__module__
    if module.startswith("torch"):
        import torch

        return value if torch.is_tensor(value) else torch.as_tensor(value)
    if module.startswith("cupy"):
        import cupy as cp

        return value if isinstance(value, cp.ndarray) else cp.asarray(value)
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array-like.") from exc


def _require_real_finite(values, *, name):
    module = type(values).__module__
    if module.startswith("torch"):
        import torch

        if torch.is_complex(values):
            raise ValueError(f"{name} must contain real numeric values.")
        if not bool(torch.all(torch.isfinite(values)).item()):
            raise ValueError(f"{name} must contain finite values.")
        return
    if module.startswith("cupy"):
        import cupy as cp

        if getattr(values.dtype, "kind", "") not in "biuf":
            raise ValueError(f"{name} must contain real numeric values.")
        if not bool(cp.all(cp.isfinite(values)).item()):
            raise ValueError(f"{name} must contain finite values.")
        return

    if getattr(values.dtype, "kind", "") not in "biuf":
        raise ValueError(f"{name} must contain real numeric values.")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError(f"{name} must contain finite values.")


def validate_glm_design_matrix(X, *, name="X"):
    """Validate a dense scalar-GLM design matrix and return its native array."""
    values = _as_native_array(X, name=name)
    if int(values.ndim) != 2:
        raise ValueError(f"{name} must be a two-dimensional design matrix.")
    if int(values.shape[0]) == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    _require_real_finite(values, name=name)
    return values


def validate_glm_sample_weight(sample_weight, n_samples, *, name="sample_weight"):
    """Validate analytic sample weights without copying GPU arrays to NumPy."""
    values = _as_native_array(sample_weight, name=name)
    if int(values.ndim) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if int(values.shape[0]) != int(n_samples):
        raise ValueError(f"{name} must have length n_samples")
    _require_real_finite(values, name=name)

    module = type(values).__module__
    if module.startswith("torch"):
        import torch

        if bool(torch.any(values < 0).item()):
            raise ValueError(f"{name} must be non-negative")
        total = float(torch.sum(values).item())
    elif module.startswith("cupy"):
        import cupy as cp

        if bool(cp.any(values < 0).item()):
            raise ValueError(f"{name} must be non-negative")
        total = float(cp.sum(values).item())
    else:
        if np.any(values < 0):
            raise ValueError(f"{name} must be non-negative")
        total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError(f"{name} must have a positive sum")
    return values
'''
Path("statgpu/glm_core/_validation.py").write_text(validation, encoding="utf-8")

# Formula alignment owns retained-row selection, then delegates all semantic
# validation to the shared GLM weight contract.
replace_once(
    "statgpu/core/formula/_alignment.py",
    '''import numpy as np

from statgpu.backends._validation import check_finite
''',
    '''import numpy as np

from statgpu.glm_core._validation import validate_glm_sample_weight
''',
)
start_marker = '''    check_finite(aligned, name="sample_weight")
'''
path = Path("statgpu/core/formula/_alignment.py")
text = path.read_text(encoding="utf-8")
start = text.index(start_marker)
end = text.index("    return aligned", start) + len("    return aligned")
text = text[:start] + '''    return validate_glm_sample_weight(
        aligned, retained_length, name="sample_weight"
    )''' + text[end:]
path.write_text(text, encoding="utf-8")

# Public GLM validates raw/direct inputs before backend conversion and uses the
# same sample-weight contract for formula and direct fits.
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''        backend = self._get_backend(backend="auto")
        backend_name = backend.name

        # Handle formula interface
''',
    '''        backend = self._get_backend(backend="auto")
        backend_name = backend.name
        from statgpu.glm_core._validation import (
            validate_glm_design_matrix,
            validate_glm_sample_weight,
        )
        fit_loss = self._resolve_loss_for_inference()

        # Handle formula interface
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''            # Formula produces numpy; convert to backend
            y_arr = self._to_array(y_arr, backend=backend_name)
            X_arr = self._to_array(X_arr, backend=backend_name)
''',
    '''            X_arr = validate_glm_design_matrix(X_arr)
            y_arr = fit_loss.validate_response(y_arr)
            # Formula produces NumPy; convert validated arrays to backend.
            y_arr = self._to_array(y_arr, backend=backend_name)
            X_arr = self._to_array(X_arr, backend=backend_name)
''',
)
replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''            # _to_array safely handles numpy/cupy/torch inputs
            y_arr = self._to_array(y, backend=backend_name)
            X_arr = self._to_array(X, backend=backend_name)
''',
    '''            X_validated = validate_glm_design_matrix(X)
            y_validated = fit_loss.validate_response(y)
            y_arr = self._to_array(y_validated, backend=backend_name)
            X_arr = self._to_array(X_validated, backend=backend_name)
''',
)
old_weight = '''        if sample_weight is not None:
            sample_weight = self._to_array(sample_weight, backend=backend_name)
            if int(sample_weight.ndim) != 1:
                raise ValueError("sample_weight must be one-dimensional")
            if int(sample_weight.shape[0]) != int(self._nobs):
                raise ValueError("sample_weight must have length n_samples")
            from statgpu.backends._validation import check_finite

            check_finite(sample_weight, name="sample_weight")
            if backend_name == "torch":
                import torch

                if bool(torch.any(sample_weight < 0).item()):
                    raise ValueError("sample_weight must be non-negative")
                weight_sum = float(torch.sum(sample_weight).item())
            elif backend_name == "cupy":
                import cupy as cp

                if bool(cp.any(sample_weight < 0).item()):
                    raise ValueError("sample_weight must be non-negative")
                weight_sum = float(cp.sum(sample_weight).item())
            else:
                if np.any(np.asarray(sample_weight) < 0):
                    raise ValueError("sample_weight must be non-negative")
                weight_sum = float(np.sum(np.asarray(sample_weight)))
            if weight_sum <= 0.0:
                raise ValueError("sample_weight must have a positive sum")

        family = self._get_family()
        fit_loss = self._resolve_loss_for_inference()
        y_arr = fit_loss.validate_response(y_arr)
'''
new_weight = '''        if sample_weight is not None:
            sample_weight = validate_glm_sample_weight(
                sample_weight, self._nobs
            )
            sample_weight = self._to_array(sample_weight, backend=backend_name)

        family = self._get_family()
        y_arr = fit_loss.validate_response(y_arr)
'''
replace_once("statgpu/linear_model/_glm_base.py", old_weight, new_weight)

# Direct IRLS normalizes/validates X and weights before shape access or casts.
replace_once(
    "statgpu/glm_core/_irls.py",
    '''    if backend == "auto":
        backend = _infer_backend(X)

    if init_coef is None:
        n_features = X.shape[1]
''',
    '''    from statgpu.glm_core._validation import (
        validate_glm_design_matrix,
        validate_glm_sample_weight,
    )

    X_validated = validate_glm_design_matrix(X)
    if backend == "auto":
        backend = _infer_backend(X_validated)
    X = _to_backend(X_validated, backend, X_validated)

    if init_coef is None:
        n_features = X.shape[1]
''',
)
replace_once(
    "statgpu/glm_core/_irls.py",
    '''    sw_work = (
        _to_backend(sample_weight, backend, X)
        if sample_weight is not None else None
    )
''',
    '''    sw_validated = (
        validate_glm_sample_weight(sample_weight, X.shape[0])
        if sample_weight is not None else None
    )
    sw_work = (
        _to_backend(sw_validated, backend, X)
        if sw_validated is not None else None
    )
''',
)

# Penalized estimators normalize X before feature access and validate raw
# weights before any reshape/backend cast.
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''        # Record number of features for sklearn compatibility
        if X is not None:
            X_arr = np.asarray(X) if not hasattr(X, 'shape') else X
            self.n_features_in_ = X_arr.shape[1] if X_arr.ndim >= 2 else 1

        self._penalty = self._resolve_penalty()
''',
    '''        from statgpu.glm_core._validation import (
            validate_glm_design_matrix,
            validate_glm_sample_weight,
        )
        X = validate_glm_design_matrix(X)
        self.n_features_in_ = int(X.shape[1])

        self._penalty = self._resolve_penalty()
''',
)
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''        _sw_arr = None
        if sample_weight is not None:
            _sw_arr = self._to_array(sample_weight, backend=backend_name).reshape(-1)
            _validate_sample_weight_backend(_sw_arr, X.shape[0], backend_name)
''',
    '''        _sw_arr = None
        if sample_weight is not None:
            sample_weight = validate_glm_sample_weight(
                sample_weight, X.shape[0]
            )
            _sw_arr = self._to_array(sample_weight, backend=backend_name)
''',
)

# CV validates non-Cox scalar designs/weights before any fold.  X is not
# replaced, preserving the established list-design identity contract.
replace_once(
    "statgpu/linear_model/penalized/_penalized_cv.py",
    '''            from statgpu.linear_model.penalized._fit_mixin import _resolve_loss_name

            resolved_loss = _resolve_loss_name(
''',
    '''            from statgpu.linear_model.penalized._fit_mixin import _resolve_loss_name
            from statgpu.glm_core._validation import (
                validate_glm_design_matrix,
                validate_glm_sample_weight,
            )

            validated_X = validate_glm_design_matrix(X)
            resolved_loss = _resolve_loss_name(
''',
)
replace_once(
    "statgpu/linear_model/penalized/_penalized_cv.py",
    '''                if int(y.shape[0]) != int(len(X)):
                    raise ValueError("Response length must match the number of X rows.")
            return self._fit_standard(X, y, sample_weight=sample_weight)
''',
    '''                if int(y.shape[0]) != int(validated_X.shape[0]):
                    raise ValueError("Response length must match the number of X rows.")
            if sample_weight is not None:
                sample_weight = validate_glm_sample_weight(
                    sample_weight, validated_X.shape[0]
                )
            return self._fit_standard(X, y, sample_weight=sample_weight)
''',
)

# Regression tests.
tests = Path("dev/tests/test_maintenance_024_025.py")
text = tests.read_text(encoding="utf-8")
marker = "# PR87_GLM_DESIGN_AND_WEIGHT_CONTRACT_TESTS"
if marker not in text:
    text += '''

# PR87_GLM_DESIGN_AND_WEIGHT_CONTRACT_TESTS
@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
@pytest.mark.parametrize(
    "bad_X,message",
    [
        (np.ones(4), "two-dimensional design matrix"),
        (np.ones((2, 2, 1)), "two-dimensional design matrix"),
        (np.empty((0, 2)), "at least one observation"),
        (np.ones((4, 2), dtype=np.complex128), "real numeric values"),
        (np.array([["a"], ["b"], ["c"], ["d"]], dtype=object), "real numeric values"),
    ],
)
def test_glm_entrypoints_reject_invalid_design_before_solver(kind, bad_X, message, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    y = np.arange(len(bad_X) if getattr(bad_X, "ndim", 0) else 4, dtype=float)
    if bad_X.shape[0] == 0:
        y = np.empty(0)
    if kind == "glm":
        model = GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        )
    elif kind == "penalized":
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        )
    else:
        model = PenalizedGLM_CV(
            loss="squared_error", penalty="l2", alpha_grid=[0.1],
            cv=2, device="cpu", max_iter=10,
        )
        monkeypatch.setattr(
            model, "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for invalid X")
            ),
        )
    with pytest.raises(ValueError, match=message):
        model.fit(bad_X, y)


def test_glm_intercept_only_zero_feature_design_is_supported():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.empty((5, 0), dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    model = GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0,
        device="cpu", compute_inference=False,
    ).fit(X, y)
    assert model.coef_.shape == (0,)
    np.testing.assert_allclose(model.intercept_, np.mean(y))


@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
@pytest.mark.parametrize(
    "bad_weight,message",
    [
        (np.ones((4, 1)), "one-dimensional"),
        (np.ones(3), "length n_samples"),
        (np.array([1.0, 1.0j, 1.0, 1.0]), "real numeric values"),
        (np.array(["1", "1", "1", "1"], dtype=object), "real numeric values"),
    ],
)
def test_glm_entrypoints_reject_invalid_sample_weight_consistently(kind, bad_weight, message, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    X = np.arange(8.0).reshape(4, 2)
    y = np.arange(4.0)
    if kind == "glm":
        model = GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="cpu", compute_inference=False,
        )
    elif kind == "penalized":
        model = PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cpu", compute_inference=False,
        )
    else:
        model = PenalizedGLM_CV(
            loss="squared_error", penalty="l2", alpha_grid=[0.1],
            cv=2, device="cpu", max_iter=10,
        )
        monkeypatch.setattr(
            model, "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for invalid weights")
            ),
        )
    with pytest.raises(ValueError, match=message):
        model.fit(X, y, sample_weight=bad_weight)


def test_direct_irls_validates_design_and_weights_before_backend_math():
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    with pytest.raises(ValueError, match="two-dimensional design matrix"):
        IRLSSolver(Gaussian()).fit(np.ones(4), np.ones(4), backend="numpy")
    with pytest.raises(ValueError, match="real numeric values"):
        IRLSSolver(Gaussian()).fit(
            np.ones((4, 1)), np.ones(4),
            sample_weight=np.array([1.0, 1.0j, 1.0, 1.0]),
            backend="numpy",
        )


def test_penalized_cv_design_validation_preserves_list_X_identity(monkeypatch):
    from statgpu.linear_model.penalized import PenalizedGLM_CV

    X = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    y = [0.0, 1.0, 2.0, 3.0]
    model = PenalizedGLM_CV(
        loss="squared_error", penalty="l2", alpha_grid=[0.1],
        cv=2, device="cpu", max_iter=10,
    )
    seen = {}
    def capture(X_arg, y_arg, sample_weight=None):
        seen["X"] = X_arg
        seen["y"] = y_arg
        return model
    monkeypatch.setattr(model, "_fit_standard", capture)
    model.fit(X, y)
    assert seen["X"] is X
    assert isinstance(seen["y"], np.ndarray)


def test_torch_glm_complex_design_and_weight_rejected_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model import GeneralizedLinearModel

    X_complex = torch.ones((4, 2), dtype=torch.complex128, device="cuda")
    y = torch.arange(4.0, dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="real numeric values"):
        GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="torch", compute_inference=False,
        ).fit(X_complex, y)

    X = torch.ones((4, 2), dtype=torch.float64, device="cuda")
    weight = torch.ones(4, dtype=torch.complex128, device="cuda")
    with pytest.raises(ValueError, match="real numeric values"):
        GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="torch", compute_inference=False,
        ).fit(X, y, sample_weight=weight)
    assert X_complex.is_cuda and weight.is_cuda


def test_cupy_penalized_glm_complex_design_and_weight_rejected_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    y = cp.arange(4.0, dtype=cp.float64)
    X_complex = cp.ones((4, 2), dtype=cp.complex128)
    with pytest.raises(ValueError, match="real numeric values"):
        PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cuda", compute_inference=False,
        ).fit(X_complex, y)

    X = cp.ones((4, 2), dtype=cp.float64)
    weight = cp.ones(4, dtype=cp.complex128)
    with pytest.raises(ValueError, match="real numeric values"):
        PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cuda", compute_inference=False,
        ).fit(X, y, sample_weight=weight)
    assert isinstance(X_complex, cp.ndarray) and isinstance(weight, cp.ndarray)
'''
    tests.write_text(text, encoding="utf-8")

replace_once(
    "CHANGELOG.md",
    '''  multicolumn, or length-mismatched responses before solver/fold dispatch.
''',
    '''  multicolumn, or length-mismatched responses before solver/fold dispatch;
  GLM design matrices and analytic weights now share backend-native real,
  finite, shape, length, and non-empty validation across model, CV, formula,
  and direct IRLS entrypoints.
''',
)
replace_once(
    "docs/en/changelog.md",
    '''  non-real, multicolumn, or length-mismatched data before solver/fold dispatch.
''',
    '''  non-real, multicolumn, or length-mismatched data before solver/fold dispatch.
  Design matrices and analytic sample weights now use the same backend-native
  real/finite/shape/length contract in model, formula, CV, and direct IRLS paths.
''',
)
replace_once(
    "docs/cn/changelog.md",
    '''  active IRLS/FISTA 编译
''',
    '''  design matrix 与 analytic sample weight 也在 model、formula、CV 和 direct IRLS
  路径中共享 backend-native 的实数、finite、shape 与 length 契约；active IRLS/FISTA 编译
''',
)
