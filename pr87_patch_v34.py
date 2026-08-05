from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:180]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Enforce a real, non-empty scalar-response dtype before family-domain tests.
replace_once(
    "statgpu/glm_core/_base.py",
    '''        ndim = int(values.ndim)
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
    '''        ndim = int(values.ndim)
        if ndim == 2 and int(values.shape[1]) == 1:
            values = values.reshape(-1)
        elif ndim != 1:
            raise ValueError(
                f"{self.name} response must be one-dimensional; "
                "a single-column (n_samples, 1) response is also accepted."
            )
        if int(values.shape[0]) == 0:
            raise ValueError(
                f"{self.name} response must contain at least one observation."
            )

        if xp.__name__ == "torch":
            import torch

            nonreal = torch.is_complex(values)
        else:
            nonreal = getattr(values.dtype, "kind", "") not in "biuf"
        if bool(nonreal.item() if hasattr(nonreal, "item") else nonreal):
            raise ValueError(
                f"{self.name} response must contain real numeric values."
            )

        try:
            invalid = xp.any(~xp.isfinite(values))
        except (TypeError, RuntimeError) as exc:
            raise ValueError(
                f"{self.name} response must contain real numeric finite values."
            ) from exc
''',
)

# Regression coverage across public entrypoints and physical GPU backends.
tests = Path("dev/tests/test_maintenance_024_025.py")
text = tests.read_text(encoding="utf-8")
marker = "# PR87_GLM_REAL_NONEMPTY_RESPONSE_TESTS"
if marker not in text:
    text += '''

# PR87_GLM_REAL_NONEMPTY_RESPONSE_TESTS
@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
@pytest.mark.parametrize(
    "bad_y",
    [
        np.array([0.0 + 0.0j, 1.0 + 0.0j, 2.0 + 1.0j]),
        np.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64[D]"),
        np.array(["0", "1", "two"], dtype=object),
    ],
)
def test_glm_entrypoints_reject_nonreal_response_with_value_error(kind, bad_y, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    X = np.arange(6.0, dtype=np.float64).reshape(3, 2)
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
            model,
            "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for a nonreal response")
            ),
        )
    with pytest.raises(ValueError, match="response must contain real numeric"):
        model.fit(X, bad_y)


@pytest.mark.parametrize("kind", ["glm", "penalized", "cv"])
def test_glm_entrypoints_reject_empty_response_before_solver(kind, monkeypatch):
    from statgpu.linear_model import GeneralizedLinearModel
    from statgpu.linear_model.penalized import (
        PenalizedGeneralizedLinearModel,
        PenalizedGLM_CV,
    )

    X = np.empty((0, 2), dtype=np.float64)
    y = np.empty(0, dtype=np.float64)
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
            model,
            "_fit_standard",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CV must not start for an empty response")
            ),
        )
    with pytest.raises(ValueError, match="at least one observation"):
        model.fit(X, y)


def test_direct_irls_rejects_complex_and_empty_response():
    from statgpu.glm_core._family import Gaussian
    from statgpu.glm_core._irls import IRLSSolver

    with pytest.raises(ValueError, match="real numeric"):
        IRLSSolver(Gaussian()).fit(
            np.ones((2, 1)), np.array([1.0 + 0.0j, 2.0 + 1.0j]),
            backend="numpy",
        )
    with pytest.raises(ValueError, match="at least one observation"):
        IRLSSolver(Gaussian()).fit(
            np.empty((0, 1)), np.empty(0), backend="numpy"
        )


def test_torch_glm_complex_response_rejected_on_device():
    torch = _require_modern_torch_cuda()
    from statgpu.linear_model import GeneralizedLinearModel

    X = torch.ones((3, 1), dtype=torch.float64, device="cuda")
    y = torch.tensor([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 1.0j], device="cuda")
    with pytest.raises(ValueError, match="real numeric"):
        GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0,
            device="torch", compute_inference=False,
        ).fit(X, y)
    assert X.is_cuda and y.is_cuda


def test_cupy_penalized_glm_complex_response_rejected_on_device():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("requires a working CuPy CUDA backend")
    except Exception:
        pytest.skip("requires a working CuPy CUDA backend")
    from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel

    X = cp.ones((3, 1), dtype=cp.float64)
    y = cp.asarray([1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 1.0j])
    with pytest.raises(ValueError, match="real numeric"):
        PenalizedGeneralizedLinearModel(
            loss="squared_error", penalty="l2", alpha=0.1,
            solver="fista", device="cuda", compute_inference=False,
        ).fit(X, y)
    assert isinstance(X, cp.ndarray) and isinstance(y, cp.ndarray)
'''
    tests.write_text(text, encoding="utf-8")

replace_once(
    "CHANGELOG.md",
    '''  GLMs now normalize single-column responses and reject multicolumn or
  length-mismatched responses before solver/fold dispatch.
''',
    '''  GLMs now normalize single-column responses and reject empty, non-real,
  multicolumn, or length-mismatched responses before solver/fold dispatch.
''',
)
replace_once(
    "docs/en/changelog.md",
    '''  accept one-dimensional or single-column input and reject multicolumn or
  length-mismatched data before solver/fold dispatch.
''',
    '''  accept non-empty real one-dimensional or single-column input and reject
  non-real, multicolumn, or length-mismatched data before solver/fold dispatch.
''',
)
replace_once(
    "docs/cn/changelog.md",
    '''  支持一维或单列输入，并在 solver/fold dispatch 前拒绝多列或长度不匹配；active IRLS/FISTA 编译
''',
    '''  支持非空实数的一维或单列输入，并在 solver/fold dispatch 前拒绝非实数、多列或长度不匹配；
  active IRLS/FISTA 编译
''',
)
