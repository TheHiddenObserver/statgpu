from pathlib import Path
import runpy

runpy.run_path("pr87_patch_v32.py", run_name="__main__")


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:160]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Preserve existing list-design support.  Response length checks only need the
# public sample count and must not require an ndarray-like ``shape`` attribute.
replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''            if int(y.shape[0]) != int(X.shape[0]):
                raise ValueError("Response length must match X.shape[0].")
''',
    '''            if int(y.shape[0]) != int(len(X)):
                raise ValueError("Response length must match the number of X rows.")
''',
)
replace_once(
    "statgpu/linear_model/penalized/_penalized_cv.py",
    '''                if int(y.shape[0]) != int(X.shape[0]):
                    raise ValueError("Response length must match X.shape[0].")
''',
    '''                if int(y.shape[0]) != int(len(X)):
                    raise ValueError("Response length must match the number of X rows.")
''',
)

# Keep public tests implementation-agnostic about whether X exposes ``shape``.
tests = Path("dev/tests/test_maintenance_024_025.py")
text = tests.read_text(encoding="utf-8")
text = text.replace(
    r'''match=r"Response length must match X\.shape\[0\]"''',
    r'''match=r"Response length must match (?:X\.shape\[0\]|the number of X rows)"''',
)
marker = "# PR87_GLM_LIST_DESIGN_LENGTH_TEST"
if marker not in text:
    text += '''

# PR87_GLM_LIST_DESIGN_LENGTH_TEST
def test_penalized_glm_cv_response_length_check_preserves_list_design_input(monkeypatch):
    from statgpu.linear_model.penalized import PenalizedGLM_CV

    X = [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
    y = [0.0, 1.0, 2.0, 3.0]
    model = PenalizedGLM_CV(
        loss="poisson", penalty="l2", alpha_grid=[0.1],
        cv=2, device="cpu", max_iter=10,
    )
    seen = {}

    def capture(X_arg, y_arg, sample_weight=None):
        seen["X"] = X_arg
        seen["y"] = y_arg
        return model

    monkeypatch.setattr(model, "_fit_standard", capture)
    result = model.fit(X, y)
    assert result is model
    assert seen["X"] is X
    assert isinstance(seen["y"], np.ndarray)
    assert seen["y"].shape == (len(X),)
'''
    tests.write_text(text, encoding="utf-8")
