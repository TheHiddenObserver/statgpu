from pathlib import Path
import runpy

# Reapply the reviewed v27 source changes to the clean pre-v27 source tree.
runpy.run_path("pr87_patch_v27.py", run_name="__main__")


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:160]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Migrate the earlier manual diagnostics test to the analytic-weight contract.
replace_once(
    "dev/tests/test_maintenance_024_025.py",
    '''    expected_ll = -0.5 * float(np.sum(weights * resid_sq))
    k = 1 + X.shape[1]
    expected_dispersion = float(np.sum(weights * resid_sq)) / (weights.sum() - k)
''',
    '''    expected_ll = -0.5 * X.shape[0] * float(
        np.sum(weights * resid_sq) / np.sum(weights)
    )
    k = 1 + X.shape[1]
    expected_dispersion = float(np.sum(weights * resid_sq)) / (X.shape[0] - k)
''',
)
replace_once(
    "dev/tests/test_maintenance_024_025.py",
    '''    expected_no_inf = -0.5 * float(np.sum(weights * (y - eta_no_inf) ** 2))
''',
    '''    expected_no_inf = -0.5 * X.shape[0] * float(
        np.sum(weights * (y - eta_no_inf) ** 2) / np.sum(weights)
    )
''',
)
replace_once(
    "dev/tests/test_maintenance_024_025.py",
    '''    np.testing.assert_allclose(weighted.bse_, scaled.bse_, rtol=1e-11, atol=1e-11)
''',
    '''    np.testing.assert_allclose(weighted._bse, scaled._bse, rtol=1e-11, atol=1e-11)
''',
)
replace_once(
    "dev/tests/test_maintenance_024_025.py",
    '''    np.testing.assert_allclose(robust.bse_, robust_scaled.bse_, rtol=1e-11, atol=1e-11)
''',
    '''    np.testing.assert_allclose(robust._bse, robust_scaled._bse, rtol=1e-11, atol=1e-11)
''',
)

# Remove the stale second implementation of weighted GLM value/gradient.
# The active GLMLoss path already uses _fused._weighted_loss_and_grad, which
# is backend-native and propagates implementation errors.
solver_utils = Path("statgpu/glm_core/_solver_utils.py")
text = solver_utils.read_text(encoding="utf-8")
start = text.index("def _weighted_loss_and_grad(loss, X, y, coef, sample_weight):")
# This helper is the final function in the module.
text = text[:start] + '''def _weighted_loss_and_grad(loss, X, y, coef, sample_weight):
    """Delegate to the single backend-native weighted GLM implementation."""
    from statgpu.glm_core._fused import _weighted_loss_and_grad as _weighted

    return _weighted(loss, X, y, coef, sample_weight)
'''
solver_utils.write_text(text, encoding="utf-8")

# Regression test: the compatibility delegate must preserve TypeError and must
# not contain a CPU roundtrip or an unweighted fallback.
tests = Path("dev/tests/test_maintenance_024_025.py")
text = tests.read_text(encoding="utf-8")
marker = "# PR87_WEIGHTED_HELPER_SINGLE_SOURCE_TESTS"
if marker not in text:
    text += '''

# PR87_WEIGHTED_HELPER_SINGLE_SOURCE_TESTS
def test_solver_utils_weighted_helper_delegates_without_silent_unweighting(monkeypatch):
    import statgpu.glm_core._fused as fused
    import statgpu.glm_core._solver_utils as solver_utils

    sentinel = RuntimeError("weighted implementation failed")

    def fail(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(fused, "_weighted_loss_and_grad", fail)
    with pytest.raises(RuntimeError, match="weighted implementation failed"):
        solver_utils._weighted_loss_and_grad(
            object(), np.ones((2, 1)), np.ones(2), np.zeros(1), np.ones(2)
        )

    source = Path("statgpu/glm_core/_solver_utils.py").read_text(encoding="utf-8")
    block = source.split(
        "def _weighted_loss_and_grad(loss, X, y, coef, sample_weight):", 1
    )[1]
    assert "_to_numpy(sample_weight)" not in block
    assert "except TypeError" not in block
    assert "statgpu.glm_core._fused" in block
'''
    tests.write_text(text, encoding="utf-8")
