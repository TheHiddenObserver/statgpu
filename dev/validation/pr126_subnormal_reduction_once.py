from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "statgpu/panel/_diagnostics.py",
    '''def _common_scaled_sumsquares(left, right, xp):\n    """Return two sums of squares normalized by one common backend scale."""\n    scale = xp.maximum(xp.max(xp.abs(left)), xp.max(xp.abs(right)))\n    scale_value = _to_float_scalar(scale)\n    if scale_value == 0.0:\n        return 0.0, 0.0, 0.0\n    left_scaled = left / scale\n    right_scaled = right / scale\n    left_ss = _to_float_scalar(xp.sum(left_scaled * left_scaled))\n    right_ss = _to_float_scalar(xp.sum(right_scaled * right_scaled))\n    return float(left_ss), float(right_ss), float(scale_value)\n''',
    '''def _scaled_unit_values(values, scale, xp):\n    """Normalize by a scalar scale without dividing by a subnormal denominator."""\n    scale_value = _to_float_scalar(scale)\n    if scale_value == 0.0:\n        return values\n    target = float(np.sqrt(np.finfo(np.float64).tiny))\n    if scale_value < target:\n        relative = scale / float(target)\n        factor = 1.0 / relative\n        values_work = values * factor\n        scale_work = scale * factor\n        return values_work / scale_work\n    return values / scale\n\n\ndef _common_scaled_sumsquares(left, right, xp):\n    """Return two sums of squares normalized by one common backend scale."""\n    scale = xp.maximum(xp.max(xp.abs(left)), xp.max(xp.abs(right)))\n    scale_value = _to_float_scalar(scale)\n    if scale_value == 0.0:\n        return 0.0, 0.0, 0.0\n    left_scaled = _scaled_unit_values(left, scale, xp)\n    right_scaled = _scaled_unit_values(right, scale, xp)\n    left_ss = _to_float_scalar(xp.sum(left_scaled * left_scaled))\n    right_ss = _to_float_scalar(xp.sum(right_scaled * right_scaled))\n    return float(left_ss), float(right_ss), float(scale_value)\n''',
    "subnormal-safe common sumsquares",
)
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''    unit = resid / scale\n    norm_sq = _to_float_scalar(xp.sum(unit * unit))\n''',
    '''    unit = _scaled_unit_values(resid, scale, xp)\n    norm_sq = _to_float_scalar(xp.sum(unit * unit))\n''',
    "subnormal-safe residual variance",
)
replace_once(
    "statgpu/panel/_diagnostics.py",
    '''    resid_scaled = resid / scale\n    centered_scaled = centered / scale\n''',
    '''    resid_scaled = _scaled_unit_values(resid, scale, xp)\n    centered_scaled = _scaled_unit_values(centered, scale, xp)\n''',
    "subnormal-safe R2",
)

# Direct Torch regression for the exact backend arithmetic that previously
# failed in the SVD working-scale path.
test_path = Path("dev/tests/test_panel_stage_c_final_review_fixes.py")
text = test_path.read_text(encoding="utf-8")
text = text.replace(
    "from statgpu.panel._diagnostics import _build_fit_statistics, _classical_model_f\n",
    "from statgpu.panel._diagnostics import (\n"
    "    _build_fit_statistics,\n"
    "    _classical_model_f,\n"
    "    _common_scaled_sumsquares,\n"
    "    _scaled_residual_r2,\n"
    "    _scaled_residual_variance,\n"
    ")\n",
    1,
)
marker = "def test_torch_subnormal_residual_reductions_use_normal_working_scale():"
if marker in text:
    raise RuntimeError("subnormal reduction regression already present")
text += r'''


def test_torch_subnormal_residual_reductions_use_normal_working_scale():
    torch = pytest.importorskip("torch")
    resid_np = np.asarray([1.0e-320, -1.0e-320, 5.0e-321, -5.0e-321])
    centered_np = np.asarray([2.0e-320, -2.0e-320, 1.0e-320, -1.0e-320])
    resid = torch.as_tensor(resid_np, dtype=torch.float64)
    centered = torch.as_tensor(centered_np, dtype=torch.float64)

    variance = _scaled_residual_variance(resid, 4, torch)
    # The variance itself underflows in float64, but the normalized reduction
    # must remain finite rather than becoming NaN/Inf on Torch.
    assert variance == 0.0
    r2_torch, degenerate = _scaled_residual_r2(resid, centered, torch)
    r2_numpy, _ = _scaled_residual_r2(resid_np, centered_np, np)
    assert not degenerate
    assert np.isfinite(r2_torch)
    assert_allclose(r2_torch, r2_numpy, rtol=3e-12, atol=1e-14)

    left_ss, right_ss, scale = _common_scaled_sumsquares(
        resid, centered, torch
    )
    assert scale > 0.0
    assert np.isfinite(left_ss) and np.isfinite(right_ss)
    left_np, right_np, _ = _common_scaled_sumsquares(
        resid_np, centered_np, np
    )
    assert_allclose([left_ss, right_ss], [left_np, right_np], rtol=3e-12, atol=1e-14)
'''
test_path.write_text(text, encoding="utf-8")

print("PR126 subnormal reduction follow-up staged")
