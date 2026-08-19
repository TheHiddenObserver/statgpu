from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:140]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Restore covariance scales through mantissa/exponent composition. Fixed
# multiplication order cannot simultaneously avoid transient overflow and
# transient underflow when coordinate/stage scales compensate each other.
p = Path("statgpu/panel/_covariance.py")
text = p.read_text(encoding="utf-8")
old = '''def _restore_coordinate_covariance(covariance, scale, xp):\n    """Restore a per-coordinate covariance scale without forming scale_i*scale_j."""\n    row = scale[:, None]\n    col = scale[None, :]\n    large = xp.maximum(row, col)\n    small = xp.minimum(row, col)\n    return _symmetrize((covariance * large) * small)\n\n\ndef _restore_scalar_covariance(covariance, scale, xp):\n    """Restore one positive scalar score scale without materializing scale**2."""\n    return _symmetrize((covariance * scale) * scale)\n\n\ndef _restore_influence_covariance(\n    covariance, covariance_scale, projection_scale, design_scale, xp\n):\n    """Restore Gram, projection, then tiny-design scales after cancellation."""\n    covariance = _restore_coordinate_covariance(covariance, covariance_scale, xp)\n    covariance = _restore_coordinate_covariance(covariance, projection_scale, xp)\n    return _restore_scalar_covariance(covariance, design_scale, xp)\n'''
new = '''def _range_safe_float_product(values, factors, xp):\n    """Multiply finite float64 factors without a range-unsafe association.\n\n    Decompose each operand with ``frexp`` and accumulate exponents separately.\n    Every mantissa stays in a compact interval, so compensating large/small\n    factors can cancel in exponent space before the one final ``ldexp``.\n    """\n    mantissa, exponent = xp.frexp(values)\n    for factor in factors:\n        if np.isscalar(factor):\n            factor_mantissa, factor_exponent = np.frexp(float(factor))\n        else:\n            factor_mantissa, factor_exponent = xp.frexp(factor)\n        mantissa = mantissa * factor_mantissa\n        exponent = exponent + factor_exponent\n    return xp.ldexp(mantissa, exponent)\n\n\ndef _restore_coordinate_covariance(covariance, scale, xp):\n    """Restore a per-coordinate covariance scale without transient range loss."""\n    row = scale[:, None]\n    col = scale[None, :]\n    return _symmetrize(_range_safe_float_product(covariance, (row, col), xp))\n\n\ndef _restore_scalar_covariance(covariance, scale, xp):\n    """Restore one positive scalar score scale without transient range loss."""\n    return _symmetrize(_range_safe_float_product(covariance, (scale, scale), xp))\n\n\ndef _restore_influence_covariance(\n    covariance, covariance_scale, projection_scale, design_scale, xp\n):\n    """Restore all delayed influence scales in one exponent-safe product."""\n    return _symmetrize(\n        _range_safe_float_product(\n            covariance,\n            (\n                covariance_scale[:, None],\n                covariance_scale[None, :],\n                projection_scale[:, None],\n                projection_scale[None, :],\n                design_scale,\n                design_scale,\n            ),\n            xp,\n        )\n    )\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("covariance restore anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# NumPy + maintained Torch complementing-scale regressions.
p = Path("dev/tests/test_panel_stage_c_covariance.py")
text = p.read_text(encoding="utf-8")
old_import = '''    _influence_rows,\n    _symmetrize,\n'''
new_import = '''    _influence_rows,\n    _restore_coordinate_covariance,\n    _restore_influence_covariance,\n    _symmetrize,\n'''
if new_import not in text:
    if old_import not in text:
        raise RuntimeError("covariance test import anchor not found")
    text = text.replace(old_import, new_import, 1)
marker = "def test_covariance_restore_compensating_scales_avoid_transient_range_loss():"
if marker not in text:
    text += '''\n\n\ndef test_covariance_restore_compensating_scales_avoid_transient_range_loss():\n    scale_np = np.asarray([1.0e200, 1.0e-200], dtype=np.float64)\n    for entry in (1.0e200, 1.0e-200):\n        covariance_np = np.asarray([[0.0, entry], [entry, 0.0]], dtype=np.float64)\n        expected = covariance_np.copy()\n        actual = _restore_coordinate_covariance(covariance_np, scale_np, np)\n        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)\n\n    # Compensation can also occur across delayed restoration stages. The first\n    # coordinate scale alone would overflow, but the projection scale restores\n    # a finite final covariance.\n    covariance_np = np.asarray([[1.0e100]], dtype=np.float64)\n    actual = _restore_influence_covariance(\n        covariance_np,\n        np.asarray([1.0e200]),\n        np.asarray([1.0e-100]),\n        1.0,\n        np,\n    )\n    np.testing.assert_allclose(actual, np.asarray([[1.0e300]]), rtol=5e-15, atol=0.0)\n\n\ndef test_torch_covariance_restore_compensating_scales_avoid_transient_range_loss():\n    torch = pytest.importorskip("torch")\n    scale = torch.as_tensor([1.0e200, 1.0e-200], dtype=torch.float64)\n    for entry in (1.0e200, 1.0e-200):\n        covariance = torch.as_tensor([[0.0, entry], [entry, 0.0]], dtype=torch.float64)\n        actual = _restore_coordinate_covariance(covariance, scale, torch)\n        np.testing.assert_allclose(\n            actual.detach().cpu().numpy(), covariance.detach().cpu().numpy(),\n            rtol=0.0, atol=0.0\n        )\n\n    covariance = torch.as_tensor([[1.0e100]], dtype=torch.float64)\n    actual = _restore_influence_covariance(\n        covariance,\n        torch.as_tensor([1.0e200], dtype=torch.float64),\n        torch.as_tensor([1.0e-100], dtype=torch.float64),\n        1.0,\n        torch,\n    )\n    np.testing.assert_allclose(\n        actual.detach().cpu().numpy(), np.asarray([[1.0e300]]),\n        rtol=5e-15, atol=0.0\n    )\n'''
    p.write_text(text, encoding="utf-8")

# Physical CuPy/Torch contract uses both overflow- and underflow-prone orders.
p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
old_import = 'from statgpu.panel._covariance import _grouped_score_sums, hac_covariance, ols_covariance, two_way_clustered_covariance\n'
new_import = '''from statgpu.panel._covariance import (\n    _grouped_score_sums,\n    _restore_coordinate_covariance,\n    _restore_influence_covariance,\n    hac_covariance,\n    ols_covariance,\n    two_way_clustered_covariance,\n)\n'''
if new_import not in text:
    if old_import not in text:
        raise RuntimeError("physical covariance import anchor not found")
    text = text.replace(old_import, new_import, 1)
audit_marker = "def _covariance_restore_range_audit(backend):"
if audit_marker not in text:
    insert_before = "def _nonfinite_covariance_guard_audit(backend):\n"
    audit = '''def _covariance_restore_range_audit(backend):\n    xp = __import__("torch") if backend == "torch" else __import__("cupy")\n    if backend == "torch":\n        scale = xp.as_tensor([1.0e200, 1.0e-200], dtype=xp.float64, device="cuda")\n        covariance = xp.as_tensor([[0.0, 1.0e200], [1.0e200, 0.0]], dtype=xp.float64, device="cuda")\n        stage_covariance = xp.as_tensor([[1.0e100]], dtype=xp.float64, device="cuda")\n        covariance_scale = xp.as_tensor([1.0e200], dtype=xp.float64, device="cuda")\n        projection_scale = xp.as_tensor([1.0e-100], dtype=xp.float64, device="cuda")\n    else:\n        scale = xp.asarray([1.0e200, 1.0e-200], dtype=xp.float64)\n        covariance = xp.asarray([[0.0, 1.0e200], [1.0e200, 0.0]], dtype=xp.float64)\n        stage_covariance = xp.asarray([[1.0e100]], dtype=xp.float64)\n        covariance_scale = xp.asarray([1.0e200], dtype=xp.float64)\n        projection_scale = xp.asarray([1.0e-100], dtype=xp.float64)\n    coordinate = _array(_restore_coordinate_covariance(covariance, scale, xp))\n    np.testing.assert_allclose(\n        coordinate, np.asarray([[0.0, 1.0e200], [1.0e200, 0.0]]),\n        rtol=0.0, atol=0.0\n    )\n    staged = _array(\n        _restore_influence_covariance(\n            stage_covariance, covariance_scale, projection_scale, 1.0, xp\n        )\n    )\n    np.testing.assert_allclose(staged, np.asarray([[1.0e300]]), rtol=5e-15, atol=0.0)\n    return {\n        "status": "success",\n        "backend": backend,\n        "coordinate_restore": float(coordinate[0, 1]),\n        "cross_stage_restore": float(staged[0, 0]),\n    }\n\n\n'''
    if insert_before not in text:
        raise RuntimeError("physical covariance audit anchor not found")
    text = text.replace(insert_before, audit + insert_before, 1)
register_anchor = '            "two_way_effect_range_gauge": _two_way_effect_range_gauge_audit(backend),\n'
register_new = register_anchor + '            "covariance_restore_range": _covariance_restore_range_audit(backend),\n'
if register_new not in text:
    if register_anchor not in text:
        raise RuntimeError("physical covariance registration anchor not found")
    text = text.replace(register_anchor, register_new, 1)
p.write_text(text, encoding="utf-8")

p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
marker = "def test_stage_c_runner_registers_covariance_restore_range_audit():"
if marker not in text:
    text += '''\n\n\ndef test_stage_c_runner_registers_covariance_restore_range_audit():\n    source = inspect.getsource(_MOD._covariance_restore_range_audit)\n    for token in ("1.0e200", "1.0e-200", "1.0e300", "cross_stage_restore"):\n        assert token in source\n    main_source = inspect.getsource(_MOD.main)\n    assert '"covariance_restore_range": _covariance_restore_range_audit(backend)' in main_source\n'''
    p.write_text(text, encoding="utf-8")

# Changelog: restoration now composes all delayed scale exponents before final rounding.
p = Path("CHANGELOG.md")
text = p.read_text(encoding="utf-8")
needle = "- **Panel covariance extreme-scale arithmetic**:"
idx = text.find(needle)
if idx < 0:
    raise RuntimeError("CHANGELOG covariance bullet not found")
line_end = text.find("\n", idx)
line = text[idx:line_end]
if "mantissa/exponent scale restoration" not in line:
    line += " Delayed covariance, projection, and design scales use mantissa/exponent scale restoration so compensating large/small factors cannot create a transient overflow or underflow before the final representable float64 covariance is rounded."
    text = text[:idx] + line + text[line_end:]
    p.write_text(text, encoding="utf-8")
