from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:120]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Reuse the shared magnitude-tiered axis-0 mean.  The previous helper only
# protected same-sign overflow and could erase a recoverable low-order column
# mean under large positive/negative cancellation.
replace_once(
    "statgpu/panel/_diagnostic_context.py",
    '''def _scaled_column_means(values, xp):\n    """Return column means without overflowing a finite reduction."""\n    n = int(values.shape[0])\n    if getattr(xp, "__name__", "") == "torch":\n        max_abs = xp.max(xp.abs(values), dim=0).values\n    else:\n        max_abs = xp.max(xp.abs(values), axis=0)\n    limit = np.finfo(np.float64).max / float(max(n, 1))\n    factor = xp.where(\n        max_abs > limit,\n        xp.full_like(max_abs, float(n)),\n        xp.ones_like(max_abs),\n    )\n    if getattr(xp, "__name__", "") == "torch":\n        summed = xp.sum(values / factor, dim=0)\n    else:\n        summed = xp.sum(values / factor, axis=0)\n    return summed * (factor / float(n))\n''',
    '''def _scaled_column_means(values, xp):\n    """Return cancellation-safe column means on the shared panel reducer."""\n    return _scaled_mean(values, xp)\n''',
)

# CPU + maintained Torch regression for the diagnostic helper used by the
# pooling-F pooled-null centering path.
p = Path("dev/tests/test_panel_diagnostic_cancellation_precision.py")
text = p.read_text(encoding="utf-8")
old_import = "from statgpu.panel._diagnostics import _scaled_group_means, _scaled_mean\n"
new_import = (
    "from statgpu.panel._diagnostic_context import _scaled_column_means\n"
    "from statgpu.panel._diagnostics import _scaled_group_means, _scaled_mean\n"
)
if new_import not in text:
    if old_import not in text:
        raise RuntimeError("diagnostic precision import anchor not found")
    text = text.replace(old_import, new_import, 1)
marker = "def test_pooling_f_column_mean_preserves_cancellation_tail_numpy():"
if marker not in text:
    text += '''\n\n\ndef test_pooling_f_column_mean_preserves_cancellation_tail_numpy():\n    values = np.asarray(\n        [[1.0e308], [1.0], [-1.0e308], [0.0], [0.0], [0.0]],\n        dtype=np.float64,\n    )\n    actual = np.asarray(_scaled_column_means(values, np), dtype=np.float64)\n    np.testing.assert_allclose(actual, np.asarray([1.0 / 6.0]), rtol=0.0, atol=0.0)\n\n\ndef test_pooling_f_column_mean_preserves_cancellation_tail_torch_cpu():\n    torch = pytest.importorskip("torch")\n    values = torch.as_tensor(\n        [[1.0e308], [1.0], [-1.0e308], [0.0], [0.0], [0.0]],\n        dtype=torch.float64,\n    )\n    actual = _to_numpy(_scaled_column_means(values, torch))\n    np.testing.assert_allclose(actual, np.asarray([1.0 / 6.0]), rtol=0.0, atol=0.0)\n'''
    p.write_text(text, encoding="utf-8")

# Physical CuPy/Torch validator exercises the same shared diagnostic reduction.
p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
old_import = '''from statgpu.panel._diagnostic_context import (\n    bp_lm_from_residuals,\n    pooling_f_from_level_arrays,\n)\n'''
new_import = '''from statgpu.panel._diagnostic_context import (\n    _scaled_column_means,\n    bp_lm_from_residuals,\n    pooling_f_from_level_arrays,\n)\n'''
if new_import not in text:
    if old_import not in text:
        raise RuntimeError("physical runner diagnostic import anchor not found")
    text = text.replace(old_import, new_import, 1)
audit_anchor = '''    # Pooling F: naive column/scalar means overflow after this common scaling,\n'''
audit_insert = '''    # The pooled-null design mean must preserve a low-order column contribution\n    # beside huge cancellation, matching the shared scalar/group mean policy.\n    cancellation_values_np = np.asarray(\n        [[1.0e308], [1.0], [-1.0e308], [0.0], [0.0], [0.0]],\n        dtype=np.float64,\n    )\n    if backend == "cupy":\n        cancellation_values = xp.asarray(cancellation_values_np)\n    elif backend == "torch":\n        cancellation_values = xp.as_tensor(cancellation_values_np, dtype=xp.float64)\n    else:\n        cancellation_values = cancellation_values_np\n    cancellation_column_mean = _array(\n        _scaled_column_means(cancellation_values, xp)\n    )\n    np.testing.assert_allclose(\n        cancellation_column_mean, np.asarray([1.0 / 6.0]), rtol=0.0, atol=0.0\n    )\n\n    # Pooling F: naive column/scalar means overflow after this common scaling,\n'''
if audit_insert not in text:
    if audit_anchor not in text:
        raise RuntimeError("diagnostic scale audit anchor not found")
    text = text.replace(audit_anchor, audit_insert, 1)
p.write_text(text, encoding="utf-8")

# Lock physical registration/source so a future runner refactor cannot drop it.
p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
contract_marker = "def test_stage_c_runner_covers_pooling_f_column_cancellation_mean():"
if contract_marker not in text:
    text += '''\n\n\ndef test_stage_c_runner_covers_pooling_f_column_cancellation_mean():\n    source = inspect.getsource(_MOD._diagnostic_scale_audit)\n    assert "_scaled_column_means" in source\n    assert "1.0e308" in source\n    assert "1.0 / 6.0" in source\n'''
    p.write_text(text, encoding="utf-8")
