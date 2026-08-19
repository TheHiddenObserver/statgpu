from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Detect product-level underflow after a common score scale.  Existing guards
# only detect a component that becomes exactly zero; a nonzero working component
# can still have every relevant self/cross product round to zero.
insert_anchor = '''def _retier_component_for_safe_gram(component, xp):\n'''
helper = '''def _row_min_nonzero(values, xp):\n    absolute = xp.abs(values)\n    sentinel = xp.full_like(absolute, float("inf"))\n    nonzero = xp.where(absolute > 0.0, absolute, sentinel)\n    if _is_torch(xp):\n        return xp.min(nonzero, dim=1).values\n    return xp.min(nonzero, axis=1)\n\n\ndef _common_scale_product_range_lost(component_sets, xp) -> bool:\n    \"\"\"Return whether a nonzero row product underflows on the common scale.\n\n    Multi-tier two-way covariance must put estimator components on one working\n    scale before inclusion-exclusion. A component can survive that division as\n    a nonzero subnormal while its self/cross product is already below the\n    smallest float64 subnormal. In that case silently evaluating the Gram would\n    discard a potentially recoverable final covariance contribution after the\n    larger estimator terms cancel, so the current float64 representation must\n    fail closed.\n    \"\"\"\n    minimum = float(np.nextafter(0.0, 1.0))\n    risk = None\n    for components in component_sets:\n        row_minima = [_row_min_nonzero(component, xp) for component in components]\n        for i, left_min in enumerate(row_minima):\n            for right_min in row_minima[: i + 1]:\n                active = xp.isfinite(left_min) & xp.isfinite(right_min)\n                # Division avoids materializing the underflowing product itself.\n                threshold = minimum / right_min\n                local = xp.any(active & (left_min < threshold))\n                risk = local if risk is None else (risk | local)\n    if risk is None:\n        return False\n    return bool(_to_float_scalar(risk))\n\n\n'''
p = Path("statgpu/panel/_covariance.py")
text = p.read_text(encoding="utf-8")
if helper not in text:
    if insert_anchor not in text:
        raise RuntimeError("covariance helper insertion anchor not found")
    text = text.replace(insert_anchor, helper + insert_anchor, 1)
p.write_text(text, encoding="utf-8")

replace_once(
    "statgpu/panel/_covariance.py",
    '''            work1 = working_components[:n1]\n            work2 = working_components[n1:n1 + n2]\n            work12 = working_components[n1 + n2:]\n\n            terms = []\n''',
    '''            work1 = working_components[:n1]\n            work2 = working_components[n1:n1 + n2]\n            work12 = working_components[n1 + n2:]\n\n            if _common_scale_product_range_lost((work1, work2, work12), xp):\n                raise FloatingPointError(\n                    "two-way cluster score expansion exceeds the float64 "\n                    "common-scale product range"\n                )\n\n            terms = []\n''',
)

# NumPy public regression: the exact CGM meat is 4*m**2, but the current common
# value scale would erase m**2 before high terms cancel. The supported float64
# contract is explicit failure rather than a silent zero covariance.
p = Path("dev/tests/test_panel_stage_c_covariance.py")
text = p.read_text(encoding="utf-8")
marker = "def test_two_way_clustered_covariance_fails_closed_on_common_scale_product_underflow():"
if marker not in text:
    text += '''\n\n\ndef test_two_way_clustered_covariance_fails_closed_on_common_scale_product_underflow():\n    amplitude = 1.0e308\n    tiny = 1.0e-100\n    X = np.full((4, 1), 0.25, dtype=np.float64)\n    resid = np.asarray([-amplitude, tiny, amplitude, tiny], dtype=np.float64)\n    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)\n    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)\n\n    # With influence rows equal to resid, CGM gives exactly 4*tiny**2, which is\n    # representable. A single common value scale, however, cannot represent the\n    # low-low product beside the 1e308 tier. Never silently publish zero.\n    expected_meat = 4.0 * tiny * tiny\n    assert expected_meat > 0.0 and np.isfinite(expected_meat)\n    with pytest.raises(FloatingPointError, match="common-scale product range"):\n        two_way_clustered_covariance(X, resid, cluster1, cluster2)\n'''
    p.write_text(text, encoding="utf-8")

# Maintained Torch CPU regression.
p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
marker = "def test_stage_c_torch_cpu_two_way_common_scale_product_underflow_fails_closed():"
if marker not in text:
    text += '''\n\n\ndef test_stage_c_torch_cpu_two_way_common_scale_product_underflow_fails_closed():\n    amplitude = 1.0e308\n    tiny = 1.0e-100\n    X = torch.full((4, 1), 0.25, dtype=torch.float64)\n    resid = torch.as_tensor([-amplitude, tiny, amplitude, tiny], dtype=torch.float64)\n    cluster1 = torch.as_tensor([0, 0, 1, 1], dtype=torch.int64)\n    cluster2 = torch.as_tensor([0, 1, 0, 1], dtype=torch.int64)\n    with pytest.raises(FloatingPointError, match="common-scale product range"):\n        two_way_clustered_covariance(\n            X, resid, cluster1, cluster2, xp=torch\n        )\n'''
    p.write_text(text, encoding="utf-8")

# Physical CuPy/Torch audit.
p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
marker = "def _common_scale_product_range_guard_audit(backend):"
if marker not in text:
    anchor = text.index("\ndef _nonfinite_covariance_guard_audit")
    audit = '''\n\ndef _common_scale_product_range_guard_audit(backend):\n    amplitude = 1.0e308\n    tiny = 1.0e-100\n    X_np = np.full((4, 1), 0.25, dtype=np.float64)\n    resid_np = np.asarray([-amplitude, tiny, amplitude, tiny], dtype=np.float64)\n    cluster1_np = np.asarray([0, 0, 1, 1], dtype=np.int64)\n    cluster2_np = np.asarray([0, 1, 0, 1], dtype=np.int64)\n    dummy_time = np.arange(4, dtype=np.int64)\n    X, resid, cluster1, _time = _to_backend(\n        X_np, resid_np, cluster1_np, dummy_time, backend\n    )\n    if backend == "numpy":\n        xp = np\n        cluster2 = cluster2_np\n    elif backend == "cupy":\n        import cupy as cp\n        xp = cp\n        cluster2 = cp.asarray(cluster2_np)\n    elif backend == "torch":\n        import torch\n        xp = torch\n        cluster2 = torch.as_tensor(cluster2_np, dtype=torch.int64)\n    else:\n        raise ValueError(backend)\n    try:\n        two_way_clustered_covariance(\n            X, resid, cluster1, cluster2, xp=xp\n        )\n    except FloatingPointError as exc:\n        if "common-scale product range" not in str(exc):\n            raise\n    else:\n        raise AssertionError(\n            f"{backend}: common-scale product underflow did not fail closed"\n        )\n    return {\n        "status": "success",\n        "backend": backend,\n        "representable_exact_meat": 4.0 * tiny * tiny,\n        "failed_closed": True,\n    }\n'''
    text = text[:anchor] + audit + text[anchor:]

old = '            "two_way_effect_normalization_overflow": _two_way_effect_normalization_overflow_audit(backend),\n            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),'
new = '            "two_way_effect_normalization_overflow": _two_way_effect_normalization_overflow_audit(backend),\n            "common_scale_product_range_guard": _common_scale_product_range_guard_audit(backend),\n            "nonfinite_covariance_guards": _nonfinite_covariance_guard_audit(backend),'
if new not in text:
    if old not in text:
        raise RuntimeError("physical product-range registry anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# Hosted contract that keeps physical coverage registered.
p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
marker = "def test_stage_c_runner_registers_common_scale_product_range_guard():"
if marker not in text:
    text += '''\n\n\ndef test_stage_c_runner_registers_common_scale_product_range_guard():\n    audit_source = inspect.getsource(_MOD._common_scale_product_range_guard_audit)\n    for token in (\n        "1.0e308",\n        "1.0e-100",\n        "common-scale product range",\n        "failed_closed",\n    ):\n        assert token in audit_source\n    main_source = inspect.getsource(_MOD.main)\n    assert (\n        '\"common_scale_product_range_guard\": _common_scale_product_range_guard_audit(backend)'\n        in main_source\n    )\n'''
    p.write_text(text, encoding="utf-8")

# Document the explicit float64 limit in both languages.
replace_once(
    "docs/en/panel/covariance.md",
    '''otherwise the three components share one minimally scaled Gram space before inclusion-exclusion:\n''',
    '''otherwise the three components share one minimally scaled Gram space before inclusion-exclusion. If that common score scale can keep every grouped component nonzero but would still make a mathematically nonzero component self/cross product underflow before inclusion-exclusion, statgpu raises `FloatingPointError` rather than silently dropping the term. This is an explicit float64 working-range boundary for cancellations spanning more exponent range than one common Gram representation can retain; it is not reported as a zero covariance.\n\nThe estimator is then\n''',
)
replace_once(
    "docs/cn/panel/covariance.md",
    '''非嵌套情形则让三个 component 共用同一个最小 Gram working scale 后再做 inclusion-exclusion：\n''',
    '''非嵌套情形则让三个 component 共用同一个最小 Gram working scale 后再做 inclusion-exclusion。如果这个 common score scale 虽然仍能把每个 grouped component 保持为非零，却会使某个数学上非零的 component self/cross product 在 inclusion-exclusion 之前先下溢为 0，statgpu 会显式抛出 `FloatingPointError`，而不会静默丢掉该项并报告零 covariance。这是当前 float64 common-Gram 表示的明确 working-range 边界。\n\n此时 estimator 仍定义为\n''',
)

# Release note.
p = Path("CHANGELOG.md")
text = p.read_text(encoding="utf-8")
old = "Public clustered, two-way clustered, HAC, and Driscoll-Kraay helpers now reject non-finite `X`/residual inputs before signed/group reductions, preventing NaN/Inf scores from being silently reinterpreted as zero contributions."
new = "Two-way multi-tier covariance also fails closed if a nonzero grouped component survives the common score scaling but a mathematically nonzero self/cross product would underflow on that common Gram scale; this prevents a representable low-order covariance remainder from being silently published as zero when the required estimator-level cancellation exceeds the supported float64 common-scale range. Public clustered, two-way clustered, HAC, and Driscoll-Kraay helpers reject non-finite `X`/residual inputs before signed/group reductions, preventing NaN/Inf scores from being silently reinterpreted as zero contributions."
if new not in text:
    if old not in text:
        raise RuntimeError("CHANGELOG product-range anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")
