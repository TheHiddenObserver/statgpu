from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:160]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "statgpu/panel/_covariance.py",
    '''def _cross_reduction_is_safe(left, right, xp) -> bool:\n    """Return whether ``left.T @ right`` has a conservative rowwise range bound.\n''',
    '''def _cross_reduction_is_safe(\n    left, right, xp, *, max_multiplier: float = 1.0\n) -> bool:\n    """Return whether ``left.T @ right`` has a conservative rowwise range bound.\n''',
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''    n_terms = max(1, int(left.shape[0]))\n    if _is_torch(xp):\n''',
    '''    n_terms = max(1, int(left.shape[0]))\n    multiplier = max(1.0, abs(float(max_multiplier)))\n    if _is_torch(xp):\n''',
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''    per_term_limit = 0.25 * float(np.finfo(np.float64).max) / float(n_terms)\n    safe = finite & (small <= float(per_term_limit) / safe_large)\n''',
    '''    # Reserve a factor-four margin for symmetric cross doubling plus the\n    # caller's finite-sample/component multiplier.\n    per_term_limit = (\n        0.25\n        * float(np.finfo(np.float64).max)\n        / (float(n_terms) * multiplier)\n    )\n    safe = finite & (small <= float(per_term_limit) / safe_large)\n''',
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''        need_compensation = (\n            not group_debias\n            and _to_float_scalar(xp.max(common_scale)) > 1.0\n        )\n''',
    '''        need_compensation = _to_float_scalar(xp.max(common_scale)) > 1.0\n''',
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''            safe_correction = all(\n                _cross_reduction_is_safe(high, low, xp)\n                and _cross_reduction_is_safe(low, low, xp)\n                for high, low in (\n                    (grouped1_high, grouped1_low),\n                    (grouped2_high, grouped2_low),\n                    (grouped12_high, grouped12_low),\n                )\n            )\n            if safe_correction:\n                def _low_order_covariance(high, low):\n                    cross = _symmetrize(high.T @ low)\n                    return _symmetrize(\n                        (2.0 * cross) + _symmetrize(low.T @ low)\n                    )\n\n                low_correction1 = _low_order_covariance(\n                    grouped1_high, grouped1_low\n                )\n                low_correction2 = _low_order_covariance(\n                    grouped2_high, grouped2_low\n                )\n                low_correction12 = _low_order_covariance(\n                    grouped12_high, grouped12_low\n                )\n''',
    '''            correction_triples = (\n                (grouped1_high, grouped1_low, correction1),\n                (grouped2_high, grouped2_low, correction2),\n                (grouped12_high, grouped12_low, correction12),\n            )\n            safe_correction = all(\n                _cross_reduction_is_safe(\n                    high, low, xp, max_multiplier=correction\n                )\n                and _cross_reduction_is_safe(\n                    low, low, xp, max_multiplier=correction\n                )\n                for high, low, correction in correction_triples\n            )\n            if safe_correction:\n                def _low_order_covariance(high, low, correction):\n                    cross = _symmetrize(high.T @ low)\n                    low_square = _symmetrize(low.T @ low)\n                    return _symmetrize(\n                        ((2.0 * cross) + low_square) * float(correction)\n                    )\n\n                low_correction1 = _low_order_covariance(\n                    grouped1_high, grouped1_low, correction1\n                )\n                low_correction2 = _low_order_covariance(\n                    grouped2_high, grouped2_low, correction2\n                )\n                low_correction12 = _low_order_covariance(\n                    grouped12_high, grouped12_low, correction12\n                )\n''',
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''        # that final-addition residual, and only for the un-debiased definition\n        # where all three CGM components have unit weight.\n''',
    '''        # that final-addition residual.  Each component keeps its own CGM\n        # finite-sample correction, so group-debiased and un-debiased definitions\n        # use the same compensated algebra without changing their statistics.\n''',
)

p = Path("dev/tests/test_panel_stage_c_covariance.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_two_way_group_debias_preserves_weighted_low_order_cancellation():
    amplitude = 1.0e154
    small = 1.0e-154
    X = np.full((4, 1), 0.5, dtype=np.float64)
    scores = np.asarray([-amplitude, amplitude, amplitude, small], dtype=np.float64)
    resid = 2.0 * scores
    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)

    actual = two_way_clustered_covariance(
        X, resid, cluster1, cluster2, group_debias=True
    )
    expected = np.asarray(
        [[6.0 * amplitude * small + 2.0 * small * small]], dtype=np.float64
    )
    assert np.all(np.isfinite(actual))
    assert_allclose(actual, expected, rtol=5e-13, atol=0.0)
'''
if "test_two_way_group_debias_preserves_weighted_low_order_cancellation" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_stage_c_torch_cpu_two_way_group_debias_preserves_low_order_term():
    amplitude = 1.0e154
    small = 1.0e-154
    X = torch.full((4, 1), 0.5, dtype=torch.float64)
    scores = torch.tensor(
        [-amplitude, amplitude, amplitude, small], dtype=torch.float64
    )
    resid = 2.0 * scores
    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)
    actual = two_way_clustered_covariance(
        X, resid, cluster1, cluster2, xp=torch, group_debias=True
    ).detach().cpu().numpy()
    expected = np.asarray(
        [[6.0 * amplitude * small + 2.0 * small * small]], dtype=np.float64
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=0.0)
'''
if "test_stage_c_torch_cpu_two_way_group_debias_preserves_low_order_term" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

# Add the same case to the maintained physical CUDA numerical primitive audit.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    resid_nonnested_np = 2.0 * nonnested_scores\n    nonnested_cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)\n    nonnested_cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)\n\n    if backend == "numpy":\n''',
    '''    resid_nonnested_np = 2.0 * nonnested_scores\n    nonnested_cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)\n    nonnested_cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)\n    nonnested_debias_scores = np.asarray(\n        [-nonnested_amplitude, nonnested_amplitude,\n         nonnested_amplitude, nonnested_small],\n        dtype=np.float64,\n    )\n    resid_nonnested_debias_np = 2.0 * nonnested_debias_scores\n\n    if backend == "numpy":\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        X_nonnested, resid_nonnested = X_nonnested_np, resid_nonnested_np\n    elif backend == "cupy":\n''',
    '''        X_nonnested, resid_nonnested = X_nonnested_np, resid_nonnested_np\n        resid_nonnested_debias = resid_nonnested_debias_np\n    elif backend == "cupy":\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        X_nonnested = cp.asarray(X_nonnested_np)\n        resid_nonnested = cp.asarray(resid_nonnested_np)\n    elif backend == "torch":\n''',
    '''        X_nonnested = cp.asarray(X_nonnested_np)\n        resid_nonnested = cp.asarray(resid_nonnested_np)\n        resid_nonnested_debias = cp.asarray(resid_nonnested_debias_np)\n    elif backend == "torch":\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        resid_nonnested = torch.as_tensor(\n            resid_nonnested_np, dtype=torch.float64, device="cuda"\n        )\n    else:\n''',
    '''        resid_nonnested = torch.as_tensor(\n            resid_nonnested_np, dtype=torch.float64, device="cuda"\n        )\n        resid_nonnested_debias = torch.as_tensor(\n            resid_nonnested_debias_np, dtype=torch.float64, device="cuda"\n        )\n    else:\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    nonnested_two_way = _array(two_way_clustered_covariance(\n        X_nonnested,\n        resid_nonnested,\n        nonnested_cluster1,\n        nonnested_cluster2,\n        xp=xp,\n    ))\n    for name, value in (("one_way", one_way),''',
    '''    nonnested_two_way = _array(two_way_clustered_covariance(\n        X_nonnested,\n        resid_nonnested,\n        nonnested_cluster1,\n        nonnested_cluster2,\n        xp=xp,\n    ))\n    nonnested_two_way_group_debias = _array(two_way_clustered_covariance(\n        X_nonnested,\n        resid_nonnested_debias,\n        nonnested_cluster1,\n        nonnested_cluster2,\n        xp=xp,\n        group_debias=True,\n    ))\n    for name, value in (("one_way", one_way),''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''("mixed_dk", mixed_dk), ("nonnested_two_way", nonnested_two_way)):\n''',
    '''("mixed_dk", mixed_dk), ("nonnested_two_way", nonnested_two_way), ("nonnested_two_way_group_debias", nonnested_two_way_group_debias)):\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    np.testing.assert_allclose(\n        nonnested_two_way,\n        np.asarray([[-4.0 * nonnested_amplitude * nonnested_small]]),\n        rtol=2e-12,\n        atol=0.0,\n    )\n    return {\n''',
    '''    np.testing.assert_allclose(\n        nonnested_two_way,\n        np.asarray([[-4.0 * nonnested_amplitude * nonnested_small]]),\n        rtol=2e-12,\n        atol=0.0,\n    )\n    np.testing.assert_allclose(\n        nonnested_two_way_group_debias,\n        np.asarray([[\n            6.0 * nonnested_amplitude * nonnested_small\n            + 2.0 * nonnested_small * nonnested_small\n        ]]),\n        rtol=5e-13,\n        atol=0.0,\n    )\n    return {\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        "nonnested_two_way_structural_cancellation": nonnested_two_way.tolist(),\n    }\n''',
    '''        "nonnested_two_way_structural_cancellation": nonnested_two_way.tolist(),\n        "nonnested_two_way_group_debias_cancellation": (\n            nonnested_two_way_group_debias.tolist()\n        ),\n    }\n''',
)

p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
old = '''        "nonnested_two_way_structural_cancellation",\n    ):\n'''
new = '''        "nonnested_two_way_structural_cancellation",\n        "nonnested_two_way_group_debias_cancellation",\n    ):\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("physical runner audit key anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text.rstrip() + "\n", encoding="utf-8")

for path in (
    "dev/tests/test_panel_stage_c_covariance.py",
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
):
    p = Path(path)
    p.write_text(p.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
