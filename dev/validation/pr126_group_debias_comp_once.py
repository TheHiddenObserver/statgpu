from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:180]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Two-way clustered covariance: retain grouped-score low parts from the first
# reduction and compensate them independently of Gram scaling.
# ---------------------------------------------------------------------------
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
    '''    per_term_limit = (\n        0.25\n        * float(np.finfo(np.float64).max)\n        / (float(n_terms) * multiplier)\n    )\n    safe = finite & (small <= float(per_term_limit) / safe_large)\n''',
)

cov_path = Path("statgpu/panel/_covariance.py")
text = cov_path.read_text(encoding="utf-8")
old_func = '''def _cluster_grouped_scores(\n    scores, codes, *, n_groups: int, nobs: int, group_debias: bool, xp\n):\n    if int(n_groups) < 2:\n        raise ValueError(\n            "clustered covariance requires at least two distinct clusters"\n        )\n    grouped = _grouped_score_sums(scores, codes, n_groups=int(n_groups), xp=xp)\n    correction = (\n        _group_debias_factor(int(n_groups), int(nobs)) if group_debias else 1.0\n    )\n    return grouped, float(correction)\n'''
new_func = '''def _cluster_grouped_scores(\n    scores, codes, *, n_groups: int, nobs: int, group_debias: bool, xp,\n    return_compensation: bool = False,\n):\n    if int(n_groups) < 2:\n        raise ValueError(\n            "clustered covariance requires at least two distinct clusters"\n        )\n    grouped = _grouped_score_sums(\n        scores,\n        codes,\n        n_groups=int(n_groups),\n        xp=xp,\n        return_compensation=return_compensation,\n    )\n    correction = (\n        _group_debias_factor(int(n_groups), int(nobs)) if group_debias else 1.0\n    )\n    if return_compensation:\n        high, low = grouped\n        return high, low, float(correction)\n    return grouped, float(correction)\n'''
if new_func not in text:
    if old_func not in text:
        raise RuntimeError("cluster grouped-score helper anchor not found")
    text = text.replace(old_func, new_func, 1)

old_calls = '''    grouped1, correction1 = _cluster_grouped_scores(\n        influence, c1, n_groups=int(len(labels1)), nobs=n,\n        group_debias=group_debias, xp=xp\n    )\n    grouped2, correction2 = _cluster_grouped_scores(\n        influence, c2, n_groups=int(len(labels2)), nobs=n,\n        group_debias=group_debias, xp=xp\n    )\n    grouped12, correction12 = _cluster_grouped_scores(\n        influence, c12, n_groups=n12, nobs=n,\n        group_debias=group_debias, xp=xp\n    )\n'''
new_calls = '''    grouped1, grouped1_low, correction1 = _cluster_grouped_scores(\n        influence, c1, n_groups=int(len(labels1)), nobs=n,\n        group_debias=group_debias, xp=xp, return_compensation=True\n    )\n    grouped2, grouped2_low, correction2 = _cluster_grouped_scores(\n        influence, c2, n_groups=int(len(labels2)), nobs=n,\n        group_debias=group_debias, xp=xp, return_compensation=True\n    )\n    grouped12, grouped12_low, correction12 = _cluster_grouped_scores(\n        influence, c12, n_groups=n12, nobs=n,\n        group_debias=group_debias, xp=xp, return_compensation=True\n    )\n'''
if new_calls not in text:
    if old_calls not in text:
        raise RuntimeError("two-way initial grouped-score calls anchor not found")
    text = text.replace(old_calls, new_calls, 1)

start_marker = '''        # The ordinary grouped score is a float64 high part.  Under extreme\n'''
end_marker = '''                common_scale = xp.ones_like(projection_scale)\n'''
start = text.find(start_marker)
if start < 0:
    raise RuntimeError("existing low-order compensation block start not found")
end = text.find(end_marker, start)
if end < 0:
    raise RuntimeError("existing low-order compensation block end not found")
end += len(end_marker)
new_block = '''        # The grouped score is represented as a float64 high part plus the\n        # exact TwoSum residual from its final positive/negative combination.\n        # This rounding loss occurs at grouping time and is independent of\n        # whether the subsequent Gram needs range scaling, so compensate any\n        # nonzero low part in the nonnested CGM combination.  Each component\n        # keeps its own finite-sample group-debias factor.\n        low_max = xp.maximum(\n            xp.maximum(\n                xp.max(xp.abs(grouped1_low)),\n                xp.max(xp.abs(grouped2_low)),\n            ),\n            xp.max(xp.abs(grouped12_low)),\n        )\n        need_compensation = _to_float_scalar(low_max) > 0.0\n        if need_compensation:\n            correction_triples = (\n                (grouped1, grouped1_low, correction1),\n                (grouped2, grouped2_low, correction2),\n                (grouped12, grouped12_low, correction12),\n            )\n            safe_correction = all(\n                _cross_reduction_is_safe(\n                    high, low, xp, max_multiplier=correction\n                )\n                and _cross_reduction_is_safe(\n                    low, low, xp, max_multiplier=correction\n                )\n                for high, low, correction in correction_triples\n            )\n            if safe_correction:\n                def _low_order_covariance(high, low, correction):\n                    cross = _symmetrize(high.T @ low)\n                    low_square = _symmetrize(low.T @ low)\n                    return _symmetrize(\n                        ((2.0 * cross) + low_square) * float(correction)\n                    )\n\n                low_correction = _stable_inclusion_exclusion(\n                    _low_order_covariance(grouped1, grouped1_low, correction1),\n                    _low_order_covariance(grouped2, grouped2_low, correction2),\n                    _low_order_covariance(grouped12, grouped12_low, correction12),\n                    xp,\n                )\n                cov_work = _restore_coordinate_covariance(\n                    cov_work, common_scale, xp\n                )\n                cov_work = _symmetrize(cov_work + low_correction)\n                common_scale = xp.ones_like(projection_scale)\n'''
text = text[:start] + new_block + text[end:]
cov_path.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Hausman: compare range/null-space norms on a common finite scale instead of
# allowing np.linalg.norm to overflow to Inf on finite vectors.
# ---------------------------------------------------------------------------
diag_path = Path("statgpu/panel/_diagnostics.py")
text = diag_path.read_text(encoding="utf-8")
anchor = '''def _restore_squared_scale(value: float, scale: float) -> float:\n'''
helper = '''def _safe_l2_norm(values) -> float:\n    """Return a float64 L2 norm without avoidable squaring overflow."""\n    values = np.asarray(values, dtype=np.float64)\n    if values.size == 0:\n        return 0.0\n    scale = float(np.max(np.abs(values)))\n    if scale == 0.0:\n        return 0.0\n    if not np.isfinite(scale):\n        return float("inf")\n    unit_norm = float(np.sqrt(np.sum((values / scale) ** 2)))\n    if unit_norm == 0.0:\n        return 0.0\n    if scale > float(np.finfo(np.float64).max) / unit_norm:\n        return float("inf")\n    return float(scale * unit_norm)\n\n\ndef _restore_squared_scale(value: float, scale: float) -> float:\n'''
if helper not in text:
    if anchor not in text:
        raise RuntimeError("Hausman safe-norm insertion anchor not found")
    text = text.replace(anchor, helper, 1)

old_range = '''    projected = basis @ (basis.T @ d)\n    null_component = d - projected\n    range_tol = _relative_tolerance(np.linalg.norm(d), factor=1024.0)\n    meta["range_tolerance"] = float(range_tol)\n    meta["nullspace_component_norm"] = float(np.linalg.norm(null_component))\n    if float(np.linalg.norm(null_component)) > range_tol:\n'''
new_range = '''    projected = basis @ (basis.T @ d)\n    null_component = d - projected\n    d_norm = _safe_l2_norm(d)\n    null_norm = _safe_l2_norm(null_component)\n    range_tol = _relative_tolerance(d_norm, factor=1024.0)\n    meta["range_tolerance"] = float(range_tol)\n    meta["nullspace_component_norm"] = float(null_norm)\n\n    comparison_scale = max(\n        float(np.max(np.abs(d))) if d.size else 0.0,\n        float(np.max(np.abs(null_component))) if null_component.size else 0.0,\n    )\n    if comparison_scale == 0.0:\n        outside_range = False\n        range_tol_normalized = 0.0\n        null_norm_normalized = 0.0\n    else:\n        d_normalized = d / comparison_scale\n        null_normalized = null_component / comparison_scale\n        d_norm_normalized = float(np.linalg.norm(d_normalized))\n        null_norm_normalized = float(np.linalg.norm(null_normalized))\n        range_tol_normalized = (\n            1024.0 * np.finfo(np.float64).eps * d_norm_normalized\n        )\n        outside_range = null_norm_normalized > range_tol_normalized\n    meta["range_comparison_scale"] = float(comparison_scale)\n    meta["range_tolerance_normalized"] = float(range_tol_normalized)\n    meta["nullspace_component_norm_normalized"] = float(null_norm_normalized)\n    if outside_range:\n'''
if new_range not in text:
    if old_range not in text:
        raise RuntimeError("Hausman range guard anchor not found")
    text = text.replace(old_range, new_range, 1)
diag_path.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Hosted regressions.
# ---------------------------------------------------------------------------
p = Path("dev/tests/test_panel_stage_c_covariance.py")
text = p.read_text(encoding="utf-8")
blocks = [r'''


def test_two_way_nonnested_low_order_correction_does_not_require_gram_scaling():
    amplitude = 1.0e150
    small = 1.0e-150
    X = np.full((4, 1), 0.5, dtype=np.float64)
    scores = np.asarray([-amplitude, small, amplitude, -small], dtype=np.float64)
    resid = 2.0 * scores
    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)
    actual = two_way_clustered_covariance(X, resid, cluster1, cluster2)
    assert_allclose(actual, np.asarray([[-4.0]]), rtol=5e-13, atol=0.0)
''', r'''


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
    expected = np.asarray([[6.0 * amplitude * small + 2.0 * small * small]])
    assert_allclose(actual, expected, rtol=5e-13, atol=0.0)
''']
for block in blocks:
    name = block.split("def ", 1)[1].split("(", 1)[0]
    if name not in text:
        text = text.rstrip() + block.rstrip() + "\n"
p.write_text(text, encoding="utf-8")

p = Path("dev/tests/test_panel_stage_b_hausman_covariance.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_hausman_quadratic_rejects_large_finite_nullspace_component():
    result = _hausman_quadratic(
        np.asarray([1.0e154, 1.0e200]),
        np.diag(np.asarray([1.0e308, 0.0])),
    )
    assert result.applicable is False
    assert "outside the identified covariance-difference range" in result.reason
    assert result.metadata["range_comparison_scale"] == 1.0e200
    assert np.isfinite(result.metadata["range_tolerance_normalized"])
    assert np.isfinite(result.metadata["nullspace_component_norm_normalized"])
'''
if "test_hausman_quadratic_rejects_large_finite_nullspace_component" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
blocks = [r'''


def test_stage_c_torch_cpu_two_way_safe_gram_preserves_low_order_term():
    amplitude = 1.0e150
    small = 1.0e-150
    X = torch.full((4, 1), 0.5, dtype=torch.float64)
    scores = torch.tensor([-amplitude, small, amplitude, -small], dtype=torch.float64)
    resid = 2.0 * scores
    c1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    c2 = np.asarray([0, 1, 0, 1], dtype=np.int64)
    actual = two_way_clustered_covariance(X, resid, c1, c2, xp=torch)
    np.testing.assert_allclose(actual.detach().cpu().numpy(), [[-4.0]], rtol=5e-13, atol=0.0)
''', r'''


def test_stage_c_torch_cpu_two_way_group_debias_preserves_low_order_term():
    amplitude = 1.0e154
    small = 1.0e-154
    X = torch.full((4, 1), 0.5, dtype=torch.float64)
    scores = torch.tensor([-amplitude, amplitude, amplitude, small], dtype=torch.float64)
    resid = 2.0 * scores
    c1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    c2 = np.asarray([0, 1, 0, 1], dtype=np.int64)
    actual = two_way_clustered_covariance(
        X, resid, c1, c2, xp=torch, group_debias=True
    )
    np.testing.assert_allclose(actual.detach().cpu().numpy(), [[6.0]], rtol=5e-13, atol=0.0)
''', r'''


def test_stage_b_torch_cpu_hausman_large_singular_range_guard():
    result = _hausman_quadratic(
        np.asarray([1.0e154, 1.0e200]),
        np.diag(np.asarray([1.0e308, 0.0])),
    )
    assert result.applicable is False
    assert "outside the identified covariance-difference range" in result.reason
''']
for block in blocks:
    name = block.split("def ", 1)[1].split("(", 1)[0]
    if name not in text:
        text = text.rstrip() + block.rstrip() + "\n"
p.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Physical validator coverage: safe-Gram and group-debiased nonnested CGM plus
# the host-side Hausman singular-range guard.
# ---------------------------------------------------------------------------
p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
replace_pairs = [
('''    resid_nonnested_np = 2.0 * nonnested_scores\n    nonnested_cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)\n''', '''    resid_nonnested_np = 2.0 * nonnested_scores\n    nonnested_safe_scores = np.asarray(\n        [-1.0e150, 1.0e-150, 1.0e150, -1.0e-150], dtype=np.float64\n    )\n    resid_nonnested_safe_np = 2.0 * nonnested_safe_scores\n    nonnested_debias_scores = np.asarray(\n        [-nonnested_amplitude, nonnested_amplitude,\n         nonnested_amplitude, nonnested_small], dtype=np.float64\n    )\n    resid_nonnested_debias_np = 2.0 * nonnested_debias_scores\n    nonnested_cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)\n'''),
('''        X_nonnested, resid_nonnested = X_nonnested_np, resid_nonnested_np\n    elif backend == "cupy":\n''', '''        X_nonnested, resid_nonnested = X_nonnested_np, resid_nonnested_np\n        resid_nonnested_safe = resid_nonnested_safe_np\n        resid_nonnested_debias = resid_nonnested_debias_np\n    elif backend == "cupy":\n'''),
('''        X_nonnested = cp.asarray(X_nonnested_np)\n        resid_nonnested = cp.asarray(resid_nonnested_np)\n    elif backend == "torch":\n''', '''        X_nonnested = cp.asarray(X_nonnested_np)\n        resid_nonnested = cp.asarray(resid_nonnested_np)\n        resid_nonnested_safe = cp.asarray(resid_nonnested_safe_np)\n        resid_nonnested_debias = cp.asarray(resid_nonnested_debias_np)\n    elif backend == "torch":\n'''),
('''        resid_nonnested = torch.as_tensor(\n            resid_nonnested_np, dtype=torch.float64, device="cuda"\n        )\n    else:\n''', '''        resid_nonnested = torch.as_tensor(\n            resid_nonnested_np, dtype=torch.float64, device="cuda"\n        )\n        resid_nonnested_safe = torch.as_tensor(\n            resid_nonnested_safe_np, dtype=torch.float64, device="cuda"\n        )\n        resid_nonnested_debias = torch.as_tensor(\n            resid_nonnested_debias_np, dtype=torch.float64, device="cuda"\n        )\n    else:\n'''),
('''    nonnested_two_way = _array(two_way_clustered_covariance(\n        X_nonnested,\n        resid_nonnested,\n        nonnested_cluster1,\n        nonnested_cluster2,\n        xp=xp,\n    ))\n    for name, value in (("one_way", one_way),''', '''    nonnested_two_way = _array(two_way_clustered_covariance(\n        X_nonnested, resid_nonnested, nonnested_cluster1, nonnested_cluster2, xp=xp\n    ))\n    nonnested_two_way_safe = _array(two_way_clustered_covariance(\n        X_nonnested, resid_nonnested_safe, nonnested_cluster1, nonnested_cluster2, xp=xp\n    ))\n    nonnested_two_way_group_debias = _array(two_way_clustered_covariance(\n        X_nonnested, resid_nonnested_debias, nonnested_cluster1, nonnested_cluster2,\n        xp=xp, group_debias=True\n    ))\n    for name, value in (("one_way", one_way),'''),
('''("mixed_dk", mixed_dk), ("nonnested_two_way", nonnested_two_way)):\n''', '''("mixed_dk", mixed_dk), ("nonnested_two_way", nonnested_two_way), ("nonnested_two_way_safe", nonnested_two_way_safe), ("nonnested_two_way_group_debias", nonnested_two_way_group_debias)):\n'''),
('''    np.testing.assert_allclose(\n        nonnested_two_way,\n        np.asarray([[-4.0 * nonnested_amplitude * nonnested_small]]),\n        rtol=2e-12,\n        atol=0.0,\n    )\n    return {\n''', '''    np.testing.assert_allclose(\n        nonnested_two_way, np.asarray([[-4.0]]), rtol=2e-12, atol=0.0\n    )\n    np.testing.assert_allclose(\n        nonnested_two_way_safe, np.asarray([[-4.0]]), rtol=5e-13, atol=0.0\n    )\n    np.testing.assert_allclose(\n        nonnested_two_way_group_debias, np.asarray([[6.0]]), rtol=5e-13, atol=0.0\n    )\n    return {\n'''),
('''        "nonnested_two_way_structural_cancellation": nonnested_two_way.tolist(),\n    }\n''', '''        "nonnested_two_way_structural_cancellation": nonnested_two_way.tolist(),\n        "nonnested_two_way_safe_gram_cancellation": nonnested_two_way_safe.tolist(),\n        "nonnested_two_way_group_debias_cancellation": nonnested_two_way_group_debias.tolist(),\n    }\n'''),
]
for old, new in replace_pairs:
    if new not in text:
        if old not in text:
            raise RuntimeError(f"physical validator anchor not found: {old[:100]!r}")
        text = text.replace(old, new, 1)

old_hausman = '''    np.testing.assert_allclose(\n        results["large"]["pvalue"], results["subnormal"]["pvalue"],\n        rtol=3e-12, atol=0.0,\n    )\n    return {"status": "success", "backend": backend, "cases": results}\n'''
new_hausman = '''    np.testing.assert_allclose(\n        results["large"]["pvalue"], results["subnormal"]["pvalue"],\n        rtol=3e-12, atol=0.0,\n    )\n    singular = _hausman_quadratic(\n        np.asarray([1.0e154, 1.0e200]),\n        np.diag(np.asarray([1.0e308, 0.0])),\n    )\n    if singular.applicable or not singular.reason or (\n        "outside the identified covariance-difference range" not in singular.reason\n    ):\n        raise AssertionError(\n            f"{backend}: large singular Hausman range guard failed: {singular}"\n        )\n    return {\n        "status": "success",\n        "backend": backend,\n        "cases": results,\n        "large_singular_range_rejected": True,\n    }\n'''
if new_hausman not in text:
    if old_hausman not in text:
        raise RuntimeError("Hausman physical audit anchor not found")
    text = text.replace(old_hausman, new_hausman, 1)
p.write_text(text, encoding="utf-8")

p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
old_keys = '''        "nonnested_two_way_structural_cancellation",\n    ):\n'''
new_keys = '''        "nonnested_two_way_structural_cancellation",\n        "nonnested_two_way_safe_gram_cancellation",\n        "nonnested_two_way_group_debias_cancellation",\n    ):\n'''
if new_keys not in text:
    if old_keys not in text:
        raise RuntimeError("physical covariance key anchor not found")
    text = text.replace(old_keys, new_keys, 1)
old_h = '''    assert audit["backend"] == "numpy"\n    for label in ("large", "subnormal"):\n'''
new_h = '''    assert audit["backend"] == "numpy"\n    assert audit["large_singular_range_rejected"] is True\n    for label in ("large", "subnormal"):\n'''
if new_h not in text:
    if old_h not in text:
        raise RuntimeError("physical Hausman contract anchor not found")
    text = text.replace(old_h, new_h, 1)
p.write_text(text, encoding="utf-8")

for path in (
    "statgpu/panel/_covariance.py",
    "statgpu/panel/_diagnostics.py",
    "dev/tests/test_panel_stage_c_covariance.py",
    "dev/tests/test_panel_stage_b_hausman_covariance.py",
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
):
    p = Path(path)
    p.write_text(p.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
