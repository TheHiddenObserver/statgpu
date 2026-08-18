from pathlib import Path


def replace_once(path, old, new, label):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"{label} anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_before(path, anchor, addition, label):
    replace_once(path, anchor, addition + anchor, label)


# Keep the private diag_floor argument source-compatible, but never let an
# absolute, dimensionful floor alter public inference.
replace_once(
    "statgpu/panel/_base.py",
    '''        # Positive absolute variance floors are dimensionful and break\n        # outcome-scale equivariance. Keep the private compatibility argument,\n        # but fail closed if a caller tries to reintroduce such a floor.\n        if diag_floor not in (None, 0, 0.0):\n            raise ValueError(\n                "positive absolute covariance diagonal floors are not supported"\n            )\n        diag = xp_maximum(diag, 0.0, xp)\n''',
    '''        # ``diag_floor`` is retained only for private-call compatibility.\n        # A positive absolute floor is dimensionful and would break outcome-scale\n        # equivariance, so it is intentionally ignored. Exact zero is handled by\n        # the explicit statistic-ratio convention below.\n        _ = diag_floor\n        diag = xp_maximum(diag, 0.0, xp)\n''',
    "private diag_floor compatibility",
)

# Stable covariance combination primitives.
replace_once(
    "statgpu/panel/_covariance.py",
    '''def _symmetrize(matrix):\n    return 0.5 * (matrix + matrix.T)\n\n\ndef _grouped_score_sums(scores, codes_np, *, n_groups: int, xp):\n''',
    '''def _symmetrize(matrix):\n    """Average a matrix with its transpose without avoidable overflow.\n\n    ``0.5 * (a + b)`` preserves subnormal entries but can overflow when both\n    finite operands are near DBL_MAX. Only those same-sign risky entries use\n    ``0.5*a + 0.5*b``; ordinary and subnormal entries retain the direct sum.\n    """\n    xp = _ensure_xp(None, matrix)\n    other = matrix.T\n    abs_left = xp.abs(matrix)\n    abs_right = xp.abs(other)\n    same_sign = ((matrix >= 0.0) & (other >= 0.0)) | (\n        (matrix < 0.0) & (other < 0.0)\n    )\n    risk = same_sign & (abs_left > float(np.finfo(np.float64).max) - abs_right)\n    left = xp.where(risk, 0.5 * matrix, matrix)\n    right = xp.where(risk, 0.5 * other, other)\n    summed = left + right\n    return xp.where(risk, summed, 0.5 * summed)\n\n\ndef _weighted_symmetric_sum(matrix, weight):\n    """Return ``weight * (matrix + matrix.T)`` without an unsafe raw sum."""\n    return _symmetrize(matrix) * (2.0 * weight)\n\n\ndef _stable_inclusion_exclusion(V1, V2, V12, xp):\n    """Return ``V1 + V2 - V12`` with cancellation before risky addition.\n\n    If either marginal entry has the same sign as the intersection entry,\n    subtract that pair first; subtraction of same-sign finite values cannot\n    overflow. If neither does, all three inclusion-exclusion terms have the same\n    sign and any overflow in their direct sum reflects an unrepresentable result.\n    """\n    same1 = ((V1 >= 0.0) & (V12 >= 0.0)) | ((V1 < 0.0) & (V12 < 0.0))\n    same2 = ((V2 >= 0.0) & (V12 >= 0.0)) | ((V2 < 0.0) & (V12 < 0.0))\n    use1 = same1\n    use2 = (~use1) & same2\n    stable_mask = use1 | use2\n\n    first = xp.where(use1, V1, V2)\n    remaining = xp.where(use1, V2, V1)\n    first_safe = xp.where(stable_mask, first, xp.zeros_like(first))\n    remaining_safe = xp.where(stable_mask, remaining, xp.zeros_like(remaining))\n    intersection_safe = xp.where(stable_mask, V12, xp.zeros_like(V12))\n    stable = (first_safe - intersection_safe) + remaining_safe\n\n    fallback_V1 = xp.where(stable_mask, xp.zeros_like(V1), V1)\n    fallback_V2 = xp.where(stable_mask, xp.zeros_like(V2), V2)\n    fallback_V12 = xp.where(stable_mask, xp.zeros_like(V12), V12)\n    direct = (fallback_V1 + fallback_V2) - fallback_V12\n    return xp.where(stable_mask, stable, direct)\n\n\ndef _grouped_score_sums(scores, codes_np, *, n_groups: int, xp):\n''',
    "stable covariance combination primitives",
)

replace_once(
    "statgpu/panel/_covariance.py",
    '''    out = xp_zeros(\n        (int(n_groups), int(scores.shape[1])),\n        dtype=xp.float64,\n        xp=xp,\n        ref_arr=scores,\n    )\n    if hasattr(out, "scatter_add_"):\n        out.scatter_add_(0, codes.unsqueeze(1).expand_as(scores), scores)\n    elif type(out).__module__.startswith("cupy"):\n        xp.add.at(out, codes, scores)\n    else:\n        np.add.at(out, codes_np, scores)\n    return out\n''',
    '''    shape = (int(n_groups), int(scores.shape[1]))\n    out = xp_zeros(shape, dtype=xp.float64, xp=xp, ref_arr=scores)\n\n    # A group can have a finite final score even when sequential same-sign\n    # partial sums overflow before later cancellation. Scale only risky\n    # group/coordinate reductions by the group size; safe groups and coordinates\n    # remain completely untouched, so tiny unrelated scores are not normalized\n    # away.\n    abs_scores = xp.abs(scores)\n    max_abs = xp_zeros(shape, dtype=xp.float64, xp=xp, ref_arr=scores)\n    index = codes.unsqueeze(1).expand_as(scores) if hasattr(codes, "unsqueeze") else None\n    if hasattr(max_abs, "scatter_reduce_"):\n        max_abs.scatter_reduce_(\n            0, index, abs_scores, reduce="amax", include_self=True\n        )\n    elif type(max_abs).__module__.startswith("cupy"):\n        xp.maximum.at(max_abs, codes, abs_scores)\n    else:\n        np.maximum.at(max_abs, codes_np, abs_scores)\n\n    counts_np = np.bincount(codes_np, minlength=int(n_groups)).astype(np.float64)\n    counts = xp_asarray(\n        counts_np, dtype=xp.float64, xp=xp, ref_arr=scores\n    ).reshape(-1, 1)\n    limit = float(np.finfo(np.float64).max) / counts\n    factor = xp.where(max_abs > limit, counts, xp.ones_like(max_abs))\n    working = scores / factor[codes]\n\n    if hasattr(out, "scatter_add_"):\n        out.scatter_add_(0, index, working)\n    elif type(out).__module__.startswith("cupy"):\n        xp.add.at(out, codes, working)\n    else:\n        np.add.at(out, codes_np, working)\n    return out * factor\n''',
    "overflow-safe grouped score reduction",
)
replace_once(
    "statgpu/panel/_covariance.py",
    "    return _symmetrize(V1 + V2 - V12)\n",
    "    return _symmetrize(_stable_inclusion_exclusion(V1, V2, V12, xp))\n",
    "two-way inclusion exclusion",
)
replace_once(
    "statgpu/panel/_covariance.py",
    "        cov = cov + w * (gamma_h + gamma_h.T)\n",
    "        cov = cov + _weighted_symmetric_sum(gamma_h, float(w))\n",
    "HAC weighted symmetric lag",
)
replace_once(
    "statgpu/panel/_covariance.py",
    "        cov = cov + weights[lag] * (gamma + gamma.T)\n",
    "        cov = cov + _weighted_symmetric_sum(gamma, weights[lag])\n",
    "DK weighted symmetric lag",
)

# NumPy regressions exercise only public covariance primitives except for one
# direct symmetrization check that protects the minimum-subnormal edge.
p = Path("dev/tests/test_panel_stage_c_covariance.py")
text = p.read_text(encoding="utf-8")
text = text.replace(
    '''    _dk_kernel_weights,\n    clustered_covariance,\n''',
    '''    _dk_kernel_weights,\n    _symmetrize,\n    clustered_covariance,\n''',
    1,
)
text += '''\n\ndef test_covariance_symmetrization_preserves_huge_and_subnormal_finite_entries():\n    huge = 1.0e308\n    tiny = np.nextafter(0.0, 1.0)\n    matrix = np.asarray([[huge, tiny], [tiny, -huge]], dtype=np.float64)\n    actual = _symmetrize(matrix)\n    assert np.all(np.isfinite(actual))\n    np.testing.assert_array_equal(actual, matrix)\n\n\ndef test_cluster_and_two_way_inclusion_exclusion_preserve_huge_finite_covariance():\n    X = np.ones((4, 1), dtype=np.float64)\n    amplitude = 1.4e154\n    resid = np.asarray([amplitude, amplitude, -amplitude, -amplitude])\n    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)\n    expected = np.asarray([[0.5 * amplitude * amplitude]], dtype=np.float64)\n\n    one_way = clustered_covariance(X, resid, groups)\n    two_way = two_way_clustered_covariance(X, resid, groups, groups)\n    assert np.all(np.isfinite(one_way))\n    assert np.all(np.isfinite(two_way))\n    np.testing.assert_allclose(one_way, expected, rtol=4e-15, atol=0.0)\n    np.testing.assert_allclose(two_way, expected, rtol=5e-15, atol=0.0)\n\n\ndef test_cluster_group_reduction_survives_finite_partial_sum_overflow():\n    X = np.ones((6, 1), dtype=np.float64) * 1.0e-154\n    resid = np.asarray([1.0e155, 1.0e155, -1.0e155, -1.0e155, 1.0, -1.0])\n    groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)\n    cov = clustered_covariance(X, resid, groups)\n    assert np.all(np.isfinite(cov))\n    np.testing.assert_allclose(cov, np.zeros((1, 1)), rtol=0.0, atol=0.0)\n\n\ndef test_hac_and_dk_weighted_lags_do_not_overflow_before_finite_cancellation():\n    n = 16\n    influence_amplitude = 3.0e153\n    X = np.ones((n, 1), dtype=np.float64)\n    resid = n * influence_amplitude * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)\n    time = np.arange(n, dtype=np.int64)\n    expected_hac = np.asarray([[influence_amplitude ** 2]], dtype=np.float64)\n    expected_dk = expected_hac * (n / (n - 1.0))\n\n    hac = hac_covariance(X, resid, bandwidth=1)\n    dk = driscoll_kraay_covariance(X, resid, time, bandwidth=1)\n    assert np.all(np.isfinite(hac))\n    assert np.all(np.isfinite(dk))\n    np.testing.assert_allclose(hac, expected_hac, rtol=8e-15, atol=0.0)\n    np.testing.assert_allclose(dk, expected_dk, rtol=8e-15, atol=0.0)\n'''
p.write_text(text, encoding="utf-8")

# Maintained Torch 2.0.1 CPU coverage, including the private diag_floor
# compatibility call that existed before this review.
p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
text = text.replace(
    "from statgpu.panel._diagnostic_context import explicit_constant_column\n",
    '''from statgpu.panel._covariance import (\n    clustered_covariance,\n    driscoll_kraay_covariance,\n    hac_covariance,\n    two_way_clustered_covariance,\n)\nfrom statgpu.panel._diagnostic_context import explicit_constant_column\n''',
    1,
)
text += '''\n\ndef test_stage_c_torch_cpu_extreme_covariance_combinations_remain_finite():\n    amplitude = 1.4e154\n    X = torch.ones((4, 1), dtype=torch.float64)\n    resid = torch.tensor([amplitude, amplitude, -amplitude, -amplitude], dtype=torch.float64)\n    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)\n    expected = np.asarray([[0.5 * amplitude * amplitude]], dtype=np.float64)\n    one_way = clustered_covariance(X, resid, groups, xp=torch)\n    two_way = two_way_clustered_covariance(X, resid, groups, groups, xp=torch)\n    assert_allclose(one_way, expected, rtol=5e-14, atol=0.0)\n    assert_allclose(two_way, expected, rtol=6e-14, atol=0.0)\n\n    X_tiny = torch.ones((6, 1), dtype=torch.float64) * 1.0e-154\n    resid_cancel = torch.tensor(\n        [1.0e155, 1.0e155, -1.0e155, -1.0e155, 1.0, -1.0],\n        dtype=torch.float64,\n    )\n    cancel_groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)\n    cancel_cov = clustered_covariance(X_tiny, resid_cancel, cancel_groups, xp=torch)\n    assert_allclose(cancel_cov, np.zeros((1, 1)), rtol=0.0, atol=0.0)\n\n    n = 16\n    influence_amplitude = 3.0e153\n    X_hac = torch.ones((n, 1), dtype=torch.float64)\n    signs = torch.where(\n        torch.arange(n) % 2 == 0,\n        torch.tensor(1.0, dtype=torch.float64),\n        torch.tensor(-1.0, dtype=torch.float64),\n    )\n    resid_hac = (n * influence_amplitude) * signs\n    time = np.arange(n, dtype=np.int64)\n    expected_hac = np.asarray([[influence_amplitude ** 2]], dtype=np.float64)\n    assert_allclose(\n        hac_covariance(X_hac, resid_hac, bandwidth=1, xp=torch),\n        expected_hac,\n        rtol=8e-14,\n        atol=0.0,\n    )\n    assert_allclose(\n        driscoll_kraay_covariance(X_hac, resid_hac, time, bandwidth=1, xp=torch),\n        expected_hac * (n / (n - 1.0)),\n        rtol=8e-14,\n        atol=0.0,\n    )\n'''
p.write_text(text, encoding="utf-8")

# Physical CuPy/Torch validator coverage for all new covariance arithmetic.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    "from statgpu.panel._covariance import ols_covariance\n",
    "from statgpu.panel._covariance import hac_covariance, ols_covariance, two_way_clustered_covariance\n",
    "physical covariance imports",
)
append_before(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    "def _zero_variance_inference_audit(backend):\n",
    '''def _covariance_extreme_scale_audit(backend):\n    amplitude = 1.4e154\n    X_np = np.ones((4, 1), dtype=np.float64)\n    resid_np = np.asarray([amplitude, amplitude, -amplitude, -amplitude])\n    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)\n    X_cancel_np = np.ones((6, 1), dtype=np.float64) * 1.0e-154\n    resid_cancel_np = np.asarray([1.0e155, 1.0e155, -1.0e155, -1.0e155, 1.0, -1.0])\n    cancel_groups = np.asarray([0, 0, 0, 0, 1, 1], dtype=np.int64)\n    n = 16\n    influence_amplitude = 3.0e153\n    X_hac_np = np.ones((n, 1), dtype=np.float64)\n    resid_hac_np = n * influence_amplitude * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)\n    time = np.arange(n, dtype=np.int64)\n\n    if backend == "numpy":\n        xp = np\n        X, resid = X_np, resid_np\n        X_cancel, resid_cancel = X_cancel_np, resid_cancel_np\n        X_hac, resid_hac = X_hac_np, resid_hac_np\n    elif backend == "cupy":\n        import cupy as cp\n        xp = cp\n        X, resid = cp.asarray(X_np), cp.asarray(resid_np)\n        X_cancel, resid_cancel = cp.asarray(X_cancel_np), cp.asarray(resid_cancel_np)\n        X_hac, resid_hac = cp.asarray(X_hac_np), cp.asarray(resid_hac_np)\n    elif backend == "torch":\n        import torch\n        xp = torch\n        X = torch.as_tensor(X_np, dtype=torch.float64, device="cuda")\n        resid = torch.as_tensor(resid_np, dtype=torch.float64, device="cuda")\n        X_cancel = torch.as_tensor(X_cancel_np, dtype=torch.float64, device="cuda")\n        resid_cancel = torch.as_tensor(resid_cancel_np, dtype=torch.float64, device="cuda")\n        X_hac = torch.as_tensor(X_hac_np, dtype=torch.float64, device="cuda")\n        resid_hac = torch.as_tensor(resid_hac_np, dtype=torch.float64, device="cuda")\n    else:\n        raise ValueError(backend)\n\n    expected_cluster = np.asarray([[0.5 * amplitude * amplitude]])\n    expected_hac = np.asarray([[influence_amplitude ** 2]])\n    one_way = _array(clustered_covariance(X, resid, groups, xp=xp))\n    two_way = _array(two_way_clustered_covariance(X, resid, groups, groups, xp=xp))\n    cancellation = _array(clustered_covariance(X_cancel, resid_cancel, cancel_groups, xp=xp))\n    hac = _array(hac_covariance(X_hac, resid_hac, bandwidth=1, xp=xp))\n    dk = _array(driscoll_kraay_covariance(X_hac, resid_hac, time, bandwidth=1, xp=xp))\n    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk)):\n        if not np.all(np.isfinite(value)):\n            raise AssertionError(f"{backend}: {name} produced non-finite covariance")\n    np.testing.assert_allclose(one_way, expected_cluster, rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(two_way, expected_cluster, rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(cancellation, np.zeros((1, 1)), rtol=0.0, atol=0.0)\n    np.testing.assert_allclose(hac, expected_hac, rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(dk, expected_hac * (n / (n - 1.0)), rtol=8e-13, atol=0.0)\n    return {\n        "status": "success",\n        "backend": backend,\n        "one_way": one_way.tolist(),\n        "two_way": two_way.tolist(),\n        "group_cancellation": cancellation.tolist(),\n        "hac": hac.tolist(),\n        "driscoll_kraay": dk.tolist(),\n    }\n\n\n''',
    "physical covariance extreme scale audit",
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''            "gram_overflow_certificate": _gram_overflow_certificate_audit(backend),\n            "zero_variance_inference": _zero_variance_inference_audit(backend),\n''',
    '''            "gram_overflow_certificate": _gram_overflow_certificate_audit(backend),\n            "covariance_extreme_scale": _covariance_extreme_scale_audit(backend),\n            "zero_variance_inference": _zero_variance_inference_audit(backend),\n''',
    "physical payload covariance audit",
)

p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
text += '''\n\ndef test_stage_c_runner_covariance_extreme_scale_audit_is_executable():\n    audit = _MOD._covariance_extreme_scale_audit("numpy")\n    assert audit["status"] == "success"\n    assert audit["backend"] == "numpy"\n    for key in ("one_way", "two_way", "group_cancellation", "hac", "driscoll_kraay"):\n        assert np.all(np.isfinite(np.asarray(audit[key], dtype=np.float64))), key\n'''
p.write_text(text, encoding="utf-8")
