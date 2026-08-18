from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:140]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "statgpu/panel/_covariance.py",
    '''def _influence_rows(X, resid, xp):\n''',
    '''def _cross_reduction_is_safe(left, right, xp) -> bool:\n    """Return whether ``left.T @ right`` has a conservative rowwise range bound.\n\n    Matrix-product terms pair values from the same observation/group row.  Use\n    per-row infinity norms so unrelated column maxima do not reject a safe\n    mixed-range cross term.  This is only a sufficient condition; uncertain\n    cases remain on the established scaled covariance path.\n    """\n    n_terms = max(1, int(left.shape[0]))\n    if _is_torch(xp):\n        left_row_max = xp.max(xp.abs(left), dim=1).values\n        right_row_max = xp.max(xp.abs(right), dim=1).values\n    else:\n        left_row_max = xp.max(xp.abs(left), axis=1)\n        right_row_max = xp.max(xp.abs(right), axis=1)\n    finite = xp.isfinite(left_row_max) & xp.isfinite(right_row_max)\n    large = xp.maximum(left_row_max, right_row_max)\n    small = xp.minimum(left_row_max, right_row_max)\n    safe_large = xp.maximum(large, xp.ones_like(large))\n    per_term_limit = 0.25 * float(np.finfo(np.float64).max) / float(n_terms)\n    safe = finite & (small <= float(per_term_limit) / safe_large)\n    return bool(_to_float_scalar(xp.all(safe)))\n\n\ndef _influence_rows(X, resid, xp):\n''',
)

replace_once(
    "statgpu/panel/_covariance.py",
    '''def _grouped_score_sums(scores, codes_np, *, n_groups: int, xp):\n    """Sum an observation-by-parameter score matrix by integer group code."""\n''',
    '''def _grouped_score_sums(\n    scores, codes_np, *, n_groups: int, xp, return_compensation: bool = False\n):\n    """Sum scores by group, optionally retaining the final cancellation residual."""\n''',
)
replace_once(
    "statgpu/panel/_covariance.py",
    '''    else:\n        np.add.at(out, codes_np, positive)\n        np.add.at(negative_out, codes_np, negative)\n    return (out + negative_out) * factor\n''',
    '''    else:\n        np.add.at(out, codes_np, positive)\n        np.add.at(negative_out, codes_np, negative)\n\n    summed = out + negative_out\n    if not return_compensation:\n        return summed * factor\n\n    # Knuth TwoSum residual for the final opposite-sign combination.  Positive\n    # and negative reductions are already accumulated separately above; this\n    # residual preserves a low-order term such as -1e154 + 1e-154 without\n    # changing the ordinary public grouped-score value.\n    virtual_negative = summed - out\n    residual = (out - (summed - virtual_negative)) + (negative_out - virtual_negative)\n    return summed * factor, residual * factor\n''',
)

old = '''    else:\n        multiplier = max(correction1, correction2, correction12)\n        (grouped1_work, grouped2_work, grouped12_work), common_scale = (\n            _common_gram_working_values(\n                [grouped1, grouped2, grouped12],\n                xp,\n                max_multiplier=multiplier,\n            )\n        )\n        V1_work = _symmetrize(\n            grouped1_work.T @ grouped1_work * float(correction1)\n        )\n        V2_work = _symmetrize(\n            grouped2_work.T @ grouped2_work * float(correction2)\n        )\n        V12_work = _symmetrize(\n            grouped12_work.T @ grouped12_work * float(correction12)\n        )\n        cov_work = _stable_inclusion_exclusion(V1_work, V2_work, V12_work, xp)\n'''
new = '''    else:\n        multiplier = max(correction1, correction2, correction12)\n        (grouped1_work, grouped2_work, grouped12_work), common_scale = (\n            _common_gram_working_values(\n                [grouped1, grouped2, grouped12],\n                xp,\n                max_multiplier=multiplier,\n            )\n        )\n        V1_work = _symmetrize(\n            grouped1_work.T @ grouped1_work * float(correction1)\n        )\n        V2_work = _symmetrize(\n            grouped2_work.T @ grouped2_work * float(correction2)\n        )\n        V12_work = _symmetrize(\n            grouped12_work.T @ grouped12_work * float(correction12)\n        )\n        cov_work = _stable_inclusion_exclusion(V1_work, V2_work, V12_work, xp)\n\n        # The ordinary grouped score is a float64 high part.  Under extreme\n        # nonnested CGM cancellation it can discard a representable low term at\n        # the final positive+negative group-sum addition; after component Gram\n        # scaling that lost term can control the final covariance.  Recover only\n        # that final-addition residual, and only for the un-debiased definition\n        # where all three CGM components have unit weight.\n        need_compensation = (\n            not group_debias\n            and _to_float_scalar(xp.max(common_scale)) > 1.0\n        )\n        if need_compensation:\n            grouped1_high, grouped1_low = _grouped_score_sums(\n                influence, c1, n_groups=int(len(labels1)), xp=xp,\n                return_compensation=True,\n            )\n            grouped2_high, grouped2_low = _grouped_score_sums(\n                influence, c2, n_groups=int(len(labels2)), xp=xp,\n                return_compensation=True,\n            )\n            grouped12_high, grouped12_low = _grouped_score_sums(\n                influence, c12, n_groups=n12, xp=xp,\n                return_compensation=True,\n            )\n            safe_correction = all(\n                _cross_reduction_is_safe(high, low, xp)\n                and _cross_reduction_is_safe(low, low, xp)\n                for high, low in (\n                    (grouped1_high, grouped1_low),\n                    (grouped2_high, grouped2_low),\n                    (grouped12_high, grouped12_low),\n                )\n            )\n            if safe_correction:\n                def _low_order_covariance(high, low):\n                    cross = _symmetrize(high.T @ low)\n                    return _symmetrize(\n                        (2.0 * cross) + _symmetrize(low.T @ low)\n                    )\n\n                correction1 = _low_order_covariance(grouped1_high, grouped1_low)\n                correction2 = _low_order_covariance(grouped2_high, grouped2_low)\n                correction12 = _low_order_covariance(\n                    grouped12_high, grouped12_low\n                )\n                correction = _stable_inclusion_exclusion(\n                    correction1, correction2, correction12, xp\n                )\n                cov_work = _restore_coordinate_covariance(\n                    cov_work, common_scale, xp\n                )\n                cov_work = _symmetrize(cov_work + correction)\n                common_scale = xp.ones_like(projection_scale)\n'''
replace_once("statgpu/panel/_covariance.py", old, new)

# Add the public NumPy regression.  The exact-SVD fixture keeps the issue inside
# CGM grouped covariance arithmetic rather than upstream least-squares roundoff.
p = Path("dev/tests/test_panel_stage_c_covariance.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_two_way_nonnested_structural_cancellation_preserves_low_group_sum():
    n = 4
    amplitude = 1.0e154
    small = 1.0e-154
    X = np.full((n, 1), 0.5, dtype=np.float64)
    target_scores = np.asarray(
        [-amplitude, small, amplitude, -small], dtype=np.float64
    )
    resid = 2.0 * target_scores
    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)
    actual = two_way_clustered_covariance(X, resid, cluster1, cluster2)
    expected = np.asarray([[-4.0 * amplitude * small]], dtype=np.float64)
    assert_allclose(actual, expected, rtol=2e-12, atol=0.0)
'''
if "test_two_way_nonnested_structural_cancellation_preserves_low_group_sum" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_stage_c_torch_cpu_two_way_nonnested_structural_cancellation():
    n = 4
    amplitude = 1.0e154
    small = 1.0e-154
    X = torch.full((n, 1), 0.5, dtype=torch.float64)
    target_scores = torch.tensor(
        [-amplitude, small, amplitude, -small], dtype=torch.float64
    )
    resid = 2.0 * target_scores
    cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)
    actual = two_way_clustered_covariance(
        X, resid, cluster1, cluster2, xp=torch
    ).detach().cpu().numpy()
    np.testing.assert_allclose(
        actual, np.asarray([[-4.0 * amplitude * small]]), rtol=2e-12, atol=0.0
    )
'''
if "test_stage_c_torch_cpu_two_way_nonnested_structural_cancellation" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

# Extend the maintained physical extreme-scale audit with the same exact-SVD
# nonnested structural-cancellation case for CuPy and Torch CUDA.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    mixed_nonmonotone_coarse = np.asarray([1, 1, 0], dtype=np.int64)\n\n    if backend == "numpy":\n''',
    '''    mixed_nonmonotone_coarse = np.asarray([1, 1, 0], dtype=np.int64)\n\n    nonnested_n = 4\n    nonnested_amplitude = 1.0e154\n    nonnested_small = 1.0e-154\n    X_nonnested_np = np.full((nonnested_n, 1), 0.5, dtype=np.float64)\n    nonnested_scores = np.asarray(\n        [-nonnested_amplitude, nonnested_small,\n         nonnested_amplitude, -nonnested_small],\n        dtype=np.float64,\n    )\n    resid_nonnested_np = 2.0 * nonnested_scores\n    nonnested_cluster1 = np.asarray([0, 0, 1, 1], dtype=np.int64)\n    nonnested_cluster2 = np.asarray([0, 1, 0, 1], dtype=np.int64)\n\n    if backend == "numpy":\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        X_mixed, resid_mixed = X_mixed_np, resid_mixed_np\n    elif backend == "cupy":\n''',
    '''        X_mixed, resid_mixed = X_mixed_np, resid_mixed_np\n        X_nonnested, resid_nonnested = X_nonnested_np, resid_nonnested_np\n    elif backend == "cupy":\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        X_mixed, resid_mixed = cp.asarray(X_mixed_np), cp.asarray(resid_mixed_np)\n    elif backend == "torch":\n''',
    '''        X_mixed, resid_mixed = cp.asarray(X_mixed_np), cp.asarray(resid_mixed_np)\n        X_nonnested = cp.asarray(X_nonnested_np)\n        resid_nonnested = cp.asarray(resid_nonnested_np)\n    elif backend == "torch":\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        X_mixed = torch.as_tensor(X_mixed_np, dtype=torch.float64, device="cuda")\n        resid_mixed = torch.as_tensor(resid_mixed_np, dtype=torch.float64, device="cuda")\n    else:\n''',
    '''        X_mixed = torch.as_tensor(X_mixed_np, dtype=torch.float64, device="cuda")\n        resid_mixed = torch.as_tensor(resid_mixed_np, dtype=torch.float64, device="cuda")\n        X_nonnested = torch.as_tensor(\n            X_nonnested_np, dtype=torch.float64, device="cuda"\n        )\n        resid_nonnested = torch.as_tensor(\n            resid_nonnested_np, dtype=torch.float64, device="cuda"\n        )\n    else:\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    mixed_dk = _array(driscoll_kraay_covariance(\n        X_mixed, resid_mixed, mixed_time, bandwidth=0, xp=xp\n    ))\n    for name, value in (("one_way", one_way),''',
    '''    mixed_dk = _array(driscoll_kraay_covariance(\n        X_mixed, resid_mixed, mixed_time, bandwidth=0, xp=xp\n    ))\n    nonnested_two_way = _array(two_way_clustered_covariance(\n        X_nonnested,\n        resid_nonnested,\n        nonnested_cluster1,\n        nonnested_cluster2,\n        xp=xp,\n    ))\n    for name, value in (("one_way", one_way),''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''("mixed_two_way_permuted", mixed_two_way_permuted), ("mixed_dk", mixed_dk)):\n''',
    '''("mixed_two_way_permuted", mixed_two_way_permuted), ("mixed_dk", mixed_dk), ("nonnested_two_way", nonnested_two_way)):\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    np.testing.assert_allclose(mixed_dk, np.asarray([[1.5e-200]]), rtol=8e-13, atol=0.0)\n    return {\n''',
    '''    np.testing.assert_allclose(mixed_dk, np.asarray([[1.5e-200]]), rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(\n        nonnested_two_way,\n        np.asarray([[-4.0 * nonnested_amplitude * nonnested_small]]),\n        rtol=2e-12,\n        atol=0.0,\n    )\n    return {\n''',
)
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''        "mixed_driscoll_kraay": mixed_dk.tolist(),\n    }\n''',
    '''        "mixed_driscoll_kraay": mixed_dk.tolist(),\n        "nonnested_two_way_structural_cancellation": nonnested_two_way.tolist(),\n    }\n''',
)

p = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = p.read_text(encoding="utf-8")
old = '''        "lag_accumulator_driscoll_kraay",\n    ):\n'''
new = '''        "lag_accumulator_driscoll_kraay",\n        "nonnested_two_way_structural_cancellation",\n    ):\n'''
if old in text and new not in text:
    text = text.replace(old, new, 1)
p.write_text(text.rstrip() + "\n", encoding="utf-8")

for path in (
    "dev/tests/test_panel_stage_c_covariance.py",
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
):
    p = Path(path)
    p.write_text(p.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")
