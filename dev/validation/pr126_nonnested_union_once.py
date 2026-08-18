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
    '''def _cross_reduction_is_safe(left, right, xp) -> bool:\n    """Return whether ``left.T @ right`` has a conservative float64 range bound.\n\n    This is deliberately only a sufficient condition.  It is used to select a\n    cancellation-preserving two-way-cluster shortcut; if the bound is not met,\n    callers retain the existing scaled covariance path.\n    """\n    n_terms = max(1, int(left.shape[0]))\n    left_max = _to_float_scalar(xp.max(xp.abs(left)))\n    right_max = _to_float_scalar(xp.max(xp.abs(right)))\n    if not np.isfinite(left_max) or not np.isfinite(right_max):\n        return False\n    if left_max == 0.0 or right_max == 0.0:\n        return True\n    # Keep a factor-four range margin, matching the Gram working-scale policy.\n    per_term_limit = 0.25 * float(np.finfo(np.float64).max) / float(n_terms)\n    return bool(left_max <= per_term_limit / right_max)\n\n\ndef _influence_rows(X, resid, xp):\n''',
)

old = '''    else:\n        multiplier = max(correction1, correction2, correction12)\n        (grouped1_work, grouped2_work, grouped12_work), common_scale = (\n            _common_gram_working_values(\n                [grouped1, grouped2, grouped12],\n                xp,\n                max_multiplier=multiplier,\n            )\n        )\n        V1_work = _symmetrize(\n            grouped1_work.T @ grouped1_work * float(correction1)\n        )\n        V2_work = _symmetrize(\n            grouped2_work.T @ grouped2_work * float(correction2)\n        )\n        V12_work = _symmetrize(\n            grouped12_work.T @ grouped12_work * float(correction12)\n        )\n        cov_work = _stable_inclusion_exclusion(V1_work, V2_work, V12_work, xp)\n'''
new = '''    else:\n        multiplier = max(correction1, correction2, correction12)\n        (grouped1_work, grouped2_work, grouped12_work), common_scale = (\n            _common_gram_working_values(\n                [grouped1, grouped2, grouped12],\n                xp,\n                max_multiplier=multiplier,\n            )\n        )\n\n        # Without group debiasing, CGM inclusion-exclusion has an equivalent\n        # observation-score form:\n        #   sum_i psi_i (s_1(i) + s_2(i) - s_12(i))'.\n        # If the component Gram path needs a large common scale, forming that\n        # score-level union first removes the *structural* repeated intersection\n        # contribution before squaring.  Use the shortcut only when the direct\n        # cross reduction has a conservative finite range bound; otherwise keep\n        # the established scaled component path.\n        use_union_scores = (\n            not group_debias\n            and _to_float_scalar(xp.max(common_scale)) > 1.0\n        )\n        if use_union_scores:\n            c1_backend = xp_asarray(\n                c1, dtype=xp.int64, xp=xp, ref_arr=influence\n            )\n            c2_backend = xp_asarray(\n                c2, dtype=xp.int64, xp=xp, ref_arr=influence\n            )\n            c12_backend = xp_asarray(\n                c12, dtype=xp.int64, xp=xp, ref_arr=influence\n            )\n            union_scores = _stable_inclusion_exclusion(\n                grouped1[c1_backend],\n                grouped2[c2_backend],\n                grouped12[c12_backend],\n                xp,\n            )\n            use_union_scores = _cross_reduction_is_safe(\n                influence, union_scores, xp\n            )\n\n        if use_union_scores:\n            cov_work = _symmetrize(influence.T @ union_scores)\n            common_scale = xp.ones_like(projection_scale)\n        else:\n            V1_work = _symmetrize(\n                grouped1_work.T @ grouped1_work * float(correction1)\n            )\n            V2_work = _symmetrize(\n                grouped2_work.T @ grouped2_work * float(correction2)\n            )\n            V12_work = _symmetrize(\n                grouped12_work.T @ grouped12_work * float(correction12)\n            )\n            cov_work = _stable_inclusion_exclusion(V1_work, V2_work, V12_work, xp)\n'''
replace_once("statgpu/panel/_covariance.py", old, new)

p = Path("dev/tests/test_panel_stage_c_covariance.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_two_way_nonnested_structural_cancellation_preserves_small_covariance():
    n = 6
    amplitude = 2.0e307
    small = 1.0e-150
    X = np.ones((n, 1), dtype=np.float64)
    target_scores = np.asarray(
        [-amplitude, 0.0, amplitude, 0.0, small, -small], dtype=np.float64
    )
    resid = float(n) * target_scores
    cluster1 = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1, 2, 3], dtype=np.int64)

    actual = two_way_clustered_covariance(X, resid, cluster1, cluster2)
    expected = np.asarray([[2.0 * small * small]], dtype=np.float64)
    assert np.all(np.isfinite(actual))
    assert_allclose(actual, expected, rtol=2e-12, atol=0.0)
'''
if "test_two_way_nonnested_structural_cancellation_preserves_small_covariance" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

p = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = p.read_text(encoding="utf-8")
block = r'''


def test_stage_c_torch_cpu_two_way_nonnested_structural_cancellation():
    n = 6
    amplitude = 2.0e307
    small = 1.0e-150
    X = torch.ones((n, 1), dtype=torch.float64)
    target_scores = torch.tensor(
        [-amplitude, 0.0, amplitude, 0.0, small, -small], dtype=torch.float64
    )
    resid = float(n) * target_scores
    cluster1 = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1, 2, 3], dtype=np.int64)
    actual = two_way_clustered_covariance(
        X, resid, cluster1, cluster2, xp=torch
    ).detach().cpu().numpy()
    np.testing.assert_allclose(
        actual, np.asarray([[2.0 * small * small]]), rtol=2e-12, atol=0.0
    )
'''
if "test_stage_c_torch_cpu_two_way_nonnested_structural_cancellation" not in text:
    p.write_text(text.rstrip() + block.rstrip() + "\n", encoding="utf-8")

# Extend the maintained physical extreme-scale audit with the same nonnested
# structural-cancellation case for CuPy and Torch CUDA.
replace_once(
    "dev/benchmarks/validate_panel_stage_c_gpu.py",
    '''    mixed_nonmonotone_coarse = np.asarray([1, 1, 0], dtype=np.int64)\n\n    if backend == "numpy":\n''',
    '''    mixed_nonmonotone_coarse = np.asarray([1, 1, 0], dtype=np.int64)\n\n    nonnested_n = 6\n    nonnested_amplitude = 2.0e307\n    nonnested_small = 1.0e-150\n    X_nonnested_np = np.ones((nonnested_n, 1), dtype=np.float64)\n    nonnested_scores = np.asarray(\n        [-nonnested_amplitude, 0.0, nonnested_amplitude, 0.0,\n         nonnested_small, -nonnested_small],\n        dtype=np.float64,\n    )\n    resid_nonnested_np = float(nonnested_n) * nonnested_scores\n    nonnested_cluster1 = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)\n    nonnested_cluster2 = np.asarray([0, 1, 0, 1, 2, 3], dtype=np.int64)\n\n    if backend == "numpy":\n''',
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
    '''    np.testing.assert_allclose(mixed_dk, np.asarray([[1.5e-200]]), rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(\n        nonnested_two_way,\n        np.asarray([[2.0 * nonnested_small * nonnested_small]]),\n        rtol=2e-12,\n        atol=0.0,\n    )\n    return {\n''',
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
