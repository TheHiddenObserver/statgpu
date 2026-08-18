from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:160]!r}")
    p.write_text(text.replace(old, new, 1))


cov = "statgpu/panel/_covariance.py"
old = '''def _paired_codes(left, right):
    pairs = np.column_stack(
        [np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)]
    )
    _, codes = np.unique(pairs, axis=0, return_inverse=True)
    return codes.astype(np.int64, copy=False)
'''
new = '''def _paired_codes(left, right):
    pairs = np.column_stack(
        [np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)]
    )
    _, codes = np.unique(pairs, axis=0, return_inverse=True)
    return codes.astype(np.int64, copy=False)


def _same_partition(left, right) -> bool:
    """Return whether two integer code vectors induce the same partition.

    Code values themselves are arbitrary labels.  Two partitions are equal iff
    they have the same number of groups and each observed left/right pair gives
    a one-to-one mapping between those groups.
    """
    left = np.asarray(left, dtype=np.int64).ravel()
    right = np.asarray(right, dtype=np.int64).ravel()
    if left.shape != right.shape:
        return False
    n_left = int(np.unique(left).size)
    n_right = int(np.unique(right).size)
    if n_left != n_right:
        return False
    pairs = np.column_stack([left, right])
    return int(np.unique(pairs, axis=0).shape[0]) == n_left
'''
replace_once(cov, old, new)

old = '''    if np.array_equal(c12, c1):
        cov_work, common_scale = _cluster_meat_from_grouped(
            grouped2, correction2, xp
        )
    elif np.array_equal(c12, c2):
        cov_work, common_scale = _cluster_meat_from_grouped(
            grouped1, correction1, xp
        )
'''
new = '''    if _same_partition(c12, c1):
        cov_work, common_scale = _cluster_meat_from_grouped(
            grouped2, correction2, xp
        )
    elif _same_partition(c12, c2):
        cov_work, common_scale = _cluster_meat_from_grouped(
            grouped1, correction1, xp
        )
'''
replace_once(cov, old, new)

# NumPy regression: fine/intersection codes are a permutation, not equal arrays.
test_cov = Path("dev/tests/test_panel_stage_c_covariance.py")
text = test_cov.read_text()
append = r'''


def test_two_way_nested_partition_detection_is_invariant_to_code_permutation():
    X = np.ones((3, 1), dtype=np.float64)
    resid = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)
    coarse = np.asarray([1, 1, 0], dtype=np.int64)
    fine = np.asarray([0, 1, 2], dtype=np.int64)

    # The fine partition equals the intersection partition, but paired-code
    # factorization orders (coarse, fine) lexicographically and therefore
    # produces a permutation of the fine integer codes. Statistical cancellation
    # is partition-based and must not depend on that arbitrary code numbering.
    reference = clustered_covariance(X, resid, coarse)
    actual = two_way_clustered_covariance(X, resid, coarse, fine)
    assert reference[0, 0] > 0.0
    np.testing.assert_allclose(reference, np.asarray([[1.0e-200]]), rtol=3e-14, atol=0.0)
    np.testing.assert_allclose(actual, reference, rtol=3e-14, atol=0.0)
'''
if "test_two_way_nested_partition_detection_is_invariant_to_code_permutation" not in text:
    test_cov.write_text(text + append)

# Torch CPU counterpart.
torch_test = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = torch_test.read_text()
append = r'''


def test_stage_c_torch_cpu_nested_partition_code_permutation_is_exact():
    X = torch.ones((3, 1), dtype=torch.float64)
    resid = torch.as_tensor([1.5e308, -1.5e308, 3.0e-100], dtype=torch.float64)
    coarse = np.asarray([1, 1, 0], dtype=np.int64)
    fine = np.asarray([0, 1, 2], dtype=np.int64)
    reference = clustered_covariance(X, resid, coarse, xp=torch)
    actual = two_way_clustered_covariance(X, resid, coarse, fine, xp=torch)
    assert_allclose(actual, reference, rtol=5e-13, atol=0.0)
'''
if "test_stage_c_torch_cpu_nested_partition_code_permutation_is_exact" not in text:
    torch_test.write_text(text + append)

# Physical CUDA audit includes the non-monotone nested code permutation.
bench = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = bench.read_text()
old = '''    mixed_time = np.asarray([0, 0, 1], dtype=np.int64)\n\n    if backend == "numpy":\n'''
new = '''    mixed_time = np.asarray([0, 0, 1], dtype=np.int64)\n    mixed_nonmonotone_coarse = np.asarray([1, 1, 0], dtype=np.int64)\n\n    if backend == "numpy":\n'''
if old not in text:
    raise RuntimeError("physical partition insertion anchor missing")
text = text.replace(old, new, 1)
old = '''    mixed_two_way = _array(two_way_clustered_covariance(\n        X_mixed, resid_mixed, mixed_unique, mixed_coarse, xp=xp\n    ))\n    mixed_dk = _array(driscoll_kraay_covariance(\n'''
new = '''    mixed_two_way = _array(two_way_clustered_covariance(\n        X_mixed, resid_mixed, mixed_unique, mixed_coarse, xp=xp\n    ))\n    mixed_two_way_permuted = _array(two_way_clustered_covariance(\n        X_mixed, resid_mixed, mixed_nonmonotone_coarse, mixed_unique, xp=xp\n    ))\n    mixed_dk = _array(driscoll_kraay_covariance(\n'''
if old not in text:
    raise RuntimeError("physical partition computation anchor missing")
text = text.replace(old, new, 1)
old = '''    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk), ("pregram_hac", pregram_hac), ("pregram_dk", pregram_dk), ("two_way_component_cancellation", component_two_way), ("tiny_design_cluster_cancellation", tiny_design_cluster), ("mixed_cluster", mixed_cluster), ("mixed_two_way", mixed_two_way), ("mixed_dk", mixed_dk)):\n'''
new = '''    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk), ("pregram_hac", pregram_hac), ("pregram_dk", pregram_dk), ("two_way_component_cancellation", component_two_way), ("tiny_design_cluster_cancellation", tiny_design_cluster), ("mixed_cluster", mixed_cluster), ("mixed_two_way", mixed_two_way), ("mixed_two_way_permuted", mixed_two_way_permuted), ("mixed_dk", mixed_dk)):\n'''
if old not in text:
    raise RuntimeError("physical partition finite-loop anchor missing")
text = text.replace(old, new, 1)
old = '''    np.testing.assert_allclose(mixed_two_way, mixed_cluster, rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(mixed_dk, np.asarray([[1.5e-200]]), rtol=8e-13, atol=0.0)\n'''
new = '''    np.testing.assert_allclose(mixed_two_way, mixed_cluster, rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(mixed_two_way_permuted, mixed_cluster, rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(mixed_dk, np.asarray([[1.5e-200]]), rtol=8e-13, atol=0.0)\n'''
text = text.replace(old, new, 1)
old = '''        "mixed_two_way": mixed_two_way.tolist(),\n        "mixed_driscoll_kraay": mixed_dk.tolist(),\n'''
new = '''        "mixed_two_way": mixed_two_way.tolist(),\n        "mixed_two_way_permuted": mixed_two_way_permuted.tolist(),\n        "mixed_driscoll_kraay": mixed_dk.tolist(),\n'''
text = text.replace(old, new, 1)
bench.write_text(text)
