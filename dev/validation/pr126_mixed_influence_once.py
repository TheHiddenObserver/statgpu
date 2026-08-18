from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:160]!r}")
    p.write_text(text.replace(old, new, 1))


cov = "statgpu/panel/_covariance.py"
old = '''def _influence_rows(X, resid, xp):
    """Return finite working influence rows and delayed positive restore scales.

    Both the tiny-design SVD factor and any response magnitude above one are
    kept outside the observation-level score.  Covariance definitions can then
    perform grouping and lag cancellation before restoring either physical
    scale.  Values at or below unit magnitude are never normalized, preserving
    ordinary and subnormal residual coordinates.
    """
    X_work, X_pinv_work, design_scale, rank = panel_svd_working_pseudoinverse(
        X, xp
    )
    resid_max = xp.max(xp.abs(resid))
    resid_scale = xp.maximum(resid_max, xp.ones_like(resid_max))
    resid_work = resid / resid_scale
    influence_work = X_pinv_work.T * resid_work[:, None]
    return (
        influence_work,
        resid_scale,
        design_scale,
        X_pinv_work,
        X_work,
        rank,
    )
'''
new = '''def _influence_rows(X, resid, xp):
    """Return finite working influence rows without normalizing the residual vector.

    The working-design pseudoinverse is scaled per parameter coordinate before
    multiplying by the original residuals.  Because each projection coordinate
    is then bounded by one, every finite residual produces a finite working
    score while unrelated small/subnormal residuals are preserved exactly as
    float64 inputs.  Projection and tiny-design scales are restored only after
    the covariance definition has performed its cancellation/reduction.
    """
    X_work, X_pinv_work, design_scale, rank = panel_svd_working_pseudoinverse(
        X, xp
    )
    projection_rows = X_pinv_work.T
    projection_work, projection_scale = _column_working_values(
        projection_rows, xp
    )
    influence_work = projection_work * resid[:, None]
    return (
        influence_work,
        projection_scale,
        design_scale,
        X_pinv_work,
        X_work,
        rank,
    )
'''
replace_once(cov, old, new)

old = '''def _restore_influence_covariance(
    covariance, coordinate_scale, residual_scale, design_scale, xp
):
    """Restore coordinate, response, then design score scales after cancellation."""
    covariance = _restore_coordinate_covariance(covariance, coordinate_scale, xp)
    covariance = _restore_scalar_covariance(covariance, residual_scale, xp)
    return _restore_scalar_covariance(covariance, design_scale, xp)
'''
new = '''def _restore_influence_covariance(
    covariance, coordinate_scale, projection_scale, design_scale, xp
):
    """Restore covariance-local, projection, then design scales after cancellation."""
    covariance = _restore_coordinate_covariance(covariance, coordinate_scale, xp)
    covariance = _restore_coordinate_covariance(covariance, projection_scale, xp)
    return _restore_scalar_covariance(covariance, design_scale, xp)
'''
replace_once(cov, old, new)

# Rename the delayed per-coordinate scale at every consumer.
p = Path(cov)
text = p.read_text().replace("residual_scale,\n        design_scale,", "projection_scale,\n        design_scale,")
text = text.replace("influence_scale, residual_scale, design_scale, xp", "influence_scale, projection_scale, design_scale, xp")
text = text.replace("adjusted_scale, residual_scale, design_scale, xp", "adjusted_scale, projection_scale, design_scale, xp")
p.write_text(text)

# One-way clustered covariance must keep finite mixed-dynamic-range scores intact
# until the group-specific reducer has cancelled them. Its component is PSD, so
# any post-group Gram overflow implies an unrepresentable final diagonal.
old = '''    influence_work, influence_scale = _column_working_values(influence, xp)
    n_clusters = int(len(labels))
    cov_work, correction = _cluster_component_from_scores(
        influence_work,
'''
new = '''    influence_work = influence
    influence_scale = xp.ones_like(projection_scale)
    n_clusters = int(len(labels))
    cov_work, correction = _cluster_component_from_scores(
        influence_work,
'''
replace_once(cov, old, new)

# NumPy regression: huge cluster cancels while a separate small, representable
# group must survive. This specifically rejects whole-residual normalization.
test_cov = Path("dev/tests/test_panel_stage_c_covariance.py")
text = test_cov.read_text()
append = r'''


def test_clustered_covariance_preserves_small_group_beside_huge_exact_cancellation():
    X = np.ones((3, 1), dtype=np.float64)
    resid = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)
    groups = np.asarray([0, 0, 1], dtype=np.int64)

    # X+ has identical 1/3 rows. Cluster 0 cancels exactly while cluster 1 has
    # score 1e-100, so the covariance is 1e-200. Scaling the entire residual
    # vector by 1.5e308 would underflow the third residual and incorrectly give 0.
    expected = np.asarray([[1.0e-200]], dtype=np.float64)
    actual = clustered_covariance(X, resid, groups)
    assert np.all(np.isfinite(actual))
    assert actual[0, 0] > 0.0
    np.testing.assert_allclose(actual, expected, rtol=3.0e-14, atol=0.0)
'''
if "test_clustered_covariance_preserves_small_group_beside_huge_exact_cancellation" not in text:
    test_cov.write_text(text + append)

# Torch CPU regression.
torch_test = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = torch_test.read_text()
append = r'''


def test_stage_c_torch_cpu_preserves_small_cluster_beside_huge_cancellation():
    X = torch.ones((3, 1), dtype=torch.float64)
    resid = torch.as_tensor([1.5e308, -1.5e308, 3.0e-100], dtype=torch.float64)
    groups = np.asarray([0, 0, 1], dtype=np.int64)
    actual = clustered_covariance(X, resid, groups, xp=torch)
    expected = np.asarray([[1.0e-200]], dtype=np.float64)
    assert float(actual[0, 0]) > 0.0
    assert_allclose(actual, expected, rtol=4.0e-13, atol=0.0)
'''
if "test_stage_c_torch_cpu_preserves_small_cluster_beside_huge_cancellation" not in text:
    torch_test.write_text(text + append)

# Physical validator.
bench = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = bench.read_text()
old = '''    tiny_cluster_groups = np.asarray([0, 0, 1, 1], dtype=np.int64)\n\n    if backend == "numpy":\n'''
new = '''    tiny_cluster_groups = np.asarray([0, 0, 1, 1], dtype=np.int64)\n\n    X_mixed_cluster_np = np.ones((3, 1), dtype=np.float64)\n    resid_mixed_cluster_np = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)\n    mixed_cluster_groups = np.asarray([0, 0, 1], dtype=np.int64)\n\n    if backend == "numpy":\n'''
if old not in text:
    raise RuntimeError("physical mixed-cluster insertion anchor missing")
text = text.replace(old, new, 1)
old = '''        X_tiny_cluster, resid_tiny_cluster = X_tiny_cluster_np, resid_tiny_cluster_np\n    elif backend == "cupy":\n'''
new = '''        X_tiny_cluster, resid_tiny_cluster = X_tiny_cluster_np, resid_tiny_cluster_np\n        X_mixed_cluster, resid_mixed_cluster = X_mixed_cluster_np, resid_mixed_cluster_np\n    elif backend == "cupy":\n'''
text = text.replace(old, new, 1)
old = '''        X_tiny_cluster = cp.asarray(X_tiny_cluster_np)\n        resid_tiny_cluster = cp.asarray(resid_tiny_cluster_np)\n    elif backend == "torch":\n'''
new = '''        X_tiny_cluster = cp.asarray(X_tiny_cluster_np)\n        resid_tiny_cluster = cp.asarray(resid_tiny_cluster_np)\n        X_mixed_cluster = cp.asarray(X_mixed_cluster_np)\n        resid_mixed_cluster = cp.asarray(resid_mixed_cluster_np)\n    elif backend == "torch":\n'''
text = text.replace(old, new, 1)
old = '''        X_tiny_cluster = torch.as_tensor(X_tiny_cluster_np, dtype=torch.float64, device="cuda")\n        resid_tiny_cluster = torch.as_tensor(resid_tiny_cluster_np, dtype=torch.float64, device="cuda")\n    else:\n'''
new = '''        X_tiny_cluster = torch.as_tensor(X_tiny_cluster_np, dtype=torch.float64, device="cuda")\n        resid_tiny_cluster = torch.as_tensor(resid_tiny_cluster_np, dtype=torch.float64, device="cuda")\n        X_mixed_cluster = torch.as_tensor(X_mixed_cluster_np, dtype=torch.float64, device="cuda")\n        resid_mixed_cluster = torch.as_tensor(resid_mixed_cluster_np, dtype=torch.float64, device="cuda")\n    else:\n'''
text = text.replace(old, new, 1)
old = '''    tiny_design_cluster = _array(\n        clustered_covariance(\n            X_tiny_cluster, resid_tiny_cluster, tiny_cluster_groups, xp=xp\n        )\n    )\n    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk), ("pregram_hac", pregram_hac), ("pregram_dk", pregram_dk), ("two_way_component_cancellation", component_two_way), ("tiny_design_cluster_cancellation", tiny_design_cluster)):\n'''
new = '''    tiny_design_cluster = _array(\n        clustered_covariance(\n            X_tiny_cluster, resid_tiny_cluster, tiny_cluster_groups, xp=xp\n        )\n    )\n    mixed_cluster = _array(\n        clustered_covariance(\n            X_mixed_cluster, resid_mixed_cluster, mixed_cluster_groups, xp=xp\n        )\n    )\n    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk), ("pregram_hac", pregram_hac), ("pregram_dk", pregram_dk), ("two_way_component_cancellation", component_two_way), ("tiny_design_cluster_cancellation", tiny_design_cluster), ("mixed_dynamic_cluster", mixed_cluster)):\n'''
if old not in text:
    raise RuntimeError("physical mixed-cluster computation anchor missing")
text = text.replace(old, new, 1)
old = '''    np.testing.assert_allclose(\n        tiny_design_cluster, np.zeros((1, 1)), rtol=0.0, atol=0.0\n    )\n    return {\n'''
new = '''    np.testing.assert_allclose(\n        tiny_design_cluster, np.zeros((1, 1)), rtol=0.0, atol=0.0\n    )\n    np.testing.assert_allclose(\n        mixed_cluster, np.asarray([[1.0e-200]], dtype=np.float64), rtol=8e-13, atol=0.0\n    )\n    return {\n'''
text = text.replace(old, new, 1)
old = '''        "tiny_design_cluster_cancellation": tiny_design_cluster.tolist(),\n    }\n'''
new = '''        "tiny_design_cluster_cancellation": tiny_design_cluster.tolist(),\n        "mixed_dynamic_cluster": mixed_cluster.tolist(),\n    }\n'''
text = text.replace(old, new, 1)
bench.write_text(text)
