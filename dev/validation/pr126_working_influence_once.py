from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:140]!r}")
    p.write_text(text.replace(old, new, 1))


cov = "statgpu/panel/_covariance.py"
old = '''def _influence_rows(X, resid, xp):
    """Return stable observation influence rows plus working projection factors."""
    X_work, X_pinv_work, design_scale, rank = panel_svd_working_pseudoinverse(
        X, xp
    )
    # X+ = design_scale * X_work+.  Multiply residuals before restoring the
    # design scale so a tiny full-rank design does not overflow X+ even when the
    # final influence and covariance remain representable.
    influence = (X_pinv_work.T * resid[:, None]) * design_scale
    return influence, X_pinv_work, X_work, rank
'''
new = '''def _influence_rows(X, resid, xp):
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
replace_once(cov, old, new)

anchor = '''def _restore_coordinate_covariance(covariance, scale, xp):
    """Restore a per-coordinate covariance scale without forming scale_i*scale_j."""
    row = scale[:, None]
    col = scale[None, :]
    large = xp.maximum(row, col)
    small = xp.minimum(row, col)
    return _symmetrize((covariance * large) * small)


'''
new_anchor = anchor + '''def _restore_scalar_covariance(covariance, scale, xp):
    """Restore one positive scalar score scale without materializing scale**2."""
    return _symmetrize((covariance * scale) * scale)


def _restore_influence_covariance(
    covariance, coordinate_scale, residual_scale, design_scale, xp
):
    """Restore coordinate, response, then design score scales after cancellation."""
    covariance = _restore_coordinate_covariance(covariance, coordinate_scale, xp)
    covariance = _restore_scalar_covariance(covariance, residual_scale, xp)
    return _restore_scalar_covariance(covariance, design_scale, xp)


'''
replace_once(cov, anchor, new_anchor)

# Cluster one-way.
old = '''    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
'''
new = '''    (
        influence,
        residual_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
'''
replace_once(cov, old, new)
old = '''    cov = _restore_coordinate_covariance(cov_work, influence_scale, xp)
    if metadata is not None:
'''
new = '''    cov = _restore_influence_covariance(
        cov_work, influence_scale, residual_scale, design_scale, xp
    )
    if metadata is not None:
'''
replace_once(cov, old, new)

# Two-way cluster.
old = '''    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
'''
new = '''    (
        influence,
        residual_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
'''
replace_once(cov, old, new)
old = '''    cov = _restore_coordinate_covariance(cov_work, influence_scale, xp)
    if metadata is not None:
'''
new = '''    cov = _restore_influence_covariance(
        cov_work, influence_scale, residual_scale, design_scale, xp
    )
    if metadata is not None:
'''
replace_once(cov, old, new)

# HAC.
old = '''    influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
'''
new = '''    (
        influence,
        residual_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
'''
replace_once(cov, old, new)
old = '''    return _restore_coordinate_covariance(cov, influence_scale, xp)
'''
new = '''    return _restore_influence_covariance(
        cov, influence_scale, residual_scale, design_scale, xp
    )
'''
replace_once(cov, old, new)

# DK.
old = '''    influence, _X_pinv, _bread, rank = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
'''
new = '''    (
        influence,
        residual_scale,
        design_scale,
        _X_pinv,
        _X_work,
        rank,
    ) = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
'''
replace_once(cov, old, new)
old = '''    cov = _restore_coordinate_covariance(cov, grouped_scale, xp)
    cov = _restore_coordinate_covariance(cov, influence_scale, xp)
    scale = float(n) / float(denom)
'''
new = '''    cov = _restore_coordinate_covariance(cov, grouped_scale, xp)
    cov = _restore_influence_covariance(
        cov, influence_scale, residual_scale, design_scale, xp
    )
    scale = float(n) / float(denom)
'''
replace_once(cov, old, new)

# HC paths.
old = '''    influence, X_pinv_work, X_work, rank = _influence_rows(X, resid, xp)
    if kind == "hc0":
        if metadata is not None:
            metadata.update(
                {
                    "covariance": "hc0",
                    "design_rank": int(rank),
                    "design_columns": int(X.shape[1]),
                }
            )
        return _symmetrize(influence.T @ influence)
'''
new = '''    (
        influence,
        residual_scale,
        design_scale,
        X_pinv_work,
        X_work,
        rank,
    ) = _influence_rows(X, resid, xp)
    if kind == "hc0":
        influence_work, influence_scale = _column_working_values(influence, xp)
        cov_work = _symmetrize(influence_work.T @ influence_work)
        if metadata is not None:
            metadata.update(
                {
                    "covariance": "hc0",
                    "design_rank": int(rank),
                    "design_columns": int(X.shape[1]),
                }
            )
        return _restore_influence_covariance(
            cov_work, influence_scale, residual_scale, design_scale, xp
        )
'''
replace_once(cov, old, new)
old = '''    if metadata is not None:
        metadata.update(
            {
                "covariance": kind,
                "design_rank": int(rank),
                "design_columns": int(X.shape[1]),
                "leverage_min": float(leverage_min),
                "leverage_max": float(leverage_max),
            }
        )
    return _symmetrize(adjusted_influence.T @ adjusted_influence)
'''
new = '''    adjusted_work, adjusted_scale = _column_working_values(
        adjusted_influence, xp
    )
    cov_work = _symmetrize(adjusted_work.T @ adjusted_work)
    if metadata is not None:
        metadata.update(
            {
                "covariance": kind,
                "design_rank": int(rank),
                "design_columns": int(X.shape[1]),
                "leverage_min": float(leverage_min),
                "leverage_max": float(leverage_max),
            }
        )
    return _restore_influence_covariance(
        cov_work, adjusted_scale, residual_scale, design_scale, xp
    )
'''
replace_once(cov, old, new)

# HC1/robust dispatcher.
old = '''    if name == "robust":
        influence, _X_pinv, _bread, _rank = _influence_rows(X, resid, xp)
        correction = hc1_correction
'''
new = '''    if name == "robust":
        (
            influence,
            residual_scale,
            design_scale,
            _X_pinv,
            _X_work,
            _rank,
        ) = _influence_rows(X, resid, xp)
        influence_work, influence_scale = _column_working_values(influence, xp)
        correction = hc1_correction
'''
replace_once(cov, old, new)
old = '''        return _symmetrize(
            influence.T @ influence * float(correction)
        )
'''
new = '''        cov_work = _symmetrize(
            influence_work.T @ influence_work * float(correction)
        )
        return _restore_influence_covariance(
            cov_work, influence_scale, residual_scale, design_scale, xp
        )
'''
replace_once(cov, old, new)

# NumPy public regression for a condition-number-one tiny design whose raw
# observation influence is unrepresentable but group scores cancel exactly.
test_cov = Path("dev/tests/test_panel_stage_c_covariance.py")
text = test_cov.read_text()
append = r'''


def test_clustered_covariance_delays_tiny_design_scale_until_after_group_cancellation():
    tiny = 1.0e-320
    X = np.ones((4, 1), dtype=np.float64) * tiny
    resid = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)
    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)

    # The single-column design has condition number one.  Original-scale
    # observation influences exceed DBL_MAX, but each cluster score is exactly
    # zero, so the mathematically defined cluster covariance is zero.
    actual = clustered_covariance(X, resid, groups)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, np.zeros((1, 1)), rtol=0.0, atol=0.0)
'''
if "test_clustered_covariance_delays_tiny_design_scale_until_after_group_cancellation" not in text:
    test_cov.write_text(text + append)

# Maintained Torch CPU version.
torch_test = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = torch_test.read_text()
append = r'''


def test_stage_c_torch_cpu_delays_tiny_design_scale_until_group_cancellation():
    tiny = 1.0e-320
    X = torch.ones((4, 1), dtype=torch.float64) * tiny
    resid = torch.as_tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64)
    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)
    actual = clustered_covariance(X, resid, groups, xp=torch)
    assert_allclose(actual, np.zeros((1, 1)), rtol=0.0, atol=0.0)
'''
if "test_stage_c_torch_cpu_delays_tiny_design_scale_until_group_cancellation" not in text:
    torch_test.write_text(text + append)

# Physical validator.
bench = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = bench.read_text()
old = '''    component_pairs = np.asarray([0, 0, 1, 1], dtype=np.int64)\n\n    if backend == "numpy":\n'''
new = '''    component_pairs = np.asarray([0, 0, 1, 1], dtype=np.int64)\n\n    tiny_design_value = 1.0e-320\n    X_tiny_cluster_np = np.ones((4, 1), dtype=np.float64) * tiny_design_value\n    resid_tiny_cluster_np = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.float64)\n    tiny_cluster_groups = np.asarray([0, 0, 1, 1], dtype=np.int64)\n\n    if backend == "numpy":\n'''
if old not in text:
    raise RuntimeError("physical tiny-design insertion anchor missing")
text = text.replace(old, new, 1)
old = '''        X_component, resid_component = X_component_np, resid_component_np\n    elif backend == "cupy":\n'''
new = '''        X_component, resid_component = X_component_np, resid_component_np\n        X_tiny_cluster, resid_tiny_cluster = X_tiny_cluster_np, resid_tiny_cluster_np\n    elif backend == "cupy":\n'''
text = text.replace(old, new, 1)
old = '''        X_component, resid_component = cp.asarray(X_component_np), cp.asarray(resid_component_np)\n    elif backend == "torch":\n'''
new = '''        X_component, resid_component = cp.asarray(X_component_np), cp.asarray(resid_component_np)\n        X_tiny_cluster = cp.asarray(X_tiny_cluster_np)\n        resid_tiny_cluster = cp.asarray(resid_tiny_cluster_np)\n    elif backend == "torch":\n'''
text = text.replace(old, new, 1)
old = '''        X_component = torch.as_tensor(X_component_np, dtype=torch.float64, device="cuda")\n        resid_component = torch.as_tensor(resid_component_np, dtype=torch.float64, device="cuda")\n    else:\n'''
new = '''        X_component = torch.as_tensor(X_component_np, dtype=torch.float64, device="cuda")\n        resid_component = torch.as_tensor(resid_component_np, dtype=torch.float64, device="cuda")\n        X_tiny_cluster = torch.as_tensor(X_tiny_cluster_np, dtype=torch.float64, device="cuda")\n        resid_tiny_cluster = torch.as_tensor(resid_tiny_cluster_np, dtype=torch.float64, device="cuda")\n    else:\n'''
text = text.replace(old, new, 1)
old = '''    component_two_way = _array(\n        two_way_clustered_covariance(\n            X_component,\n            resid_component,\n            component_unique,\n            component_pairs,\n            xp=xp,\n        )\n    )\n    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk), ("pregram_hac", pregram_hac), ("pregram_dk", pregram_dk), ("two_way_component_cancellation", component_two_way)):\n'''
new = '''    component_two_way = _array(\n        two_way_clustered_covariance(\n            X_component,\n            resid_component,\n            component_unique,\n            component_pairs,\n            xp=xp,\n        )\n    )\n    tiny_design_cluster = _array(\n        clustered_covariance(\n            X_tiny_cluster, resid_tiny_cluster, tiny_cluster_groups, xp=xp\n        )\n    )\n    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk), ("pregram_hac", pregram_hac), ("pregram_dk", pregram_dk), ("two_way_component_cancellation", component_two_way), ("tiny_design_cluster_cancellation", tiny_design_cluster)):\n'''
if old not in text:
    raise RuntimeError("physical tiny-design computation anchor missing")
text = text.replace(old, new, 1)
old = '''    np.testing.assert_allclose(\n        component_two_way, component_reference, rtol=8e-13, atol=0.0\n    )\n    return {\n'''
new = '''    np.testing.assert_allclose(\n        component_two_way, component_reference, rtol=8e-13, atol=0.0\n    )\n    np.testing.assert_allclose(\n        tiny_design_cluster, np.zeros((1, 1)), rtol=0.0, atol=0.0\n    )\n    return {\n'''
text = text.replace(old, new, 1)
old = '''        "two_way_component_cancellation": component_two_way.tolist(),\n    }\n'''
new = '''        "two_way_component_cancellation": component_two_way.tolist(),\n        "tiny_design_cluster_cancellation": tiny_design_cluster.tolist(),\n    }\n'''
text = text.replace(old, new, 1)
bench.write_text(text)
