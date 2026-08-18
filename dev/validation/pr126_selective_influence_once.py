from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:180]!r}")
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


def _column_working_values(values, xp):
    """Return per-coordinate unit values with restore scales bounded below by one.

    Only coordinates whose absolute magnitude exceeds one are normalized.  This
    keeps ordinary/tiny coordinates on their original scale while ensuring that
    Gram and lag products are formed from values with absolute magnitude at most
    one.  The restore scales are therefore all >= 1, so restoring a finite final
    covariance cannot overflow before that final value is reached.
    """
    if _is_torch(xp):
        max_abs = xp.max(xp.abs(values), dim=0).values
    else:
        max_abs = xp.max(xp.abs(values), axis=0)
    scale = xp.maximum(max_abs, xp.ones_like(max_abs))
    return values / scale, scale
'''
new = '''def _column_abs_max(values, xp):
    if _is_torch(xp):
        return xp.max(xp.abs(values), dim=0).values
    return xp.max(xp.abs(values), axis=0)


def _projection_product_working_values(projection_rows, resid, xp):
    """Scale projection coordinates only when projection*residual could overflow.

    The scale is the smallest per-coordinate factor needed to leave a factor-two
    range margin.  If the raw multiplication is already safe, the projection is
    left bit-for-bit unchanged; in particular, ordinary/subnormal-design paths
    do not pay an avoidable normalization round trip.
    """
    projection_max = _column_abs_max(projection_rows, xp)
    resid_max = xp.max(xp.abs(resid))
    resid_bound = xp.maximum(resid_max, xp.ones_like(resid_max))
    safe_projection = (0.5 * float(np.finfo(np.float64).max)) / resid_bound
    one = xp.ones_like(projection_max)
    scale = xp.where(
        projection_max > safe_projection,
        projection_max / safe_projection,
        one,
    )
    return projection_rows / scale, scale


def _gram_working_values(values, xp, *, max_multiplier: float = 1.0):
    """Apply only the per-coordinate scaling required for a safe Gram reduction."""
    n_terms = max(1, int(values.shape[0]))
    multiplier = max(1.0, abs(float(max_multiplier)))
    threshold = float(
        np.sqrt(0.25 * np.finfo(np.float64).max / (float(n_terms) * multiplier))
    )
    max_abs = _column_abs_max(values, xp)
    one = xp.ones_like(max_abs)
    scale = xp.where(max_abs > threshold, max_abs / threshold, one)
    return values / scale, scale


def _common_gram_working_values(values_list, xp, *, max_multiplier: float = 1.0):
    """Put several grouped-score matrices on one minimally scaled Gram space."""
    max_abs = _column_abs_max(values_list[0], xp)
    n_terms = int(values_list[0].shape[0])
    for values in values_list[1:]:
        max_abs = xp.maximum(max_abs, _column_abs_max(values, xp))
        n_terms = max(n_terms, int(values.shape[0]))
    multiplier = max(1.0, abs(float(max_multiplier)))
    threshold = float(
        np.sqrt(0.25 * np.finfo(np.float64).max / (float(max(1, n_terms)) * multiplier))
    )
    one = xp.ones_like(max_abs)
    scale = xp.where(max_abs > threshold, max_abs / threshold, one)
    return [values / scale for values in values_list], scale


def _influence_rows(X, resid, xp):
    """Return finite influence rows with only unavoidable product scaling delayed.

    The SVD working-design pseudoinverse is rescaled only if multiplying a
    coordinate by the largest finite residual could overflow.  The original
    residual vector is never globally normalized, so unrelated tiny/subnormal
    residual contributions survive beside very large observations.
    """
    X_work, X_pinv_work, design_scale, rank = panel_svd_working_pseudoinverse(
        X, xp
    )
    projection_rows = X_pinv_work.T
    projection_work, projection_scale = _projection_product_working_values(
        projection_rows, resid, xp
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


def _cluster_component_from_scores(
    scores, codes, *, n_groups: int, nobs: int, group_debias: bool, xp
):
    """Return one cluster meat on an already-safe common score scale."""
    if int(n_groups) < 2:
        raise ValueError(
            "clustered covariance requires at least two distinct clusters"
        )
    grouped = _grouped_score_sums(scores, codes, n_groups=int(n_groups), xp=xp)
    correction = (
        _group_debias_factor(int(n_groups), int(nobs)) if group_debias else 1.0
    )
    meat = _symmetrize(grouped.T @ grouped * float(correction))
    return meat, float(correction)
'''
new = '''def _restore_influence_covariance(
    covariance, covariance_scale, projection_scale, design_scale, xp
):
    """Restore Gram, projection, then tiny-design scales after cancellation."""
    covariance = _restore_coordinate_covariance(covariance, covariance_scale, xp)
    covariance = _restore_coordinate_covariance(covariance, projection_scale, xp)
    return _restore_scalar_covariance(covariance, design_scale, xp)


def _cluster_grouped_scores(
    scores, codes, *, n_groups: int, nobs: int, group_debias: bool, xp
):
    if int(n_groups) < 2:
        raise ValueError(
            "clustered covariance requires at least two distinct clusters"
        )
    grouped = _grouped_score_sums(scores, codes, n_groups=int(n_groups), xp=xp)
    correction = (
        _group_debias_factor(int(n_groups), int(nobs)) if group_debias else 1.0
    )
    return grouped, float(correction)


def _cluster_meat_from_grouped(grouped, correction: float, xp):
    grouped_work, grouped_scale = _gram_working_values(
        grouped, xp, max_multiplier=correction
    )
    meat = _symmetrize(grouped_work.T @ grouped_work * float(correction))
    return meat, grouped_scale
'''
replace_once(cov, old, new)

# One-way cluster: group before any Gram normalization.
old = '''    (
        influence,
        residual_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
    n_clusters = int(len(labels))
    cov_work, correction = _cluster_component_from_scores(
        influence_work,
        cluster_idx,
        n_groups=n_clusters,
        nobs=int(n),
        group_debias=group_debias,
        xp=xp,
    )
    cov = _restore_influence_covariance(
        cov_work, influence_scale, residual_scale, design_scale, xp
    )
'''
new = '''    (
        influence,
        projection_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    n_clusters = int(len(labels))
    grouped, correction = _cluster_grouped_scores(
        influence,
        cluster_idx,
        n_groups=n_clusters,
        nobs=int(n),
        group_debias=group_debias,
        xp=xp,
    )
    cov_work, grouped_scale = _cluster_meat_from_grouped(
        grouped, correction, xp
    )
    cov = _restore_influence_covariance(
        cov_work, grouped_scale, projection_scale, design_scale, xp
    )
'''
replace_once(cov, old, new)

# Two-way cluster: group raw finite scores first, algebraically short-circuit
# nested partitions, otherwise use one common minimally scaled Gram space.
start = '''    # All three Cameron-Gelbach-Miller components must be combined on one
    # common finite score scale. Restoring each component first can produce
    # Inf - Inf even when the inclusion-exclusion result itself is finite.
    (
        influence,
        residual_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
    V1_work, correction1 = _cluster_component_from_scores(
        influence_work,
        c1,
        n_groups=int(len(labels1)),
        nobs=n,
        group_debias=group_debias,
        xp=xp,
    )
    V2_work, correction2 = _cluster_component_from_scores(
        influence_work,
        c2,
        n_groups=int(len(labels2)),
        nobs=n,
        group_debias=group_debias,
        xp=xp,
    )
    V12_work, correction12 = _cluster_component_from_scores(
        influence_work,
        c12,
        n_groups=n12,
        nobs=n,
        group_debias=group_debias,
        xp=xp,
    )
    cov_work = _stable_inclusion_exclusion(V1_work, V2_work, V12_work, xp)
    cov = _restore_influence_covariance(
        cov_work, influence_scale, residual_scale, design_scale, xp
    )
'''
newstart = '''    # All Cameron-Gelbach-Miller grouping is performed before covariance-scale
    # normalization. Nested cluster dimensions are simplified algebraically, so
    # an exactly cancelling marginal/intersection pair never has to be rounded.
    (
        influence,
        projection_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    grouped1, correction1 = _cluster_grouped_scores(
        influence, c1, n_groups=int(len(labels1)), nobs=n,
        group_debias=group_debias, xp=xp
    )
    grouped2, correction2 = _cluster_grouped_scores(
        influence, c2, n_groups=int(len(labels2)), nobs=n,
        group_debias=group_debias, xp=xp
    )
    grouped12, correction12 = _cluster_grouped_scores(
        influence, c12, n_groups=n12, nobs=n,
        group_debias=group_debias, xp=xp
    )
    if np.array_equal(c12, c1):
        cov_work, common_scale = _cluster_meat_from_grouped(
            grouped2, correction2, xp
        )
    elif np.array_equal(c12, c2):
        cov_work, common_scale = _cluster_meat_from_grouped(
            grouped1, correction1, xp
        )
    else:
        multiplier = max(correction1, correction2, correction12)
        (grouped1_work, grouped2_work, grouped12_work), common_scale = (
            _common_gram_working_values(
                [grouped1, grouped2, grouped12],
                xp,
                max_multiplier=multiplier,
            )
        )
        V1_work = _symmetrize(
            grouped1_work.T @ grouped1_work * float(correction1)
        )
        V2_work = _symmetrize(
            grouped2_work.T @ grouped2_work * float(correction2)
        )
        V12_work = _symmetrize(
            grouped12_work.T @ grouped12_work * float(correction12)
        )
        cov_work = _stable_inclusion_exclusion(V1_work, V2_work, V12_work, xp)
    cov = _restore_influence_covariance(
        cov_work, common_scale, projection_scale, design_scale, xp
    )
'''
replace_once(cov, start, newstart)

# HAC: minimally Gram-scale only after the finite influence product exists.
old = '''    (
        influence,
        residual_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
    max_terms = int(bandwidth) + 1
'''
new = '''    (
        influence,
        projection_scale,
        design_scale,
        _X_pinv,
        _X_work,
        _rank,
    ) = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _gram_working_values(
        influence, xp, max_multiplier=2.0
    )
    max_terms = int(bandwidth) + 1
'''
replace_once(cov, old, new)
old = '''    return _restore_influence_covariance(
        cov, influence_scale, residual_scale, design_scale, xp
    )
'''
new = '''    return _restore_influence_covariance(
        cov, influence_scale, projection_scale, design_scale, xp
    )
'''
replace_once(cov, old, new)

# DK: group finite raw scores first, then minimally normalize grouped Gram space.
old = '''    (
        influence,
        residual_scale,
        design_scale,
        _X_pinv,
        _X_work,
        rank,
    ) = _influence_rows(X, resid, xp)
    influence_work, influence_scale = _column_working_values(influence, xp)
    k_columns = int(X.shape[1])
'''
new = '''    (
        influence,
        projection_scale,
        design_scale,
        _X_pinv,
        _X_work,
        rank,
    ) = _influence_rows(X, resid, xp)
    k_columns = int(X.shape[1])
'''
replace_once(cov, old, new)
old = '''    grouped = _grouped_score_sums(
        influence_work, time_codes, n_groups=n_periods, xp=xp
    )
    grouped_work, grouped_scale = _column_working_values(grouped, xp)
'''
new = '''    grouped = _grouped_score_sums(
        influence, time_codes, n_groups=n_periods, xp=xp
    )
    grouped_work, grouped_scale = _gram_working_values(
        grouped, xp, max_multiplier=2.0
    )
'''
replace_once(cov, old, new)
old = '''    cov = _restore_coordinate_covariance(cov, grouped_scale, xp)
    cov = _restore_influence_covariance(
        cov, influence_scale, residual_scale, design_scale, xp
    )
'''
new = '''    cov = _restore_influence_covariance(
        cov, grouped_scale, projection_scale, design_scale, xp
    )
'''
replace_once(cov, old, new)

# HC paths: only minimally Gram-normalize; no scale if the existing finite Gram
# is already safe, preserving historical subnormal-design precision.
old = '''    (
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
'''
new = '''    (
        influence,
        projection_scale,
        design_scale,
        X_pinv_work,
        X_work,
        rank,
    ) = _influence_rows(X, resid, xp)
    if kind == "hc0":
        influence_work, influence_scale = _gram_working_values(influence, xp)
        cov_work = _symmetrize(influence_work.T @ influence_work)
'''
replace_once(cov, old, new)
old = '''        return _restore_influence_covariance(
            cov_work, influence_scale, residual_scale, design_scale, xp
        )
'''
new = '''        return _restore_influence_covariance(
            cov_work, influence_scale, projection_scale, design_scale, xp
        )
'''
replace_once(cov, old, new)
old = '''    adjusted_work, adjusted_scale = _column_working_values(
        adjusted_influence, xp
    )
'''
new = '''    adjusted_work, adjusted_scale = _gram_working_values(
        adjusted_influence, xp
    )
'''
replace_once(cov, old, new)
old = '''    return _restore_influence_covariance(
        cov_work, adjusted_scale, residual_scale, design_scale, xp
    )
'''
new = '''    return _restore_influence_covariance(
        cov_work, adjusted_scale, projection_scale, design_scale, xp
    )
'''
replace_once(cov, old, new)

# HC1 robust dispatcher.
old = '''        (
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
new = '''        (
            influence,
            projection_scale,
            design_scale,
            _X_pinv,
            _X_work,
            _rank,
        ) = _influence_rows(X, resid, xp)
        correction = hc1_correction
'''
replace_once(cov, old, new)
old = '''        cov_work = _symmetrize(
            influence_work.T @ influence_work * float(correction)
        )
        return _restore_influence_covariance(
            cov_work, influence_scale, residual_scale, design_scale, xp
        )
'''
new = '''        influence_work, influence_scale = _gram_working_values(
            influence, xp, max_multiplier=float(correction)
        )
        cov_work = _symmetrize(
            influence_work.T @ influence_work * float(correction)
        )
        return _restore_influence_covariance(
            cov_work, influence_scale, projection_scale, design_scale, xp
        )
'''
replace_once(cov, old, new)

# Regressions for mixed dynamic range and nested two-way/DK grouping.
test_cov = Path("dev/tests/test_panel_stage_c_covariance.py")
text = test_cov.read_text()
append = r'''


def test_clustered_covariance_preserves_small_group_beside_huge_exact_cancellation():
    X = np.ones((3, 1), dtype=np.float64)
    resid = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)
    groups = np.asarray([0, 0, 1], dtype=np.int64)
    actual = clustered_covariance(X, resid, groups)
    expected = np.asarray([[1.0e-200]], dtype=np.float64)
    assert actual[0, 0] > 0.0
    np.testing.assert_allclose(actual, expected, rtol=3.0e-14, atol=0.0)


def test_two_way_nested_cluster_preserves_small_component_after_huge_cancellation():
    X = np.ones((3, 1), dtype=np.float64)
    resid = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)
    unique = np.asarray([0, 1, 2], dtype=np.int64)
    coarse = np.asarray([0, 0, 1], dtype=np.int64)
    reference = clustered_covariance(X, resid, coarse)
    actual = two_way_clustered_covariance(X, resid, unique, coarse)
    np.testing.assert_allclose(actual, reference, rtol=3.0e-14, atol=0.0)


def test_dk_groups_before_gram_scaling_preserves_small_period_score():
    X = np.ones((3, 1), dtype=np.float64)
    resid = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)
    time = np.asarray([0, 0, 1], dtype=np.int64)
    actual = driscoll_kraay_covariance(X, resid, time, bandwidth=0)
    expected = np.asarray([[1.5e-200]], dtype=np.float64)
    assert actual[0, 0] > 0.0
    np.testing.assert_allclose(actual, expected, rtol=4.0e-14, atol=0.0)
'''
for name in [
    "test_clustered_covariance_preserves_small_group_beside_huge_exact_cancellation",
    "test_two_way_nested_cluster_preserves_small_component_after_huge_cancellation",
    "test_dk_groups_before_gram_scaling_preserves_small_period_score",
]:
    if name in text:
        raise RuntimeError(f"unexpected preexisting staged test: {name}")
test_cov.write_text(text + append)

# Torch CPU analogs.
torch_test = Path("dev/tests/test_panel_stage_b_torch_cpu.py")
text = torch_test.read_text()
append = r'''


def test_stage_c_torch_cpu_preserves_mixed_range_cluster_and_dk_scores():
    X = torch.ones((3, 1), dtype=torch.float64)
    resid = torch.as_tensor([1.5e308, -1.5e308, 3.0e-100], dtype=torch.float64)
    coarse = np.asarray([0, 0, 1], dtype=np.int64)
    unique = np.asarray([0, 1, 2], dtype=np.int64)
    time = np.asarray([0, 0, 1], dtype=np.int64)
    clustered = clustered_covariance(X, resid, coarse, xp=torch)
    two_way = two_way_clustered_covariance(X, resid, unique, coarse, xp=torch)
    dk = driscoll_kraay_covariance(X, resid, time, bandwidth=0, xp=torch)
    assert_allclose(clustered, np.asarray([[1.0e-200]]), rtol=5.0e-13, atol=0.0)
    assert_allclose(two_way, clustered, rtol=5.0e-13, atol=0.0)
    assert_allclose(dk, np.asarray([[1.5e-200]]), rtol=6.0e-13, atol=0.0)
'''
if "test_stage_c_torch_cpu_preserves_mixed_range_cluster_and_dk_scores" not in text:
    torch_test.write_text(text + append)

# Physical validator extends the existing extreme-scale audit.
bench = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = bench.read_text()
old = '''    tiny_cluster_groups = np.asarray([0, 0, 1, 1], dtype=np.int64)\n\n    if backend == "numpy":\n'''
new = '''    tiny_cluster_groups = np.asarray([0, 0, 1, 1], dtype=np.int64)\n\n    X_mixed_np = np.ones((3, 1), dtype=np.float64)\n    resid_mixed_np = np.asarray([1.5e308, -1.5e308, 3.0e-100], dtype=np.float64)\n    mixed_coarse = np.asarray([0, 0, 1], dtype=np.int64)\n    mixed_unique = np.asarray([0, 1, 2], dtype=np.int64)\n    mixed_time = np.asarray([0, 0, 1], dtype=np.int64)\n\n    if backend == "numpy":\n'''
if old not in text:
    raise RuntimeError("physical mixed insertion anchor missing")
text = text.replace(old, new, 1)
old = '''        X_tiny_cluster, resid_tiny_cluster = X_tiny_cluster_np, resid_tiny_cluster_np\n    elif backend == "cupy":\n'''
new = '''        X_tiny_cluster, resid_tiny_cluster = X_tiny_cluster_np, resid_tiny_cluster_np\n        X_mixed, resid_mixed = X_mixed_np, resid_mixed_np\n    elif backend == "cupy":\n'''
text = text.replace(old, new, 1)
old = '''        X_tiny_cluster = cp.asarray(X_tiny_cluster_np)\n        resid_tiny_cluster = cp.asarray(resid_tiny_cluster_np)\n    elif backend == "torch":\n'''
new = '''        X_tiny_cluster = cp.asarray(X_tiny_cluster_np)\n        resid_tiny_cluster = cp.asarray(resid_tiny_cluster_np)\n        X_mixed, resid_mixed = cp.asarray(X_mixed_np), cp.asarray(resid_mixed_np)\n    elif backend == "torch":\n'''
text = text.replace(old, new, 1)
old = '''        X_tiny_cluster = torch.as_tensor(X_tiny_cluster_np, dtype=torch.float64, device="cuda")\n        resid_tiny_cluster = torch.as_tensor(resid_tiny_cluster_np, dtype=torch.float64, device="cuda")\n    else:\n'''
new = '''        X_tiny_cluster = torch.as_tensor(X_tiny_cluster_np, dtype=torch.float64, device="cuda")\n        resid_tiny_cluster = torch.as_tensor(resid_tiny_cluster_np, dtype=torch.float64, device="cuda")\n        X_mixed = torch.as_tensor(X_mixed_np, dtype=torch.float64, device="cuda")\n        resid_mixed = torch.as_tensor(resid_mixed_np, dtype=torch.float64, device="cuda")\n    else:\n'''
text = text.replace(old, new, 1)
old = '''    tiny_design_cluster = _array(\n        clustered_covariance(\n            X_tiny_cluster, resid_tiny_cluster, tiny_cluster_groups, xp=xp\n        )\n    )\n    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk), ("pregram_hac", pregram_hac), ("pregram_dk", pregram_dk), ("two_way_component_cancellation", component_two_way), ("tiny_design_cluster_cancellation", tiny_design_cluster)):\n'''
new = '''    tiny_design_cluster = _array(\n        clustered_covariance(\n            X_tiny_cluster, resid_tiny_cluster, tiny_cluster_groups, xp=xp\n        )\n    )\n    mixed_cluster = _array(clustered_covariance(X_mixed, resid_mixed, mixed_coarse, xp=xp))\n    mixed_two_way = _array(two_way_clustered_covariance(\n        X_mixed, resid_mixed, mixed_unique, mixed_coarse, xp=xp\n    ))\n    mixed_dk = _array(driscoll_kraay_covariance(\n        X_mixed, resid_mixed, mixed_time, bandwidth=0, xp=xp\n    ))\n    for name, value in (("one_way", one_way), ("two_way", two_way), ("group_cancellation", cancellation), ("hac", hac), ("dk", dk), ("lag_hac", lag_hac), ("lag_dk", lag_dk), ("pregram_hac", pregram_hac), ("pregram_dk", pregram_dk), ("two_way_component_cancellation", component_two_way), ("tiny_design_cluster_cancellation", tiny_design_cluster), ("mixed_cluster", mixed_cluster), ("mixed_two_way", mixed_two_way), ("mixed_dk", mixed_dk)):\n'''
if old not in text:
    raise RuntimeError("physical mixed computation anchor missing")
text = text.replace(old, new, 1)
old = '''    np.testing.assert_allclose(\n        tiny_design_cluster, np.zeros((1, 1)), rtol=0.0, atol=0.0\n    )\n    return {\n'''
new = '''    np.testing.assert_allclose(\n        tiny_design_cluster, np.zeros((1, 1)), rtol=0.0, atol=0.0\n    )\n    np.testing.assert_allclose(mixed_cluster, np.asarray([[1.0e-200]]), rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(mixed_two_way, mixed_cluster, rtol=8e-13, atol=0.0)\n    np.testing.assert_allclose(mixed_dk, np.asarray([[1.5e-200]]), rtol=8e-13, atol=0.0)\n    return {\n'''
text = text.replace(old, new, 1)
old = '''        "tiny_design_cluster_cancellation": tiny_design_cluster.tolist(),\n    }\n'''
new = '''        "tiny_design_cluster_cancellation": tiny_design_cluster.tolist(),\n        "mixed_cluster": mixed_cluster.tolist(),\n        "mixed_two_way": mixed_two_way.tolist(),\n        "mixed_driscoll_kraay": mixed_dk.tolist(),\n    }\n'''
text = text.replace(old, new, 1)
bench.write_text(text)
