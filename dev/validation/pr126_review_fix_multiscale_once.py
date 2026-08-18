from pathlib import Path


def replace_between(path, start_marker, end_marker, replacement):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError(f"replace_between anchors not found in {path}")
    p.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def append_once(path, marker, block):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if marker not in text:
        p.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


grouped = r"""
def _grouped_score_sums(
    scores, codes_np, *, n_groups: int, xp, return_compensation: bool = False
):
    "Sum scores by group while preserving recursively separated magnitude tiers."
    codes_np = np.asarray(codes_np, dtype=np.int64).ravel()
    if codes_np.shape[0] != int(scores.shape[0]):
        raise ValueError("group codes must match the number of score rows")
    if int(n_groups) <= 0:
        raise ValueError("at least one group is required")
    codes = xp_asarray(
        codes_np,
        dtype=xp.int64,
        xp=xp,
        ref_arr=scores,
    )
    shape = (int(n_groups), int(scores.shape[1]))
    index = (
        codes.unsqueeze(1).expand_as(scores)
        if hasattr(codes, "unsqueeze")
        else None
    )

    def _group_abs_max(values):
        out = xp_zeros(shape, dtype=xp.float64, xp=xp, ref_arr=scores)
        absolute = xp.abs(values)
        if hasattr(out, "scatter_reduce_"):
            out.scatter_reduce_(
                0, index, absolute, reduce="amax", include_self=True
            )
        elif type(out).__module__.startswith("cupy"):
            xp.maximum.at(out, codes, absolute)
        else:
            np.maximum.at(out, codes_np, absolute)
        return out

    def _signed_parts(values):
        positive = xp.where(values > 0.0, values, xp.zeros_like(values))
        negative = xp.where(values < 0.0, values, xp.zeros_like(values))
        positive_out = xp_zeros(
            shape, dtype=xp.float64, xp=xp, ref_arr=scores
        )
        negative_out = xp_zeros(
            shape, dtype=xp.float64, xp=xp, ref_arr=scores
        )
        if hasattr(positive_out, "scatter_add_"):
            positive_out.scatter_add_(0, index, positive)
            negative_out.scatter_add_(0, index, negative)
        elif type(positive_out).__module__.startswith("cupy"):
            xp.add.at(positive_out, codes, positive)
            xp.add.at(negative_out, codes, negative)
        else:
            np.add.at(positive_out, codes_np, positive)
            np.add.at(negative_out, codes_np, negative)
        return positive_out, negative_out

    def _two_sum(left, right):
        summed = left + right
        virtual_right = summed - left
        residual = (
            left - (summed - virtual_right)
        ) + (right - virtual_right)
        return summed, residual

    max_abs = _group_abs_max(scores)
    counts_np = np.bincount(
        codes_np, minlength=int(n_groups)
    ).astype(np.float64)
    counts = xp_asarray(
        counts_np,
        dtype=xp.float64,
        xp=xp,
        ref_arr=scores,
    ).reshape(-1, 1)
    limit = float(np.finfo(np.float64).max) / counts
    factor = xp.where(max_abs > limit, counts, xp.ones_like(max_abs))
    remaining = scores / factor[codes]
    split_ratio = xp.minimum(
        counts * float(8.0 * np.finfo(np.float64).eps),
        xp.full_like(counts, 0.25),
    )

    tiers = []
    for _ in range(128):
        tier_max = _group_abs_max(remaining)
        threshold = tier_max * split_ratio
        tail_mask = (
            (xp.abs(remaining) < threshold[codes])
            & (remaining != 0.0)
        )
        main = xp.where(tail_mask, xp.zeros_like(remaining), remaining)
        positive_out, negative_out = _signed_parts(main)
        tier_sum, tier_error = _two_sum(positive_out, negative_out)
        tiers.append((tier_sum, tier_error))
        if not bool(_to_float_scalar(xp.any(tail_mask))):
            break
        remaining = xp.where(
            tail_mask, remaining, xp.zeros_like(remaining)
        )
    else:
        raise RuntimeError(
            "grouped score cancellation exceeded the float64 tier budget"
        )

    def _collapse(parts):
        total = xp.zeros_like(tiers[0][0])
        correction = xp.zeros_like(total)
        for part in parts:
            summed, error = _two_sum(total, part)
            total = summed
            correction = correction + error
        summed, error = _two_sum(total, correction)
        return summed + error

    lower_parts = []
    for tier_sum, tier_error in reversed(tiers[1:]):
        lower_parts.extend((tier_error, tier_sum))
    lower_parts.append(tiers[0][1])
    low = _collapse(lower_parts)
    high = tiers[0][0]

    if return_compensation:
        return high * factor, low * factor

    summed, error = _two_sum(high, low)
    return (summed + error) * factor


"""
replace_between(
    "statgpu/panel/_covariance.py",
    "def _grouped_score_sums(",
    "def _factorize_1d_labels",
    grouped,
)

diag = Path("statgpu/panel/_diagnostics.py")
text = diag.read_text(encoding="utf-8")
old_start = text.find("    basis = eigvecs[:, positive]\n")
old_end = text.find(
    "    # Evaluate d' D+ d in standardized eigencoordinates",
    old_start,
)
if old_start < 0 or old_end < 0:
    raise RuntimeError("Hausman range block anchors not found")
range_block = r"""
    basis = eigvecs[:, positive]

    # Range membership is scale invariant.  Normalize d before the dense
    # eigenspace projection so several finite O(DBL_MAX) coordinates cannot
    # overflow in basis.T @ d before the range guard is evaluated.
    d_scale = float(np.max(np.abs(d))) if d.size else 0.0
    if not np.isfinite(d_scale):
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=float(rank),
            reason="coefficient difference contains non-finite values",
            metadata=meta,
        )
    d_work = d if d_scale == 0.0 else d / d_scale
    coordinates_work = basis.T @ d_work
    projected_work = basis @ coordinates_work
    null_work = d_work - projected_work

    d_norm_normalized = float(np.linalg.norm(d_work))
    null_norm_work = float(np.linalg.norm(null_work))
    range_tol_work = (
        1024.0 * np.finfo(np.float64).eps * d_norm_normalized
    )

    def _restore_d_scale(value: float) -> float:
        value = float(value)
        if value == 0.0 or d_scale == 0.0:
            return 0.0
        if d_scale > float(np.finfo(np.float64).max) / abs(value):
            return float(np.copysign(np.inf, value))
        return float(value * d_scale)

    d_norm = _restore_d_scale(d_norm_normalized)
    null_norm = _restore_d_scale(null_norm_work)
    range_tol = _restore_d_scale(range_tol_work)
    meta["range_tolerance"] = float(range_tol)
    meta["nullspace_component_norm"] = float(null_norm)

    comparison_factor = max(
        float(np.max(np.abs(d_work))) if d_work.size else 0.0,
        float(np.max(np.abs(null_work))) if null_work.size else 0.0,
    )
    comparison_scale = _restore_d_scale(comparison_factor)
    if comparison_factor == 0.0:
        outside_range = False
        range_tol_normalized = 0.0
        null_norm_normalized = 0.0
    else:
        d_normalized = d_work / comparison_factor
        null_normalized = null_work / comparison_factor
        d_norm_comparison = float(np.linalg.norm(d_normalized))
        null_norm_normalized = float(np.linalg.norm(null_normalized))
        range_tol_normalized = (
            1024.0 * np.finfo(np.float64).eps * d_norm_comparison
        )
        outside_range = (
            null_norm_normalized > range_tol_normalized
        )
    meta["range_comparison_scale"] = float(comparison_scale)
    meta["range_tolerance_normalized"] = float(range_tol_normalized)
    meta["nullspace_component_norm_normalized"] = float(
        null_norm_normalized
    )
    if outside_range:
        return _inapplicable(
            null=null,
            alternative=alternative,
            distribution="chi2",
            df=float(rank),
            reason=(
                "coefficient difference has a component outside the "
                "identified covariance-difference range"
            ),
            metadata=meta,
        )

"""
text = text[:old_start] + range_block + text[old_end:]

old = """    coordinates = basis.T @ d
    if matrix_scale == 0.0:
        standardized = coordinates
    else:
        standardized = coordinates / np.sqrt(matrix_scale)
        standardized = standardized / np.sqrt(eigvals_work[positive])
    statistic = float(np.sum(standardized * standardized))
"""
new = """    eig_standardized = (
        coordinates_work / np.sqrt(eigvals_work[positive])
    )
    if d_scale == 0.0:
        standardized = eig_standardized
    else:
        scale_ratio = d_scale / np.sqrt(matrix_scale)
        standardized = eig_standardized * scale_ratio
    statistic = float(np.sum(standardized * standardized))
"""
if old not in text:
    raise RuntimeError("Hausman quadratic block anchor not found")
text = text.replace(old, new, 1)
diag.write_text(text, encoding="utf-8")

append_once(
    "dev/tests/test_panel_stage_c_covariance.py",
    "test_grouped_score_multiscale_cancellation_survives_three_levels",
    r"""
def test_grouped_score_multiscale_cancellation_survives_three_levels():
    scores = np.asarray(
        [[1.0e154], [-1.0e154], [1.0e138], [1.0], [-1.0e138], [-1.0]],
        dtype=np.float64,
    )
    cluster1 = np.asarray([0, 0, 0, 1, 0, 0], dtype=np.int64)
    cluster2 = np.asarray([0, 0, 0, 1, 0, 1], dtype=np.int64)

    grouped = _grouped_score_sums(
        scores,
        cluster1,
        n_groups=2,
        xp=np,
    )
    np.testing.assert_array_equal(
        grouped, np.asarray([[-1.0], [1.0]])
    )

    X = np.full((6, 1), 1.0 / 6.0, dtype=np.float64)
    actual = two_way_clustered_covariance(
        X,
        scores.ravel(),
        cluster1,
        cluster2,
    )
    np.testing.assert_allclose(
        actual, np.zeros((1, 1)), rtol=0.0, atol=0.0
    )
""",
)
append_once(
    "dev/tests/test_panel_stage_c_covariance.py",
    "test_one_way_and_dk_preserve_small_score_after_same_sign_swallow",
    r"""
def test_one_way_and_dk_preserve_small_score_after_same_sign_swallow():
    scores = np.asarray([1.0e154, -1.0e154, 1.0, 1.0])
    groups = np.asarray([0, 0, 0, 1], dtype=np.int64)
    X = np.full((4, 1), 0.5, dtype=np.float64)
    resid = 2.0 * scores

    grouped = _grouped_score_sums(
        scores[:, None], groups, n_groups=2, xp=np
    )
    np.testing.assert_array_equal(
        grouped, np.asarray([[1.0], [1.0]])
    )
    one_way = clustered_covariance(X, resid, groups)
    dk = driscoll_kraay_covariance(
        X, resid, groups, bandwidth=0
    )
    np.testing.assert_allclose(
        one_way, np.asarray([[2.0]]), rtol=2e-15, atol=0.0
    )
    np.testing.assert_allclose(
        dk, np.asarray([[8.0 / 3.0]]), rtol=3e-15, atol=0.0
    )
""",
)
append_once(
    "dev/tests/test_panel_stage_b_hausman_covariance.py",
    "test_hausman_quadratic_scales_dense_projection_before_range_check",
    r"""
def test_hausman_quadratic_scales_dense_projection_before_range_check():
    basis = np.full(4, 0.5, dtype=np.float64)
    covariance = 1.0e308 * np.outer(basis, basis)
    difference = np.asarray(
        [
            1.0e308 + 1.0e300,
            1.0e308 - 1.0e300,
            1.0e308,
            1.0e308,
        ],
        dtype=np.float64,
    )
    result = _hausman_quadratic(difference, covariance)
    assert result.applicable is False
    assert (
        "outside the identified covariance-difference range"
        in result.reason
    )
    assert np.isfinite(
        result.metadata["range_tolerance_normalized"]
    )
    assert np.isfinite(
        result.metadata["nullspace_component_norm_normalized"]
    )
""",
)
append_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    "test_stage_c_torch_cpu_multiscale_group_cancellation",
    r"""
def test_stage_c_torch_cpu_multiscale_group_cancellation():
    scores = torch.tensor(
        [1.0e154, -1.0e154, 1.0, 1.0],
        dtype=torch.float64,
    )
    groups = np.asarray([0, 0, 0, 1], dtype=np.int64)
    X = torch.full((4, 1), 0.5, dtype=torch.float64)
    resid = 2.0 * scores

    grouped = _grouped_score_sums(
        scores[:, None], groups, n_groups=2, xp=torch
    ).detach().cpu().numpy()
    np.testing.assert_array_equal(
        grouped, np.asarray([[1.0], [1.0]])
    )
    one_way = clustered_covariance(
        X, resid, groups, xp=torch
    ).detach().cpu().numpy()
    dk = driscoll_kraay_covariance(
        X, resid, groups, bandwidth=0, xp=torch
    ).detach().cpu().numpy()
    np.testing.assert_allclose(
        one_way, np.asarray([[2.0]]), rtol=2e-15, atol=0.0
    )
    np.testing.assert_allclose(
        dk, np.asarray([[8.0 / 3.0]]), rtol=3e-15, atol=0.0
    )
""",
)

runner = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = runner.read_text(encoding="utf-8")
marker = "\ndef _hausman_scale_audit(backend):\n"
physical_helper = r"""
def _multiscale_grouping_audit(backend):
    scores_np = np.asarray(
        [1.0e154, -1.0e154, 1.0, 1.0], dtype=np.float64
    )
    groups = np.asarray([0, 0, 0, 1], dtype=np.int64)
    X_np = np.full((4, 1), 0.5, dtype=np.float64)
    resid_np = 2.0 * scores_np
    dummy = np.arange(4, dtype=np.int64)
    X, resid, _entity, _time = _to_backend(
        X_np, resid_np, dummy, dummy, backend
    )

    deep_scores_np = np.asarray(
        [1.0e154, -1.0e154, 1.0e138, 1.0, -1.0e138, -1.0],
        dtype=np.float64,
    )
    cluster1 = np.asarray([0, 0, 0, 1, 0, 0], dtype=np.int64)
    cluster2 = np.asarray([0, 0, 0, 1, 0, 1], dtype=np.int64)
    X_deep_np = np.full((6, 1), 1.0 / 6.0, dtype=np.float64)
    deep_dummy = np.arange(6, dtype=np.int64)
    X_deep, deep_scores, _entity2, _time2 = _to_backend(
        X_deep_np,
        deep_scores_np,
        deep_dummy,
        deep_dummy,
        backend,
    )

    if backend == "numpy":
        xp = np
        scores = scores_np
    elif backend == "cupy":
        import cupy as cp
        xp = cp
        scores = cp.asarray(scores_np)
    elif backend == "torch":
        import torch
        xp = torch
        scores = torch.as_tensor(
            scores_np, dtype=torch.float64, device="cuda"
        )
    else:
        raise ValueError(backend)

    grouped = _array(
        _grouped_score_sums(
            scores[:, None], groups, n_groups=2, xp=xp
        )
    )
    one_way = _array(
        clustered_covariance(X, resid, groups, xp=xp)
    )
    dk = _array(
        driscoll_kraay_covariance(
            X, resid, groups, bandwidth=0, xp=xp
        )
    )
    deep_two_way = _array(
        two_way_clustered_covariance(
            X_deep,
            deep_scores,
            cluster1,
            cluster2,
            xp=xp,
        )
    )
    np.testing.assert_array_equal(
        grouped, np.asarray([[1.0], [1.0]])
    )
    np.testing.assert_allclose(
        one_way, np.asarray([[2.0]]), rtol=8e-13, atol=0.0
    )
    np.testing.assert_allclose(
        dk, np.asarray([[8.0 / 3.0]]), rtol=8e-13, atol=0.0
    )
    np.testing.assert_allclose(
        deep_two_way, np.zeros((1, 1)), rtol=0.0, atol=0.0
    )
    return {
        "status": "success",
        "backend": backend,
        "grouped": grouped.tolist(),
        "one_way": one_way.tolist(),
        "driscoll_kraay": dk.tolist(),
        "deep_two_way": deep_two_way.tolist(),
    }


"""
if "_multiscale_grouping_audit" not in text:
    if marker not in text:
        raise RuntimeError("physical runner Hausman anchor not found")
    text = text.replace(marker, "\n" + physical_helper.strip() + "\n\n" + marker.lstrip(), 1)

old = """    dense = _hausman_quadratic(
        np.asarray([1.0e154, 1.0e154]),
        np.full((2, 2), 1.0e308, dtype=np.float64),
    )
"""
new = old + """    dense_basis = np.full(4, 0.5, dtype=np.float64)
    dense_range = _hausman_quadratic(
        np.asarray(
            [
                1.0e308 + 1.0e300,
                1.0e308 - 1.0e300,
                1.0e308,
                1.0e308,
            ],
            dtype=np.float64,
        ),
        1.0e308 * np.outer(dense_basis, dense_basis),
    )
    if dense_range.applicable or not dense_range.reason or (
        "outside the identified covariance-difference range"
        not in dense_range.reason
    ):
        raise AssertionError(
            f"{backend}: dense Hausman range projection failed closed"
        )
"""
if "dense_range = _hausman_quadratic" not in text:
    if old not in text:
        raise RuntimeError("physical Hausman dense anchor not found")
    text = text.replace(old, new, 1)
old_return = """        "dense_large_statistic": float(dense.statistic),
    }
"""
new_return = """        "dense_large_statistic": float(dense.statistic),
        "dense_projection_range_rejected": True,
    }
"""
if '"dense_projection_range_rejected"' not in text:
    if old_return not in text:
        raise RuntimeError("physical Hausman return anchor not found")
    text = text.replace(old_return, new_return, 1)

old_payload = """            "covariance_extreme_scale": _covariance_extreme_scale_audit(backend),
            "hausman_scale": _hausman_scale_audit(backend),
"""
new_payload = """            "covariance_extreme_scale": _covariance_extreme_scale_audit(backend),
            "multiscale_grouping": _multiscale_grouping_audit(backend),
            "hausman_scale": _hausman_scale_audit(backend),
"""
if '"multiscale_grouping": _multiscale_grouping_audit' not in text:
    if old_payload not in text:
        raise RuntimeError("physical numerical primitive anchor not found")
    text = text.replace(old_payload, new_payload, 1)
runner.write_text(text, encoding="utf-8")

contract = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = contract.read_text(encoding="utf-8")
old = """    assert audit["large_singular_range_rejected"] is True
    np.testing.assert_allclose(audit["dense_large_statistic"], 1.0, rtol=5e-13, atol=0.0)
"""
new = """    assert audit["large_singular_range_rejected"] is True
    assert audit["dense_projection_range_rejected"] is True
    np.testing.assert_allclose(audit["dense_large_statistic"], 1.0, rtol=5e-13, atol=0.0)
"""
if 'audit["dense_projection_range_rejected"]' not in text:
    if old not in text:
        raise RuntimeError("physical Hausman contract anchor not found")
    text = text.replace(old, new, 1)
contract.write_text(text, encoding="utf-8")
append_once(
    "dev/tests/test_panel_stage_c_physical_runner_contract.py",
    "test_stage_c_runner_multiscale_grouping_audit_is_executable",
    r"""
def test_stage_c_runner_multiscale_grouping_audit_is_executable():
    audit = _MOD._multiscale_grouping_audit("numpy")
    assert audit["status"] == "success"
    assert audit["backend"] == "numpy"
    np.testing.assert_array_equal(
        np.asarray(audit["grouped"]),
        np.asarray([[1.0], [1.0]]),
    )
    np.testing.assert_allclose(
        np.asarray(audit["one_way"]),
        np.asarray([[2.0]]),
        rtol=2e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(audit["driscoll_kraay"]),
        np.asarray([[8.0 / 3.0]]),
        rtol=3e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(audit["deep_two_way"]),
        np.zeros((1, 1)),
        rtol=0.0,
        atol=0.0,
    )
""",
)

for stale in (
    "dev/validation/pr126_grouped_tail_fixup_once.py",
    "dev/validation/pr126_hausman_dense_fixup_once.py",
    "dev/validation/pr126_hausman_dense_once.py",
    "dev/validation/pr126_nonnested_union_fixup_once.py",
    "dev/validation/pr126_nonnested_union_once.py",
    "dev/validation/pr126_review_fix_multiscale_once.py",
    ".github/workflows/pr126-review-fix-multiscale.yml",
):
    Path(stale).unlink(missing_ok=True)
