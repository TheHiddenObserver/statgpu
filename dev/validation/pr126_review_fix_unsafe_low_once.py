from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:160]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if marker not in text:
        p.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


old = r'''        if need_compensation:
            correction_triples = (
                (grouped1, grouped1_low, correction1),
                (grouped2, grouped2_low, correction2),
                (grouped12, grouped12_low, correction12),
            )
            safe_correction = all(
                _cross_reduction_is_safe(
                    high, low, xp, max_multiplier=correction
                )
                and _cross_reduction_is_safe(
                    low, low, xp, max_multiplier=correction
                )
                for high, low, correction in correction_triples
            )
            if safe_correction:
                def _low_order_covariance(high, low, correction):
                    cross = _symmetrize(high.T @ low)
                    low_square = _symmetrize(low.T @ low)
                    return _symmetrize(
                        ((2.0 * cross) + low_square) * float(correction)
                    )

                low_correction = _stable_inclusion_exclusion(
                    _low_order_covariance(grouped1, grouped1_low, correction1),
                    _low_order_covariance(grouped2, grouped2_low, correction2),
                    _low_order_covariance(grouped12, grouped12_low, correction12),
                    xp,
                )
                cov_work = _restore_coordinate_covariance(
                    cov_work, common_scale, xp
                )
                cov_work = _symmetrize(cov_work + low_correction)
                common_scale = xp.ones_like(projection_scale)
'''
new = r'''        if need_compensation:
            # First try to keep the low-order expansion in the *same* coordinate
            # scale as the high Gram.  This is the important path when individual
            # physical high-low cross terms would overflow but the CGM
            # inclusion-exclusion is finite: cancellation happens before any
            # physical-scale restoration instead of silently dropping the tail.
            low1_work = grouped1_low / common_scale
            low2_work = grouped2_low / common_scale
            low12_work = grouped12_low / common_scale
            low_work_triples = (
                (grouped1_low, low1_work),
                (grouped2_low, low2_work),
                (grouped12_low, low12_work),
            )
            low_underflowed = any(
                bool(
                    _to_float_scalar(
                        xp.any((low != 0.0) & (low_work == 0.0))
                    )
                )
                for low, low_work in low_work_triples
            )

            if not low_underflowed:
                def _low_order_covariance_work(high_work, low_work, correction):
                    cross = high_work.T @ low_work
                    low_square = _symmetrize(low_work.T @ low_work)
                    return _symmetrize(
                        (
                            cross
                            + cross.T
                            + low_square
                        )
                        * float(correction)
                    )

                low_correction_work = _stable_inclusion_exclusion(
                    _low_order_covariance_work(
                        grouped1_work, low1_work, correction1
                    ),
                    _low_order_covariance_work(
                        grouped2_work, low2_work, correction2
                    ),
                    _low_order_covariance_work(
                        grouped12_work, low12_work, correction12
                    ),
                    xp,
                )
                cov_work = _symmetrize(cov_work + low_correction_work)
            else:
                # If dividing by the high-Gram coordinate scale would erase a
                # nonzero low part, preserve it on the physical score scale.
                # Such a fallback is used only after proving every required
                # high-low and low-low reduction safe.  Never fail open by
                # omitting the low component.
                correction_triples = (
                    (grouped1, grouped1_low, correction1),
                    (grouped2, grouped2_low, correction2),
                    (grouped12, grouped12_low, correction12),
                )
                safe_correction = all(
                    _cross_reduction_is_safe(
                        high, low, xp, max_multiplier=correction
                    )
                    and _cross_reduction_is_safe(
                        low, low, xp, max_multiplier=correction
                    )
                    for high, low, correction in correction_triples
                )
                if not safe_correction:
                    raise FloatingPointError(
                        "two-way cluster low-order correction cannot be "
                        "evaluated safely without losing a nonzero tail"
                    )

                def _low_order_covariance(high, low, correction):
                    cross = high.T @ low
                    low_square = _symmetrize(low.T @ low)
                    return _symmetrize(
                        (
                            cross
                            + cross.T
                            + low_square
                        )
                        * float(correction)
                    )

                low_correction = _stable_inclusion_exclusion(
                    _low_order_covariance(
                        grouped1, grouped1_low, correction1
                    ),
                    _low_order_covariance(
                        grouped2, grouped2_low, correction2
                    ),
                    _low_order_covariance(
                        grouped12, grouped12_low, correction12
                    ),
                    xp,
                )
                cov_work = _restore_coordinate_covariance(
                    cov_work, common_scale, xp
                )
                cov_work = _symmetrize(cov_work + low_correction)
                common_scale = xp.ones_like(projection_scale)
'''
replace_once("statgpu/panel/_covariance.py", old, new)

append_once(
    "dev/tests/test_panel_stage_c_covariance.py",
    "test_two_way_nonnested_unsafe_cross_cancels_before_restore",
    r'''
def test_two_way_nonnested_unsafe_cross_cancels_before_restore():
    amplitude = 1.0e200
    low1 = 1.0e108
    low2 = np.nextafter(low1, np.inf)
    scores = np.asarray(
        [
            -amplitude, low1, amplitude, -low1,
            -amplitude, -low2, amplitude, low2,
        ],
        dtype=np.float64,
    )
    cluster1 = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)
    X = np.full((8, 1), 0.5, dtype=np.float64)

    actual = two_way_clustered_covariance(
        X, 4.0 * scores, cluster1, cluster2
    )
    expected = np.asarray(
        [[4.0 * amplitude * (low2 - low1)]], dtype=np.float64
    )
    assert np.isfinite(expected[0, 0])
    assert_allclose(actual, expected, rtol=4e-12, atol=0.0)
''',
)

append_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    "test_stage_c_torch_cpu_two_way_unsafe_cross_cancels_before_restore",
    r'''
def test_stage_c_torch_cpu_two_way_unsafe_cross_cancels_before_restore():
    amplitude = 1.0e200
    low1 = 1.0e108
    low2 = np.nextafter(low1, np.inf)
    scores_np = np.asarray(
        [
            -amplitude, low1, amplitude, -low1,
            -amplitude, -low2, amplitude, low2,
        ],
        dtype=np.float64,
    )
    cluster1 = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    cluster2 = np.asarray([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)
    X = torch.full((8, 1), 0.5, dtype=torch.float64)
    resid = torch.as_tensor(4.0 * scores_np, dtype=torch.float64)
    actual = two_way_clustered_covariance(
        X, resid, cluster1, cluster2, xp=torch
    ).detach().cpu().numpy()
    expected = np.asarray(
        [[4.0 * amplitude * (low2 - low1)]], dtype=np.float64
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=0.0)
''',
)

runner = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = runner.read_text(encoding="utf-8")
old = r'''    deep_two_way = _array(
        two_way_clustered_covariance(
            X_deep,
            deep_scores,
            cluster1,
            cluster2,
            xp=xp,
        )
    )
    np.testing.assert_array_equal(
'''
new = r'''    deep_two_way = _array(
        two_way_clustered_covariance(
            X_deep,
            deep_scores,
            cluster1,
            cluster2,
            xp=xp,
        )
    )

    unsafe_amplitude = 1.0e200
    unsafe_low1 = 1.0e108
    unsafe_low2 = np.nextafter(unsafe_low1, np.inf)
    unsafe_scores_np = np.asarray(
        [
            -unsafe_amplitude, unsafe_low1, unsafe_amplitude, -unsafe_low1,
            -unsafe_amplitude, -unsafe_low2, unsafe_amplitude, unsafe_low2,
        ],
        dtype=np.float64,
    )
    unsafe_c1 = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    unsafe_c2 = np.asarray([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)
    unsafe_X_np = np.full((8, 1), 0.5, dtype=np.float64)
    unsafe_dummy = np.arange(8, dtype=np.int64)
    unsafe_X, unsafe_resid, _ue, _ut = _to_backend(
        unsafe_X_np,
        4.0 * unsafe_scores_np,
        unsafe_dummy,
        unsafe_dummy,
        backend,
    )
    unsafe_two_way = _array(
        two_way_clustered_covariance(
            unsafe_X,
            unsafe_resid,
            unsafe_c1,
            unsafe_c2,
            xp=xp,
        )
    )
    unsafe_expected = np.asarray(
        [[4.0 * unsafe_amplitude * (unsafe_low2 - unsafe_low1)]],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
'''
if "unsafe_two_way = _array(" not in text:
    if old not in text:
        raise RuntimeError("multiscale physical audit insertion anchor not found")
    text = text.replace(old, new, 1)
old = r'''    np.testing.assert_allclose(
        deep_two_way, np.zeros((1, 1)), rtol=0.0, atol=0.0
    )
    return {
'''
new = r'''    np.testing.assert_allclose(
        deep_two_way, np.zeros((1, 1)), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        unsafe_two_way, unsafe_expected, rtol=4e-12, atol=0.0
    )
    return {
'''
if "unsafe_two_way, unsafe_expected" not in text:
    if old not in text:
        raise RuntimeError("multiscale physical audit assertion anchor not found")
    text = text.replace(old, new, 1)
old = r'''        "deep_two_way": deep_two_way.tolist(),
    }
'''
new = r'''        "deep_two_way": deep_two_way.tolist(),
        "unsafe_cross_two_way": unsafe_two_way.tolist(),
    }
'''
if '"unsafe_cross_two_way"' not in text:
    if old not in text:
        raise RuntimeError("multiscale physical audit return anchor not found")
    text = text.replace(old, new, 1)
runner.write_text(text, encoding="utf-8")

contract = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = contract.read_text(encoding="utf-8")
old = r'''    np.testing.assert_allclose(
        np.asarray(audit["deep_two_way"]),
        np.zeros((1, 1)),
        rtol=0.0,
        atol=0.0,
    )
'''
new = r'''    np.testing.assert_allclose(
        np.asarray(audit["deep_two_way"]),
        np.zeros((1, 1)),
        rtol=0.0,
        atol=0.0,
    )
    unsafe_amplitude = 1.0e200
    unsafe_low1 = 1.0e108
    unsafe_low2 = np.nextafter(unsafe_low1, np.inf)
    np.testing.assert_allclose(
        np.asarray(audit["unsafe_cross_two_way"]),
        np.asarray(
            [[4.0 * unsafe_amplitude * (unsafe_low2 - unsafe_low1)]],
            dtype=np.float64,
        ),
        rtol=4e-12,
        atol=0.0,
    )
'''
if 'audit["unsafe_cross_two_way"]' not in text:
    if old not in text:
        raise RuntimeError("physical runner contract multiscale anchor not found")
    text = text.replace(old, new, 1)
contract.write_text(text, encoding="utf-8")

for stale in (
    "dev/validation/pr126_review_fix_unsafe_low_once.py",
    ".github/workflows/pr126-review-fix-unsafe-low.yml",
):
    Path(stale).unlink(missing_ok=True)
