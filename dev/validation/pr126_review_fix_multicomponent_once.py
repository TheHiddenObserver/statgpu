from pathlib import Path


def replace_between(path, start_marker, end_marker, replacement):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError(f"replace_between anchors not found in {path}")
    p.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"replace anchor not found in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path, marker, block):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if marker not in text:
        p.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


path = "statgpu/panel/_covariance.py"
text = Path(path).read_text(encoding="utf-8")
marker = "\n\ndef _grouped_score_sums(\n"
helper = r'''

def _stable_matrix_expansion_sum(terms, xp):
    """Sum finite matrix terms with a floating-point expansion.

    The caller must put terms on a common range-safe working scale.  General
    TwoSum then keeps cancellation residuals as separate expansion components,
    so a smaller covariance tier is not discarded before later large terms
    cancel.  This is used only for nonnested multiway-cluster inclusion-
    exclusion, where cancellation across marginal/intersection components is
    part of the estimator definition.
    """
    if not terms:
        raise ValueError("at least one matrix term is required")

    partials = []
    for term in terms:
        carry = term
        next_partials = []
        for partial in partials:
            summed = carry + partial
            virtual_partial = summed - carry
            residual = (
                carry - (summed - virtual_partial)
            ) + (partial - virtual_partial)
            next_partials.append(residual)
            carry = summed
        next_partials.append(carry)
        partials = next_partials

    # Grow-expansion keeps residual components before the final carry.  At this
    # point all large cancellation has already occurred; an ascending estimate
    # performs the one unavoidable float64 rounding of the final covariance.
    total = xp.zeros_like(partials[0])
    for partial in partials:
        total = total + partial
    return _symmetrize(total)
'''
if "def _stable_matrix_expansion_sum(" not in text:
    if marker not in text:
        raise RuntimeError("matrix expansion helper anchor not found")
    text = text.replace(marker, helper + marker, 1)
Path(path).write_text(text, encoding="utf-8")

replace_once(
    path,
    "def _grouped_score_sums(\n    scores, codes_np, *, n_groups: int, xp, return_compensation: bool = False\n):",
    "def _grouped_score_sums(\n    scores, codes_np, *, n_groups: int, xp, return_compensation: bool = False,\n    return_components: bool = False,\n):",
)
replace_once(
    path,
    "    if int(n_groups) <= 0:\n        raise ValueError(\"at least one group is required\")\n",
    "    if int(n_groups) <= 0:\n        raise ValueError(\"at least one group is required\")\n    if return_compensation and return_components:\n        raise ValueError(\n            \"return_compensation and return_components are mutually exclusive\"\n        )\n",
)
replace_once(
    path,
    "    def _collapse(parts):\n",
    "    if return_components:\n        # Keep every recursively separated tier and every TwoSum residual as a\n        # distinct expansion component.  Do not recombine lower tiers here:\n        # two-way inclusion-exclusion may cancel much larger components later.\n        components = []\n        for tier_sum, tier_error in tiers:\n            components.extend((tier_sum * factor, tier_error * factor))\n        return components\n\n    def _collapse(parts):\n",
)

cluster_helper = r'''def _cluster_grouped_scores(
    scores,
    codes,
    *,
    n_groups: int,
    nobs: int,
    group_debias: bool,
    xp,
    return_compensation: bool = False,
    return_components: bool = False,
):
    if int(n_groups) < 2:
        raise ValueError(
            "clustered covariance requires at least two distinct clusters"
        )
    grouped = _grouped_score_sums(
        scores,
        codes,
        n_groups=int(n_groups),
        xp=xp,
        return_compensation=return_compensation,
        return_components=return_components,
    )
    correction = (
        _group_debias_factor(int(n_groups), int(nobs)) if group_debias else 1.0
    )
    if return_components:
        return grouped, float(correction)
    if return_compensation:
        high, low = grouped
        return high, low, float(correction)
    return grouped, float(correction)


'''
replace_between(
    path,
    "def _cluster_grouped_scores(\n",
    "def _cluster_meat_from_grouped",
    cluster_helper,
)

nonnested = r'''    if nested_c1 or nested_c2:
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
        if nested_c1:
            cov_work, common_scale = _cluster_meat_from_grouped(
                grouped2, correction2, xp
            )
        else:
            cov_work, common_scale = _cluster_meat_from_grouped(
                grouped1, correction1, xp
            )
    else:
        components1, correction1 = _cluster_grouped_scores(
            influence,
            c1,
            n_groups=int(len(labels1)),
            nobs=n,
            group_debias=group_debias,
            xp=xp,
            return_components=True,
        )
        components2, correction2 = _cluster_grouped_scores(
            influence,
            c2,
            n_groups=int(len(labels2)),
            nobs=n,
            group_debias=group_debias,
            xp=xp,
            return_components=True,
        )
        components12, correction12 = _cluster_grouped_scores(
            influence,
            c12,
            n_groups=n12,
            nobs=n,
            group_debias=group_debias,
            xp=xp,
            return_components=True,
        )

        extra_max = xp.zeros_like(xp.max(xp.abs(components1[0])))
        for components in (components1, components2, components12):
            for component in components[1:]:
                extra_max = xp.maximum(extra_max, xp.max(xp.abs(component)))

        if _to_float_scalar(extra_max) == 0.0:
            (grouped1_work, grouped2_work, grouped12_work), common_scale = (
                _common_gram_working_values(
                    [components1[0], components2[0], components12[0]],
                    xp,
                    max_multiplier=max(correction1, correction2, correction12),
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
            cov_work = _stable_inclusion_exclusion(
                V1_work, V2_work, V12_work, xp
            )
        else:
            all_components = components1 + components2 + components12
            term_count = sum(
                len(components) * (len(components) + 1) // 2
                for components in (components1, components2, components12)
            )
            max_correction = max(correction1, correction2, correction12)
            working_components, common_scale = _common_gram_working_values(
                all_components,
                xp,
                max_multiplier=(
                    2.0 * float(max(1, term_count)) * float(max_correction)
                ),
            )

            lost_component = False
            for original, working in zip(all_components, working_components):
                if bool(
                    _to_float_scalar(
                        xp.any((original != 0.0) & (working == 0.0))
                    )
                ):
                    lost_component = True
                    break
            if lost_component:
                raise FloatingPointError(
                    "two-way cluster score expansion exceeds the float64 "
                    "common-scale dynamic range"
                )

            n1 = len(components1)
            n2 = len(components2)
            work1 = working_components[:n1]
            work2 = working_components[n1:n1 + n2]
            work12 = working_components[n1 + n2:]

            terms = []

            def _append_component_terms(components, correction, sign):
                coefficient = float(sign) * float(correction)
                for i, left in enumerate(components):
                    terms.append(
                        _symmetrize(left.T @ left) * coefficient
                    )
                    for right in components[:i]:
                        cross = left.T @ right
                        terms.append((cross + cross.T) * coefficient)

            _append_component_terms(work1, correction1, 1.0)
            _append_component_terms(work2, correction2, 1.0)
            _append_component_terms(work12, correction12, -1.0)
            cov_work = _stable_matrix_expansion_sum(terms, xp)
'''
replace_between(
    path,
    "    if nested_c1 or nested_c2:\n",
    "    cov = _restore_influence_covariance(\n",
    nonnested,
)

append_once(
    "dev/tests/test_panel_stage_c_covariance.py",
    "test_two_way_nonnested_preserves_third_magnitude_component",
    r'''
def test_two_way_nonnested_preserves_third_magnitude_component():
    amplitude = 2.0 ** 660
    middle = 2.0 ** 600
    tiny = 2.0 ** 350
    scores = np.asarray(
        [
            -amplitude, middle, tiny, amplitude, -middle, -tiny,
            -amplitude, -middle, amplitude, middle,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        dtype=np.float64,
    )
    cluster1 = np.asarray(
        [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    cluster2 = np.asarray(
        [0, 1, 1, 0, 1, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    X = np.full((16, 1), 0.5, dtype=np.float64)
    actual = two_way_clustered_covariance(
        X, 8.0 * scores, cluster1, cluster2
    )
    expected = np.asarray([[-4.0 * amplitude * tiny]], dtype=np.float64)
    assert np.isfinite(expected[0, 0])
    assert_allclose(actual, expected, rtol=3e-12, atol=0.0)
''',
)

append_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    "test_stage_c_torch_cpu_two_way_preserves_third_magnitude_component",
    r'''
def test_stage_c_torch_cpu_two_way_preserves_third_magnitude_component():
    amplitude = 2.0 ** 660
    middle = 2.0 ** 600
    tiny = 2.0 ** 350
    scores_np = np.asarray(
        [
            -amplitude, middle, tiny, amplitude, -middle, -tiny,
            -amplitude, -middle, amplitude, middle,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        dtype=np.float64,
    )
    cluster1 = np.asarray(
        [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    cluster2 = np.asarray(
        [0, 1, 1, 0, 1, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    X = torch.full((16, 1), 0.5, dtype=torch.float64)
    resid = torch.as_tensor(8.0 * scores_np, dtype=torch.float64)
    actual = two_way_clustered_covariance(
        X, resid, cluster1, cluster2, xp=torch
    ).detach().cpu().numpy()
    expected = np.asarray([[-4.0 * amplitude * tiny]], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=0.0)
''',
)

runner = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = runner.read_text(encoding="utf-8")
old = r'''    unsafe_expected = np.asarray(
        [[4.0 * unsafe_amplitude * (unsafe_low2 - unsafe_low1)]],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(
'''
new = r'''    unsafe_expected = np.asarray(
        [[4.0 * unsafe_amplitude * (unsafe_low2 - unsafe_low1)]],
        dtype=np.float64,
    )

    tier_amplitude = 2.0 ** 660
    tier_middle = 2.0 ** 600
    tier_tiny = 2.0 ** 350
    tier_scores_np = np.asarray(
        [
            -tier_amplitude, tier_middle, tier_tiny,
            tier_amplitude, -tier_middle, -tier_tiny,
            -tier_amplitude, -tier_middle, tier_amplitude, tier_middle,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        dtype=np.float64,
    )
    tier_c1 = np.asarray(
        [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    tier_c2 = np.asarray(
        [0, 1, 1, 0, 1, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    tier_X_np = np.full((16, 1), 0.5, dtype=np.float64)
    tier_dummy = np.arange(16, dtype=np.int64)
    tier_X, tier_resid, _te, _tt = _to_backend(
        tier_X_np,
        8.0 * tier_scores_np,
        tier_dummy,
        tier_dummy,
        backend,
    )
    tier_two_way = _array(
        two_way_clustered_covariance(
            tier_X, tier_resid, tier_c1, tier_c2, xp=xp
        )
    )
    tier_expected = np.asarray(
        [[-4.0 * tier_amplitude * tier_tiny]], dtype=np.float64
    )

    np.testing.assert_array_equal(
'''
if "tier_two_way = _array(" not in text:
    if old not in text:
        raise RuntimeError("physical tier fixture insertion anchor not found")
    text = text.replace(old, new, 1)
old = r'''    np.testing.assert_allclose(
        unsafe_two_way, unsafe_expected, rtol=4e-12, atol=0.0
    )
    return {
'''
new = r'''    np.testing.assert_allclose(
        unsafe_two_way, unsafe_expected, rtol=4e-12, atol=0.0
    )
    np.testing.assert_allclose(
        tier_two_way, tier_expected, rtol=3e-12, atol=0.0
    )
    return {
'''
if "tier_two_way, tier_expected" not in text:
    if old not in text:
        raise RuntimeError("physical tier assertion anchor not found")
    text = text.replace(old, new, 1)
old = r'''        "unsafe_cross_two_way": unsafe_two_way.tolist(),
    }
'''
new = r'''        "unsafe_cross_two_way": unsafe_two_way.tolist(),
        "third_tier_two_way": tier_two_way.tolist(),
    }
'''
if '"third_tier_two_way"' not in text:
    if old not in text:
        raise RuntimeError("physical tier return anchor not found")
    text = text.replace(old, new, 1)
runner.write_text(text, encoding="utf-8")

contract = Path("dev/tests/test_panel_stage_c_physical_runner_contract.py")
text = contract.read_text(encoding="utf-8")
anchor = r'''    np.testing.assert_allclose(
        np.asarray(audit["unsafe_cross_two_way"]),
        np.asarray(
            [[4.0 * unsafe_amplitude * (unsafe_low2 - unsafe_low1)]],
            dtype=np.float64,
        ),
        rtol=4e-12,
        atol=0.0,
    )
'''
addition = anchor + r'''    tier_amplitude = 2.0 ** 660
    tier_tiny = 2.0 ** 350
    np.testing.assert_allclose(
        np.asarray(audit["third_tier_two_way"]),
        np.asarray([[-4.0 * tier_amplitude * tier_tiny]], dtype=np.float64),
        rtol=3e-12,
        atol=0.0,
    )
'''
if 'audit["third_tier_two_way"]' not in text:
    if anchor not in text:
        raise RuntimeError("physical runner contract tier anchor not found")
    text = text.replace(anchor, addition, 1)
contract.write_text(text, encoding="utf-8")

for stale in (
    "dev/validation/pr126_review_fix_multicomponent_once.py",
    ".github/workflows/pr126-review-fix-multicomponent.yml",
):
    Path(stale).unlink(missing_ok=True)
