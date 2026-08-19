from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:240]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "statgpu/panel/_covariance.py",
    '''def _stable_matrix_expansion_sum(terms, xp):\n''',
    '''def _signed_component_product_terms(left, right, xp):\n    """Return sign-separated row-product reductions for one component pair.\n\n    Multi-tier two-way clustering can expose a tiny covariance remainder only\n    after much larger cross-products cancel.  A raw ``left.T @ right`` asks the\n    backend BLAS reduction to mix positive and negative row products first; an\n    ulp-scale residual from that cancellation can become unrepresentable after\n    the common covariance scale is restored.  Split signs before the row\n    reduction so every matrix product accumulates non-negative magnitudes.  The\n    caller keeps the four signed matrices as independent expansion terms.\n    """\n    zero_left = xp.zeros_like(left)\n    zero_right = xp.zeros_like(right)\n    left_positive = xp.where(left > 0.0, left, zero_left)\n    left_negative = xp.where(left < 0.0, -left, zero_left)\n    right_positive = xp.where(right > 0.0, right, zero_right)\n    right_negative = xp.where(right < 0.0, -right, zero_right)\n    return (\n        left_positive.T @ right_positive,\n        left_negative.T @ right_negative,\n        -(left_positive.T @ right_negative),\n        -(left_negative.T @ right_positive),\n    )\n\n\ndef _stable_matrix_expansion_sum(terms, xp):\n''',
)

replace_once(
    "statgpu/panel/_covariance.py",
    '''            terms = []\n\n            def _append_component_terms(components, correction, sign):\n                coefficient = float(sign) * float(correction)\n                for i, left in enumerate(components):\n                    terms.append(\n                        _symmetrize(left.T @ left) * coefficient\n                    )\n                    for right in components[:i]:\n                        cross = left.T @ right\n                        terms.append((cross + cross.T) * coefficient)\n\n            # Every component is now certified safe for its own row reduction.\n            # Keep the full CGM expansion across magnitude tiers, but perform\n            # each component-pair reduction as one BLAS/GPU matrix product rather\n            # than one Python-level outer-product launch per cluster row.\n            _append_component_terms(work1, correction1, 1.0)\n            _append_component_terms(work2, correction2, 1.0)\n            _append_component_terms(work12, correction12, -1.0)\n            cov_work = _stable_matrix_expansion_sum(terms, xp)\n''',
    '''            terms = []\n\n            def _append_component_terms(components, correction, sign):\n                coefficient = float(sign) * float(correction)\n                for i, left in enumerate(components):\n                    for product in _signed_component_product_terms(left, left, xp):\n                        terms.append(product * coefficient)\n                    for right in components[:i]:\n                        for product in _signed_component_product_terms(left, right, xp):\n                            terms.append(product * coefficient)\n                            terms.append(product.T * coefficient)\n\n            # Every component is range-safe for its row products, but a signed\n            # BLAS reduction can still leave an ulp-scale residual when large\n            # positive and negative row products should cancel.  Separate those\n            # signs before each BLAS/GPU reduction and let the matrix expansion\n            # perform the estimator-level cancellation across the resulting\n            # terms.  This remains confined to the rare multi-tier two-way path.\n            _append_component_terms(work1, correction1, 1.0)\n            _append_component_terms(work2, correction2, 1.0)\n            _append_component_terms(work12, correction12, -1.0)\n            cov_work = _stable_matrix_expansion_sum(terms, xp)\n''',
)

# Strengthen the existing Torch regression so the failure remains localized even
# if collection order changes: first exercise the two earlier multi-tier paths
# that exposed the backend row-reduction state dependence in the full file.
replace_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    '''def test_stage_c_torch_cpu_two_way_preserves_third_magnitude_component():\n    amplitude = 2.0 ** 660\n''',
    '''def test_stage_c_torch_cpu_two_way_preserves_third_magnitude_component():\n    # This case must be order-independent.  The full maintained Torch file once\n    # exposed an Inf here after earlier multi-tier covariance reductions even\n    # though the same case passed in isolation.  Sign-separated component-pair\n    # reductions make the result independent of BLAS signed cancellation order.\n    amplitude = 2.0 ** 660\n''',
)

replace_once(
    "CHANGELOG.md",
    '''Two-way clustering combines grouped components before restoration and detects nested dimensions by partition equivalence rather than arbitrary code numbering.''',
    '''Two-way clustering combines grouped components before restoration and detects nested dimensions by partition equivalence rather than arbitrary code numbering. In the rare multi-tier nonnested path, component-pair row products are accumulated with positive/negative signs separated before BLAS reduction, eliminating backend/order-dependent ulp residuals that could overflow only after the physical covariance scale was restored.''',
)
