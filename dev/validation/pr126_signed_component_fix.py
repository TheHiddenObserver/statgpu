from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"anchor not found in {path}: {old[:240]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, addition: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if marker in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    p.write_text(text + "\n" + addition.strip() + "\n", encoding="utf-8")


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

# ols_covariance is itself public.  Apply the same finite-input contract at its
# dispatch boundary so nonrobust/HC/robust calls cannot bypass the direct helper
# guards (nonrobust can otherwise ignore a NaN residual when a scale is supplied).
replace_once(
    "statgpu/panel/_covariance.py",
    '''    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()\n    if X.ndim != 2 or resid.shape[0] != X.shape[0]:\n        raise ValueError("X and resid must have matching observation counts")\n    n = int(X.shape[0])\n\n    if metadata is not None:\n        metadata.clear()\n''',
    '''    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()\n    if X.ndim != 2 or resid.shape[0] != X.shape[0]:\n        raise ValueError("X and resid must have matching observation counts")\n    _validate_covariance_finite_inputs(X, resid, xp)\n    n = int(X.shape[0])\n\n    if metadata is not None:\n        metadata.clear()\n''',
)

# Strengthen the existing Torch regression so the failure remains localized even
# if collection order changes.
replace_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    '''def test_stage_c_torch_cpu_two_way_preserves_third_magnitude_component():\n    amplitude = 2.0 ** 660\n''',
    '''def test_stage_c_torch_cpu_two_way_preserves_third_magnitude_component():\n    # This case must be order-independent.  The full maintained Torch file once\n    # exposed an Inf here after earlier multi-tier covariance reductions even\n    # though the same case passed in isolation.  Sign-separated component-pair\n    # reductions make the result independent of BLAS signed cancellation order.\n    amplitude = 2.0 ** 660\n''',
)

append_once(
    "dev/tests/test_panel_stage_c_edge_contracts.py",
    "test_public_ols_covariance_rejects_nonfinite_inputs_numpy",
    r'''
def test_public_ols_covariance_rejects_nonfinite_inputs_numpy():
    from statgpu.panel import ols_covariance

    X = np.column_stack([np.ones(6), np.arange(6.0)])
    resid = np.linspace(-0.3, 0.4, 6)
    resid[2] = np.nan
    with pytest.raises(ValueError, match="X and resid must contain only finite values"):
        ols_covariance(X, resid, cov_type="nonrobust", scale=1.0)

    X_bad = X.copy()
    X_bad[1, 1] = np.inf
    with pytest.raises(ValueError, match="X and resid must contain only finite values"):
        ols_covariance(X_bad, np.zeros(6), cov_type="hc0")
''',
)

append_once(
    "dev/tests/test_panel_stage_c_torch_cpu.py",
    "test_stage_c_public_ols_covariance_rejects_nonfinite_torch_cpu",
    r'''
def test_stage_c_public_ols_covariance_rejects_nonfinite_torch_cpu():
    from statgpu.panel import ols_covariance

    X = torch.column_stack(
        [torch.ones(6, dtype=torch.float64), torch.arange(6, dtype=torch.float64)]
    )
    resid = torch.linspace(-0.3, 0.4, 6, dtype=torch.float64)
    resid[2] = float("nan")
    with pytest.raises(ValueError, match="X and resid must contain only finite values"):
        ols_covariance(X, resid, cov_type="nonrobust", scale=1.0)
''',
)

replace_once(
    "CHANGELOG.md",
    '''Two-way clustering combines grouped components before restoration and detects nested dimensions by partition equivalence rather than arbitrary code numbering.''',
    '''Two-way clustering combines grouped components before restoration and detects nested dimensions by partition equivalence rather than arbitrary code numbering. In the rare multi-tier nonnested path, component-pair row products are accumulated with positive/negative signs separated before BLAS reduction, eliminating backend/order-dependent ulp residuals that could overflow only after the physical covariance scale was restored.''',
)
replace_once(
    "CHANGELOG.md",
    '''Public clustered, two-way clustered, HAC, and Driscoll-Kraay helpers now reject non-finite `X`/residual inputs before signed/group reductions, preventing NaN/Inf scores from being silently reinterpreted as zero contributions.''',
    '''Public `ols_covariance`, clustered, two-way clustered, HAC, and Driscoll-Kraay helpers now reject non-finite `X`/residual inputs before numerical reductions, preventing NaN/Inf scores from being silently ignored, propagated into published covariance, or reinterpreted as zero contributions.''',
)
