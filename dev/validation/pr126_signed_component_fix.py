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
    '''def _dyadic_float_terms(value: float):\n    """Return an exact finite-float coefficient as signed powers of two."""\n    value = float(value)\n    if not np.isfinite(value):\n        raise ValueError("expansion coefficient must be finite")\n    if value == 0.0:\n        return []\n    sign = -1.0 if value < 0.0 else 1.0\n    numerator, denominator = abs(value).as_integer_ratio()\n    denominator_shift = int(denominator).bit_length() - 1\n    powers = []\n    bit = 0\n    while numerator:\n        if numerator & 1:\n            powers.append(sign * (2.0 ** (bit - denominator_shift)))\n        numerator >>= 1\n        bit += 1\n    powers.reverse()\n    return powers\n\n\ndef _two_sum_matrix(left, right):\n    summed = left + right\n    virtual_right = summed - left\n    error = (left - (summed - virtual_right)) + (right - virtual_right)\n    return summed, error\n\n\ndef _twofold_matrix_sum(terms, xp):\n    """Accumulate row-level matrix terms with two compensated error streams."""\n    if not terms:\n        raise ValueError("at least one matrix term is required")\n    total = xp.zeros_like(terms[0])\n    correction = xp.zeros_like(total)\n    residual_correction = xp.zeros_like(total)\n    for term in terms:\n        total, error = _two_sum_matrix(total, term)\n        correction, second_error = _two_sum_matrix(correction, error)\n        residual_correction, third_error = _two_sum_matrix(\n            residual_correction, second_error\n        )\n        residual_correction = residual_correction + third_error\n    return _stable_matrix_expansion_sum(\n        [residual_correction, correction, total], xp\n    )\n\n\ndef _stable_matrix_expansion_sum(terms, xp):\n''',
)

replace_once(
    "statgpu/panel/_covariance.py",
    '''            terms = []\n\n            def _append_component_terms(components, correction, sign):\n                coefficient = float(sign) * float(correction)\n                for i, left in enumerate(components):\n                    terms.append(\n                        _symmetrize(left.T @ left) * coefficient\n                    )\n                    for right in components[:i]:\n                        cross = left.T @ right\n                        terms.append((cross + cross.T) * coefficient)\n\n            # Every component is now certified safe for its own row reduction.\n            # Keep the full CGM expansion across magnitude tiers, but perform\n            # each component-pair reduction as one BLAS/GPU matrix product rather\n            # than one Python-level outer-product launch per cluster row.\n            _append_component_terms(work1, correction1, 1.0)\n            _append_component_terms(work2, correction2, 1.0)\n            _append_component_terms(work12, correction12, -1.0)\n            cov_work = _stable_matrix_expansion_sum(terms, xp)\n''',
    '''            terms = []\n\n            def _append_scaled_row_term(term, coefficient):\n                for dyadic in _dyadic_float_terms(coefficient):\n                    terms.append(term * float(dyadic))\n\n            def _append_component_terms(components, correction, sign):\n                coefficient = float(sign) * float(correction)\n                for i, left in enumerate(components):\n                    for row in range(int(left.shape[0])):\n                        vector = left[row]\n                        outer = vector[:, None] * vector[None, :]\n                        _append_scaled_row_term(outer, coefficient)\n                    for right in components[:i]:\n                        for row in range(int(left.shape[0])):\n                            cross = left[row, :, None] * right[row, None, :]\n                            _append_scaled_row_term(cross + cross.T, coefficient)\n\n            # This branch is already the rare multi-tier precision fallback.\n            # Keep row products out of signed BLAS reductions and decompose float\n            # debias corrections into exact dyadic factors; then accumulate all\n            # estimator-level terms with two compensated error streams. Ordinary\n            # single-tier/vectorized covariance paths are unchanged.\n            _append_component_terms(work1, correction1, 1.0)\n            _append_component_terms(work2, correction2, 1.0)\n            _append_component_terms(work12, correction12, -1.0)\n            cov_work = _twofold_matrix_sum(terms, xp)\n''',
)

replace_once(
    "statgpu/panel/_covariance.py",
    '''    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()\n    if X.ndim != 2 or resid.shape[0] != X.shape[0]:\n        raise ValueError("X and resid must have matching observation counts")\n    n = int(X.shape[0])\n\n    if metadata is not None:\n        metadata.clear()\n''',
    '''    resid = xp_asarray(resid, dtype=xp.float64, xp=xp, ref_arr=X).ravel()\n    if X.ndim != 2 or resid.shape[0] != X.shape[0]:\n        raise ValueError("X and resid must have matching observation counts")\n    _validate_covariance_finite_inputs(X, resid, xp)\n    n = int(X.shape[0])\n\n    if metadata is not None:\n        metadata.clear()\n''',
)

replace_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    '''def test_stage_c_torch_cpu_two_way_preserves_third_magnitude_component():\n    amplitude = 2.0 ** 660\n''',
    '''def test_stage_c_torch_cpu_two_way_preserves_third_magnitude_component():\n    # This case must be order-independent. The full maintained Torch file once\n    # exposed an Inf here after earlier multi-tier covariance reductions even\n    # though the same case passed in isolation. The rare row-level compensated\n    # fallback keeps estimator cancellation ahead of backend reduction rounding.\n    amplitude = 2.0 ** 660\n''',
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
    '''Two-way clustering combines grouped components before restoration and detects nested dimensions by partition equivalence rather than arbitrary code numbering. In the rare multi-tier nonnested precision fallback, grouped row outer products remain separate terms, float group-debias corrections are decomposed into exact power-of-two factors, and two compensated error streams preserve magnitude-tier/CGM/debias cancellation without changing ordinary vectorized paths.''',
)
replace_once(
    "CHANGELOG.md",
    '''Public clustered, two-way clustered, HAC, and Driscoll-Kraay helpers now reject non-finite `X`/residual inputs before signed/group reductions, preventing NaN/Inf scores from being silently reinterpreted as zero contributions.''',
    '''Public `ols_covariance`, clustered, two-way clustered, HAC, and Driscoll-Kraay helpers now reject non-finite `X`/residual inputs before numerical reductions, preventing NaN/Inf scores from being silently ignored, propagated into published covariance, or reinterpreted as zero contributions.''',
)
