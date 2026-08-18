from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"replace anchor not found in {path}: {old[:160]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, block: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if marker not in text:
        p.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


linalg = Path("statgpu/panel/_linalg.py")
text = linalg.read_text(encoding="utf-8")
anchor = "\n\ndef panel_svd_pseudoinverse(X, xp):\n"
helper = r'''

def panel_working_pseudoinverse(X, xp):
    """Return a working-design pseudoinverse with a certified Gram fast path.

    The SVD remains the authoritative fallback and rank policy.  For designs
    whose Gram matrix is range-safe and whose eigenvalue ratio is far above the
    SVD rank boundary, the existing ``_GRAM_CERTIFIED_MIN_EIGEN_RATIO`` policy
    certifies a normal-equation solve.  Besides avoiding an unnecessary SVD,
    this preserves exact row symmetries (for example a constant binary-exact
    design) that can otherwise be perturbed at the U-vector rounding level and
    then amplified by extreme but finite residuals.
    """
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    if int(X.shape[-1]) == 0:
        raise ValueError("panel design must contain at least one column")

    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    n = max(1, int(X_work.shape[0]))
    k = int(X_work.shape[1])
    namespace = getattr(xp, "__name__", "")

    # Certify the Gram path only when forming X'X itself has a conservative
    # factor-four range margin.  Extreme designs stay on the SVD path without
    # ever materializing an overflowing Gram matrix.
    max_abs = xp.max(xp.abs(X_work))
    gram_limit = float(
        np.sqrt(0.25 * np.finfo(np.float64).max / float(n))
    )
    gram_range_safe = bool(
        _to_float_scalar(
            xp.isfinite(max_abs) & (max_abs <= float(gram_limit))
        )
    )

    if gram_range_safe:
        gram = X_work.T @ X_work
        gram_finite = bool(_to_float_scalar(xp.all(xp.isfinite(gram))))
        if gram_finite:
            eigenvalues = xp.linalg.eigvalsh(gram)
            smallest = eigenvalues[0]
            largest = eigenvalues[-1]
            certified = bool(
                _to_float_scalar(
                    xp.isfinite(smallest)
                    & xp.isfinite(largest)
                    & (largest > 0.0)
                    & (
                        smallest
                        > largest * float(_GRAM_CERTIFIED_MIN_EIGEN_RATIO)
                    )
                )
            )
            if certified:
                rhs = X_work.T
                if namespace == "torch" and hasattr(xp.linalg, "solve_ex"):
                    X_pinv_work, info = xp.linalg.solve_ex(
                        gram, rhs, check_errors=False
                    )
                    solve_ok = bool(
                        _to_float_scalar(
                            (info == 0) & xp.all(xp.isfinite(X_pinv_work))
                        )
                    )
                    if solve_ok:
                        return X_work, X_pinv_work, design_scale, k
                else:
                    X_pinv_work = xp.linalg.solve(gram, rhs)
                    if bool(
                        _to_float_scalar(xp.all(xp.isfinite(X_pinv_work)))
                    ):
                        return X_work, X_pinv_work, design_scale, k

    # Uncertified, range-risky, or failed certified solves retain the shared SVD
    # cutoff exactly; the Gram path never expands the accepted rank region.
    U, Vh, inverse_values, rank = _svd_inverse_factors(X_work, xp)
    X_pinv_work = (Vh.T * inverse_values) @ U.T
    return X_work, X_pinv_work, design_scale, rank
'''
if "def panel_working_pseudoinverse(" not in text:
    if anchor not in text:
        raise RuntimeError("panel working pseudoinverse insertion anchor not found")
    text = text.replace(anchor, helper + anchor, 1)
linalg.write_text(text, encoding="utf-8")

replace_once(
    "statgpu/panel/_covariance.py",
    "    panel_svd_pseudoinverse,\n    panel_svd_working_pseudoinverse,\n",
    "    panel_svd_pseudoinverse,\n    panel_svd_working_pseudoinverse,\n    panel_working_pseudoinverse,\n",
)
replace_once(
    "statgpu/panel/_covariance.py",
    "    X_work, X_pinv_work, design_scale, rank = panel_svd_working_pseudoinverse(\n        X, xp\n    )\n",
    "    X_work, X_pinv_work, design_scale, rank = panel_working_pseudoinverse(\n        X, xp\n    )\n",
)
replace_once(
    "dev/tests/test_panel_stage_c_covariance.py",
    "    _grouped_score_sums,\n",
    "    _grouped_score_sums,\n    _influence_rows,\n",
)
append_once(
    "dev/tests/test_panel_stage_c_covariance.py",
    "test_influence_rows_certified_gram_preserves_constant_design_symmetry",
    r'''
def test_influence_rows_certified_gram_preserves_constant_design_symmetry():
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
    X = np.full((16, 1), 0.5, dtype=np.float64)
    influence, projection_scale, design_scale, *_ = _influence_rows(
        X, 8.0 * scores, np
    )
    np.testing.assert_array_equal(influence[:, 0], scores)
    np.testing.assert_array_equal(projection_scale, np.ones(1))
    assert float(np.asarray(design_scale)) == 1.0
''',
)

Path("dev/validation/pr126_review_fix_certified_projection_once.py").unlink(missing_ok=True)
