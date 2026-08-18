from pathlib import Path


def replace_between(path, start_marker, end_marker, replacement):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError(f"replace_between anchors not found in {path}")
    p.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def insert_once(path, marker, block):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    signature = block.strip().splitlines()[0]
    if signature in text:
        return
    pos = text.find(marker)
    if pos < 0:
        raise RuntimeError(f"insert anchor not found in {path}")
    p.write_text(text[:pos] + block + text[pos:], encoding="utf-8")


def append_once(path, marker, block):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if marker not in text:
        p.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


linalg_replacement = r'''def panel_working_pseudoinverse(X, xp):
    """Return a working-design pseudoinverse with one-sync Gram certification.

    The Gram candidate is uniformly scaled before ``X'X`` is formed, so the
    certification calculation itself is range-safe without a host-side branch.
    All finite/spectrum/solve checks remain backend-native until one final
    boolean transfer.  Certified full-rank designs use the normal-equation
    candidate; every uncertified case falls back to the shared SVD cutoff.
    """
    if getattr(X, "ndim", None) != 2:
        raise ValueError("panel design must be two-dimensional")
    if int(X.shape[-1]) == 0:
        raise ValueError("panel design must contain at least one column")

    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    n = max(1, int(X_work.shape[0]))
    k = int(X_work.shape[1])
    namespace = getattr(xp, "__name__", "")

    max_abs = xp.max(xp.abs(X_work))
    gram_limit = float(
        np.sqrt(0.25 * np.finfo(np.float64).max / float(n))
    )
    one = xp.ones_like(max_abs)
    gram_scale = xp.where(
        max_abs > float(gram_limit),
        max_abs / float(gram_limit),
        one,
    )
    X_gram = X_work / gram_scale
    gram = X_gram.T @ X_gram
    rhs = X_gram.T
    gram_finite = xp.all(xp.isfinite(gram))
    rhs_finite = xp.all(xp.isfinite(rhs))

    if namespace == "torch":
        identity = xp.eye(k, dtype=X.dtype, device=X.device)
    else:
        identity = xp.eye(k, dtype=X.dtype)
    spectrum_gram = xp.where(gram_finite, gram, identity)
    eigenvalues = xp.linalg.eigvalsh(spectrum_gram)
    smallest = eigenvalues[0]
    largest = eigenvalues[-1]
    certified = (
        gram_finite
        & rhs_finite
        & xp.isfinite(smallest)
        & xp.isfinite(largest)
        & (largest > 0.0)
        & (
            smallest
            > largest * float(_GRAM_CERTIFIED_MIN_EIGEN_RATIO)
        )
    )

    safe_gram = xp.where(certified, gram, identity)
    safe_rhs = xp.where(certified, rhs, xp.zeros_like(rhs))
    if namespace == "torch" and hasattr(xp.linalg, "solve_ex"):
        candidate, info = xp.linalg.solve_ex(
            safe_gram, safe_rhs, check_errors=False
        )
        certified = certified & (info == 0)
    else:
        candidate = xp.linalg.solve(safe_gram, safe_rhs)
    candidate = candidate / gram_scale
    certified = certified & xp.all(xp.isfinite(candidate))

    # This is the only certification transfer on the accepted Gram path.
    if bool(_to_float_scalar(certified)):
        return X_work, candidate, design_scale, k

    U, Vh, inverse_values, rank = _svd_inverse_factors(X_work, xp)
    X_pinv_work = (Vh.T * inverse_values) @ U.T
    return X_work, X_pinv_work, design_scale, rank


'''
replace_between(
    "statgpu/panel/_linalg.py",
    "def panel_working_pseudoinverse(X, xp):\n",
    "def panel_svd_pseudoinverse(X, xp):\n",
    linalg_replacement,
)

risk_helper = r'''
def _component_row_reduction_needs_expansion(component_sets, xp) -> bool:
    """Return whether a BLAS row reduction can hide a recoverable component.

    Component-pair Grams are vectorized whenever every nonzero grouped value is
    comfortably above the roundoff floor induced by the largest value in the
    same coordinate.  A self-product squares the value ratio, so the relevant
    value threshold is ``sqrt(n * eps)`` rather than ``n * eps``.  A factor-16
    margin covers reduction order and the later inclusion-exclusion.  Only the
    rare high-dynamic-range case falls back to explicit row outer products.
    """
    risk = None
    eps = float(np.finfo(np.float64).eps)
    for components in component_sets:
        for component in components:
            n_rows = max(1, int(component.shape[0]))
            absolute = xp.abs(component)
            max_abs = _column_abs_max(component, xp)
            sentinel = xp.full_like(absolute, float(np.inf))
            nonzero = xp.where(absolute > 0.0, absolute, sentinel)
            if _is_torch(xp):
                min_nonzero = xp.min(nonzero, dim=0).values
            else:
                min_nonzero = xp.min(nonzero, axis=0)
            ratio_floor = float(np.sqrt(16.0 * float(n_rows) * eps))
            local = xp.any(
                xp.isfinite(min_nonzero)
                & (min_nonzero < max_abs * ratio_floor)
            )
            risk = local if risk is None else (risk | local)
    if risk is None:
        return False
    return bool(_to_float_scalar(risk))


'''
insert_once(
    "statgpu/panel/_covariance.py",
    "def _grouped_score_sums(\n",
    risk_helper,
)

p = Path("statgpu/panel/_covariance.py")
text = p.read_text(encoding="utf-8")
old = r'''            lost_component = False
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
'''
new = r'''            lost_component_backend = None
            for original, working in zip(all_components, working_components):
                local_loss = xp.any(
                    (original != 0.0) & (working == 0.0)
                )
                lost_component_backend = (
                    local_loss
                    if lost_component_backend is None
                    else (lost_component_backend | local_loss)
                )
            if bool(_to_float_scalar(lost_component_backend)):
                raise FloatingPointError(
                    "two-way cluster score expansion exceeds the float64 "
                    "common-scale dynamic range"
                )
'''
if old not in text:
    raise RuntimeError("lost-component anchor not found")
text = text.replace(old, new, 1)

old = r'''            terms = []

            def _append_component_terms(components, correction, sign):
                coefficient = float(sign) * float(correction)
                n_rows = int(components[0].shape[0])
                for i, left in enumerate(components):
                    for row in range(n_rows):
                        left_row = left[row]
                        terms.append(
                            (left_row[:, None] * left_row[None, :])
                            * coefficient
                        )
                    for right in components[:i]:
                        for row in range(n_rows):
                            left_row = left[row]
                            right_row = right[row]
                            cross = left_row[:, None] * right_row[None, :]
                            terms.append(
                                (cross + cross.T) * coefficient
                            )

            # Do not reduce over group rows with a BLAS Gram here.  A component
            # may contain, for example, A in one group and b in another; A^2+b^2
            # would round away b^2 before a later +V1+V2-V12 cancellation removes
            # A^2.  Keep every row outer product as an expansion term until the
            # complete CGM expression has cancelled structurally.
            _append_component_terms(work1, correction1, 1.0)
            _append_component_terms(work2, correction2, 1.0)
            _append_component_terms(work12, correction12, -1.0)
            cov_work = _stable_matrix_expansion_sum(terms, xp)
'''
new = r'''            terms = []
            rowwise_expansion = _component_row_reduction_needs_expansion(
                (components1, components2, components12), xp
            )

            def _append_component_terms(components, correction, sign):
                coefficient = float(sign) * float(correction)
                if not rowwise_expansion:
                    for i, left in enumerate(components):
                        terms.append(
                            _symmetrize(left.T @ left) * coefficient
                        )
                        for right in components[:i]:
                            cross = left.T @ right
                            terms.append((cross + cross.T) * coefficient)
                    return

                n_rows = int(components[0].shape[0])
                for i, left in enumerate(components):
                    for row in range(n_rows):
                        left_row = left[row]
                        terms.append(
                            (left_row[:, None] * left_row[None, :])
                            * coefficient
                        )
                    for right in components[:i]:
                        for row in range(n_rows):
                            left_row = left[row]
                            right_row = right[row]
                            cross = left_row[:, None] * right_row[None, :]
                            terms.append(
                                (cross + cross.T) * coefficient
                            )

            # Ordinary grouped scores keep all compensation components but use
            # vectorized component-pair Grams.  Explicit group-row outer products
            # are reserved for a certified dynamic-range risk where BLAS could
            # erase a small row contribution before CGM cancellation exposes it.
            _append_component_terms(work1, correction1, 1.0)
            _append_component_terms(work2, correction2, 1.0)
            _append_component_terms(work12, correction12, -1.0)
            cov_work = _stable_matrix_expansion_sum(terms, xp)
'''
if old not in text:
    raise RuntimeError("rowwise expansion anchor not found")
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# Regression: normal-scale nonnested data may have a nonzero TwoSum residual,
# but must not launch one outer-product kernel per group row.
test_path = Path("dev/tests/test_panel_stage_c_covariance.py")
test_text = test_path.read_text(encoding="utf-8")
if "import statgpu.panel._covariance as covariance_module" not in test_text:
    test_text = test_text.replace(
        "import pytest\n",
        "import pytest\n\nimport statgpu.panel._covariance as covariance_module\n",
        1,
    )
test_path.write_text(test_text, encoding="utf-8")
append_once(
    "dev/tests/test_panel_stage_c_covariance.py",
    "test_two_way_ordinary_compensation_stays_vectorized",
    r'''
def test_two_way_ordinary_compensation_stays_vectorized(monkeypatch):
    rng = np.random.default_rng(126)
    scores = rng.normal(size=32)
    cluster1 = np.repeat(np.arange(4), 8)
    cluster2 = np.tile(np.arange(8), 4)
    X = np.ones((32, 1), dtype=np.float64)
    resid = 32.0 * scores

    components = _grouped_score_sums(
        scores[:, None], cluster1, n_groups=4, xp=np,
        return_components=True,
    )
    assert any(np.any(component != 0.0) for component in components[1:])

    observed_term_counts = []
    original = covariance_module._stable_matrix_expansion_sum

    def wrapped(terms, xp):
        observed_term_counts.append(len(terms))
        return original(terms, xp)

    monkeypatch.setattr(
        covariance_module, "_stable_matrix_expansion_sum", wrapped
    )
    covariance_module.two_way_clustered_covariance(
        X, resid, cluster1, cluster2
    )
    assert observed_term_counts
    # Two components in each of three cluster dimensions require only
    # 3 vectorized component-pair terms per dimension, not O(group-count) terms.
    assert max(observed_term_counts) <= 9
''',
)

for stale in (
    "dev/validation/pr126_review_fix_perf_once.py",
    ".github/workflows/pr126-review-fix-perf.yml",
):
    Path(stale).unlink(missing_ok=True)
