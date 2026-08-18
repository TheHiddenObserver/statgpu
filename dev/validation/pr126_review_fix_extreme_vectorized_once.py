from pathlib import Path


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


retier_helper = r'''
def _retier_component_for_safe_gram(component, xp):
    """Split one grouped component into globally Gram-safe magnitude tiers.

    Local group summation tiers protect cancellation *within* each cluster.  A
    later Gram reduction also sums across cluster rows, so singleton/intersection
    groups can still place vastly different magnitudes in the same component.
    Split those rows by a per-coordinate ``sqrt(n * eps)`` threshold before the
    BLAS reduction.  Each returned tier can then use a vectorized Gram without
    erasing a smaller row contribution that CGM cancellation may later expose.
    """
    n_rows = max(1, int(component.shape[0]))
    ratio_floor = float(
        np.sqrt(16.0 * float(n_rows) * float(np.finfo(np.float64).eps))
    )
    remaining = component
    tiers = []
    for _ in range(128):
        max_abs = _column_abs_max(remaining, xp)
        threshold = max_abs * ratio_floor
        tail_mask = (
            (xp.abs(remaining) < threshold[None, :])
            & (remaining != 0.0)
        )
        tiers.append(
            xp.where(tail_mask, xp.zeros_like(remaining), remaining)
        )
        if not bool(_to_float_scalar(xp.any(tail_mask))):
            break
        remaining = xp.where(
            tail_mask, remaining, xp.zeros_like(remaining)
        )
    else:
        raise RuntimeError(
            "two-way cluster global magnitude tiers exceeded the float64 budget"
        )
    return tiers


def _retier_component_sets_for_safe_gram(component_sets, xp):
    refined = []
    for components in component_sets:
        current = []
        for component in components:
            current.extend(_retier_component_for_safe_gram(component, xp))
        refined.append(current)
    return tuple(refined)


'''
insert_once(
    "statgpu/panel/_covariance.py",
    "def _grouped_score_sums(\n",
    retier_helper,
)

p = Path("statgpu/panel/_covariance.py")
text = p.read_text(encoding="utf-8")
old = r'''        else:
            all_components = components1 + components2 + components12
            component_sets = (components1, components2, components12)
            max_rows = max(
                int(components[0].shape[0]) for components in component_sets
            )
'''
new = r'''        else:
            component_sets = (components1, components2, components12)
            if _component_row_reduction_needs_expansion(component_sets, xp):
                component_sets = _retier_component_sets_for_safe_gram(
                    component_sets, xp
                )
                components1, components2, components12 = component_sets

            all_components = components1 + components2 + components12
            max_rows = max(
                int(components[0].shape[0]) for components in component_sets
            )
'''
if old not in text:
    raise RuntimeError("component-set retiering anchor not found")
text = text.replace(old, new, 1)

old = r'''            terms = []
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
new = r'''            terms = []

            def _append_component_terms(components, correction, sign):
                coefficient = float(sign) * float(correction)
                for i, left in enumerate(components):
                    terms.append(
                        _symmetrize(left.T @ left) * coefficient
                    )
                    for right in components[:i]:
                        cross = left.T @ right
                        terms.append((cross + cross.T) * coefficient)

            # Every component is now certified safe for its own row reduction.
            # Keep the full CGM expansion across magnitude tiers, but perform
            # each component-pair reduction as one BLAS/GPU matrix product rather
            # than one Python-level outer-product launch per cluster row.
            _append_component_terms(work1, correction1, 1.0)
            _append_component_terms(work2, correction2, 1.0)
            _append_component_terms(work12, correction12, -1.0)
            cov_work = _stable_matrix_expansion_sum(terms, xp)
'''
if old not in text:
    raise RuntimeError("rowwise-to-tiered vectorization anchor not found")
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

append_once(
    "dev/tests/test_panel_stage_c_covariance.py",
    "test_two_way_extreme_many_groups_retiers_before_vectorized_gram",
    r'''
def test_two_way_extreme_many_groups_retiers_before_vectorized_gram(monkeypatch):
    n = 256
    large = 2.0 ** 500
    small = 2.0 ** 400
    scores = np.where(np.arange(n) % 2 == 0, large, small).astype(np.float64)
    cluster1 = np.repeat(np.arange(16), 16)
    cluster2 = np.tile(np.arange(16), 16)
    X = np.ones((n, 1), dtype=np.float64)
    resid = float(n) * scores

    observed_term_counts = []
    original = covariance_module._stable_matrix_expansion_sum

    def wrapped(terms, xp):
        observed_term_counts.append(len(terms))
        return original(terms, xp)

    monkeypatch.setattr(
        covariance_module, "_stable_matrix_expansion_sum", wrapped
    )
    actual = covariance_module.two_way_clustered_covariance(
        X, resid, cluster1, cluster2
    )
    assert np.all(np.isfinite(actual))
    assert observed_term_counts
    # The fallback complexity is tied to magnitude tiers, not 256 intersection
    # rows.  This fixture previously produced roughly one thousand row terms.
    assert max(observed_term_counts) <= 30
''',
)

for stale in (
    "dev/validation/pr126_review_fix_extreme_vectorized_once.py",
    ".github/workflows/pr126-review-fix-extreme-vectorized.yml",
):
    Path(stale).unlink(missing_ok=True)
