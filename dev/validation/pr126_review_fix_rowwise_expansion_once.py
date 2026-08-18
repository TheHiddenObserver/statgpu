from pathlib import Path

p = Path("statgpu/panel/_covariance.py")
text = p.read_text(encoding="utf-8")

old_count = '''            term_count = sum(
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
'''
new_count = '''            component_sets = (components1, components2, components12)
            max_rows = max(
                int(components[0].shape[0]) for components in component_sets
            )
            # Count scalar products rather than matrix terms: an off-diagonal
            # expansion pair contributes both u v' and v u'.  The existing
            # common-Gram scaler already multiplies its bound by max_rows, so
            # convert the total product count to an equivalent per-max-row
            # multiplier and keep a factor-of-two margin for intermediate sums.
            product_count = sum(
                int(components[0].shape[0]) * (len(components) ** 2)
                for components in component_sets
            )
            max_correction = max(correction1, correction2, correction12)
            product_multiplier = (
                2.0
                * float(max(1, product_count))
                / float(max(1, max_rows))
                * float(max_correction)
            )
            working_components, common_scale = _common_gram_working_values(
                all_components,
                xp,
                max_multiplier=product_multiplier,
            )
'''
if old_count not in text:
    raise RuntimeError("multicomponent scaling-count anchor not found")
text = text.replace(old_count, new_count, 1)

old_terms = '''            terms = []

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
new_terms = '''            terms = []

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
if old_terms not in text:
    raise RuntimeError("multicomponent Gram-term anchor not found")
text = text.replace(old_terms, new_terms, 1)
p.write_text(text, encoding="utf-8")

Path("dev/validation/pr126_review_fix_rowwise_expansion_once.py").unlink(missing_ok=True)
