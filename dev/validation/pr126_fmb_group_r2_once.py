from pathlib import Path

p = Path('statgpu/panel/_diagnostics.py')
text = p.read_text(encoding='utf-8')
anchor = '''def _scaled_mean(values, xp):
    """Return a backend-native mean without overflowing the raw sum."""
    scale = xp.max(xp.abs(values))
    safe_scale = xp.where(scale > 0.0, scale, xp.ones_like(scale))
    return xp.mean(values / safe_scale) * safe_scale


'''
addition = anchor + '''def _scaled_group_means(values, groups, xp):
    """Return group means after one global scaling to protect group sums."""
    scale = xp.max(xp.abs(values))
    safe_scale = xp.where(scale > 0.0, scale, xp.ones_like(scale))
    return group_means(values / safe_scale, groups, xp=xp) * safe_scale


'''
if anchor not in text:
    raise RuntimeError('scaled mean anchor not found')
text = text.replace(anchor, addition, 1)

old = '''def _demean_matrix(X, entity_codes, xp):
    out = X.clone() if getattr(xp, "__name__", "") == "torch" else X.copy()
    for j in range(int(X.shape[1])):
        out[:, j] = X[:, j] - group_means(X[:, j], entity_codes, xp=xp)
    return out
'''
new = '''def _demean_matrix(X, entity_codes, xp):
    out = X.clone() if getattr(xp, "__name__", "") == "torch" else X.copy()
    for j in range(int(X.shape[1])):
        out[:, j] = X[:, j] - _scaled_group_means(
            X[:, j], entity_codes, xp
        )
    return out
'''
if old not in text:
    raise RuntimeError('diagnostic demean anchor not found')
text = text.replace(old, new, 1)

old = '''    y_mean_aligned = group_means(y, entity_codes, xp=xp)
    X_mean_aligned = X.clone() if getattr(xp, "__name__", "") == "torch" else X.copy()
    for j in range(int(X.shape[1])):
        X_mean_aligned[:, j] = group_means(X[:, j], entity_codes, xp=xp)
'''
new = '''    y_mean_aligned = _scaled_group_means(y, entity_codes, xp)
    X_mean_aligned = X.clone() if getattr(xp, "__name__", "") == "torch" else X.copy()
    for j in range(int(X.shape[1])):
        X_mean_aligned[:, j] = _scaled_group_means(
            X[:, j], entity_codes, xp
        )
'''
if old not in text:
    raise RuntimeError('diagnostic group mean block anchor not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# Extend the maintained Torch/NumPy large finite-level regression to the
# optional entity-aware within/between result surface.
p = Path('dev/tests/test_fama_macbeth_inference_matrix.py')
text = p.read_text(encoding='utf-8')
anchor = '''    assert actual.fit_statistics_.rsquared_overall == 0.0
    assert actual.fit_statistics_.metadata["degenerate_total_ss"]["overall"] is True
    np.testing.assert_allclose(
'''
replacement = '''    assert actual.fit_statistics_.rsquared_overall == 0.0
    assert actual.fit_statistics_.metadata["degenerate_total_ss"]["overall"] is True

    entity_ids = np.tile(np.arange(3, dtype=np.int64), 4)
    entity_model = FamaMacBeth(bandwidth=0).fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=torch.as_tensor(time_ids, dtype=torch.int64),
        entity_ids=torch.as_tensor(entity_ids, dtype=torch.int64),
    )
    stats = entity_model.fit_statistics_
    assert stats.rsquared_overall == 0.0
    assert stats.rsquared_between == 0.0
    assert stats.rsquared_within == 0.0
    assert stats.metadata["degenerate_total_ss"] == {
        "within": True,
        "between": True,
        "overall": True,
    }
    np.testing.assert_allclose(
'''
if anchor not in text:
    raise RuntimeError('large finite-level entity regression anchor not found')
p.write_text(text.replace(anchor, replacement, 1), encoding='utf-8')

# Physical validation: exercise the same entity-aware fit-stat surface on both
# CuPy and Torch CUDA, in addition to the Gram fallback and covariance checks.
p = Path('dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py')
text = p.read_text(encoding='utf-8')
anchor = '''    np.testing.assert_allclose(
        _public_array(actual.coef_)[0],
        _public_array(reference.coef_)[0],
        rtol=5e-13,
        atol=0.0,
    )

    # Separately exercise the scaled coefficient covariance at a magnitude
'''
replacement = '''    np.testing.assert_allclose(
        _public_array(actual.coef_)[0],
        _public_array(reference.coef_)[0],
        rtol=5e-13,
        atol=0.0,
    )

    entity_ids = np.tile(np.arange(x_period.size, dtype=np.int64), n_periods)
    entity_model = FamaMacBeth(bandwidth=0, device=_device(backend)).fit(
        Xb, yb, time_ids=time_ids, entity_ids=entity_ids
    )
    stats = entity_model.fit_statistics_
    if (
        stats.rsquared_overall != 0.0
        or stats.rsquared_between != 0.0
        or stats.rsquared_within != 0.0
        or stats.metadata.get("degenerate_total_ss")
        != {"within": True, "between": True, "overall": True}
    ):
        raise AssertionError(
            f"entity-aware scaled R2 drifted on {backend}: {stats}"
        )

    # Separately exercise the scaled coefficient covariance at a magnitude
'''
if anchor not in text:
    raise RuntimeError('physical numeric entity anchor not found')
text = text.replace(anchor, replacement, 1)
old = '''        "common_intercept": float(_public_array(actual.coef_)[0]),
        "scaled_covariance_slope_variance": float(
'''
new = '''        "common_intercept": float(_public_array(actual.coef_)[0]),
        "entity_aware_r2": {
            "within": float(stats.rsquared_within),
            "between": float(stats.rsquared_between),
            "overall": float(stats.rsquared_overall),
        },
        "scaled_covariance_slope_variance": float(
'''
if old not in text:
    raise RuntimeError('physical numeric payload anchor not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# Record that the fit-statistic hardening covers optional entity-aware R2 too.
p = Path('CHANGELOG.md')
text = p.read_text(encoding='utf-8')
marker = '- **Fama-MacBeth numerical stability**:'
pos = text.find(marker)
if pos < 0:
    raise RuntimeError('numeric stability changelog marker not found')
line_end = text.find('\n', pos)
line = text[pos:line_end]
if 'entity-aware' not in line:
    line += ' Parameter-based overall/within/between R² also uses scaled mean and group-mean reductions so finite large-level panels do not produce overflow-driven NaN fit statistics, including when `entity_ids` is supplied.'
    text = text[:pos] + line + text[line_end:]
p.write_text(text, encoding='utf-8')
