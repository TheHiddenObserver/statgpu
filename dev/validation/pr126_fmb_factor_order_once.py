from pathlib import Path

# Shared panel SVD: do not scale y. Apply inverse singular values to U.T before
# the projection, which prevents the avoidable U.T@y overflow without destroying
# small representable response components.
p = Path("statgpu/panel/_linalg.py")
text = p.read_text(encoding="utf-8")
start = text.index("def _scaled_lstsq_rhs")
end = text.index("def panel_svd_pseudoinverse", start)
text = text[:start] + text[end:]
old = '''def panel_lstsq(X, y, xp):
    """Return the minimum-norm least-squares solution under the panel SVD policy."""
    U, Vh, inverse_values, rank = _svd_inverse_factors(X, xp)
    y_scaled, response_scale = _scaled_lstsq_rhs(y, xp, batched=False)
    projected = U.T @ y_scaled
    if getattr(projected, "ndim", 1) == 1:
        scaled = inverse_values * projected
        params = Vh.T @ scaled
        return params * response_scale, rank
    scaled = inverse_values.reshape(-1, 1) * projected
    params = Vh.T @ scaled
    return params * response_scale, rank
'''
new = '''def panel_lstsq(X, y, xp):
    """Return the minimum-norm least-squares solution under the panel SVD policy."""
    U, Vh, inverse_values, rank = _svd_inverse_factors(X, xp)
    # Algebraically this is diag(1/s) @ U.T @ y. Apply 1/s to the
    # orthonormal rows before the reduction so a large projection that is
    # cancelled by a singular value (e.g. an intercept column) never has to be
    # represented as an overflowing intermediate. Unlike response
    # normalization, this does not discard unrelated tiny finite entries of y.
    weighted_u_t = inverse_values.reshape(-1, 1) * U.T
    scaled = weighted_u_t @ y
    return Vh.T @ scaled, rank
'''
if old not in text:
    raise RuntimeError("panel_lstsq anchor not found")
text = text.replace(old, new, 1)
old = '''    inverse_values = _inverse_values(singular_values, retained, xp)
    y_scaled, response_scale = _scaled_lstsq_rhs(y, xp, batched=False)
    projected = U.T @ y_scaled
    if getattr(projected, "ndim", 1) == 1:
        scaled = inverse_values * projected
        params = Vh.T @ scaled
        return params * response_scale, rank_backend
    scaled = inverse_values.reshape(-1, 1) * projected
    params = Vh.T @ scaled
    return params * response_scale, rank_backend
'''
new = '''    inverse_values = _inverse_values(singular_values, retained, xp)
    weighted_u_t = inverse_values.reshape(-1, 1) * U.T
    scaled = weighted_u_t @ y
    return Vh.T @ scaled, rank_backend
'''
if old not in text:
    raise RuntimeError("deferred-rank anchor not found")
text = text.replace(old, new, 1)
old = '''    y_scaled, response_scale = _scaled_lstsq_rhs(y, xp, batched=True)
    rhs = y_scaled[..., None] if getattr(y_scaled, "ndim", None) == 2 else y_scaled
    projected = xp.matmul(xp.swapaxes(U, -2, -1), rhs)
    scaled = inverse_values[..., None] * projected
    params = xp.matmul(xp.swapaxes(Vh, -2, -1), scaled)
    if getattr(y, "ndim", None) == 2:
        params = params[..., 0] * response_scale
    else:
        params = params * response_scale
    return params, ranks
'''
new = '''    rhs = y[..., None] if getattr(y, "ndim", None) == 2 else y
    weighted_u_t = xp.swapaxes(U, -2, -1) * inverse_values[..., :, None]
    scaled = xp.matmul(weighted_u_t, rhs)
    params = xp.matmul(xp.swapaxes(Vh, -2, -1), scaled)
    if getattr(y, "ndim", None) == 2:
        params = params[..., 0]
    return params, ranks
'''
if old not in text:
    raise RuntimeError("batched lstsq anchor not found")
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# Fama-MacBeth covariance: normalized covariance entries are bounded in
# magnitude by 1 under both supported definitions. Multiply each symmetric pair
# by the larger coordinate scale first, then the smaller one, so a tiny scale
# cannot underflow before a compensating large scale is applied.
p = Path("statgpu/panel/_fama_macbeth.py")
text = p.read_text(encoding="utf-8")
old = '''        cov_params = (
            cov_scaled * safe_centered_scale[:, None]
        ) * safe_centered_scale[None, :]
'''
new = '''        scale_row = safe_centered_scale[:, None]
        scale_col = safe_centered_scale[None, :]
        scale_large = xp.maximum(scale_row, scale_col)
        scale_small = xp.minimum(scale_row, scale_col)
        cov_params = (cov_scaled * scale_large) * scale_small
'''
if old not in text:
    raise RuntimeError("FMB covariance rescale anchor not found")
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# Linalg regressions: upgrade the mixed-range case to the minimum positive
# subnormal, while retaining the previous large-intercept overflow regression.
p = Path("dev/tests/test_fama_macbeth_batched_solver.py")
text = p.read_text(encoding="utf-8")
text = text.replace(
    'y = np.asarray([1.0e308, 1.0e-20], dtype=np.float64)',
    'y = np.asarray([1.7e308, np.nextafter(0.0, 1.0)], dtype=np.float64)',
)
text = text.replace(
    'np.testing.assert_allclose(params[0], 1.0e308, rtol=5e-15, atol=0.0)',
    'np.testing.assert_allclose(params[0], 1.7e308, rtol=5e-15, atol=0.0)',
)
text = text.replace(
    'np.testing.assert_allclose(params[1], 1.0e-20, rtol=5e-15, atol=0.0)',
    'assert params[1] == np.nextafter(0.0, 1.0)',
)
old = '''    y = torch.tensor(
        [[1.0e308, 1.0e-20], [-1.0e308, -1.0e-20]],
        dtype=torch.float64,
    )
    params, ranks = panel_lstsq_batched(X, y, torch)
    assert ranks.tolist() == [2, 2]
    expected = np.asarray(
        [[1.0e308, 1.0e-20], [-1.0e308, -1.0e-20]], dtype=np.float64
    )
    np.testing.assert_allclose(
        params.detach().cpu().numpy(), expected, rtol=5e-15, atol=0.0
    )
'''
new = '''    tiny = float(np.nextafter(0.0, 1.0))
    y = torch.tensor(
        [[1.7e308, tiny], [-1.7e308, -tiny]],
        dtype=torch.float64,
    )
    params, ranks = panel_lstsq_batched(X, y, torch)
    assert ranks.tolist() == [2, 2]
    actual = params.detach().cpu().numpy()
    np.testing.assert_allclose(actual[:, 0], np.asarray([1.7e308, -1.7e308]), rtol=5e-15, atol=0.0)
    assert actual[0, 1] == tiny
    assert actual[1, 1] == -tiny
'''
if old not in text:
    raise RuntimeError("Torch mixed-range test anchor not found")
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

# FMB regression for symmetric cross covariance when one coordinate scale is the
# minimum subnormal. The small diagonal variance is genuinely below float64 and
# may be zero, but the cross covariance is representable and must be symmetric.
p = Path("dev/tests/test_fama_macbeth_review_fixes.py")
text = p.read_text(encoding="utf-8")
append = r'''


def _subnormal_cross_covariance_fixture():
    tiny = float(np.nextafter(0.0, 1.0))
    X_period = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    period_betas = np.asarray(
        [[0.0, -1.0e154, -tiny], [0.0, 0.0, 0.0], [0.0, 1.0e154, tiny]],
        dtype=np.float64,
    )
    design = np.column_stack([np.ones(X_period.shape[0]), X_period])
    X = np.tile(X_period, (3, 1))
    y = np.concatenate([design @ beta for beta in period_betas])
    time = np.repeat(np.arange(3), X_period.shape[0])
    expected_cross = (1.0e154 * tiny) / 3.0
    return X, y, time, expected_cross


def test_fama_macbeth_covariance_rescale_preserves_subnormal_cross_symmetry_numpy():
    X, y, time, expected_cross = _subnormal_cross_covariance_fixture()
    model = FamaMacBeth(cov_type="nonrobust", device="cpu").fit(X, y, time_ids=time)
    cov = _to_numpy(model.cov_params_)
    assert expected_cross > 0.0
    np.testing.assert_allclose(cov[1, 2], expected_cross, rtol=5e-13, atol=0.0)
    np.testing.assert_allclose(cov[2, 1], expected_cross, rtol=5e-13, atol=0.0)
    np.testing.assert_allclose(cov, cov.T, rtol=0.0, atol=0.0)


def test_fama_macbeth_covariance_rescale_preserves_subnormal_cross_symmetry_torch_cpu():
    torch = pytest.importorskip("torch")
    X, y, time, expected_cross = _subnormal_cross_covariance_fixture()
    model = FamaMacBeth(cov_type="nonrobust").fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        time_ids=time,
    )
    cov = _to_numpy(model.cov_params_)
    np.testing.assert_allclose(cov[1, 2], expected_cross, rtol=5e-13, atol=0.0)
    np.testing.assert_allclose(cov[2, 1], expected_cross, rtol=5e-13, atol=0.0)
    np.testing.assert_allclose(cov, cov.T, rtol=0.0, atol=0.0)
'''
text += append
p.write_text(text, encoding="utf-8")

# Physical GPU validator: add the same asymmetric-underflow negative control.
p = Path("dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py")
text = p.read_text(encoding="utf-8")
needle = '''    if not np.all(np.isfinite(_public_array(mixed.pvalues_))):
        raise AssertionError("zero-variance coefficient leaked non-finite p-value")

'''
insert = '''    tiny = float(np.nextafter(0.0, 1.0))
    expected_subnormal_betas = np.asarray(
        [[0.0, -1.0e154, -tiny], [0.0, 0.0, 0.0], [0.0, 1.0e154, tiny]],
        dtype=np.float64,
    )
    X_sub = np.tile(X_period, (3, 1))
    y_sub = np.concatenate([design @ beta for beta in expected_subnormal_betas])
    X_sub_b, y_sub_b = _arrays(X_sub, y_sub, backend)
    subnormal = FamaMacBeth(cov_type="nonrobust", device=_device(backend)).fit(
        X_sub_b, y_sub_b, time_ids=time_mixed
    )
    sub_cov = _public_array(subnormal.cov_params_)
    expected_cross = 1.0e154 * tiny / 3.0
    np.testing.assert_allclose(sub_cov[1, 2], expected_cross, rtol=3e-12, atol=0.0)
    np.testing.assert_allclose(sub_cov[2, 1], expected_cross, rtol=3e-12, atol=0.0)
    np.testing.assert_allclose(sub_cov, sub_cov.T, rtol=0.0, atol=0.0)

'''
if needle not in text:
    raise RuntimeError("physical subnormal insertion anchor not found")
text = text.replace(needle, needle + insert, 1)
old = '''        "mixed_scale_zero_variance_statistic": float(
            _public_array(mixed.tvalues_)[0]
        ),
'''
new = '''        "mixed_scale_zero_variance_statistic": float(
            _public_array(mixed.tvalues_)[0]
        ),
        "subnormal_cross_covariance": float(sub_cov[1, 2]),
'''
if old not in text:
    raise RuntimeError("physical payload anchor not found")
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

print("PR126 factor-order review patch applied")
