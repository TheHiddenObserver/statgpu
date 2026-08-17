from pathlib import Path

p = Path('statgpu/panel/_linalg.py')
text = p.read_text(encoding='utf-8')
anchor = '''def _svd_inverse_factors(X, xp):
    """Return SVD factors, inverse singular values, and shared numerical rank."""
    U, singular_values, Vh = xp.linalg.svd(X, full_matrices=False)
    retained, rank = _rank_mask(X, singular_values, xp)
    return U, Vh, _inverse_values(singular_values, retained, xp), rank


'''
addition = anchor + '''def _scaled_lstsq_rhs(y, xp, *, batched: bool):
    """Scale least-squares responses before orthogonal projection.

    The least-squares map is linear in ``y``.  Scaling each independent
    response/target by its maximum absolute value prevents an otherwise
    representable coefficient from being lost when the orthogonal projection
    ``U.T @ y`` accumulates large finite observations beyond float64 range.
    """
    namespace = getattr(xp, "__name__", "")
    ndim = int(getattr(y, "ndim", 0))
    if batched:
        if ndim == 2:
            scale = (
                xp.max(xp.abs(y), dim=1).values
                if namespace == "torch"
                else xp.max(xp.abs(y), axis=1)
            )
            safe_scale = xp.where(scale > 0.0, scale, xp.ones_like(scale))
            return y / safe_scale[:, None], safe_scale[:, None]
        if ndim == 3:
            scale = (
                xp.max(xp.abs(y), dim=1).values
                if namespace == "torch"
                else xp.max(xp.abs(y), axis=1)
            )
            safe_scale = xp.where(scale > 0.0, scale, xp.ones_like(scale))
            return y / safe_scale[:, None, :], safe_scale[:, None, :]
        raise ValueError("batched panel response must have shape (batch, n_obs[, n_targets])")

    if ndim == 1:
        scale = xp.max(xp.abs(y))
        safe_scale = xp.where(scale > 0.0, scale, xp.ones_like(scale))
        return y / safe_scale, safe_scale
    if ndim == 2:
        scale = (
            xp.max(xp.abs(y), dim=0).values
            if namespace == "torch"
            else xp.max(xp.abs(y), axis=0)
        )
        safe_scale = xp.where(scale > 0.0, scale, xp.ones_like(scale))
        return y / safe_scale[None, :], safe_scale[None, :]
    raise ValueError("panel response must be one- or two-dimensional")


'''
if anchor not in text:
    raise RuntimeError('SVD factor anchor not found')
text = text.replace(anchor, addition, 1)

old = '''def panel_lstsq(X, y, xp):
    """Return the minimum-norm least-squares solution under the panel SVD policy."""
    U, Vh, inverse_values, rank = _svd_inverse_factors(X, xp)
    projected = U.T @ y
    if getattr(projected, "ndim", 1) == 1:
        scaled = inverse_values * projected
    else:
        scaled = inverse_values.reshape(-1, 1) * projected
    return Vh.T @ scaled, rank
'''
new = '''def panel_lstsq(X, y, xp):
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
if old not in text:
    raise RuntimeError('panel_lstsq anchor not found')
text = text.replace(old, new, 1)

old = '''    inverse_values = _inverse_values(singular_values, retained, xp)
    projected = U.T @ y
    if getattr(projected, "ndim", 1) == 1:
        scaled = inverse_values * projected
    else:
        scaled = inverse_values.reshape(-1, 1) * projected
    return Vh.T @ scaled, rank_backend
'''
new = '''    inverse_values = _inverse_values(singular_values, retained, xp)
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
if old not in text:
    raise RuntimeError('deferred lstsq anchor not found')
text = text.replace(old, new, 1)

old = '''    rhs = y[..., None] if getattr(y, "ndim", None) == 2 else y
    projected = xp.matmul(xp.swapaxes(U, -2, -1), rhs)
    scaled = inverse_values[..., None] * projected
    params = xp.matmul(xp.swapaxes(Vh, -2, -1), scaled)
    if getattr(y, "ndim", None) == 2:
        params = params[..., 0]
    return params, ranks
'''
new = '''    y_scaled, response_scale = _scaled_lstsq_rhs(y, xp, batched=True)
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
if old not in text:
    raise RuntimeError('batched lstsq anchor not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# Direct shared solver regressions plus a tall Fama-MacBeth fallback case.
p = Path('dev/tests/test_fama_macbeth_batched_solver.py')
text = p.read_text(encoding='utf-8')
anchor = '''def test_panel_lstsq_gram_certified_matches_svd_for_well_conditioned_numpy():
'''
addition = '''def test_panel_lstsq_scales_large_finite_rhs_before_svd_projection():
    n = 16
    X = np.ones((n, 1), dtype=np.float64)
    y = np.full(n, 6.0e307, dtype=np.float64)
    params, rank = panel_lstsq(X, y, np)
    assert rank == 1
    assert np.isfinite(params[0])
    np.testing.assert_allclose(params[0], 6.0e307, rtol=5e-15, atol=0.0)


def test_panel_lstsq_batched_scales_large_finite_rhs_before_projection():
    torch = pytest.importorskip("torch")
    n = 16
    X = torch.ones((2, n, 1), dtype=torch.float64)
    y = torch.stack(
        [
            torch.full((n,), 6.0e307, dtype=torch.float64),
            torch.full((n,), -6.0e307, dtype=torch.float64),
        ]
    )
    params, ranks = panel_lstsq_batched(X, y, torch)
    assert ranks.tolist() == [1, 1]
    assert torch.all(torch.isfinite(params))
    np.testing.assert_allclose(
        params.detach().cpu().numpy()[:, 0],
        np.asarray([6.0e307, -6.0e307]),
        rtol=5e-15,
        atol=0.0,
    )


def test_panel_lstsq_deferred_rank_scales_large_finite_rhs_before_projection():
    from statgpu.panel._linalg import panel_lstsq_deferred_rank

    n = 16
    X = np.ones((n, 1), dtype=np.float64)
    y = np.full(n, 6.0e307, dtype=np.float64)
    params, rank = panel_lstsq_deferred_rank(X, y, np)
    assert int(rank) == 1
    assert np.isfinite(params[0])
    np.testing.assert_allclose(params[0], 6.0e307, rtol=5e-15, atol=0.0)


'''
if anchor not in text:
    raise RuntimeError('batched solver insertion anchor not found')
p.write_text(text.replace(anchor, addition + anchor, 1), encoding='utf-8')

# Make the maintained FMB extreme fixture tall enough that the unscaled SVD
# fallback itself would overflow U.T@y, while the true intercept remains finite.
p = Path('dev/tests/test_fama_macbeth_inference_matrix.py')
text = p.read_text(encoding='utf-8')
old = '''def _large_common_intercept_fixture(n_periods=4):
    x_period = np.asarray([-1.0, 0.0, 1.0])
'''
new = '''def _large_common_intercept_fixture(n_periods=4):
    x_period = np.linspace(-1.0, 1.0, 16, dtype=np.float64)
'''
if old not in text:
    raise RuntimeError('large FMB fixture anchor not found')
text = text.replace(old, new, 1)
old = '''    entity_ids = np.tile(np.arange(3, dtype=np.int64), 4)
'''
new = '''    entity_ids = np.tile(np.arange(X.shape[0] // 4, dtype=np.int64), 4)
'''
if old not in text:
    raise RuntimeError('entity fixture anchor not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')

# Physical validator: same tall period must validate the complete Gram->SVD
# chain on both CuPy and Torch CUDA.
p = Path('dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py')
text = p.read_text(encoding='utf-8')
start = text.find('def _numeric_stability_case(backend: str):')
if start < 0:
    raise RuntimeError('physical numeric case not found')
pos = text.find('    x_period = np.asarray([-1.0, 0.0, 1.0])', start)
if pos < 0:
    raise RuntimeError('physical x_period anchor not found')
text = text[:pos] + '    x_period = np.linspace(-1.0, 1.0, 16, dtype=np.float64)' + text[pos + len('    x_period = np.asarray([-1.0, 0.0, 1.0])'):]
p.write_text(text, encoding='utf-8')

# Changelog and EN/CN docs.
p = Path('CHANGELOG.md')
text = p.read_text(encoding='utf-8')
marker = '- **Fama-MacBeth numerical stability**:'
pos = text.find(marker)
if pos < 0:
    raise RuntimeError('numeric stability changelog marker missing')
line_end = text.find('\n', pos)
line = text[pos:line_end]
if 'SVD projection' not in line:
    line += ' The shared serial/deferred/batched panel SVD solvers also scale response RHS values before orthogonal projection, preventing representable coefficients from being lost when `U^T y` would overflow.'
    text = text[:pos] + line + text[line_end:]
p.write_text(text, encoding='utf-8')

for path, needle, replacement in [
    (
        'docs/en/panel/fama-macbeth.md',
        'A non-finite batched Gram right-hand side or candidate solution is treated as uncertified and routed through the rank-revealing SVD fallback.',
        'A non-finite batched Gram right-hand side or candidate solution is treated as uncertified and routed through the rank-revealing SVD fallback. The shared serial, deferred-rank, and batched SVD solvers scale each response/target before the orthogonal projection and rescale the final coefficient, so a representable solution is not lost merely because the unscaled `U^T y` accumulation would overflow.'
    ),
    (
        'docs/cn/panel/fama-macbeth.md',
        '若批量 Gram 右端项或候选解出现非有限值，该时期会被视为 uncertified，并转入秩揭示 SVD 回退。',
        '若批量 Gram 右端项或候选解出现非有限值，该时期会被视为 uncertified，并转入秩揭示 SVD 回退。共享的 serial、deferred-rank 与 batched SVD solver 会在正交投影前按 response/target 缩放，并在最终 coefficient 上还原该尺度，因此不会仅因为未缩放的 `U^T y` 累加溢出而丢失本来可表示的解。'
    ),
]:
    p = Path(path)
    doc = p.read_text(encoding='utf-8')
    if needle not in doc:
        raise RuntimeError(f'doc SVD scaling anchor not found: {path}')
    p.write_text(doc.replace(needle, replacement, 1), encoding='utf-8')
