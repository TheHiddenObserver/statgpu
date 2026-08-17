from pathlib import Path

# Trigger carrier for the one-shot tiny-design review workflow.
p = Path("statgpu/panel/_linalg.py")
text = p.read_text(encoding="utf-8")
anchor = '''def panel_svd_pseudoinverse(X, xp):
'''
helper = '''def _lstsq_working_design(X, xp, *, batched: bool):
    """Raise a tiny design to a safe SVD working scale without changing rank.

    Multiplying a least-squares design by a positive scalar leaves the relative
    singular-value rank policy unchanged.  For designs below sqrt(float64 tiny),
    use the minimum scalar that raises the largest element to that threshold.
    The final coefficient is multiplied by the same scalar to recover the
    original parameterization.  This avoids overflowing 1/s for retained tiny
    singular values while preserving every coefficient that could still create
    a nonzero float64 response in the original design.
    """
    namespace = getattr(xp, "__name__", "")
    target = float(np.sqrt(np.finfo(np.float64).tiny))
    if batched:
        view = xp.abs(X).reshape(int(X.shape[0]), -1)
        max_abs = (
            xp.max(view, dim=1).values
            if namespace == "torch"
            else xp.max(view, axis=1)
        )
        safe_max = xp.where(max_abs > 0.0, max_abs, xp.ones_like(max_abs))
        factor = xp.where(
            (max_abs > 0.0) & (max_abs < target),
            target / safe_max,
            xp.ones_like(max_abs),
        )
        return X * factor[:, None, None], factor

    max_abs = xp.max(xp.abs(X))
    safe_max = xp.where(max_abs > 0.0, max_abs, xp.ones_like(max_abs))
    factor = xp.where(
        (max_abs > 0.0) & (max_abs < target),
        target / safe_max,
        xp.ones_like(max_abs),
    )
    return X * factor, factor


'''
if anchor not in text:
    raise RuntimeError("linalg insertion anchor missing")
text = text.replace(anchor, helper + anchor, 1)
old = '''def panel_lstsq(X, y, xp):
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
new = '''def panel_lstsq(X, y, xp):
    """Return the minimum-norm least-squares solution under the panel SVD policy."""
    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    U, Vh, inverse_values, rank = _svd_inverse_factors(X_work, xp)
    # Algebraically this is diag(1/s) @ U.T @ y. Apply 1/s to the
    # orthonormal rows before the reduction so a large projection that is
    # cancelled by a singular value (e.g. an intercept column) never has to be
    # represented as an overflowing intermediate. Unlike response
    # normalization, this does not discard unrelated tiny finite entries of y.
    weighted_u_t = inverse_values.reshape(-1, 1) * U.T
    scaled = weighted_u_t @ y
    return (Vh.T @ scaled) * design_scale, rank
'''
if old not in text:
    raise RuntimeError("panel_lstsq anchor missing")
text = text.replace(old, new, 1)
old = '''    U, singular_values, Vh = xp.linalg.svd(X, full_matrices=False)
    retained, rank_backend = _rank_mask_backend(X, singular_values, xp)
    inverse_values = _inverse_values(singular_values, retained, xp)
    weighted_u_t = inverse_values.reshape(-1, 1) * U.T
    scaled = weighted_u_t @ y
    return Vh.T @ scaled, rank_backend
'''
new = '''    X_work, design_scale = _lstsq_working_design(X, xp, batched=False)
    U, singular_values, Vh = xp.linalg.svd(X_work, full_matrices=False)
    retained, rank_backend = _rank_mask_backend(X_work, singular_values, xp)
    inverse_values = _inverse_values(singular_values, retained, xp)
    weighted_u_t = inverse_values.reshape(-1, 1) * U.T
    scaled = weighted_u_t @ y
    return (Vh.T @ scaled) * design_scale, rank_backend
'''
if old not in text:
    raise RuntimeError("deferred rank anchor missing")
text = text.replace(old, new, 1)
old = '''    U, singular_values, Vh = xp.linalg.svd(X, full_matrices=False)
    cutoff_scale = (
        max(int(X.shape[-2]), int(X.shape[-1]))
        * np.finfo(np.float64).eps
    )
'''
new = '''    X_work, design_scale = _lstsq_working_design(X, xp, batched=True)
    U, singular_values, Vh = xp.linalg.svd(X_work, full_matrices=False)
    cutoff_scale = (
        max(int(X_work.shape[-2]), int(X_work.shape[-1]))
        * np.finfo(np.float64).eps
    )
'''
if old not in text:
    raise RuntimeError("batched SVD anchor missing")
text = text.replace(old, new, 1)
old = '''    params = xp.matmul(xp.swapaxes(Vh, -2, -1), scaled)
    if getattr(y, "ndim", None) == 2:
        params = params[..., 0]
    return params, ranks
'''
new = '''    params = xp.matmul(xp.swapaxes(Vh, -2, -1), scaled)
    if getattr(y, "ndim", None) == 2:
        params = params[..., 0] * design_scale[:, None]
    else:
        params = params * design_scale[:, None, None]
    return params, ranks
'''
if old not in text:
    raise RuntimeError("batched rescale anchor missing")
text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")

p = Path("dev/tests/test_fama_macbeth_batched_solver.py")
text = p.read_text(encoding="utf-8")
anchor = '''def test_panel_lstsq_preserves_mixed_dynamic_range_identity_rhs():
'''
insert = '''def test_panel_lstsq_rescales_tiny_full_rank_design():
    tiny = 1.0e-320
    X = np.eye(2, dtype=np.float64) * tiny
    y = np.asarray([tiny, 2.0 * tiny], dtype=np.float64)
    params, rank = panel_lstsq(X, y, np)
    assert rank == 2
    np.testing.assert_allclose(params, np.asarray([1.0, 2.0]), rtol=5e-14, atol=0.0)


def test_panel_lstsq_deferred_rank_rescales_tiny_full_rank_design():
    from statgpu.panel._linalg import panel_lstsq_deferred_rank

    tiny = 1.0e-320
    X = np.eye(2, dtype=np.float64) * tiny
    y = np.asarray([tiny, 2.0 * tiny], dtype=np.float64)
    params, rank = panel_lstsq_deferred_rank(X, y, np)
    assert int(rank) == 2
    np.testing.assert_allclose(params, np.asarray([1.0, 2.0]), rtol=5e-14, atol=0.0)


def test_panel_lstsq_batched_rescales_tiny_full_rank_design_torch_cpu():
    torch = pytest.importorskip("torch")
    tiny = 1.0e-320
    X = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1) * tiny
    y = torch.tensor([[tiny, 2.0 * tiny], [2.0 * tiny, -tiny]], dtype=torch.float64)
    params, ranks = panel_lstsq_batched(X, y, torch)
    assert ranks.tolist() == [2, 2]
    np.testing.assert_allclose(
        params.detach().cpu().numpy(),
        np.asarray([[1.0, 2.0], [2.0, -1.0]]),
        rtol=5e-13,
        atol=0.0,
    )


'''
if anchor not in text:
    raise RuntimeError("test insertion anchor missing")
text = text.replace(anchor, insert + anchor, 1)
p.write_text(text, encoding="utf-8")

p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
old = '''from statgpu.panel._covariance import ols_covariance
'''
new = '''from statgpu.panel._covariance import ols_covariance
from statgpu.panel._linalg import panel_lstsq, panel_lstsq_batched
'''
if old not in text:
    raise RuntimeError("validator import anchor missing")
text = text.replace(old, new, 1)
anchor = '''def _fit_rank(model):
'''
helper = '''def _tiny_design_lstsq_audit(backend):
    tiny = 1.0e-320
    X = np.eye(2, dtype=np.float64) * tiny
    y = np.asarray([tiny, 2.0 * tiny], dtype=np.float64)
    entity = np.arange(2, dtype=np.int64)
    time = np.arange(2, dtype=np.int64)
    Xb, yb, _eb, _tb = _to_backend(X, y, entity, time, backend)
    if backend == "torch":
        params, ranks = panel_lstsq_batched(Xb[None, ...], yb[None, ...], __import__("torch"))
        rank = int(_array(ranks)[0])
        params_np = _array(params)[0]
    else:
        xp = np if backend == "numpy" else __import__("cupy")
        params, rank = panel_lstsq(Xb, yb, xp)
        rank = int(rank)
        params_np = _array(params)
    if rank != 2:
        raise AssertionError(f"{backend}: tiny full-rank design rank drifted to {rank}")
    np.testing.assert_allclose(params_np, np.asarray([1.0, 2.0]), rtol=5e-11, atol=0.0)
    return {"status": "success", "backend": backend, "rank": rank, "params": params_np.tolist()}


'''
if anchor not in text:
    raise RuntimeError("validator insertion anchor missing")
text = text.replace(anchor, helper + anchor, 1)
needle = '''            "level_constant_contract": _level_constant_contract_audit(backend),
'''
replacement = '''            "level_constant_contract": _level_constant_contract_audit(backend),
            "tiny_design_lstsq": _tiny_design_lstsq_audit(backend),
'''
if needle not in text:
    raise RuntimeError("validator payload anchor missing")
text = text.replace(needle, replacement, 1)
p.write_text(text, encoding="utf-8")

print("PR126 tiny-design patch applied")