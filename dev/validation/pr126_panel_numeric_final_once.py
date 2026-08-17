from pathlib import Path


def replace_once(text, old, new, label):
    if old not in text:
        raise RuntimeError(f"{label} anchor missing")
    return text.replace(old, new, 1)


# Shared least-squares policy: (1) raise collectively tiny designs to a safe
# working scale without changing the relative SVD rank rule; (2) make the Gram
# certificate inspect batch finiteness before eigvalsh so the performance gate
# can never preempt the SVD fallback with a backend linalg error.
p = Path("statgpu/panel/_linalg.py")
text = p.read_text(encoding="utf-8")
if "def _lstsq_working_design" not in text:
    anchor = "def panel_svd_pseudoinverse(X, xp):\n"
    helper = '''def _lstsq_working_design(X, xp, *, batched: bool):
    """Return an SVD working design plus its positive rescaling factor.

    Uniform positive rescaling leaves the relative panel rank cutoff unchanged.
    If an entire independent design lies below ``sqrt(DBL_MIN)``, raise its
    largest element to that threshold before forming inverse retained singular
    values.  This prevents ``1 / s`` overflow for a full-rank subnormal design;
    the final least-squares coefficient is multiplied by the same factor to
    restore the original parameterization.
    """
    namespace = getattr(xp, "__name__", "")
    target = float(np.sqrt(np.finfo(np.float64).tiny))

    def _factor(max_abs):
        safe_max = xp.where(max_abs > 0.0, max_abs, xp.ones_like(max_abs))
        return xp.where(
            (max_abs > 0.0) & (max_abs < target),
            target / safe_max,
            xp.ones_like(max_abs),
        )

    if batched:
        view = xp.abs(X).reshape(int(X.shape[0]), -1)
        max_abs = (
            xp.max(view, dim=1).values
            if namespace == "torch"
            else xp.max(view, axis=1)
        )
        factor = _factor(max_abs)
        return X * factor[:, None, None], factor

    max_abs = xp.max(xp.abs(X))
    factor = _factor(max_abs)
    return X * factor, factor


'''
    text = replace_once(text, anchor, helper + anchor, "working-design insertion")

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
if old in text:
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
if old in text:
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
if old in text:
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
if old in text:
    text = text.replace(old, new, 1)

old = '''    transpose = xp.swapaxes(X, -2, -1)
    gram = xp.matmul(transpose, X)
    rhs_input = y[..., None] if getattr(y, "ndim", None) == 2 else y
    rhs = xp.matmul(transpose, rhs_input)
    rhs_finite_view = xp.isfinite(rhs).reshape(int(rhs.shape[0]), -1)
    rhs_finite = (
        xp.all(rhs_finite_view, dim=1)
        if namespace == "torch"
        else xp.all(rhs_finite_view, axis=1)
    )

    eigenvalues = xp.linalg.eigvalsh(gram)
    smallest = eigenvalues[..., 0]
    largest = eigenvalues[..., -1]
    certified = (
        xp.isfinite(smallest)
        & xp.isfinite(largest)
        & (largest > 0.0)
        & (smallest > largest * ratio)
        & rhs_finite
    )

    k = int(X.shape[-1])
    if namespace == "torch":
        identity = xp.eye(k, dtype=X.dtype, device=X.device)
    else:
        identity = xp.eye(k, dtype=X.dtype)
    safe_gram = xp.where(certified[..., None, None], gram, identity)
'''
new = '''    transpose = xp.swapaxes(X, -2, -1)
    gram = xp.matmul(transpose, X)
    rhs_input = y[..., None] if getattr(y, "ndim", None) == 2 else y
    rhs = xp.matmul(transpose, rhs_input)

    gram_finite_view = xp.isfinite(gram).reshape(int(gram.shape[0]), -1)
    gram_finite = (
        xp.all(gram_finite_view, dim=1)
        if namespace == "torch"
        else xp.all(gram_finite_view, axis=1)
    )
    rhs_finite_view = xp.isfinite(rhs).reshape(int(rhs.shape[0]), -1)
    rhs_finite = (
        xp.all(rhs_finite_view, dim=1)
        if namespace == "torch"
        else xp.all(rhs_finite_view, axis=1)
    )

    k = int(X.shape[-1])
    if namespace == "torch":
        identity = xp.eye(k, dtype=X.dtype, device=X.device)
    else:
        identity = xp.eye(k, dtype=X.dtype)
    # A non-finite Gram matrix is already enough to reject the performance
    # certificate.  Substitute identity only for the spectrum calculation so
    # eigvalsh itself cannot abort before the caller reaches the SVD fallback.
    spectrum_gram = xp.where(gram_finite[..., None, None], gram, identity)
    eigenvalues = xp.linalg.eigvalsh(spectrum_gram)
    smallest = eigenvalues[..., 0]
    largest = eigenvalues[..., -1]
    certified = (
        gram_finite
        & xp.isfinite(smallest)
        & xp.isfinite(largest)
        & (largest > 0.0)
        & (smallest > largest * ratio)
        & rhs_finite
    )

    safe_gram = xp.where(certified[..., None, None], gram, identity)
'''
if old in text:
    text = text.replace(old, new, 1)
else:
    if "spectrum_gram = xp.where(gram_finite" not in text:
        raise RuntimeError("Gram pre-spectrum anchor missing")

p.write_text(text, encoding="utf-8")

# Focused regressions.
p = Path("dev/tests/test_fama_macbeth_batched_solver.py")
text = p.read_text(encoding="utf-8")
if "test_panel_lstsq_rescales_tiny_full_rank_design" not in text:
    anchor = "def test_panel_lstsq_preserves_mixed_dynamic_range_identity_rhs():\n"
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


def test_gram_certificate_defers_nonfinite_gram_to_svd_numpy():
    X = (np.eye(2, dtype=np.float64) * 1.0e200)[None, ...]
    y = np.asarray([[1.0e200, 2.0e200]], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        _candidate, certified = panel_lstsq_gram_certified_batched(X, y, np)
    assert np.asarray(certified, dtype=bool).tolist() == [False]
    params, rank = panel_lstsq(X[0], y[0], np)
    assert rank == 2
    np.testing.assert_allclose(params, np.asarray([1.0, 2.0]), rtol=5e-14, atol=0.0)


def test_gram_certificate_defers_nonfinite_gram_to_svd_torch_cpu():
    torch = pytest.importorskip("torch")
    X = torch.eye(2, dtype=torch.float64).reshape(1, 2, 2) * 1.0e200
    y = torch.tensor([[1.0e200, 2.0e200]], dtype=torch.float64)
    _candidate, certified = panel_lstsq_gram_certified_batched(X, y, torch)
    assert certified.tolist() == [False]
    params, ranks = panel_lstsq_batched(X, y, torch)
    assert ranks.tolist() == [2]
    np.testing.assert_allclose(
        params.detach().cpu().numpy()[0], np.asarray([1.0, 2.0]), rtol=5e-13, atol=0.0
    )


'''
    text = replace_once(text, anchor, insert + anchor, "linalg test insertion")
p.write_text(text, encoding="utf-8")

# Physical Stage-C validator: add direct GPU audits for both new shared-linalg
# branches. These are coefficient-solver checks only; covariance can genuinely
# be outside float64 when the design pseudoinverse itself is unrepresentable.
p = Path("dev/benchmarks/validate_panel_stage_c_gpu.py")
text = p.read_text(encoding="utf-8")
if "panel_lstsq_gram_certified_batched" not in text:
    text = replace_once(
        text,
        "from statgpu.panel._covariance import ols_covariance\n",
        "from statgpu.panel._covariance import ols_covariance\nfrom statgpu.panel._linalg import (\n    panel_lstsq,\n    panel_lstsq_batched,\n    panel_lstsq_gram_certified_batched,\n)\n",
        "physical validator import",
    )
if "def _tiny_design_lstsq_audit" not in text:
    anchor = "def _fit_rank(model):\n"
    helper = '''def _tiny_design_lstsq_audit(backend):
    tiny = 1.0e-320
    X = np.eye(2, dtype=np.float64) * tiny
    y = np.asarray([tiny, 2.0 * tiny], dtype=np.float64)
    entity = np.arange(2, dtype=np.int64)
    time = np.arange(2, dtype=np.int64)
    Xb, yb, _eb, _tb = _to_backend(X, y, entity, time, backend)
    if backend == "torch":
        import torch
        params, ranks = panel_lstsq_batched(Xb[None, ...], yb[None, ...], torch)
        rank = int(_array(ranks)[0])
        params_np = _array(params)[0]
    else:
        import cupy as cp
        params, rank = panel_lstsq(Xb, yb, cp)
        rank = int(rank)
        params_np = _array(params)
    if rank != 2:
        raise AssertionError(f"{backend}: tiny full-rank design rank drifted to {rank}")
    np.testing.assert_allclose(params_np, np.asarray([1.0, 2.0]), rtol=5e-11, atol=0.0)
    return {"status": "success", "backend": backend, "rank": rank, "params": params_np.tolist()}


def _gram_overflow_certificate_audit(backend):
    X = np.eye(2, dtype=np.float64) * 1.0e200
    y = np.asarray([1.0e200, 2.0e200], dtype=np.float64)
    entity = np.arange(2, dtype=np.int64)
    time = np.arange(2, dtype=np.int64)
    Xb, yb, _eb, _tb = _to_backend(X, y, entity, time, backend)
    if backend == "torch":
        import torch
        _candidate, certified = panel_lstsq_gram_certified_batched(
            Xb[None, ...], yb[None, ...], torch
        )
        params, ranks = panel_lstsq_batched(Xb[None, ...], yb[None, ...], torch)
        rank = int(_array(ranks)[0])
        params_np = _array(params)[0]
    else:
        import cupy as cp
        _candidate, certified = panel_lstsq_gram_certified_batched(
            Xb[None, ...], yb[None, ...], cp
        )
        params, rank = panel_lstsq(Xb, yb, cp)
        rank = int(rank)
        params_np = _array(params)
    if bool(_array(certified)[0]):
        raise AssertionError(f"{backend}: non-finite Gram batch was incorrectly certified")
    if rank != 2:
        raise AssertionError(f"{backend}: Gram-overflow SVD fallback rank drifted to {rank}")
    np.testing.assert_allclose(params_np, np.asarray([1.0, 2.0]), rtol=5e-11, atol=0.0)
    return {"status": "success", "backend": backend, "rank": rank, "params": params_np.tolist()}


'''
    text = replace_once(text, anchor, helper + anchor, "physical numerical audit insertion")
if 'payload["numerical_primitives"]' not in text:
    anchor = '        payload["level_constant_contract"] = level_constant_audit\n'
    addition = '''        payload["level_constant_contract"] = level_constant_audit
        payload["numerical_primitives"] = {
            "tiny_design_lstsq": _tiny_design_lstsq_audit(backend),
            "gram_overflow_certificate": _gram_overflow_certificate_audit(backend),
        }
'''
    text = replace_once(text, anchor, addition, "physical payload insertion")
p.write_text(text, encoding="utf-8")

print("PR126 final shared-panel numerical patch applied")
