from pathlib import Path

path = Path("statgpu/panel/_covariance.py")
text = path.read_text(encoding="utf-8")
old = '''    leverage_np = np.asarray(_to_numpy(leverage), dtype=np.float64).ravel()
    tol = 256.0 * np.finfo(np.float64).eps
    if leverage_np.size and float(np.min(leverage_np)) < -tol:
        raise ValueError("HC2/HC3 leverage is materially negative")
    if leverage_np.size and float(np.max(leverage_np)) > 1.0 + tol:
        raise ValueError("HC2/HC3 leverage is materially greater than one")
    leverage_np = np.clip(leverage_np, 0.0, 1.0)
    denominator_np = 1.0 - leverage_np
    if denominator_np.size and float(np.min(denominator_np)) <= tol:
        raise ValueError("HC2/HC3 covariance is undefined when leverage is numerically one")
    denominator = xp_asarray(
        denominator_np,
        dtype=xp.float64,
        xp=xp,
        ref_arr=X,
    )
'''
new = '''    leverage_min = _to_float_scalar(xp.min(leverage))
    leverage_max = _to_float_scalar(xp.max(leverage))
    tol = 256.0 * np.finfo(np.float64).eps
    if leverage_min < -tol:
        raise ValueError("HC2/HC3 leverage is materially negative")
    if leverage_max > 1.0 + tol:
        raise ValueError("HC2/HC3 leverage is materially greater than one")
    if _is_torch(xp):
        leverage = xp.clamp(leverage, min=0.0, max=1.0)
    else:
        leverage = xp.clip(leverage, 0.0, 1.0)
    denominator = 1.0 - leverage
    denominator_min = _to_float_scalar(xp.min(denominator))
    if denominator_min <= tol:
        raise ValueError("HC2/HC3 covariance is undefined when leverage is numerically one")
'''
if old not in text:
    raise SystemExit("expected HC leverage block not found")
text = text.replace(old, new, 1)
old_meta = '''                "leverage_min": float(leverage_np.min()) if leverage_np.size else None,
                "leverage_max": float(leverage_np.max()) if leverage_np.size else None,
'''
new_meta = '''                "leverage_min": float(leverage_min),
                "leverage_max": float(leverage_max),
'''
if old_meta not in text:
    raise SystemExit("expected HC leverage metadata block not found")
text = text.replace(old_meta, new_meta, 1)
path.write_text(text, encoding="utf-8")
