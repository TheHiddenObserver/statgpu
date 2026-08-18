from pathlib import Path

p = Path("statgpu/panel/_covariance.py")
text = p.read_text(encoding="utf-8")
old = '''    if hasattr(out, "scatter_add_"):\n        out.scatter_add_(0, index, working)\n    elif type(out).__module__.startswith("cupy"):\n        xp.add.at(out, codes, working)\n    else:\n        np.add.at(out, codes_np, working)\n    return out * factor\n'''
new = '''    # Accumulate signs separately at the safe working scale. This avoids\n    # observation-order paths such as +a,+a,-a,-a, where each partial sum is\n    # finite after scaling but sequential rounding can leave a huge spurious\n    # cancellation residual. The final positive/negative combination is one\n    # opposite-sign addition per group/coordinate.\n    positive = xp.where(working > 0.0, working, xp.zeros_like(working))\n    negative = xp.where(working < 0.0, working, xp.zeros_like(working))\n    negative_out = xp_zeros(shape, dtype=xp.float64, xp=xp, ref_arr=scores)\n    if hasattr(out, "scatter_add_"):\n        out.scatter_add_(0, index, positive)\n        negative_out.scatter_add_(0, index, negative)\n    elif type(out).__module__.startswith("cupy"):\n        xp.add.at(out, codes, positive)\n        xp.add.at(negative_out, codes, negative)\n    else:\n        np.add.at(out, codes_np, positive)\n        np.add.at(negative_out, codes_np, negative)\n    return (out + negative_out) * factor\n'''
if old not in text:
    raise RuntimeError("grouped score staged anchor missing")
p.write_text(text.replace(old, new, 1), encoding="utf-8")
