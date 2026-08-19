from pathlib import Path

p = Path("statgpu/panel/_utils.py")
text = p.read_text(encoding="utf-8")
old = '''        shift = xp.sum(time_effects * t_counts) / float(values.shape[0])\n        time_effects = time_effects - shift\n        entity_effects = entity_effects + shift\n\n        residual = values - entity_effects[e_idx] - time_effects[t_idx]\n'''
new = '''        residual = values - entity_effects[e_idx] - time_effects[t_idx]\n'''
if old in text:
    text = text.replace(old, new, 1)
elif "xp.sum(time_effects * t_counts)" in text:
    raise RuntimeError("unexpected count-weighted normalization layout")
# If the first helper renamed t_counts but failed to remove the shift, fail rather
# than silently accepting a NameError-producing staged tree.
if "xp.sum(time_effects * t_counts)" in text:
    raise RuntimeError("count-weighted normalization survived staging patch")
p.write_text(text, encoding="utf-8")
