from pathlib import Path
import runpy

# Apply the complete shared-panel numerical patch first.
runpy.run_path("dev/validation/pr126_panel_numeric_final_once.py", run_name="__main__")

# Torch CPU can overflow target/subnormal directly even when the mathematical
# quotient is finite.  Form the dimensionless ratio in the opposite direction
# first; max_abs/target is normal-range for every positive float64 subnormal,
# and its reciprocal gives the same finite design scale.
p = Path("statgpu/panel/_linalg.py")
text = p.read_text(encoding="utf-8")
old = '''    def _factor(max_abs):
        safe_max = xp.where(max_abs > 0.0, max_abs, xp.ones_like(max_abs))
        return xp.where(
            (max_abs > 0.0) & (max_abs < target),
            target / safe_max,
            xp.ones_like(max_abs),
        )
'''
new = '''    def _factor(max_abs):
        relative = max_abs / float(target)
        safe_relative = xp.where(
            relative > 0.0, relative, xp.ones_like(relative)
        )
        return xp.where(
            (max_abs > 0.0) & (max_abs < target),
            1.0 / safe_relative,
            xp.ones_like(max_abs),
        )
'''
if old not in text:
    raise RuntimeError("Torch-safe tiny-design scale anchor missing")
p.write_text(text.replace(old, new, 1), encoding="utf-8")
print("PR126 Torch-safe tiny-design scaling applied")
