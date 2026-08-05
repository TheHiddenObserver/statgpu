from pathlib import Path
import runpy

runpy.run_path("pr87_patch_v28.py", run_name="__main__")

path = Path("dev/tests/test_maintenance_024_025.py")
text = path.read_text(encoding="utf-8")
anchor = '''def test_solver_utils_weighted_helper_delegates_without_silent_unweighting(monkeypatch):
    import statgpu.glm_core._fused as fused
'''
replacement = '''def test_solver_utils_weighted_helper_delegates_without_silent_unweighting(monkeypatch):
    from pathlib import Path
    import statgpu.glm_core._fused as fused
'''
if anchor not in text:
    raise RuntimeError("weighted helper test import anchor missing")
path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")
