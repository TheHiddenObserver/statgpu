from pathlib import Path

path = Path("dev/tests/test_maintenance_024_025.py")
text = path.read_text(encoding="utf-8")
old = '''    import statgpu.backends as backends
    from statgpu.linear_model import GeneralizedLinearModel
'''
new = '''    import statgpu.linear_model._glm_base as glm_module
    from statgpu.linear_model import GeneralizedLinearModel
'''
if text.count(old) != 2:
    raise RuntimeError("GLM module import anchors missing")
text = text.replace(old, new, 2)
old = '''    original_to_numpy = backends._to_numpy
'''
new = '''    original_to_numpy = glm_module._to_numpy
'''
if text.count(old) != 2:
    raise RuntimeError("local _to_numpy anchors missing")
text = text.replace(old, new, 2)
old = '''    monkeypatch.setattr(backends, "_to_numpy", guarded_to_numpy)
'''
new = '''    monkeypatch.setattr(glm_module, "_to_numpy", guarded_to_numpy)
'''
if text.count(old) != 2:
    raise RuntimeError("monkeypatch target anchors missing")
path.write_text(text.replace(old, new, 2), encoding="utf-8")
