from pathlib import Path

path = Path("dev/tests/test_maintenance_024_025.py")
text = path.read_text(encoding="utf-8")
old = "bool(torch.allclose(value, weights).item())"
new = "bool(torch.allclose(value, weights))"
if old not in text:
    raise RuntimeError("Torch allclose assertion anchor missing")
text = text.replace(old, new, 1)
old = "bool(cp.allclose(value, weights).item())"
new = "bool(cp.allclose(value, weights))"
if old not in text:
    raise RuntimeError("CuPy allclose assertion anchor missing")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
