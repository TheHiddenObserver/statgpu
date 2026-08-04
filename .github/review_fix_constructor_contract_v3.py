from pathlib import Path
import runpy

path = Path(".github/review_fix_constructor_contract.py")
text = path.read_text(encoding="utf-8")
anchor = '''p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
'''
replacement = '''p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
text = replace_once(
    text,
    "import functools\\nimport inspect\\n",
    "import copy\\nimport functools\\nimport inspect\\n",
    "base copy import",
)
'''
if text.count(anchor) != 1:
    raise SystemExit(f"base import anchor count={text.count(anchor)}")
path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")
runpy.run_path(".github/review_fix_constructor_contract_v2.py", run_name="__main__")
