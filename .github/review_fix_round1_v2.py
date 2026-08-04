from __future__ import annotations

import runpy
from pathlib import Path

path = Path(".github/review_fix_round1.py")
text = path.read_text(encoding="utf-8")
old = '''    start = text.index(f"def {function_name}():")
    end = text.index("\\n\\nclass ", start)
    block = text[start:end]
    lines = block.splitlines()
'''
new = '''    tree = ast.parse(text)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == function_name
    )
    source_lines = text.splitlines(keepends=True)
    start = sum(len(line) for line in source_lines[: node.lineno - 1])
    end = sum(len(line) for line in source_lines[: node.end_lineno])
    block = text[start:end].rstrip("\\n")
    lines = block.splitlines()
'''
if text.count(old) != 1:
    raise SystemExit(f"round-one loader anchor count={text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
runpy.run_path(str(path), run_name="__main__")
