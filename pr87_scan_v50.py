from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def changed_python_files() -> list[Path]:
    subprocess.run(["git", "fetch", "origin", "master", "--depth=1"], check=True)
    output = subprocess.check_output(
        ["git", "diff", "--name-only", "origin/master...HEAD"],
        text=True,
    )
    return [
        Path(line)
        for line in output.splitlines()
        if line.endswith(".py") and Path(line).is_file()
    ]


def handler_names(handler: ast.ExceptHandler) -> set[str]:
    node = handler.type
    if node is None:
        return {"BaseException"}
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return {".".join(reversed(parts))}
    if isinstance(node, ast.Tuple):
        names: set[str] = set()
        for item in node.elts:
            fake = ast.ExceptHandler(type=item, name=None, body=[])
            names.update(handler_names(fake))
        return names
    return {ast.unparse(node)}


def text_for(lines: list[str], start: int, end: int) -> str:
    lo = max(start - 2, 1)
    hi = min(end + 2, len(lines))
    return "\n".join(f"{i:5d}: {lines[i - 1]}" for i in range(lo, hi + 1))


def main() -> None:
    files = changed_python_files()
    print(f"CHANGED_PYTHON_FILES={len(files)}")
    findings = 0
    for path in files:
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            print(f"SYNTAX_ERROR {path}:{exc.lineno}: {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            try_text = ast.get_source_segment(source, node) or ""
            risky_ops = any(
                marker in try_text
                for marker in (
                    "torch.linalg",
                    "cp.linalg",
                    "cupy.linalg",
                    "cuda",
                    ".to(",
                    ".item()",
                    "xp.linalg",
                    "_to_numpy",
                )
            )
            for handler in node.handlers:
                names = handler_names(handler)
                broad = bool(
                    names
                    & {
                        "RuntimeError",
                        "Exception",
                        "BaseException",
                        "torch.RuntimeError",
                    }
                )
                if not broad:
                    continue
                findings += 1
                tag = "RISKY" if risky_ops else "BROAD"
                print(
                    f"\n[{tag}] {path}:{handler.lineno} catches {sorted(names)}; "
                    f"try={node.lineno}-{getattr(node, 'end_lineno', node.lineno)}"
                )
                print(text_for(lines, node.lineno, getattr(node, "end_lineno", node.lineno)))
    print(f"\nBROAD_HANDLER_COUNT={findings}")


if __name__ == "__main__":
    main()
