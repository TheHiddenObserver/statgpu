from pathlib import Path

for path, old, new in (
    (
        "docs/en/panel/diagnostics.md",
        "> Last updated: 2026-08-18  \n",
        "> Last updated: 2026-08-18<br>\n",
    ),
    (
        "docs/cn/panel/diagnostics.md",
        "> 最后更新：2026-08-18  \n",
        "> 最后更新：2026-08-18<br>\n",
    ),
):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"followup date anchor missing in {path}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")

print("PR126 evidence followup formatting fixed")
