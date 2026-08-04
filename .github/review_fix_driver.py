from pathlib import Path

source_path = Path('.github/review_fix_patch.py')
source = source_path.read_text(encoding='utf-8')
start = source.index('def replace_once(')
end = source.index('\n\n\n# ---------------------------------------------------------------------------', start)
replacement = '''def replace_once(text: str, old: str, new: str, label: str) -> str:
    candidates = [(old, new)]
    for width in (4, 8, 12, 16):
        prefix = " " * width
        old_indented = "\\n".join(
            prefix + line if line else line for line in old.splitlines()
        )
        new_indented = "\\n".join(
            prefix + line if line else line for line in new.splitlines()
        )
        candidates.append((old_indented, new_indented))

    for old_candidate, new_candidate in candidates:
        count = text.count(old_candidate)
        if count == 1:
            return text.replace(old_candidate, new_candidate, 1)
    counts = [text.count(candidate) for candidate, _ in candidates]
    raise RuntimeError(f"{label}: no unique anchor; candidate counts={counts}")
'''
source = source[:start] + replacement + source[end:]
exec(compile(source, str(source_path), 'exec'))
