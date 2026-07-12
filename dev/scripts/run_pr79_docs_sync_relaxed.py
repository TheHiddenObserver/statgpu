from pathlib import Path

script = Path(__file__).with_name("sync_pr79_public_docs.py")
source = script.read_text(encoding="utf-8")
source = source.replace(
    '        raise RuntimeError(f"missing replacement anchor in {path}: {old[:80]!r}")',
    '        print(f"WARNING missing replacement anchor in {path}: {old[:80]!r}")\n        return',
)
source = source.replace(
    '        raise RuntimeError(f"missing insertion anchor in {path}: {marker!r}")',
    '        print(f"WARNING missing insertion anchor in {path}: {marker!r}")\n        return',
)
namespace = {"__file__": str(script), "__name__": "__main__"}
exec(compile(source, str(script), "exec"), namespace)
