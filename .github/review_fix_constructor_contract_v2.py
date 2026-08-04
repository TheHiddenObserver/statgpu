from pathlib import Path
import runpy

path = Path(".github/review_fix_constructor_contract.py")
text = path.read_text(encoding="utf-8")
old = '''def offset(lines, lineno, col):
    return sum(len(line) for line in lines[: lineno - 1]) + col
'''
new = '''def offset(lines, lineno, col):
    # ast column offsets are UTF-8 byte offsets. Convert the line-local byte
    # position back to a Python character index before slicing source text.
    line = lines[lineno - 1]
    prefix = line.encode("utf-8")[:col].decode("utf-8")
    return sum(len(item) for item in lines[: lineno - 1]) + len(prefix)
'''
if text.count(old) != 1:
    raise SystemExit(f"offset anchor count={text.count(old)}")
text = text.replace(old, new, 1)
old = '''            start = offset(self.lines, node.lineno, node.col_offset)
            end = offset(self.lines, node.end_lineno, node.end_col_offset)
            self.replacements.append((start, end, f"self._{node.attr}"))
'''
new = '''            start = offset(self.lines, node.lineno, node.col_offset)
            end = offset(self.lines, node.end_lineno, node.end_col_offset)
            expected = f"self.{node.attr}"
            source = "".join(self.lines)
            if source[start:end] != expected:
                raise SystemExit(
                    f"unsafe attribute span {self.module}:{node.lineno}: "
                    f"{source[start:end]!r} != {expected!r}"
                )
            self.replacements.append((start, end, f"self._{node.attr}"))
'''
if text.count(old) != 1:
    raise SystemExit(f"replacement anchor count={text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
runpy.run_path(str(path), run_name="__main__")
