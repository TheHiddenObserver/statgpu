from pathlib import Path
import compileall
import runpy

runpy.run_path(".github/review_fix_constructor_regressions_v2.py", run_name="__main__")

p = Path("statgpu/linear_model/penalized/_fit_mixin.py")
text = p.read_text(encoding="utf-8")
old = '''        self._penalty_kwargs = self.penalty_kwargs
        self._loss_kwargs = self.loss_kwargs
'''
new = '''        self._penalty_kwargs = (
            self.penalty_kwargs if self.penalty_kwargs is not None else {}
        )
        self._loss_kwargs = self.loss_kwargs if self.loss_kwargs is not None else {}
'''
if text.count(old) != 1:
    raise SystemExit(f"fit kwargs normalization anchor count={text.count(old)}")
p.write_text(text.replace(old, new, 1), encoding="utf-8")

if not compileall.compile_file(str(p), quiet=1):
    raise SystemExit("penalized fit mixin failed to compile")
