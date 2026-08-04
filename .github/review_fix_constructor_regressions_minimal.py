from pathlib import Path
import ast
import compileall
import runpy

# Apply the depth-aware constructor fix and cross-version test updates.
runpy.run_path(".github/review_fix_constructor_regressions.py", run_name="__main__")


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


# Use a collision-free private slot for the compute_inference control.
p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
anchor = '''    _NORMALIZED_CONSTRUCTOR_PARAMS = frozenset({
'''
insert = '''    _NORMALIZED_PRIVATE_NAMES = {
        "compute_inference": "_compute_inference_enabled",
    }

    @classmethod
    def _normalized_private_name(cls, name):
        return cls._NORMALIZED_PRIVATE_NAMES.get(name, f"_{name}")

    _NORMALIZED_CONSTRUCTOR_PARAMS = frozenset({
'''
text = replace_once(text, anchor, insert, "private-name mapping")
text = text.replace(
    'private_name = f"_{name}"',
    'private_name = type(self)._normalized_private_name(name)',
)
old = '''                if key in self._NORMALIZED_CONSTRUCTOR_PARAMS:
                    setattr(self, f"_{key}", value)
'''
new = '''                if key in self._NORMALIZED_CONSTRUCTOR_PARAMS:
                    setattr(self, self._normalized_private_name(key), value)
'''
text = replace_once(text, old, new, "deferred mapped private name")
p.write_text(text, encoding="utf-8")


# Minimal byte-offset rewrite: change control reads but preserve method calls and
# all surrounding source formatting/comments.
def char_offset(lines, lineno, byte_col):
    line = lines[lineno - 1]
    prefix = line.encode("utf-8")[:byte_col].decode("utf-8")
    return sum(len(item) for item in lines[: lineno - 1]) + len(prefix)


for path in Path("statgpu").rglob("*.py"):
    source = path.read_text(encoding="utf-8")
    if "self._compute_inference" not in source:
        continue
    tree = ast.parse(source)
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    lines = source.splitlines(keepends=True)
    replacements = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "_compute_inference"
        ):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        start = char_offset(lines, node.lineno, node.col_offset)
        end = char_offset(lines, node.end_lineno, node.end_col_offset)
        expected = "self._compute_inference"
        if source[start:end] != expected:
            raise SystemExit(
                f"unsafe compute_inference span {path}:{node.lineno}: "
                f"{source[start:end]!r}"
            )
        replacements.append((start, end, "self._compute_inference_enabled"))
    for start, end, replacement in sorted(replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    if replacements:
        path.write_text(source, encoding="utf-8")


# Synchronize directly replaced public kwargs at the fit boundary while keeping
# None normalized to the runtime empty mapping.
p = Path("statgpu/linear_model/penalized/_fit_mixin.py")
text = p.read_text(encoding="utf-8")
anchor = '''        if formula is not None:
'''
insert = '''        # Direct public parameter replacement is part of the refit API.
        self._penalty_kwargs = (
            self.penalty_kwargs if self.penalty_kwargs is not None else {}
        )
        self._loss_kwargs = self.loss_kwargs if self.loss_kwargs is not None else {}

        if formula is not None:
'''
text = replace_once(text, anchor, insert, "fit kwargs synchronization")
p.write_text(text, encoding="utf-8")


# Static safety: normalized private names must not collide with methods.
normalized_names = {
    "device", "cov_type", "hac_maxlags", "gpu_memory_cleanup", "solver",
    "cpu_solver", "stopping", "inference_method", "simultaneous_method",
    "n_bootstrap", "enable_simultaneous_inference", "simultaneous_alpha",
    "simultaneous_n_bootstrap", "simultaneous_include_intercept", "method",
    "admm_rho", "alpha_min_ratio", "cd_kkt_check_every", "compute_inference",
    "cv", "fit_intercept", "gpu_cv_mixed_precision", "max_iter", "n_alphas",
    "tol", "n_Cs", "C_min_ratio", "penalty_kwargs", "loss_kwargs", "epsilon",
    "ties", "acknowledge_approx", "refine_top_k", "batch_size",
    "min_effective_weight", "quantile", "cv_strategy",
}
private_map = {"compute_inference": "_compute_inference_enabled"}
method_names = set()
for path in Path("statgpu").rglob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method_names.update(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
collisions = sorted(
    (name, private_map.get(name, f"_{name}"))
    for name in normalized_names
    if private_map.get(name, f"_{name}") in method_names
)
if collisions:
    raise SystemExit(f"normalized private-name collisions remain: {collisions}")

for path in Path("statgpu").rglob("*.py"):
    if not compileall.compile_file(str(path), quiet=1):
        raise SystemExit(f"compile failed: {path}")
for path in (
    "dev/tests/test_core_contracts.py",
    "dev/tests/test_pr80_cv_fit_boundary.py",
    "dev/tests/test_pr80_fit_boundary.py",
    "dev/tests/test_maintenance_024_025.py",
):
    if not compileall.compile_file(path, quiet=1):
        raise SystemExit(f"test compile failed: {path}")
