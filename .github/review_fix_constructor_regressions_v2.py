from pathlib import Path
import ast
import compileall
import runpy

# Apply the depth-aware constructor changes and test updates first.
runpy.run_path(".github/review_fix_constructor_regressions.py", run_name="__main__")


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


p = Path("statgpu/_base.py")
text = p.read_text(encoding="utf-8")
anchor = '''    _NORMALIZED_CONSTRUCTOR_PARAMS = frozenset({
'''
insert = '''    _NORMALIZED_PRIVATE_NAMES = {
        # ``_compute_inference`` is an established method name across model
        # families, so the constructor control needs a collision-free slot.
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


# Rewrite boolean/control references without touching method calls.
class ComputeInferenceTransformer(ast.NodeTransformer):
    def visit_Call(self, node):
        # Preserve the established method call ``self._compute_inference()``.
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "_compute_inference"
        ):
            node.args = [self.visit(arg) for arg in node.args]
            node.keywords = [self.visit(keyword) for keyword in node.keywords]
            return node
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "_compute_inference"
        ):
            node.attr = "_compute_inference_enabled"
        return node


for path in Path("statgpu").rglob("*.py"):
    source = path.read_text(encoding="utf-8")
    if "self._compute_inference" not in source:
        continue
    tree = ast.parse(source)
    updated = ComputeInferenceTransformer().visit(tree)
    ast.fix_missing_locations(updated)
    rendered = ast.unparse(updated) + "\n"
    path.write_text(rendered, encoding="utf-8")


# Public kwargs may be replaced directly between fits; synchronize them at the
# maintained fit boundary before group validation and penalty construction.
p = Path("statgpu/linear_model/penalized/_fit_mixin.py")
text = p.read_text(encoding="utf-8")
anchor = '''        if formula is not None:
'''
insert = '''        # Direct public parameter replacement is part of the established
        # refit contract. Keep runtime aliases synchronized before any group
        # validation, loss construction, or penalty resolution.
        self._penalty_kwargs = self.penalty_kwargs
        self._loss_kwargs = self.loss_kwargs

        if formula is not None:
'''
text = replace_once(text, anchor, insert, "penalized fit kwargs synchronization")
p.write_text(text, encoding="utf-8")


# Guard against future normalized parameter/method collisions.
normalized = set()
base_tree = ast.parse(Path("statgpu/_base.py").read_text(encoding="utf-8"))
for node in ast.walk(base_tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_NORMALIZED_CONSTRUCTOR_PARAMS":
                if isinstance(node.value, (ast.Set, ast.Call)):
                    values = node.value.args[0].elts if isinstance(node.value, ast.Call) else node.value.elts
                    normalized = {
                        item.value for item in values if isinstance(item, ast.Constant)
                    }
method_names = set()
for path in Path("statgpu").rglob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method_names.update(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
private_map = {"compute_inference": "_compute_inference_enabled"}
collisions = sorted(
    (name, private_map.get(name, f"_{name}"))
    for name in normalized
    if private_map.get(name, f"_{name}") in method_names
)
if collisions:
    raise SystemExit(f"normalized private-name collisions remain: {collisions}")

for path in Path("statgpu").rglob("*.py"):
    if not compileall.compile_file(str(path), quiet=1):
        raise SystemExit(f"compile failed: {path}")
