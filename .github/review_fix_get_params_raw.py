from pathlib import Path
import ast
import compileall

NORMALIZED = {
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
PRIVATE_TO_PUBLIC = {f"_{name}": name for name in NORMALIZED}
PRIVATE_TO_PUBLIC["_compute_inference_enabled"] = "compute_inference"


def char_offset(lines, lineno, byte_col):
    line = lines[lineno - 1]
    prefix = line.encode("utf-8")[:byte_col].decode("utf-8")
    return sum(len(item) for item in lines[: lineno - 1]) + len(prefix)


class GetParamsVisitor(ast.NodeVisitor):
    def __init__(self, source, lines, path):
        self.source = source
        self.lines = lines
        self.path = path
        self.function_stack = []
        self.replacements = []

    def visit_FunctionDef(self, node):
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node):
        if (
            self.function_stack
            and self.function_stack[-1] == "get_params"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr in PRIVATE_TO_PUBLIC
        ):
            start = char_offset(self.lines, node.lineno, node.col_offset)
            end = char_offset(self.lines, node.end_lineno, node.end_col_offset)
            expected = f"self.{node.attr}"
            if self.source[start:end] != expected:
                raise SystemExit(
                    f"unsafe get_params span {self.path}:{node.lineno}: "
                    f"{self.source[start:end]!r} != {expected!r}"
                )
            self.replacements.append(
                (start, end, f"self.{PRIVATE_TO_PUBLIC[node.attr]}")
            )
        self.generic_visit(node)


for path in Path("statgpu").rglob("*.py"):
    source = path.read_text(encoding="utf-8")
    if "def get_params" not in source:
        continue
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    visitor = GetParamsVisitor(source, lines, path)
    visitor.visit(tree)
    for start, end, replacement in sorted(visitor.replacements, reverse=True):
        source = source[:start] + replacement + source[end:]
    if visitor.replacements:
        path.write_text(source, encoding="utf-8")

# Static postcondition: no custom get_params reads normalized private fields.
offenders = []
for path in Path("statgpu").rglob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function_stack = []

    class Audit(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            function_stack.append(node.name)
            self.generic_visit(node)
            function_stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Attribute(self, node):
            if (
                function_stack
                and function_stack[-1] == "get_params"
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and node.attr in PRIVATE_TO_PUBLIC
            ):
                offenders.append((path.as_posix(), node.lineno, node.attr))
            self.generic_visit(node)

    Audit().visit(tree)
if offenders:
    raise SystemExit(f"normalized private get_params reads remain: {offenders}")

p = Path("dev/tests/test_maintenance_024_025.py")
text = p.read_text(encoding="utf-8")
text += r'''


def test_custom_get_params_do_not_expose_normalized_private_values():
    import ast
    from pathlib import Path

    normalized_private = {
        "_device", "_cov_type", "_hac_maxlags", "_gpu_memory_cleanup",
        "_solver", "_cpu_solver", "_stopping", "_inference_method",
        "_simultaneous_method", "_n_bootstrap",
        "_enable_simultaneous_inference", "_simultaneous_alpha",
        "_simultaneous_n_bootstrap", "_simultaneous_include_intercept",
        "_method", "_admm_rho", "_alpha_min_ratio", "_cd_kkt_check_every",
        "_compute_inference_enabled", "_cv", "_fit_intercept",
        "_gpu_cv_mixed_precision", "_max_iter", "_n_alphas", "_tol",
        "_n_Cs", "_C_min_ratio", "_penalty_kwargs", "_loss_kwargs",
        "_epsilon", "_ties", "_acknowledge_approx", "_refine_top_k",
        "_batch_size", "_min_effective_weight", "_quantile", "_cv_strategy",
    }
    offenders = []
    for path in Path("statgpu").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        stack = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Attribute(self, node):
                if (
                    stack
                    and stack[-1] == "get_params"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "self"
                    and node.attr in normalized_private
                ):
                    offenders.append((path.as_posix(), node.lineno, node.attr))
                self.generic_visit(node)

        Visitor().visit(tree)
    assert offenders == []


def test_tsne_nondefault_get_params_preserve_raw_identity():
    from sklearn.base import clone
    from statgpu.unsupervised import TSNE

    max_iter = np.int64(300)
    device = "".join(("c", "pu"))
    model = TSNE(max_iter=max_iter, device=device)
    params = model.get_params(deep=False)
    assert params["max_iter"] is max_iter
    assert params["device"] is device

    cloned = clone(model)
    cloned_params = cloned.get_params(deep=False)
    assert isinstance(cloned_params["max_iter"], np.integer)
    assert cloned_params["device"] == "cpu"
'''
p.write_text(text, encoding="utf-8")

for path in Path("statgpu").rglob("*.py"):
    if not compileall.compile_file(str(path), quiet=1):
        raise SystemExit(f"compile failed: {path}")
if not compileall.compile_file(str(p), quiet=1):
    raise SystemExit("maintenance tests failed to compile")
