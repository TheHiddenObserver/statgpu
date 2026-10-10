"""Parse and execute explicitly bounded, progressive documentation examples.

A named tutorial consists of every Python fence through its matching end tag.
Dependencies are declared immediately before the start tag; the runner supplies
no imports, fixtures, data, or fitted state that the reader has not been shown.
Legacy single-fence extraction is opt-in for unchanged reference/guide pages.
"""

import re
from dataclasses import dataclass

_TAG = re.compile(
    r"<!--\s*(learner-example|api-example|inference-example|example|example-end|/example|example-requires)"
    r"\s*:\s*([^>]*?)\s*-->"
)
_FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
_STARTS = {"learner-example", "api-example", "inference-example", "example"}
_NAME = re.compile(r"[\w-]+\Z")


@dataclass(frozen=True)
class DocExample:
    name: str
    blocks: tuple[str, ...]
    lines: tuple[int, ...]
    requires: tuple[str, ...]
    start: int
    end: int
    explicit_end: bool

    @property
    def code(self):
        return "\n\n".join(self.blocks)


def parse_examples(text, source="<documentation>", *, allow_legacy=False):
    """Return named examples, rejecting ambiguous boundaries and dependencies."""
    tags = list(_TAG.finditer(text))
    examples = {}
    pending = ()
    pending_end = None
    index = 0
    while index < len(tags):
        tag = tags[index]
        kind, name = tag.group(1, 2)
        if kind == "example-requires":
            assert not pending, f"{source}: repeated example-requires declaration"
            pending = tuple(part.strip() for part in name.split(","))
            assert all(_NAME.fullmatch(part) for part in pending), (
                f"{source}: malformed example-requires declaration"
            )
            assert len(set(pending)) == len(pending), f"{source}: duplicate dependency"
            pending_end = tag.end()
            index += 1
            continue
        assert kind in _STARTS, f"{source}: unexpected end tag {name}"
        assert _NAME.fullmatch(name), f"{source}: malformed example name {name}"
        assert name not in examples, f"{source}: duplicate example {name}"
        if pending:
            assert not text[pending_end:tag.start()].strip(), (
                f"{source}: example-requires must immediately precede {name}"
            )
        next_tag = tags[index + 1] if index + 1 < len(tags) else None
        explicit_end = next_tag is not None and next_tag.group(1) in {"example-end", "/example"}
        if explicit_end:
            assert next_tag.group(2) == name, f"{source}: mismatched end for {name}"
            content_end = next_tag.start()
            end = next_tag.end()
            index += 2
        else:
            assert allow_legacy, f"{source}: missing end tag for {name}"
            # Historical reference pages define one immediately following fence.
            # Never enable this fallback for restructured learner model pages.
            first = _FENCE.match(text, tag.end() + len(text[tag.end():]) - len(text[tag.end():].lstrip()))
            assert first is not None and first.group(1).strip() == "python", (
                f"{source}: legacy example {name} must start with one Python fence"
            )
            content_end = end = first.end()
            index += 1
        body = text[tag.end():content_end]
        fences = list(_FENCE.finditer(body))
        assert len(re.findall(r"^```", body, re.MULTILINE)) == 2 * len(fences), (
            f"{source}: unclosed fence in {name}"
        )
        python = [fence for fence in fences if fence.group(1).strip() == "python"]
        assert python, f"{source}: example {name} has no Python steps"
        blocks = tuple(fence.group(2) for fence in python)
        assert all(block.strip() for block in blocks), f"{source}: empty Python step in {name}"
        lines = tuple(text.count("\n", 0, tag.end() + fence.start(2)) + 1 for fence in python)
        examples[name] = DocExample(name, blocks, lines, pending, tag.start(), end, explicit_end)
        pending, pending_end = (), None
    assert not pending, f"{source}: orphan example-requires declaration"

    visiting, visited = set(), set()

    def visit(name):
        assert name in examples, f"{source}: unknown example dependency {name}"
        assert name not in visiting, f"{source}: cyclic example dependency {name}"
        if name in visited:
            return
        visiting.add(name)
        for dependency in examples[name].requires:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)

    for name in examples:
        visit(name)
    return examples


def example_code(text, name, source="<documentation>", *, allow_legacy=False):
    """Return only the selected example's steps for AST/contract inspection."""
    examples = parse_examples(text, source, allow_legacy=allow_legacy)
    assert name in examples, f"{source}: missing example {name}"
    return examples[name].code


def run_example(text, name, source="<documentation>", *, allow_legacy=False):
    """Run declared dependencies and every tutorial step, in a fresh namespace."""
    examples = parse_examples(text, source, allow_legacy=allow_legacy)
    assert name in examples, f"{source}: missing example {name}"
    namespace = {}
    executed = set()

    def run(current):
        if current in executed:
            return
        example = examples[current]
        for dependency in example.requires:
            run(dependency)
        for block, line in zip(example.blocks, example.lines):
            # Keep the real document line in exceptions without injecting setup.
            compiled = compile("\n" * (line - 1) + block, str(source), "exec")
            exec(compiled, namespace)  # noqa: S102 - repository-owned documentation
        executed.add(current)

    run(name)
    return namespace
