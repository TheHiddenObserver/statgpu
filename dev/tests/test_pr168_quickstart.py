"""Execute the bilingual CPU quickstarts without optional dependencies."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("en", "cn")
OPTIONAL_MODULES = ("cupy", "torch", "pandas", "patsy", "sklearn", "statsmodels")


def _quickstart(language):
    path = ROOT / f"docs/{language}/getting-started/quickstart.md"
    return path, path.read_text(encoding="utf-8")


def _fences(text, language):
    return list(re.finditer(
        rf"^```{language}\s*\n(.*?)^```\s*$", text, re.MULTILINE | re.DOTALL,
    ))


def _device_values(code):
    values = set()
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "device":
                values.add(ast.literal_eval(keyword.value))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "set_device":
            values.add(ast.literal_eval(node.args[0]))
    return values


# An import finder simulates genuinely absent optional packages. Inserting None
# into sys.modules would instead break SciPy's array-api Torch type detection.
_BASE_ONLY_RUNNER = r'''
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import warnings

payload = json.load(sys.stdin)
blocked = set(payload["blocked"])
assert not any(name.split(".")[0] in blocked for name in sys.modules)

class HideOptionalDependencies:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(
                f"No module named '{fullname}'", name=fullname,
            )

sys.meta_path.insert(0, HideOptionalDependencies())
for name in blocked:
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as exc:
        assert exc.name == name
    else:
        raise AssertionError(f"Optional dependency was not hidden: {name}")

# Import the actual base dependencies after hiding the extras. Do not preload
# SciPy or statgpu in the pytest process to bypass a broken base-only import.
import numpy as np
import scipy
import joblib
import statgpu as sg
from statgpu._config import _get_configured_device

assert Path(sg.__file__).resolve().is_relative_to(Path(payload["root"]).resolve())
initial_device = _get_configured_device()
assert initial_device is sg.Device.AUTO

def run(code):
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(output):
            exec(compile(code, payload["path"], "exec"), namespace)
    assert not caught, [str(warning.message) for warning in caught]
    assert _get_configured_device() is initial_device
    assert not any(name.split(".")[0] in blocked for name in sys.modules)
    return namespace, output.getvalue()

if payload["kind"] == "device":
    namespace, output = run(payload["code"])
    assert output.strip() == "cpu"
    assert sg.get_device() is sg.Device.CPU
elif payload["kind"] == "minimal":
    random_state = np.random.get_state()
    namespace, output = run(payload["code"])
    repeat, repeated_output = run(payload["code"])
    assert repeated_output == output
    for before, after in zip(random_state, np.random.get_state()):
        np.testing.assert_array_equal(before, after)

    X, y, model = (namespace[name] for name in ("X", "y", "model"))
    assert isinstance(model, sg.LinearRegression)
    assert model.device == "cpu"
    assert X.shape == (1000, 20) and y.shape == (1000,)
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    np.testing.assert_array_equal(X, repeat["X"])
    np.testing.assert_array_equal(y, repeat["y"])

    # Independently reconstruct the intercept-augmented least-squares solution
    # and R-squared; no second statgpu fit or optional validation library.
    design = np.column_stack((np.ones(len(X)), X))
    expected = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = design @ expected
    score = 1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)
    np.testing.assert_allclose(model.coef_, expected[1:], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model.intercept_, expected[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model.predict(X), prediction, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model.score(X, y), score, rtol=1e-12, atol=1e-12)
    assert np.isfinite(score) and 0.99 < score < 1.0

    # Verify the values the reader sees, including NumPy's display rounding.
    lines = output.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("[") and lines[0].endswith("]")
    printed_prediction = np.fromstring(lines[0][1:-1], sep=" ")
    assert printed_prediction.shape == (3,)
    assert np.isfinite(printed_prediction).all()
    np.testing.assert_allclose(printed_prediction, prediction[:3], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(float(lines[1]), score, rtol=1e-12, atol=1e-12)
else:
    assert payload["kind"] == "optional"
    namespace, _ = run(payload["setup"])
    for code, device, backend in payload["examples"]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                exec(compile(code, payload["path"], "exec"), namespace)
            except RuntimeError as exc:
                assert backend.lower() in str(exc).lower(), str(exc)
            else:
                raise AssertionError(f"Explicit {device} silently succeeded without {backend}")
        assert not caught, [str(warning.message) for warning in caught]
        assert namespace["model"].device == device
        assert _get_configured_device() is initial_device
        assert not any(name.split(".")[0] in blocked for name in sys.modules)
'''


def _run_base_only(language, kind, **payload):
    path, _ = _quickstart(language)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT), env.get("PYTHONPATH"),
    )))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[variable] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_BASE_ONLY_RUNNER)],
        input=json.dumps({
            "root": str(ROOT), "path": str(path), "blocked": OPTIONAL_MODULES,
            "kind": kind, **payload,
        }),
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("language", LANGUAGES)
def test_first_example_is_deterministic_base_only_cpu_least_squares(language):
    _, text = _quickstart(language)
    first = _fences(text, "python")[0].group(1)
    assert _device_values(first) == {"cpu"}
    _run_base_only(language, "minimal", code=first)


@pytest.mark.parametrize("language", LANGUAGES)
def test_cpu_device_control_restores_the_automatic_global_policy(language):
    _, text = _quickstart(language)
    code = _fences(text, "python")[1].group(1)
    assert _device_values(code) == {"cpu", "auto"}
    _run_base_only(language, "device", code=code)


@pytest.mark.parametrize("language", LANGUAGES)
def test_gpu_examples_are_separate_opt_ins_and_raise_without_the_backend(language):
    _, text = _quickstart(language)
    fences = _fences(text, "python")
    gpu_heading = re.search(r"^## .*GPU.*$", text, re.MULTILINE)
    assert gpu_heading is not None and gpu_heading.start() > fences[1].end()
    examples = []
    for fence in fences[2:]:
        assert fence.start() > gpu_heading.end()
        code = fence.group(1)
        devices = _device_values(code)
        assert len(devices) == 1
        device = devices.pop()
        assert device in {"cuda", "torch"}
        examples.append((code, device, {"cuda": "CuPy", "torch": "PyTorch"}[device]))
    assert {device for _, device, _ in examples} == {"cuda", "torch"}
    _run_base_only(language, "optional", setup=fences[0].group(1), examples=examples)


def _dependency_metadata():
    # The dependency entries are literal string lists. Keep this small parser
    # compatible with supported Python 3.9, which has no stdlib tomllib.
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    base = re.search(r"^dependencies\s*=\s*(\[[^]]*\])", text, re.MULTILINE)
    section = re.search(
        r"^\[project\.optional-dependencies\]\n(.*?)(?=^\[|\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    assert base is not None and section is not None
    extras = {
        name: ast.literal_eval(requirements)
        for name, requirements in re.findall(
            r"^([\w-]+)\s*=\s*(\[.*\])\s*$", section.group(1), re.MULTILINE,
        )
    }
    return ast.literal_eval(base.group(1)), extras


@pytest.mark.parametrize("language", LANGUAGES)
def test_base_install_and_gpu_extras_match_project_dependencies(language):
    _, text = _quickstart(language)
    install = _fences(text, "bash")[0].group(1)
    assert re.search(r"\bpip install -e \.\s*$", install, re.MULTILINE)
    assert not re.search(r"(?:statgpu|\.)\[", install)

    base, extras = _dependency_metadata()
    base_names = {re.match(r"[\w-]+", requirement)[0] for requirement in base}
    assert {"numpy", "scipy", "joblib"} <= base_names
    assert not base_names.intersection({
        "cupy", "cupy-cuda11x", "cupy-cuda12x", "torch", "pandas", "patsy",
        "scikit-learn", "statsmodels",
    })
    documented = set(re.findall(r"(?:statgpu|\.)\[([\w-]+)\]", text))
    assert documented == {"gpu11", "gpu12", "torch"}
    assert documented <= extras.keys()
    for extra in ("gpu11", "gpu12"):
        assert len(extras[extra]) == 1 and extras[extra][0].startswith("cupy-cuda")
        assert extras[extra][0] in text
    torch_requirement = next(item for item in extras["torch"] if item.startswith("torch>="))
    assert f"PyTorch {torch_requirement.split('>=')[1]}" in text


@pytest.mark.parametrize("language", LANGUAGES)
def test_torch_guides_link_contributor_evidence_and_preserve_user_references(language):
    path = ROOT / f"docs/{language}/guides/pytorch-backend.md"
    text = path.read_text(encoding="utf-8")
    target = "../../../dev/references/model-validation.md#torch-backend"
    assert target in text
    reference = (path.parent / target.split("#")[0]).resolve()
    assert reference.is_file()
    reference_text = reference.read_text(encoding="utf-8")
    assert "## Torch backend" in reference_text
    assert "dev/benchmarks/" not in text
    for link in (
        "device-and-memory.md#current-smoothing-and-spline-exceptions",
        "implemented-methods.md",
        "https://pytorch.org/docs/",
    ):
        assert link in text
    if language == "en":
        assert '<a id="performance-and-validation-evidence"></a>' in text
        assert "exact commit SHA" not in text


def test_contributor_reference_retains_torch_evidence_checklist():
    text = (ROOT / "dev/references/model-validation.md").read_text(encoding="utf-8")
    section = text.split("## Torch backend", 1)[1].split("\n## ", 1)[0]
    for topic in (
        "commit SHA", "fingerprint", "Python", "Torch", "CUDA", "driver",
        "GPU model", "synchronized timing", "accuracy", "statistical parity",
        "passed", "failed", "skipped", "physical CUDA", "results/", "dev/benchmarks/",
    ):
        assert topic in section
    report = re.search(r"\]\(([^)#]*torch_backend_final_report\.md)\)", section)
    assert report is not None
    assert (ROOT / "dev/references" / report.group(1)).resolve().is_file()
