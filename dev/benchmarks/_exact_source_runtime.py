"""Exact-checkout runtime import provenance for benchmark suites."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


_PROBE = r"""
import hashlib
import importlib
import json
from pathlib import Path
import sys

payload = json.loads(sys.argv[1])
root = Path(payload["repo_root"]).resolve()
failures = []
modules = {}

for name in payload["modules"]:
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    if origin is None:
        failures.append(f"{name}: imported module has no __file__")
        continue
    path = Path(origin).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        failures.append(f"{name}: imported outside checkout: {path}")
        relative = None
    modules[name] = {
        "path": str(path),
        "relative_path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

expected_statgpu = (root / "statgpu" / "__init__.py").resolve()
actual_statgpu = modules.get("statgpu", {}).get("path")
if actual_statgpu is not None and Path(actual_statgpu).resolve() != expected_statgpu:
    failures.append(
        "statgpu import root mismatch: "
        f"expected {expected_statgpu}, got {actual_statgpu}"
    )

print(
    json.dumps(
        {
            "repo_root": str(root),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "pythonpath": list(sys.path),
            "modules": modules,
            "gate_failures": failures,
            "passed": not failures,
        },
        sort_keys=True,
    )
)
"""


def _git_root() -> Path:
    """Return the exact checkout root used by the current benchmark command."""
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    ).resolve()


def _exact_source_env(root: Path) -> dict[str, str]:
    """Build a child environment that resolves this checkout before installs."""
    env = os.environ.copy()
    retained = []
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        try:
            if Path(entry).resolve() == root:
                continue
        except OSError:
            pass
        retained.append(entry)
    env["PYTHONPATH"] = os.pathsep.join([str(root), *retained])
    env["PYTHONNOUSERSITE"] = "1"
    return env


def prepare_exact_source_runtime(
    module_names: Iterable[str],
) -> tuple[Path, dict[str, str], dict, list[str]]:
    """Bind child processes to this checkout and audit actual imported files.

    The returned environment must be passed unchanged to every benchmark child
    or sub-runner. The probe imports the requested modules in a fresh Python
    process under that environment, verifies every ``__file__`` lies inside the
    current Git checkout, and hashes the files that Python actually imported.
    """
    modules = tuple(dict.fromkeys(str(name) for name in module_names))
    if not modules or "statgpu" not in modules:
        raise ValueError("module_names must include 'statgpu'")

    root = _git_root()
    env = _exact_source_env(root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            json.dumps({"repo_root": str(root), "modules": modules}),
        ],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    failures: list[str] = []
    provenance: dict
    try:
        stdout_lines = [
            line for line in completed.stdout.splitlines() if line.strip()
        ]
        provenance = json.loads(stdout_lines[-1])
    except Exception as exc:
        provenance = {
            "repo_root": str(root),
            "modules": {},
            "passed": False,
            "gate_failures": [],
            "probe_stdout": completed.stdout[-4000:],
            "probe_stderr": completed.stderr[-4000:],
        }
        failures.append(
            f"runtime import probe returned invalid JSON: {type(exc).__name__}: {exc}"
        )

    if completed.returncode != 0:
        failures.append(f"runtime import probe returncode={completed.returncode}")
    if completed.stderr.strip():
        provenance["probe_stderr"] = completed.stderr[-4000:]
    failures.extend(str(item) for item in provenance.get("gate_failures") or [])
    if provenance.get("repo_root") != str(root):
        failures.append(
            "runtime import probe repo_root mismatch: "
            f"expected {root}, got {provenance.get('repo_root')}"
        )
    if not bool(provenance.get("passed", False)):
        failures.append("runtime import provenance did not pass")

    failures = list(dict.fromkeys(failures))
    provenance["gate_failures"] = failures
    provenance["passed"] = not failures
    provenance["python_no_user_site"] = env.get("PYTHONNOUSERSITE")
    provenance["effective_pythonpath"] = env.get("PYTHONPATH", "")
    return root, env, provenance, failures


__all__ = ["prepare_exact_source_runtime"]
