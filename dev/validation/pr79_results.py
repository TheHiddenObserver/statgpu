#!/usr/bin/env python3
"""
PR79 GPU Validation - Results Aggregation.

Reads structured JSON result files from a local results directory
and generates exit_decision.json, review_summary.md, and a
CLI summary table.

Usage:
    python dev/validation/pr79_results.py --results-dir results/pr79/20260721T120000Z
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_latest_results(base_dir=None):
    # type: (Optional[str]) -> Optional[Path]
    """Find the most recent results directory."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent / "results" / "pr79"
    else:
        base_dir = Path(base_dir)

    if not base_dir.exists():
        return None

    run_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir()],
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def load_result_file(results_dir, subdir, filename):
    # type: (Path, str, str) -> Optional[Any]
    """Load a JSON result file, returning None if not found."""
    path = results_dir / subdir / filename
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def generate_exit_decision(results_dir):
    # type: (Path) -> Dict[str, Any]
    """Generate exit_decision.json per Section 20 of the test plan.

    Reads all available result files and produces a structured
    decision record.
    """
    # Load available data
    env_probe = load_result_file(results_dir, "environment", "backend_probe.json")
    test_inventory = load_result_file(results_dir, "collection", "test_inventory.json")
    gate_a_json = load_result_file(results_dir, "junit", "gate_a.json")
    gate_b_json = load_result_file(results_dir, "junit", "gate_b.json")
    findings_json = load_result_file(results_dir, "failures", "findings.json")

    # Determine what gates ran
    phases_run = []
    for subdir in ["junit", "parity", "device", "memory", "performance", "external"]:
        s = results_dir / subdir
        if s.exists() and list(s.iterdir()):
            phases_run.append(subdir)

    # Parse gate A results from JUnit XML
    gate_a = _parse_gate_results(results_dir, "gate_a")
    gate_b = _parse_gate_results(results_dir, "gate_b")

    # Determine backend availability
    cupy_available = False
    torch_cuda_available = False
    if env_probe:
        cupy_available = env_probe.get("cupy", {}).get("available", False)
        torch_cuda_available = env_probe.get("torch", {}).get("available", False)

    # Count findings by severity
    critical_count = 0
    high_count = 0
    medium_count = 0
    if findings_json:
        for f in findings_json:
            sev = f.get("severity", "").upper()
            if sev == "CRITICAL":
                critical_count += 1
            elif sev == "HIGH":
                high_count += 1
            elif sev == "MEDIUM":
                medium_count += 1

    # Determine overall decision
    decisions = []
    ready = True

    if not gate_a.get("pass", False):
        decisions.append("Gate A (GPU smoke) not passed")
        ready = False
    if gate_b.get("total", 0) > 0 and not gate_b.get("pass", False):
        decisions.append("Gate B (numerical correctness) not passed")
        ready = False
    if critical_count > 0:
        decisions.append(f"{critical_count} CRITICAL findings open")
        ready = False
    if high_count > 0:
        decisions.append(f"{high_count} HIGH findings open")
        ready = False

    exit_decision = {
        "base_sha": "a4879fb4d9fb183efc01f147cd2cc501691f28c4",
        "head_sha": "e30cec6768a734a0d61dfec44b6b4884adf9a880",
        "cupy_cuda_available": cupy_available,
        "torch_cuda_available": torch_cuda_available,
        "phases_run": phases_run,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "mandatory_gpu_skips": gate_a.get("skipped", -1),
        "critical_open": critical_count,
        "high_open": high_count,
        "medium_open": medium_count,
        "cpu_ci_green": None,  # Set manually after GitHub CI check
        "gpu_correctness_green": gate_a.get("pass", False) and gate_b.get("pass", False),
        "gpu_memory_green": None,  # Set after Gate E
        "gpu_performance_reviewed": None,  # Set after Gate F
        "external_validation_green": None,  # Set after Gate G
        "decision": "READY_FOR_REVIEW" if ready else "BLOCKED",
        "blocking_issues": decisions,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    # Write exit decision
    out_path = results_dir / "final" / "exit_decision.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(exit_decision, f, indent=2)

    return exit_decision


def _parse_gate_results(results_dir, gate_name):
    # type: (Path, str) -> Dict[str, Any]
    """Parse gate results from JUnit XML."""
    import xml.etree.ElementTree as ET

    junit_path = results_dir / "junit" / f"{gate_name}.xml"
    if not junit_path.exists():
        return {"pass": None, "total": 0, "passed": 0, "failed": 0, "skipped": 0}

    try:
        tree = ET.parse(str(junit_path))
        root = tree.getroot()
        total = int(root.get("tests", 0))
        skipped = int(root.get("skipped", 0))
        errors = int(root.get("errors", 0))
        failures = int(root.get("failures", 0))
        passed = total - errors - failures - skipped

        # Gate condition: tests executed and zero failures
        gate_pass = (passed > 0 and total > 0 and errors == 0 and failures == 0)

        return {
            "pass": gate_pass,
            "total": total,
            "passed": passed,
            "failed": errors + failures,
            "skipped": skipped,
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
        }
    except Exception as e:
        return {"pass": False, "error": str(e)}


def generate_summary_md(results_dir):
    # type: (Path) -> str
    """Generate review_summary.md from all available data."""
    exit_dec = generate_exit_decision(results_dir)
    env_probe = load_result_file(results_dir, "environment", "backend_probe.json")

    lines = [
        f"# PR79 GPU Validation - Review Summary",
        f"",
        f"**Generated**: {exit_dec['generated_at']}",
        f"",
        f"## SHAs",
        f"",
        f"| Role | SHA |",
        f"|------|-----|",
        f"| Base | `{exit_dec['base_sha'][:12]}` |",
        f"| Head | `{exit_dec['head_sha'][:12]}` |",
        f"",
        f"## Environment",
        f"",
        f"| Component | Status |",
        f"|-----------|--------|",
        f"| CuPy CUDA | {'✅' if exit_dec['cupy_cuda_available'] else '❌'} |",
        f"| Torch CUDA | {'✅' if exit_dec['torch_cuda_available'] else '❌'} |",
    ]

    if env_probe:
        gpu_name = ""
        try:
            gpu_summary = (results_dir / "environment" / "gpu_summary.csv").read_text()
            gpu_name = gpu_summary.split(",")[0] if gpu_summary else ""
        except Exception:
            pass
        lines.extend([
            f"| GPU | {gpu_name} |",
            f"| CuPy version | {env_probe.get('cupy', {}).get('version', 'N/A')} |",
            f"| Torch version | {env_probe.get('torch', {}).get('version', 'N/A')} |",
            f"| NumPy version | {env_probe.get('numpy', {}).get('version', 'N/A')} |",
        ])

    lines.extend([
        f"",
        f"## Gate Results",
        f"",
        f"| Gate | Passed | Failed | Skipped | Total | Status |",
        f"|------|--------|--------|---------|-------|--------|",
    ])
    for gate_name in ["A", "B"]:
        g = exit_dec.get(f"gate_{gate_name.lower()}", {})
        if g.get("total", 0) > 0:
            status = "✅ PASS" if g.get("pass") else "❌ FAIL"
            lines.append(
                f"| Gate {gate_name} | {g.get('passed', 0)} | {g.get('failed', 0)} | "
                f"{g.get('skipped', 0)} | {g.get('total', 0)} | {status} |"
            )
        else:
            lines.append(f"| Gate {gate_name} | — | — | — | — | Not run |")

    lines.extend([
        f"",
        f"## Findings Summary",
        f"",
        f"| Severity | Count |",
        f"|----------|-------|",
        f"| CRITICAL | {exit_dec['critical_open']} |",
        f"| HIGH | {exit_dec['high_open']} |",
        f"| MEDIUM | {exit_dec['medium_open']} |",
        f"",
        f"## Decision",
        f"",
        f"**{exit_dec['decision']}**",
    ])

    if exit_dec["blocking_issues"]:
        lines.append(f"")
        lines.append(f"### Blocking Issues")
        for issue in exit_dec["blocking_issues"]:
            lines.append(f"- {issue}")

    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"### Status Fields")
    lines.append(f"")
    for field in ["cpu_ci_green", "gpu_correctness_green", "gpu_memory_green",
                   "gpu_performance_reviewed", "external_validation_green"]:
        val = exit_dec.get(field)
        status = "✅" if val else ("❌" if val is False else "⏳")
        lines.append(f"- {status} {field}: {val}")

    md_content = "\n".join(lines)

    # Write to file
    out_path = results_dir / "final" / "review_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content)

    return md_content


def print_cli_summary(results_dir):
    # type: (Path) -> None
    """Print a compact CLI summary table."""
    exit_dec = generate_exit_decision(results_dir)

    print(f"\n{'='*60}")
    print(f"  PR79 GPU Validation Results")
    print(f"{'='*60}")
    print(f"  Decision: {exit_dec['decision']}")
    print(f"  CuPy CUDA: {'YES' if exit_dec['cupy_cuda_available'] else 'NO'}")
    print(f"  Torch CUDA: {'YES' if exit_dec['torch_cuda_available'] else 'NO'}")
    print(f"  Phases run: {', '.join(exit_dec['phases_run']) or 'none'}")
    print(f"  Critical: {exit_dec['critical_open']}  "
          f"High: {exit_dec['high_open']}  "
          f"Medium: {exit_dec['medium_open']}")
    print(f"{'='*60}")

    # Gate summary
    for gate_name in ["A", "B"]:
        g = exit_dec.get(f"gate_{gate_name.lower()}", {})
        if g.get("total", 0) > 0:
            status = "PASS" if g.get("pass") else "FAIL"
            print(f"  Gate {gate_name}: {g['passed']}/{g['total']} {status} "
                  f"(skip: {g['skipped']}, fail: {g['failed']})")

    if exit_dec["blocking_issues"]:
        print(f"\n  Blocking issues:")
        for issue in exit_dec["blocking_issues"]:
            print(f"    - {issue}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PR79 GPU Validation - Results Aggregation",
    )
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to results directory (default: auto-find latest)")
    parser.add_argument("--latest", action="store_true",
                        help="Use latest results directory")

    args = parser.parse_args()

    if args.latest or args.results_dir is None:
        results_dir = find_latest_results(args.results_dir)
    else:
        results_dir = Path(args.results_dir)

    if results_dir is None:
        print("No results found. Run the orchestrator first.")
        return 1

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    print(f"Using results: {results_dir}")

    # Generate outputs
    summary_md = generate_summary_md(results_dir)
    print(f"\nGenerated: {results_dir / 'final' / 'review_summary.md'}")
    print(f"Generated: {results_dir / 'final' / 'exit_decision.json'}")

    # Print CLI summary
    print_cli_summary(results_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
