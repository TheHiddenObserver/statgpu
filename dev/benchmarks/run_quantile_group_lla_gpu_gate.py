#!/usr/bin/env python3
"""Exact-source wrapper for PR #166 Quantile Group/low-level LLA CUDA gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GROUP_RUNNER = Path(__file__).resolve().with_name("validate_quantile_group_lla_gpu.py")
SCALAR_RUNNER = Path(__file__).resolve().with_name("validate_quantile_scalar_lla_gpu.py")
GROUP_SCHEMA_VERSION = 2
SCALAR_SCHEMA_VERSION = 1


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _require_clean_source() -> str:
    sha = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain"):
        raise RuntimeError(
            "PR166 Quantile Group/LLA physical acceptance requires a clean exact-source worktree"
        )
    return sha


def _run_inner(runner: Path, output: Path):
    subprocess.run(
        [sys.executable, str(runner), "--output", str(output)],
        cwd=REPO_ROOT,
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def _validate_inner(payload, *, source_sha, label, schema_version):
    if payload.get("schema_version") != schema_version:
        raise RuntimeError(
            f"PR166 {label} schema mismatch: "
            f"{payload.get('schema_version')!r} != {schema_version!r}"
        )
    if payload.get("source_sha") != source_sha:
        raise RuntimeError(
            f"PR166 {label} source mismatch: "
            f"{payload.get('source_sha')!r} != {source_sha!r}"
        )
    if payload.get("source_clean") is not True:
        raise RuntimeError(f"PR166 {label} did not record clean source")
    if payload.get("status") != "success":
        raise RuntimeError(
            f"PR166 {label} did not succeed: {payload.get('status')!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="dev/reviews/pr166_quantile_group_lla_gpu.json"
    )
    args = parser.parse_args()

    source_before = _require_clean_source()
    with tempfile.TemporaryDirectory(prefix="statgpu-pr166-quantile-lla-") as temp_dir:
        temp_dir = Path(temp_dir)
        group_payload = _run_inner(
            GROUP_RUNNER, temp_dir / "quantile-group-lla-gpu.json"
        )
        scalar_payload = _run_inner(
            SCALAR_RUNNER, temp_dir / "quantile-scalar-lla-gpu.json"
        )

    source_after = _require_clean_source()
    if source_after != source_before:
        raise RuntimeError(
            "PR166 Quantile Group/LLA source changed during physical validation: "
            f"{source_before} -> {source_after}"
        )
    _validate_inner(
        group_payload,
        source_sha=source_before,
        label="Quantile Group Proximal IRLS-LLA inner runner",
        schema_version=GROUP_SCHEMA_VERSION,
    )
    _validate_inner(
        scalar_payload,
        source_sha=source_before,
        label="Quantile low-level FISTA-LLA inner runner",
        schema_version=SCALAR_SCHEMA_VERSION,
    )

    payload = dict(group_payload)
    payload["low_level_scalar_fista_lla"] = scalar_payload
    payload["source_sha_before"] = source_before
    payload["source_sha_after_execution"] = source_after
    payload["source_clean_before"] = True
    payload["source_clean_after_execution"] = True
    payload["evidence_wrapper"] = str(
        Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    )

    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
