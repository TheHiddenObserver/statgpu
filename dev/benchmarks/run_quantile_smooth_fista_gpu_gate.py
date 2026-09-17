#!/usr/bin/env python3
"""Exact-source wrapper for the PR #166 smooth Quantile FISTA CUDA gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve().with_name("validate_quantile_smooth_fista_gpu.py")
EXPECTED_SCHEMA_VERSION = 1


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def _require_clean_source() -> str:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if status:
        raise RuntimeError(
            "PR166 smooth-FISTA physical acceptance requires a clean exact-source worktree"
        )
    return sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="dev/reviews/pr166_quantile_smooth_fista_gpu.json",
    )
    args = parser.parse_args()

    source_before = _require_clean_source()
    with tempfile.TemporaryDirectory(prefix="statgpu-pr166-smooth-fista-") as temp_dir:
        temp_output = Path(temp_dir) / "pr166-smooth-fista-gpu.json"
        subprocess.run(
            [sys.executable, str(RUNNER), "--output", str(temp_output)],
            cwd=REPO_ROOT,
            check=True,
        )
        payload = json.loads(temp_output.read_text(encoding="utf-8"))

    source_after = _require_clean_source()
    if source_after != source_before:
        raise RuntimeError(
            "PR166 smooth-FISTA source HEAD changed during physical validation: "
            f"{source_before} -> {source_after}"
        )
    if payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            "PR166 smooth-FISTA inner-runner schema mismatch: "
            f"{payload.get('schema_version')!r} != {EXPECTED_SCHEMA_VERSION!r}"
        )
    if payload.get("source_sha") != source_before:
        raise RuntimeError(
            "PR166 smooth-FISTA inner-runner source SHA mismatch: "
            f"{payload.get('source_sha')!r} != {source_before!r}"
        )
    if payload.get("source_clean") is not True:
        raise RuntimeError("PR166 smooth-FISTA inner runner did not record a clean source")
    if payload.get("status") != "success":
        raise RuntimeError(
            "PR166 smooth-FISTA inner runner did not succeed: "
            f"{payload.get('status')!r}"
        )

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
