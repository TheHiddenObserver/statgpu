"""
Remote test runner for PR #54 review fixes.

Runs three test phases on the remote GPU server:
1. Functional tests (pytest)
2. Precision regression tests (compare against baseline)
3. Performance benchmarks (timing comparison)

Usage:
    set STATGPU_SSH_HOST=hz-4.matpool.com
    set STATGPU_SSH_PORT=28838
    set STATGPU_SSH_USER=root
    set STATGPU_SSH_PASSWORD=<password>
    python dev/tests/_remote_review_test.py [--phase=all|functional|precision|performance]
"""
import os
import sys
import time
import json
import paramiko


def get_ssh_config():
    """Read SSH config from environment variables."""
    config = {
        "hostname": os.environ.get("STATGPU_SSH_HOST"),
        "port": int(os.environ.get("STATGPU_SSH_PORT", "22")),
        "username": os.environ.get("STATGPU_SSH_USER"),
        "password": os.environ.get("STATGPU_SSH_PASSWORD"),
    }
    missing = [k for k, v in config.items() if v is None]
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {missing}\n"
            "Set STATGPU_SSH_HOST, STATGPU_SSH_PORT, STATGPU_SSH_USER, STATGPU_SSH_PASSWORD"
        )
    return config


def run_remote_command(client, cmd, timeout=300):
    """Execute command on remote server, return (exit_code, stdout, stderr)."""
    full_cmd = (
        "source /root/miniconda3/etc/profile.d/conda.sh && "
        "conda activate myconda && "
        f"{cmd}"
    )
    stdin, stdout, stderr = client.exec_command(full_cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, stdout.read().decode(), stderr.read().decode()


def upload_file(client, local_path, remote_path):
    """Upload a file via SFTP."""
    sftp = client.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()


def run_functional_tests(client):
    """Phase 1: Run pytest test suites."""
    test_commands = [
        ("PR54 dispatch table tests",
         "cd /root/statgpu && python -m pytest dev/tests/test_pr54_dispatch_table.py -v --tb=short 2>&1"),
        ("Review fixes tests",
         "cd /root/statgpu && python -m pytest dev/tests/test_review_fixes.py -v --tb=short 2>&1"),
    ]
    all_passed = True
    for name, cmd in test_commands:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print("=" * 60)
        exit_code, stdout, stderr = run_remote_command(client, cmd, timeout=600)
        print(stdout)
        if stderr:
            print(f"STDERR:\n{stderr}")
        if exit_code != 0:
            print(f"FAILED (exit code {exit_code})")
            all_passed = False
        else:
            print("PASSED")
    return all_passed


def run_precision_tests(client, mode="verify"):
    """Phase 2: Precision regression tests."""
    print(f"\n{'='*60}")
    print(f"Precision regression: {mode}")
    print("=" * 60)
    # Upload the precision test script
    upload_file(
        client,
        "dev/tests/_precision_regression.py",
        "/root/statgpu/dev/tests/_precision_regression.py",
    )
    # Also upload baseline if verifying
    if mode == "verify" and os.path.exists("dev/tests/_precision_baseline.json"):
        upload_file(
            client,
            "dev/tests/_precision_baseline.json",
            "/root/statgpu/dev/tests/_precision_baseline.json",
        )

    cmd = f"cd /root/statgpu && python dev/tests/_precision_regression.py --mode={mode} 2>&1"
    exit_code, stdout, stderr = run_remote_command(client, cmd, timeout=900)
    print(stdout)
    if stderr:
        print(f"STDERR:\n{stderr}")

    if mode == "baseline":
        # Download baseline back
        sftp = client.open_sftp()
        try:
            sftp.get(
                "/root/statgpu/dev/tests/_precision_baseline.json",
                "dev/tests/_precision_baseline.json",
            )
            print("Baseline downloaded to dev/tests/_precision_baseline.json")
        except FileNotFoundError:
            print("WARNING: Baseline file not found on remote")
        sftp.close()

    return exit_code == 0


def run_performance_tests(client):
    """Phase 3: Performance benchmark comparison."""
    print(f"\n{'='*60}")
    print("Performance benchmark")
    print("=" * 60)

    # Upload the precision test script (it contains performance configs too)
    upload_file(
        client,
        "dev/tests/_precision_regression.py",
        "/root/statgpu/dev/tests/_precision_regression.py",
    )

    # Run performance-only mode
    cmd = "cd /root/statgpu && python dev/tests/_precision_regression.py --mode=performance 2>&1"
    exit_code, stdout, stderr = run_remote_command(client, cmd, timeout=600)
    print(stdout)
    if stderr:
        print(f"STDERR:\n{stderr}")
    return exit_code == 0


def main():
    phase = "all"
    for arg in sys.argv[1:]:
        if arg.startswith("--phase="):
            phase = arg.split("=", 1)[1]

    config = get_ssh_config()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(**config)
    print(f"Connected to {config['hostname']}:{config['port']}")

    # 1. Upload modified files
    files_to_upload = [
        (
            "statgpu/linear_model/_penalized_cv.py",
            "/root/statgpu/statgpu/linear_model/_penalized_cv.py",
        ),
        (
            "statgpu/backends/_array_ops.py",
            "/root/statgpu/statgpu/backends/_array_ops.py",
        ),
    ]
    for local, remote in files_to_upload:
        if os.path.exists(local):
            upload_file(client, local, remote)
            print(f"Uploaded: {local} -> {remote}")
        else:
            print(f"WARNING: {local} not found, skipping")

    # 2. Clear __pycache__
    run_remote_command(
        client,
        "find /root/statgpu -name '__pycache__' -exec rm -rf {} + 2>/dev/null; true",
    )
    print("Cleared __pycache__")

    # 3. Run requested phases
    all_ok = True
    if phase in ("all", "functional"):
        all_ok &= run_functional_tests(client)
    if phase in ("all", "precision"):
        all_ok &= run_precision_tests(client, mode="verify")
    if phase in ("all", "performance"):
        all_ok &= run_performance_tests(client)

    client.close()
    print(f"\n{'='*60}")
    print(f"Result: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
