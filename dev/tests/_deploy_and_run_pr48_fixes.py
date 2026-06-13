"""
Deploy PR48 fixes to remote GPU server and run three-backend tests.

Credentials are read from environment variables (never hardcoded).

Usage:
    set STATGPU_SSH_HOST=hz-4.matpool.com
    set STATGPU_SSH_PORT=28838
    set STATGPU_SSH_USER=root
    set STATGPU_SSH_PASS=<password>
    python dev/tests/_deploy_and_run_pr48_fixes.py
"""

from __future__ import annotations

import os
import sys
import time
import paramiko


def get_ssh_credentials():
    """Read SSH credentials from environment variables."""
    host = os.environ.get("STATGPU_SSH_HOST", "hz-4.matpool.com")
    port = int(os.environ.get("STATGPU_SSH_PORT", "28838"))
    user = os.environ.get("STATGPU_SSH_USER", "root")
    password = os.environ.get("STATGPU_SSH_PASS", "")
    if not password:
        raise RuntimeError(
            "STATGPU_SSH_PASS environment variable not set. "
            "Set it before running this script."
        )
    return host, port, user, password


def run_remote(ssh, cmd, timeout=300):
    """Run a command on the remote server and return stdout+stderr."""
    print(f"  >> {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out, err


def upload_file(sftp, local_path, remote_path):
    """Upload a single file via SFTP."""
    sftp.put(local_path, remote_path)


def deploy_code(ssh, sftp):
    """Upload modified source files to remote server."""
    base_local = os.path.join(os.path.dirname(__file__), "..", "..", "statgpu")
    base_local = os.path.normpath(base_local)
    base_remote = "/root/statgpu/statgpu/statgpu/"

    # Files modified in this PR48 fix round
    files_to_upload = [
        # Panel module
        "panel/__init__.py",
        "panel/_fixed_effects.py",
        "panel/_random_effects.py",
        "panel/_utils.py",
        "panel/_covariance.py",
        # ANOVA
        "anova/__init__.py",
        "anova/_oneway.py",
        # GAM
        "semiparametric/__init__.py",
        "semiparametric/_gam.py",
        # IRLS
        "glm_core/_irls.py",
        # KernelRidge
        "nonparametric/kernel_methods/_krr.py",
        # Covariance
        "covariance/_empirical.py",
    ]

    print(f"\n=== Deploying {len(files_to_upload)} files ===")
    for f in files_to_upload:
        local = os.path.join(base_local, f)
        remote = base_remote + f.replace("\\", "/")
        if os.path.exists(local):
            # Ensure remote directory exists
            remote_dir = os.path.dirname(remote)
            run_remote(ssh, f"mkdir -p {remote_dir}")
            upload_file(sftp, local, remote)
            print(f"  OK: {f}")
        else:
            print(f"  SKIP (not found): {local}")

    # Also upload the test file
    test_local = os.path.join(os.path.dirname(__file__), "test_pr48_fixes.py")
    test_remote = "/root/statgpu/dev/tests/test_pr48_fixes.py"
    if os.path.exists(test_local):
        run_remote(ssh, "mkdir -p /root/statgpu/dev/tests/")
        upload_file(sftp, test_local, test_remote)
        print(f"  OK: test_pr48_fixes.py")


def run_tests(ssh):
    """Run the test suite on the remote server."""
    conda_prefix = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate myconda && "

    # Clear Python bytecode cache to ensure fresh imports
    print("\n=== Clearing cache ===")
    out, err = run_remote(ssh, "find /root/statgpu -name '__pycache__' -exec rm -rf {} + 2>/dev/null; echo 'Cache cleared'")
    print(f"  {out.strip()}")

    # Install package in development mode
    print("\n=== Installing statgpu in dev mode ===")
    out, err = run_remote(ssh, conda_prefix + "cd /root/statgpu && pip install -e . 2>&1 | tail -5", timeout=120)
    print(f"  {out.strip()}")

    # Run the PR48 fix tests with verbose output
    print("\n=== Running PR48 fix tests (numpy + cupy + torch) ===")
    cmd = (
        conda_prefix
        + "cd /root/statgpu && python -m pytest dev/tests/test_pr48_fixes.py -v --tb=short 2>&1"
    )
    out, err = run_remote(ssh, cmd, timeout=300)
    print(out)
    if err and "error" in err.lower():
        print(f"STDERR:\n{err}")

    # Run the existing panel/anova/covariance tests
    print("\n=== Running existing tests ===")
    cmd = (
        conda_prefix
        + "cd /root/statgpu && python -m pytest dev/tests/test_panel.py dev/tests/test_anova.py dev/tests/test_covariance.py dev/tests/test_backward_compat_imports.py dev/tests/test_splines.py -q --tb=short 2>&1"
    )
    out, err = run_remote(ssh, cmd, timeout=300)
    print(out)

    return out


def run_benchmarks(ssh):
    """Run three-backend performance comparison on remote server."""
    conda_prefix = "source /root/miniconda3/etc/profile.d/conda.sh && conda activate myconda && "

    benchmark_script = '''
import time
import numpy as np

def to_backend(arr, backend):
    if backend == "cupy":
        import cupy
        return cupy.asarray(arr)
    elif backend == "torch":
        import torch
        return torch.tensor(arr, dtype=torch.float64, device="cuda")
    return arr

def to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy()
    return np.asarray(arr)

results = {}

# --- ANOVA benchmark ---
print("\\n=== ANOVA Benchmark ===")
from statgpu.anova import f_oneway
rng = np.random.default_rng(42)
groups_np = [rng.standard_normal(1000) + i for i in range(5)]

for backend in ["numpy", "cupy", "torch"]:
    groups = [to_backend(g, backend) for g in groups_np]
    # Warmup
    f_oneway(*groups, backend=backend)
    # Timed
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        f_oneway(*groups, backend=backend)
        times.append(time.perf_counter() - t0)
    mean_t = np.mean(times)
    std_t = np.std(times)
    results[f"anova_{backend}"] = (mean_t, std_t)
    print(f"  {backend:>8}: {mean_t*1000:.2f} +/- {std_t*1000:.2f} ms")

# --- PanelOLS benchmark ---
print("\\n=== PanelOLS Benchmark ===")
from statgpu.panel import PanelOLS
rng = np.random.default_rng(42)
n, k = 2000, 10
X_np = rng.standard_normal((n, k))
eids_np = np.repeat(np.arange(100), 20)
y_np = X_np @ np.ones(k) + rng.standard_normal(n) * 0.1

for backend in ["numpy", "cupy", "torch"]:
    X = to_backend(X_np, backend)
    y = to_backend(y_np, backend)
    eids = to_backend(eids_np, backend)
    # Warmup
    m = PanelOLS(entity_effects=True)
    m.fit(X, y, entity_ids=eids)
    # Timed
    times = []
    for _ in range(10):
        m = PanelOLS(entity_effects=True)
        t0 = time.perf_counter()
        m.fit(X, y, entity_ids=eids)
        times.append(time.perf_counter() - t0)
    mean_t = np.mean(times)
    std_t = np.std(times)
    results[f"panel_{backend}"] = (mean_t, std_t)
    print(f"  {backend:>8}: {mean_t*1000:.2f} +/- {std_t*1000:.2f} ms")

# --- RandomEffects benchmark ---
print("\\n=== RandomEffects Benchmark ===")
from statgpu.panel import RandomEffects

for backend in ["numpy", "cupy", "torch"]:
    X = to_backend(X_np, backend)
    y = to_backend(y_np, backend)
    eids = to_backend(eids_np, backend)
    # Warmup
    m = RandomEffects()
    m.fit(X, y, entity_ids=eids)
    # Timed
    times = []
    for _ in range(10):
        m = RandomEffects()
        t0 = time.perf_counter()
        m.fit(X, y, entity_ids=eids)
        times.append(time.perf_counter() - t0)
    mean_t = np.mean(times)
    std_t = np.std(times)
    results[f"re_{backend}"] = (mean_t, std_t)
    print(f"  {backend:>8}: {mean_t*1000:.2f} +/- {std_t*1000:.2f} ms")

# --- GAM benchmark ---
print("\\n=== GAM Benchmark ===")
from statgpu.semiparametric import GAM
rng = np.random.default_rng(42)
X_np = rng.standard_normal((1000, 3))
y_np = np.sin(X_np[:, 0]) + 0.5 * X_np[:, 1]**2 + rng.standard_normal(1000) * 0.1

for backend in ["numpy", "cupy", "torch"]:
    X = to_backend(X_np, backend)
    y = to_backend(y_np, backend)
    # Warmup
    gam = GAM(n_splines=15, lam=1.0)
    gam.fit(X, y)
    # Timed
    times = []
    for _ in range(5):
        gam = GAM(n_splines=15, lam=1.0)
        t0 = time.perf_counter()
        gam.fit(X, y)
        times.append(time.perf_counter() - t0)
    mean_t = np.mean(times)
    std_t = np.std(times)
    results[f"gam_{backend}"] = (mean_t, std_t)
    print(f"  {backend:>8}: {mean_t*1000:.2f} +/- {std_t*1000:.2f} ms")

# --- Precision comparison ---
print("\\n=== Precision Comparison (PanelOLS) ===")
from statgpu.panel import PanelOLS
rng = np.random.default_rng(42)
n, k = 500, 5
X_np = rng.standard_normal((n, k))
eids_np = np.repeat(np.arange(50), 10)
y_np = X_np @ np.array([1.0, -0.5, 0.3, 0.0, -0.1]) + rng.standard_normal(n) * 0.01

m_np = PanelOLS(entity_effects=True)
m_np.fit(X_np, y_np, entity_ids=eids_np)
coef_np = m_np.coef_

for backend in ["cupy", "torch"]:
    X = to_backend(X_np, backend)
    y = to_backend(y_np, backend)
    eids = to_backend(eids_np, backend)
    m = PanelOLS(entity_effects=True)
    m.fit(X, y, entity_ids=eids)
    coef_bk = to_numpy(m.coef_)
    max_diff = np.max(np.abs(coef_np - coef_bk))
    print(f"  numpy vs {backend}: max |coef diff| = {max_diff:.2e}")

print("\\n=== All benchmarks complete ===")
'''

    print("\n=== Running three-backend benchmarks ===")
    # Write the benchmark script to a temp file on remote
    escaped_script = benchmark_script.replace("'", "'\\''")
    run_remote(ssh, f"cat > /tmp/bench_pr48.py << 'ENDSCRIPT'\n{benchmark_script}\nENDSCRIPT")

    cmd = conda_prefix + "cd /root/statgpu && python /tmp/bench_pr48.py 2>&1"
    out, err = run_remote(ssh, cmd, timeout=600)
    print(out)
    if err and "error" in err.lower():
        print(f"STDERR:\n{err}")

    return out


def main():
    host, port, user, password = get_ssh_credentials()
    print(f"Connecting to {user}@{host}:{port} ...")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username=user, password=password, timeout=30)
    print("Connected!")

    try:
        sftp = ssh.open_sftp()

        # 1. Deploy code
        deploy_code(ssh, sftp)
        sftp.close()

        # 2. Run tests
        test_output = run_tests(ssh)

        # 3. Run benchmarks
        bench_output = run_benchmarks(ssh)

        # 4. Save results locally
        results_path = os.path.join(os.path.dirname(__file__), "pr48_remote_results.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("=" * 72 + "\n")
            f.write("PR48 Fix Round - Remote Test & Benchmark Results\n")
            f.write("=" * 72 + "\n\n")
            f.write("--- TEST RESULTS ---\n")
            f.write(test_output)
            f.write("\n\n--- BENCHMARK RESULTS ---\n")
            f.write(bench_output)
        print(f"\nResults saved to: {results_path}")

    finally:
        ssh.close()
        print("SSH connection closed.")


if __name__ == "__main__":
    main()
