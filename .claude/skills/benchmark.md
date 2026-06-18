---
name: benchmark
description: Run three-backend benchmarks on remote GPU server with precision and timing comparison
---

# Benchmark Skill

When the user invokes this skill, create and run benchmarks comparing
statgpu across numpy/cupy/torch backends and external frameworks.

## Input

The user provides:
- Module or functions to benchmark
- Data scales (default: n=5000, 20000, 50000)
- External comparison targets (default: sklearn, scipy)

## Workflow

### Step 1: Create benchmark script

Create `dev/tests/bench_<module>.py` with:

```python
import time, json, os
import numpy as np

def timeit(func, n_repeats=3):
    times = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return result, float(np.median(times))

def _to_np(x):
    if hasattr(x, 'get'): return x.get()
    if hasattr(x, 'cpu'): return x.detach().cpu().numpy()
    return np.asarray(x)

def warmup_gpu():
    """Warmup GPU to avoid JIT/first-call overhead."""
    try:
        import cupy as cp
        a = cp.ones((100, 100), dtype=cp.float64)
        _ = a @ a
        cp.cuda.Stream.null.synchronize()
        del a
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.ones(100, 100, dtype=torch.float64, device='cuda')
            _ = a @ a
            torch.cuda.synchronize()
            del a
            torch.cuda.empty_cache()
    except Exception:
        pass
```

### Step 2: Run locally first

```bash
PYTHONPATH=. python dev/tests/bench_<module>.py
```

Verify it runs without errors.

### Step 3: Upload to remote and run

```python
import paramiko

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('hz-4.matpool.com', port=28838, username='root', password='<from memory>')

sftp = ssh.open_sftp()
sftp.put(local_path, remote_path)
sftp.close()

channel = ssh.invoke_shell()
time.sleep(2)
channel.recv(4096)
channel.send('source /root/miniconda3/etc/profile.d/conda.sh && conda activate myconda && cd /root/statgpu && python dev/tests/bench_<module>.py\n')

# Read output...
ssh.close()
```

### Step 4: Save results

Save JSON results to `results/bench_<module>.json` for frontend use.

## Key metrics to report

| Metric | Description |
|--------|-------------|
| `time_ms` | Median execution time in milliseconds |
| `speedup` | `time_external / time_statgpu` |
| `max_diff` | `max(abs(ours - theirs))` for precision |
| `corr` | Correlation between result matrices |

## Remote server notes

- Server: hz-4.matpool.com:28838
- Conda env: `myconda` (has all deps, never use pip install)
- GPU: Tesla P100-16GB
- Package location: `/root/statgpu/`
- Always warmup GPU before timing
- Use `paramiko` for SSH (password auth)
