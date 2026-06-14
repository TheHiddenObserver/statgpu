# dev/ — Development Workspace

This directory contains development tools, tests, benchmarks, and planning documents for statgpu. It is **not** part of the installable package.

## Directory Structure

```
dev/
├── tests/              # Integration and feature tests
│   ├── test_*.py       # Active test scripts (run locally or remotely)
│   └── _archive/       # Historical test artifacts (auto-organized)
│       ├── other/              # One-off debug/tmp scripts
│       ├── remote_runners/     # Old remote execution scripts
│       └── bench_outputs/      # Old benchmark output logs
│
├── benchmarks/         # Performance benchmarks and cross-framework comparisons
│   ├── benchmark_*.py  # Active benchmark scripts
│   └── _archive/       # Old debug/deploy/tmp scripts
│
├── scripts/            # Utility and automation scripts
│   ├── remote_config.py         # Remote server config loader
│   ├── remote_config_local.py   # Local overrides (gitignored, no passwords)
│   ├── sync_*.py                # Remote sync utilities
│   ├── profile_*.py             # Profiling scripts
│   └── _archive/                # Old scripts
│
├── docs/               # Technical reports and analysis documents
├── plans/              # Development planning and roadmap
├── manual/             # Manual ad-hoc testing scripts
├── results/            # Benchmark result data (JSON)
├── comparisons/        # Cross-language validation (R vs Python)
├── validation/         # Formal validation scripts
└── design/             # Design documents
```

## Remote GPU Testing

### Server Access

Remote testing uses SSH to a GPU server. Configuration is loaded from (in priority order):

1. **Environment variables** (recommended for CI):
   ```bash
   export STATGPU_REMOTE_HOST="hz-4.matpool.com"
   export STATGPU_REMOTE_PORT="28838"
   export STATGPU_REMOTE_USER="root"
   export STATGPU_REMOTE_PASSWORD="your_password"
   ```

2. **Local config file** (`dev/scripts/remote_config_local.py`, gitignored):
   ```python
   HOST = "hz-4.matpool.com"
   PORT = 28838
   USERNAME = "root"
   # Password should be set via STATGPU_REMOTE_PASSWORD env var
   ```

### Remote Environment

- **Conda env**: `myconda` — all dependencies pre-installed, **do not use pip install**
- **Python**: `/root/miniconda3/envs/myconda/bin/python`
- **Activate**: `source /root/miniconda3/etc/profile.d/conda.sh && conda activate myconda`
- **GPU**: Tesla P100-SXM2-16GB (or check current instance)
- **Source upload**: `statgpu/` package → `/root/statgpu/`

### Typical Remote Workflow

```python
# In a script using paramiko
import sys; sys.path.insert(0, 'dev/scripts')
from remote_config import get_remote_config, REMOTE_WORK_DIR

config = get_remote_config()
ssh = paramiko.SSHClient()
ssh.connect(config['host'], port=config['port'],
            username=config['username'], password=config['password'],
            look_for_keys=False, allow_agent=False)

# Run commands
stdin, stdout, stderr = ssh.exec_command(
    'source /root/miniconda3/etc/profile.d/conda.sh && '
    'conda activate myconda && '
    'cd /root/statgpu && python -m pytest tests/'
)
```

## Package Architecture (for reference)

statgpu is a GPU-accelerated statistics library with a pluggable backend system:

```
statgpu/
├── _config.py          # Device management (CPU/CUDA/TORCH/AUTO)
├── _base.py            # BaseEstimator (sklearn-compatible interface)
├── backends/           # Array-library abstraction (NumPy/CuPy/Torch)
├── linear_model/       # Regression & classification (largest module)
├── glm_core/           # GLM engine: families, links, solvers (IRLS, FISTA, ADMM, L-BFGS)
├── penalties/          # Penalty registry (L1, L2, SCAD, MCP, Group, Adaptive)
├── survival/           # Cox PH with CUDA/Triton kernels
├── inference/          # Statistical inference, p-value adjustment, bootstrap
├── unsupervised/       # PCA, KMeans, DBSCAN, tSNE, UMAP, NMF, GMM, etc.
├── panel/              # Panel data models (fixed/random effects)
├── nonparametric/      # KDE, kernel regression, splines
├── feature_selection/  # Knockoff filter, stepwise selection
├── covariance/         # LedoitWolf, OAS
├── anova/              # One-way ANOVA
├── metrics/            # Classification metrics (ROC, AUC)
├── diagnostics/        # Regression diagnostics
├── semiparametric/     # GAM
└── core/               # Formula parser, design matrix
```

**Backend dispatch**: Two patterns coexist:
1. **OO**: `self._get_backend()` → `backend.xp.sum()`, `backend.xp.linalg.solve()`
2. **Functional**: `_xp(arr)` runtime detection → `_sigmoid()`, `_soft_threshold()`, etc.

**Device auto-selection**: CuPy CUDA > Torch CUDA > NumPy CPU

## Archive Policy

Files are archived (not deleted) when they become obsolete:
- **`_` prefix files**: One-off debug/deploy/tmp scripts → `_archive/other/`
- **Remote runners**: `run_remote_*`, `upload_*`, `*_remote_runner.py` → `_archive/remote_runners/`
- **Old benchmarks**: Versioned output logs → `_archive/bench_outputs/`
- **`scripts/tmp/`**: Scratch scripts → `_archive/tmp/`

Archived files remain accessible for reference but should not be run.
