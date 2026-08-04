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
├── validation/         # Formal validation and documentation-contract scripts
└── design/             # Architecture and module-ownership documents
```

The canonical developer architecture entry point is
[`dev/design/ARCHITECTURE.md`](design/ARCHITECTURE.md). Its
[Survival / Cox section](design/ARCHITECTURE.md#5-survival--cox-architecture)
defines the current Cox module ownership, canonical statistical source,
CV/refit reuse contract, and inference boundaries.

## Remote GPU Testing

### Server Access

Remote testing uses SSH to a GPU server. Configuration is loaded from (in priority order):

1. **Environment variables** (recommended for CI):
   ```bash
   export STATGPU_REMOTE_HOST="<your-gpu-server>"
   export STATGPU_REMOTE_PORT="<ssh-port>"
   export STATGPU_REMOTE_USER="<username>"
   export STATGPU_REMOTE_PASSWORD="<password>"
   ```

2. **Local config file** (`dev/scripts/remote_config_local.py`, gitignored):
   ```python
   HOST = "<your-gpu-server>"
   PORT = <ssh-port>
   USERNAME = "<username>"
   # Password should be set via STATGPU_REMOTE_PASSWORD env var
   ```

See `dev/scripts/remote_config_local.example.py` for a template.

### Remote Environment

- **Conda env**: `myconda` — all dependencies pre-installed, **do not use pip install**
- **Python**: conda env Python (see `remote_config.py` for paths)
- **Activate**: `source <conda-path>/etc/profile.d/conda.sh && conda activate myconda`
- **GPU**: check the active instance with `nvidia-smi`
- **Source upload**: local `statgpu/` package → isolated remote work directory

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
    'source <conda-path>/etc/profile.d/conda.sh && '
    'conda activate myconda && '
    f'cd {REMOTE_WORK_DIR} && python -m pytest dev/tests/'
)
```

## Package Architecture (for reference)

statgpu is a GPU-accelerated statistics library with a pluggable backend system:

```
statgpu/
├── _config.py          # Device management (CPU/CUDA/TORCH/AUTO)
├── _base.py            # BaseEstimator
├── backends/           # Array-library abstraction (NumPy/CuPy/Torch)
├── linear_model/       # Regression, GLM, broad penalized estimators
├── glm_core/           # GLM families, links, and IRLS/loss infrastructure
├── penalties/          # Penalty registry (L1, L2, SCAD, MCP, Group, Adaptive)
├── survival/           # Canonical CoxPH/CoxPHCV, risk sets, prediction, inference adapters
├── inference/          # Shared distributions, results, covariance/Wald policy, resampling
├── unsupervised/       # PCA, KMeans, DBSCAN, tSNE, UMAP, NMF, GMM, etc.
├── panel/              # Panel data models
├── nonparametric/      # KDE, kernel regression, splines
├── feature_selection/  # Knockoff filter, stepwise selection
├── covariance/         # Covariance estimation
├── anova/              # ANOVA methods
├── metrics/            # Statistical and predictive metrics
├── diagnostics/        # Regression diagnostics
├── semiparametric/     # GAM
└── core/               # Formula parser and design matrices
```

**Backend dispatch**: two patterns coexist:

1. **OO**: `self._get_backend()` → `backend.xp.sum()`, `backend.xp.linalg.solve()`;
2. **functional**: runtime array detection plus shared array operations for solver and kernel code.

**Device auto-selection**: CuPy CUDA > Torch CUDA > NumPy CPU.
Explicit `device="cuda"` and `device="torch"` requests must not silently select another backend.

### Cox developer entry point

Before modifying Cox behavior, read
[`dev/design/ARCHITECTURE.md#5-survival--cox-architecture`](design/ARCHITECTURE.md#5-survival--cox-architecture).
The key ownership rules are:

- `statgpu/survival/_risk_sets.py` is the canonical statistical-definition layer;
- `statgpu/survival/_cox.py` owns the public `CoxPH` boundary and fitted-state transaction;
- `statgpu/survival/_cox_counting.py` owns prepared states and the canonical Newton solver;
- `statgpu/survival/_cox_cv.py` must select penalties and final-refit through `CoxPH`, not maintain a second optimizer;
- generic covariance-spectrum and joint-Wald policy belongs in `statgpu/inference/`;
- specialized CUDA/Torch/legacy routes must be validated against the canonical primitives and must not redefine public support claims.

Relevant maintained entry points include:

- tests under `dev/tests/test_cox*.py` and `dev/tests/test_survival_risk_sets.py`;
- survival benchmarks under `dev/benchmarks/benchmark_*cox*.py` and related Exact-ties runners;
- auditable structured artifacts under `results/benchmark_frontend_sources/`.

## Archive Policy

Files are archived, not deleted, when they become obsolete:

- **`_` prefix files**: one-off debug/deploy/tmp scripts → `_archive/other/`;
- **remote runners**: `run_remote_*`, `upload_*`, `*_remote_runner.py` → `_archive/remote_runners/`;
- **old benchmarks**: versioned output logs → `_archive/bench_outputs/`;
- **`scripts/tmp/`**: scratch scripts → `_archive/tmp/`.

Archived files remain accessible for reference but should not be run as maintained validation.
