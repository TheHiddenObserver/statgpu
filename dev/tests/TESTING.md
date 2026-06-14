# Testing Guide

## Running Tests

### Local (CPU/NumPy)

```bash
# From project root
python -m pytest statgpu/tests/ -v

# Run specific module tests
python dev/tests/test_ridge_cv.py
python dev/tests/test_logistic.py
python dev/tests/test_unsupervised_pca.py
```

### Remote (GPU)

Remote tests require SSH access to a GPU server. See `dev/README.md` for connection setup.

```python
# Quick remote smoke test
python dev/tests/test_v10_import_smoke.py

# Full backend comparison
python dev/tests/test_backend_comparison.py

# Specific feature tests
python dev/tests/test_coxph_cv.py
python dev/tests/test_elasticnet_cv.py
```

### Test Categories

| Category | Files | Description |
|----------|-------|-------------|
| Core models | `test_ridge_cv.py`, `test_logistic.py`, `test_linear.py` | Basic model correctness |
| GLM family | `test_cox.py`, `test_cox_cv.py`, `test_coxph_cv.py` | Cox proportional hazards |
| Penalties | `test_penalties_and_exports.py`, `test_penalties_phase1.py` | L1/L2/SCAD/MCP/Group |
| Inference | `test_inference_*.py` (4 files) | KDE, multiple testing, resampling, performance guard |
| Unsupervised | `test_unsupervised_*.py` (11 files) | PCA, KMeans, DBSCAN, tSNE, UMAP, NMF, GMM, etc. |
| Backend | `test_backend_comparison.py`, `test_backends.py` | CuPy/Torch/NumPy parity |
| Feature selection | `test_feature_selection_knockoff.py` | Knockoff FDR control |
| Nonparametric | `test_nonparametric_kernel_regression.py` | Kernel regression |
| External consistency | `test_external_consistency.py` | sklearn/statsmodels compatibility |
| Torch backend | `test_torch_backend_cv.py`, `test_torch_backend_full.py` | Torch-specific tests |
| Cross-backend | `test_ordered_cross_backend.py` | Ordered logit across backends |
| Smoke | `test_v10_import_smoke.py` | Quick import sanity check |

## Remote Testing Workflow

### Using Paramiko (Python)

```python
import paramiko, sys, os
sys.path.insert(0, 'dev/scripts')
from remote_config import get_remote_config, REMOTE_WORK_DIR

config = get_remote_config()
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(config['host'], port=config['port'],
            username=config['username'], password=config['password'],
            look_for_keys=False, allow_agent=False)

# Upload code
sftp = ssh.open_sftp()
sftp.put('local_file.py', REMOTE_WORK_DIR + '/dev/tests/remote_test.py')
sftp.close()

# Run test (adjust conda path for your environment)
cmd = (
    'source <conda-path>/etc/profile.d/conda.sh && '
    'conda activate myconda && '
    f'cd {REMOTE_WORK_DIR} && python dev/tests/remote_test.py'
)
stdin, stdout, stderr = ssh.exec_command(cmd)
print(stdout.read().decode())
```

### Important Notes

- **Never pip install** on remote — `myconda` env has all dependencies
- **Clear `__pycache__`** before running benchmarks: `find . -name __pycache__ -exec rm -rf {} +`
- **Source upload path**: local `statgpu/` → remote `REMOTE_WORK_DIR` (see `remote_config.py`)

## Benchmark Tests

Benchmark scripts compare statgpu performance against:
- **R**: glmnet, survival packages (via `comparisons/` scripts)
- **scikit-learn**: Ridge, Lasso, LogisticRegression
- **statsmodels**: GLM families
- **lifelines**: Cox PH

See `dev/benchmarks/RESULTS.md` for historical results.
