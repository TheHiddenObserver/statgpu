# Remote GPU Evaluation Regression and External-Probability Benchmark (2026-04-05)

## Environment

- Host: hz-4.matpool.com:27613
- Remote node: P4M1mJ
- GPU: Tesla P100-SXM2-16GB (16280 MiB)
- Driver: 460.32.03
- Python env: /root/miniconda3/envs/myconda/bin/python
- CuPy: 13.6.0
- Remote workdir: /tmp/statgpu_gpu_eval_1775385151

## Regression Results (GPU Evaluation Related)

### 1) GPU evaluation regression core
Command:

pytest dev/tests/test_logistic.py -q -k "gpu_evaluation_outputs_are_gpu_backed or gpu_batch_evaluation_outputs_are_gpu_backed or gpu_external_probability_evaluation_module" -ra

Result:

- 3 passed, 21 deselected in 1.75s
- exit code: 0

### 2) Full GPU logistic suite
Command:

pytest dev/tests/test_logistic.py -q -k "gpu" -ra

Result:

- 5 passed, 19 deselected in 1.57s
- exit code: 0

## External-Probability Interface Timing Benchmark

Dataset:

- n_samples: 300000
- threshold: 0.5
- include_curves: True
- repeats: 5 (with warmup)

### Timing (mean over 5 runs)

| Implementation | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | Notes |
|---|---:|---:|---:|---:|---|
| statgpu (CPU, NumPy backend) | 120.334 | 0.250 | 119.848 | 120.552 | evaluate_binary_classification |
| statgpu (GPU, CuPy backend) | 8.781 | 0.043 | 8.714 | 8.825 | evaluate_binary_classification |
| sklearn (CPU metrics组合) | 242.434 | 3.136 | 237.638 | 245.569 | confusion_matrix + roc_curve + roc_auc_score + precision_recall_curve + average_precision_score |

Derived speedups:

- statgpu GPU vs statgpu CPU: 13.704x
- statgpu CPU vs sklearn CPU: 2.015x

### Numerical Consistency

- statgpu CPU vs statgpu GPU:
  - ROC AUC abs diff: 0.0
  - AP abs diff: 0.0
- statgpu CPU vs sklearn CPU:
  - ROC AUC abs diff: 1.1102230246251565e-16
  - AP abs diff: 1.1102230246251565e-16

## Framework Coverage Notes

### sklearn

- Corresponding implementation exists (组合式接口) and was benchmarked.

### statsmodels

- Installed: 0.14.6
- No full external-probability evaluation equivalent API found:
  - has_roc_auc: False
  - has_average_precision: False
  - has_roc_curve: False
  - has_precision_recall_curve: False
- Therefore, no fully equivalent ROC+PR+AUC+AP one-shot or组合式对等 benchmark was run purely within statsmodels API.

### R

- Rscript not found on remote host.
- Therefore, no R benchmark could be executed.
