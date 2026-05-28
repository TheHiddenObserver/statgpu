# statgpu Agent CLI

## Basic Usage

```bash
statgpu-agent data.csv --target outcome
```

## All Options

```bash
statgpu-agent data.csv \
    --target outcome \
    --task auto \
    --device auto \
    --output report.md \
    --output-json report.json \
    --output-notebook report.ipynb \
    --cv 5 \
    --multiple-testing bh \
    --alpha 0.05
```

## Examples

### Regression Analysis
```bash
statgpu-agent data.csv --target price --output report.md
```

### Binary Classification
```bash
statgpu-agent data.csv --target churn --task binary --output report.md
```

### Survival Analysis
```bash
statgpu-agent data.csv --time survival_time --event death --task survival
```

### Unsupervised (No Target)
```bash
statgpu-agent data.csv --task unsupervised
```

### With Multiple Testing Correction
```bash
# BH correction for exploratory analysis
statgpu-agent data.csv --target outcome --multiple-testing bh

# Holm correction for confirmatory analysis
statgpu-agent data.csv --target outcome --multiple-testing holm
```

### Full Output (All Three Formats)
```bash
statgpu-agent data.csv --target outcome \
    --output report.md \
    --output-json report.json \
    --output-notebook analysis.ipynb
```

### Disable Cross-Validation
```bash
statgpu-agent data.csv --target outcome --cv 0
```

### Force CPU Mode
```bash
statgpu-agent data.csv --target outcome --device cpu
```

## Python Module

```bash
python -m statgpu.agent data.csv --target outcome --device cpu
```
