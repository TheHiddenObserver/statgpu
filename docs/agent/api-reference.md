# statgpu Agent API Reference

## Main Class

### `StatGPUAnalysisAgent`

```python
from statgpu.agent import StatGPUAnalysisAgent

agent = StatGPUAnalysisAgent(
    device="auto",           # "auto", "cpu", "cuda", "torch"
    cov_type="hc3",          # "nonrobust", "hc0", "hc1", "hc2", "hc3", "hac"
    random_state=0,          # Random seed for reproducibility
    max_categories=20,       # Max categorical levels before collapsing
    include_regularized=True, # Include Ridge alongside OLS
    include_unsupervised_diagnostics=True,  # Run PCA diagnostic
    gpu_memory_cleanup=False,  # Clean GPU memory after fitting
    cv_folds=5,              # Cross-validation folds (0 to disable)
    multiple_testing_method="none",  # "none", "bh", "by", "holm", "bonferroni", "hochberg"
    alpha=0.05,              # Significance level for multiple testing
)
```

### Methods

#### `analyze(data=None, X=None, y=None, target=None, task="auto", time=None, event=None, feature_names=None, feature_columns=None) → AnalysisResult`

Run automatic analysis on the provided data.

**Parameters:**
- `data` — Table-like input (DataFrame, dict, list-of-dicts, structured array)
- `X` — Feature matrix (numpy array)
- `y` — Target vector
- `target` — Column name for target (when using `data=`)
- `task` — "auto", "regression", "classification", "binary", "poisson", "survival", "unsupervised"
- `time` — Column name or array for survival time
- `event` — Column name or array for survival event indicator
- `feature_names` — Feature names (when using `X=`)
- `feature_columns` — Specific feature columns to use (when using `data=`)

#### `analyze_csv(path, target=None, task="auto", time=None, event=None, feature_columns=None) → AnalysisResult`

Run analysis on a CSV file.

---

## Data Classes

### `AnalysisResult`

| Field | Type | Description |
|-------|------|-------------|
| `profile` | `DataProfile` | Data characteristics |
| `plan` | `AnalysisPlan` | Planned methods and rationale |
| `models` | `List[ModelResult]` | Fitted model results |
| `warnings` | `List[str]` | Validation warnings |
| `recommendations` | `List[str]` | Suggested next steps |
| `validation_trace` | `List[dict]` | Self-correction history |
| `comparison` | `ModelComparison` | Model ranking (if multiple models) |

**Methods:**
- `to_markdown(max_terms=12)` → str
- `to_dict(include_estimators=False)` → dict
- `save_markdown(path, max_terms=12)` → None
- `save_json(path, include_estimators=False)` → None
- `save_notebook(data_source, output_path)` → None

### `DataProfile`

| Field | Type | Description |
|-------|------|-------------|
| `n_samples` | `int` | Number of samples |
| `n_features` | `int` | Number of features |
| `task_type` | `str` | Inferred task type |
| `feature_names` | `List[str]` | Feature names |
| `target_name` | `Optional[str]` | Target column name |
| `device` | `str` | Compute device used |
| `dropped_rows` | `int` | Rows dropped due to missing data |
| `imputed_values` | `int` | Missing values imputed |
| `encoded_features` | `Dict[str, List[str]]` | Categorical encoding map |
| `target_summary` | `Dict[str, Any]` | Target statistics |

### `ModelResult`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Model name |
| `task_type` | `str` | Task type |
| `estimator` | `Any` | Fitted estimator object |
| `metrics` | `Dict[str, Any]` | Performance metrics |
| `coefficients` | `List[Dict]` | Coefficient table |
| `diagnostics` | `Dict[str, Any]` | Model diagnostics |
| `warnings` | `List[str]` | Model-specific warnings |
| `error` | `Optional[str]` | Error message if fitting failed |
| `cv_results` | `Optional[CVResult]` | Cross-validation results |

### `ModelComparison`

| Field | Type | Description |
|-------|------|-------------|
| `ranking_metric` | `str` | Metric used for ranking |
| `ranking` | `List[Tuple[str, float]]` | (model_name, score) sorted by rank |
| `best_model` | `str` | Best model name |
| `delta_table` | `List[Dict]` | Pairwise differences from best |

### `CVResult`

| Field | Type | Description |
|-------|------|-------------|
| `n_folds` | `int` | Number of CV folds |
| `metric_name` | `str` | Metric name |
| `fold_scores` | `List[float]` | Per-fold scores |
| `mean` | `float` | Mean score |
| `std` | `float` | Standard deviation |
| `ci_low` | `float` | 95% CI lower bound |
| `ci_high` | `float` | 95% CI upper bound |

---

## CLI

```bash
statgpu-agent data.csv --target outcome [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--target` | None | Target column |
| `--time` | None | Survival time column |
| `--event` | None | Survival event column |
| `--task` | auto | Task type |
| `--device` | auto | Compute device |
| `--output` | None | Markdown output path |
| `--output-json` | None | JSON artifact path |
| `--output-notebook` | None | Notebook output path |
| `--cv` | 5 | CV folds (0 to disable) |
| `--multiple-testing` | none | Correction method |
| `--alpha` | 0.05 | Significance level |
