# statgpu Agent Architecture

## Overview

The statgpu agent is a deterministic automatic statistical analysis pipeline. It profiles data, infers the analysis task, selects and fits models, validates results, and generates structured reports.

## Architecture

```
Profiler → Planner (MethodPruner) → Runner → Validator → Reporter
                 │                    │     ▲
                 │ Proactive pruning  │     │ Self-correction safety net
                 ▼                    └─────┘
          MethodPruner            (while loop, max 3 rounds)
```

### Two-Stage Design

1. **Proactive pruning** (planner stage): Excludes unsuitable methods based on data characteristics before fitting. Avoids wasted computation.
2. **Self-correction safety net** (runner/validator stage): If fitting fails despite pruning, diagnoses the issue and retries with corrected methods (GeneAgent pattern).

## Module Structure

| Module | File | Responsibility |
|--------|------|----------------|
| Profiler | `_profiler.py` | Data ingestion, type inference, missing value imputation, categorical encoding |
| Planner | `_planner.py` | Task inference, method registry, pruning rules |
| Runner | `_runner.py` | Model fitting, coefficient extraction, multi-candidate competition |
| Validator | `_validator.py` | Validation checks, multiple testing correction, regression diagnostics |
| Reporter | `_reporter.py` | Markdown, JSON, and Jupyter notebook output |
| Config | `_config.py` | Centralized configuration (replaces hardcoded thresholds) |
| Memory | `_memory.py` | Lightweight memory system for workflow reuse |
| Cross-Validation | `_cross_validation.py` | K-fold CV evaluation |
| Model Comparison | `_model_comparison.py` | Side-by-side model ranking |
| Coordinator | `_analysis.py` | Orchestrates all modules, defines data classes |

## Data Flow

```
Input (CSV/DataFrame/array)
  → PreparedData (X, y, time, event, feature_names)
  → DataProfile (n_samples, n_features, task_type, ...)
  → AnalysisPlan (methods, rationale)
  → List[ModelResult] (metrics, coefficients, diagnostics)
  → AnalysisResult (profile, plan, models, warnings, recommendations, validation_trace)
```

## Key Design Decisions

1. **No external multi-agent framework**: Nature papers (GeneAgent, Co-Scientist, DeepRare) all use custom orchestration, not LangGraph/AutoGen/CrewAI.
2. **GeneAgent self-correction pattern**: while loop + diagnosis rules, max 3 rounds.
3. **Proactive pruning over reactive correction**: MethodPruner excludes methods before fitting.
4. **Extensible via registries**: TaskRegistry, MethodRegistry, PruningRuleRegistry, ValidationRuleRegistry for plugin-style extension.
5. **Four extensibility registries**: New models register themselves, agent core code unchanged.

## Extensibility

See [extensibility.md](extensibility.md) for details on the four registries:
- `MethodRegistry` — register analysis methods
- `TaskRegistry` — register new task types
- `PruningRuleRegistry` — register pruning rules
- `ValidationRuleRegistry` — register validation rules
