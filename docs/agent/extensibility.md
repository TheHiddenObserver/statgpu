# Agent Extensibility Guide

## Overview

The statgpu agent uses five pluggable registries. New models, task types, and rules can be registered without modifying agent core code.

## 1. MethodRegistry — Register Analysis Methods

```python
from statgpu.agent import MethodRegistry

# Register a new method for a task type
MethodRegistry.register(
    "regression",           # task_type
    "ElasticNet",           # method name
    factory=lambda: ElasticNet(alpha=0.5, l1_ratio=0.5),  # factory function
    priority=2,             # lower = higher priority
    prerequisites=[],       # optional prerequisites
)

# Query registered methods
methods = MethodRegistry.get_method_names("regression")
# → ["LinearRegression", "Ridge(alpha=1.0)", "ElasticNet"]
```

## 2. TaskRegistry — Register New Task Types

```python
from statgpu.agent import TaskRegistry

# Register a new task type
TaskRegistry.register(
    "causal_inference",     # task name
    inference_fn=lambda prepared: (
        hasattr(prepared, 'treatment_column') and prepared.y is not None
    ),
    description="Causal inference with treatment/control",
)
```

## 3. PruningRuleRegistry — Register Pruning Rules

```python
from statgpu.agent import PruningRuleRegistry

# Register a pruning rule for a method
PruningRuleRegistry.register(
    "ElasticNet",           # method name
    rule=lambda prepared: prepared.X.shape[0] < 10,  # exclude when True
    reason="Sample size too small for ElasticNet",
)
```

## 4. ValidationRuleRegistry — Register Validation Rules

```python
from statgpu.agent import ValidationRuleRegistry

# Register a validation rule for a task type
def aft_specific_checks(prepared, models):
    warnings = []
    if any("AFT" in m.name for m in models):
        if hasattr(prepared, 'time') and prepared.time is not None:
            if np.any(prepared.time <= 0):
                warnings.append("AFT requires positive survival times.")
    return warnings

ValidationRuleRegistry.register("survival", aft_specific_checks)
```

## Adding a New Model (Complete Example)

To add ElasticNet support, add this to `statgpu/linear_model/_elasticnet.py`:

```python
from statgpu.agent import MethodRegistry, PruningRuleRegistry

# Register method
MethodRegistry.register(
    "regression", "ElasticNet",
    factory=lambda: ElasticNet(alpha=0.5, l1_ratio=0.5),
    priority=2,
)

# Register pruning rule
PruningRuleRegistry.register(
    "ElasticNet",
    rule=lambda prepared: prepared.X.shape[0] < 10,
    reason="n < 10: too few samples for ElasticNet",
)
```

The agent will automatically discover and use ElasticNet without any changes to `_analysis.py`.
