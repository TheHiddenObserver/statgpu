# Agent Design Decisions

## Why GeneAgent Pattern (Not LangGraph)?

**Decision**: Use a simple while loop + diagnosis rules instead of LangGraph or other multi-agent frameworks.

**Reasoning**:
1. Nature papers (Robin, Co-Scientist, DeepRare, GeneAgent, BioMedAgent) all use custom orchestration, not general frameworks
2. GeneAgent's "generate → verify → modify → summarize" is the simplest and most effective pattern
3. The core pain point is "what to do when a model fails" — a while loop + diagnosis rules handles this
4. Zero external dependencies, ~150 lines of code
5. LangGraph adds complexity (graph state machine, checkpointing) that statistical analysis doesn't need

## Why Proactive Pruning?

**Decision**: Exclude unsuitable methods before fitting, not just after failure.

**Reasoning**:
- Reactive: try OLS → fail → try Ridge → fail → try Lasso (3 fits, 2 wasted)
- Proactive: check p > n → skip OLS → try Ridge directly (1 fit)
- Pruning rules are cheap (just data shape checks), model fitting is expensive

## Why Optional Multiple Testing?

**Decision**: Default to no correction (`multiple_testing_method="none"`).

**Reasoning**:
- Correction method depends on analysis purpose (exploratory vs confirmatory)
- FDR (BH) for exploratory, FWER (Holm) for confirmatory, none for pre-specified hypotheses
- Users need to see raw p-values to make informed decisions
- The agent suggests correction when feature count >= 10

## Why Three Output Formats?

**Decision**: Markdown + JSON + Jupyter Notebook.

**Reasoning**:
- **Markdown**: Human-readable report (primary output)
- **JSON**: Machine-readable artifact for audit trails and reproducibility
- **Notebook**: Executable analysis record (CellVoyager pattern), users can review and modify each step

## Why Five Registries?

**Decision**: MethodRegistry, TaskRegistry, PruningRuleRegistry, ValidationRuleRegistry, ReportTemplateRegistry.

**Reasoning**:
- statgpu will add more models (AFT, NegativeBinomial, causal inference, time series)
- Without registries, every new model requires modifying agent core code
- With registries, new models self-register: `MethodRegistry.register("survival", "AFT", factory=...)`
- Agent core code never needs to change for new models

## Why Self-Correction Has a Hard Limit?

**Decision**: `max_correction_rounds=3` prevents infinite loops.

**Reasoning**:
- GeneAgent uses iterative verification, but unbounded iteration is risky
- 3 rounds covers: initial attempt → first correction → fallback
- Each round is logged in `validation_trace` for auditability
- If all 3 rounds fail, the agent reports the error honestly
