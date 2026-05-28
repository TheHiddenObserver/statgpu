"""Agentic automatic statistical analysis for statgpu."""

from ._analysis import (
    AnalysisPlan,
    AnalysisResult,
    DataProfile,
    ModelResult,
    PreparedData,
    StatGPUAnalysisAgent,
)
from ._config import AgentConfig
from ._cross_validation import CVResult, AgentCrossValidator
from ._memory import AnalysisMemory, MemoryStore
from ._model_comparison import ModelComparison, ModelComparator
from ._planner import MethodRegistry, TaskRegistry, PruningRuleRegistry, MethodPruner
from ._validator import ValidationRuleRegistry

AutoAnalysisAgent = StatGPUAnalysisAgent

__all__ = [
    "AgentConfig",
    "AgentCrossValidator",
    "AnalysisMemory",
    "AnalysisPlan",
    "AnalysisResult",
    "AutoAnalysisAgent",
    "CVResult",
    "DataProfile",
    "MemoryStore",
    "MethodComparator",
    "MethodPruner",
    "MethodRegistry",
    "ModelComparison",
    "ModelComparator",
    "ModelResult",
    "PreparedData",
    "PruningRuleRegistry",
    "StatGPUAnalysisAgent",
    "TaskRegistry",
    "ValidationRuleRegistry",
]
