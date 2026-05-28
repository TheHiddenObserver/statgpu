"""Agentic automatic statistical analysis for statgpu."""

from ._analysis import (
    AnalysisPlan,
    AnalysisResult,
    DataProfile,
    ModelResult,
    StatGPUAnalysisAgent,
)

AutoAnalysisAgent = StatGPUAnalysisAgent

__all__ = [
    "AnalysisPlan",
    "AnalysisResult",
    "AutoAnalysisAgent",
    "DataProfile",
    "ModelResult",
    "StatGPUAnalysisAgent",
]
