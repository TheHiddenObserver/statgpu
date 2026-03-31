"""
Feature selection methods for statgpu.
"""

from ._stepwise import StepwiseSelector, stepwise_selection

__all__ = ['StepwiseSelector', 'stepwise_selection']
