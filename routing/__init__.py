"""
Routing module for LLM Router.

This module contains components for query difficulty estimation,
model selection, and routing logic.
"""

from .difficulty import QueryDifficultyEstimator

__all__ = ["QueryDifficultyEstimator"]

