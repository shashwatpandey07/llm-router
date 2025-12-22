"""
LLM Router - A cost-aware, accuracy-preserving LLM routing and serving system.

This package provides abstractions and implementations for various LLM providers.
"""

from .base import BaseLLM
from .local import LocalLLM

__all__ = ["BaseLLM", "LocalLLM"]

