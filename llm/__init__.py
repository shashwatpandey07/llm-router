"""
LLM Router - A cost-aware, accuracy-preserving LLM routing and serving system.

This package provides abstractions and implementations for various LLM providers.
"""

from .base import BaseLLM
from .local import LocalLLM
from .openai_llm import OpenAILLM

__all__ = ["BaseLLM", "LocalLLM", "OpenAILLM"]

