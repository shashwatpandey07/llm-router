"""
Base LLM abstraction interface.

This module defines the core interface that all LLM implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict


class BaseLLM(ABC):
    """
    Abstract base class for all LLM implementations.
    
    This interface ensures consistent behavior across different LLM providers
    (local models, OpenAI, Anthropic, etc.) and is critical for the routing system.
    """
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> Dict:
        """
        Generate text completion from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing:
            {
                "text": str,              # Generated text
                "input_tokens": int,       # Number of input tokens
                "output_tokens": int,      # Number of output tokens
                "latency_ms": float        # Generation latency in milliseconds
            }
        """
        pass

