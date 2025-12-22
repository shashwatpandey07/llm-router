"""
Remote LLM implementation using OpenAI GPT-4o.

Provides real API-backed inference with cost and latency tracking.
"""

import time
from typing import Dict
from openai import OpenAI

from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """
    OpenAI GPT-4o LLM wrapper.
    
    Implements BaseLLM interface for seamless integration with the routing system.
    Tracks real token usage and calculates actual USD cost.
    """
    
    # Pricing (USD per 1K tokens) — GPT-4o pricing as of 2024
    INPUT_COST_PER_1K = 0.005   # $0.005 per 1K input tokens
    OUTPUT_COST_PER_1K = 0.015  # $0.015 per 1K output tokens
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Initialize the OpenAI LLM wrapper.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: "gpt-4o")
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 256) -> Dict:
        """
        Generate text completion using OpenAI API.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with text, token counts, latency, cost, and metadata
        """
        start = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0  # Deterministic generation
        )
        
        end = time.time()
        
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        
        # Calculate actual cost in USD
        cost_usd = (
            (input_tokens / 1000) * self.INPUT_COST_PER_1K +
            (output_tokens / 1000) * self.OUTPUT_COST_PER_1K
        )
        
        return {
            "text": response.choices[0].message.content.strip(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": (end - start) * 1000,
            "model": self.model,
            "device": "openai_api",
            "cost_usd": round(cost_usd, 6)
        }

