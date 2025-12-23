"""
Local LLM implementation using llama.cpp (GGUF models).

This module provides a local LLM implementation optimized for Mac (Metal) and CPU
using llama.cpp backend for 10-50x faster inference.
"""

import time
import os
import sys
from typing import Dict
from llama_cpp import Llama

from .base import BaseLLM


class LocalLLM(BaseLLM):
    """
    Local LLM using llama.cpp (GGUF models).
    
    Optimized for Mac (Metal) and CPU with quantized models.
    Provides 10-50x faster inference compared to transformers backend.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the local LLM with a GGUF model.
        
        Args:
            model_path: Path to GGUF model file (e.g., "models/phi-2.Q4_K_M.gguf")
        """
        self.model_path = model_path
        
        # Initialize llama.cpp with Metal acceleration
        # Note: You may see "skipping kernel_*_bf16" messages during initialization.
        # These are harmless - they just mean bf16 kernels aren't supported,
        # but f32 kernels work fine and Metal acceleration is still active.
        # To suppress them, redirect stderr: python3 script.py 2>/dev/null
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,          # Context window
            n_threads=8,          # CPU threads (tune based on your CPU)
            n_gpu_layers=-1,      # Use Metal GPU fully (-1 = all layers)
            verbose=False        # Suppress llama.cpp verbose output
        )
    
    def generate(self, prompt: str, max_tokens: int = 256) -> Dict:
        """
        Generate text completion from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with text, token counts, latency metrics, model, and device
        """
        start = time.time()
        
        # Format prompt for better instruction following
        # Phi-2 is a base model, so we add a simple instruction prefix
        formatted_prompt = f"Answer the following question concisely:\n\n{prompt}\n\nAnswer:"
        
        # Generate using llama.cpp
        # Add stop sequences to prevent model from continuing indefinitely
        # Phi-2 is a base model trained on textbooks, so it tends to generate "Exercise" sections
        # Also stop on code blocks and class definitions
        stop_sequences = [
            "\n\nExercise",
            "\n\nQuestion", 
            "\n\nProblem",
            "\nExercise",
            "\n\n```",  # Code blocks
            "\n```",    # Code blocks
            "\ndef ",   # Python function definitions
            "\nclass ", # Python class definitions
            "\nimport ", # Python imports
            "\nfrom ",  # Python imports
            "<|endoftext|>"
        ]
        
        output = self.llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.0,      # Deterministic (greedy) generation
            stop=stop_sequences   # Stop on common continuation patterns
        )
        
        end = time.time()
        
        # Extract results
        text = output["choices"][0]["text"]
        usage = output["usage"]
        
        # Clean up the response
        # Remove any code-like patterns that might have slipped through
        cleaned_text = text.strip()
        
        # If response starts with code-like patterns, try to extract the actual answer
        if cleaned_text.startswith("def ") or cleaned_text.startswith("class ") or cleaned_text.startswith("import "):
            # This looks like code, try to find the actual answer
            # Split by newlines and find the first non-code line
            lines = cleaned_text.split("\n")
            answer_lines = []
            for line in lines:
                line_stripped = line.strip()
                # Skip code-like lines
                if (line_stripped.startswith("def ") or 
                    line_stripped.startswith("class ") or 
                    line_stripped.startswith("import ") or
                    line_stripped.startswith("from ") or
                    line_stripped.startswith("@") or
                    line_stripped.startswith("#")):
                    continue
                # If we find a normal line, use it and everything after
                if line_stripped and not line_stripped.startswith("    ") and not line_stripped.startswith("\t"):
                    answer_lines = lines[lines.index(line):]
                    break
            
            if answer_lines:
                cleaned_text = "\n".join(answer_lines).strip()
            else:
                # If we can't find a good answer, return a fallback
                cleaned_text = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        return {
            "text": cleaned_text,
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "latency_ms": (end - start) * 1000,
            "model": self.model_path,
            "device": "metal/cpu"
        }
