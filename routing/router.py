"""
LLM Router - Cost-aware routing and cascading system.

This module implements the core routing logic that selects which model to call
based on query difficulty, with escalation and cost optimization.
"""

from typing import Dict, Optional
from llm.base import BaseLLM
from .difficulty import QueryDifficultyEstimator


class LLMRouter:
    """
    Routes queries to appropriate LLM models based on difficulty estimation.
    
    Implements cost-aware routing policy:
    - Easy queries (< 0.3) → local GGUF model
    - Medium queries (0.3-0.6) → local model, check confidence, escalate if needed
    - Hard queries (≥ 0.6) → large API model directly
    """
    
    def __init__(
        self,
        difficulty_estimator: QueryDifficultyEstimator,
        local_llm: BaseLLM,
        remote_llm: Optional[BaseLLM] = None
    ):
        """
        Initialize the LLM router.
        
        Args:
            difficulty_estimator: QueryDifficultyEstimator instance
            local_llm: Local LLM instance (e.g., LocalLLM with GGUF model)
            remote_llm: Remote/API LLM instance (optional, for escalation)
        """
        self.difficulty_estimator = difficulty_estimator
        self.local_llm = local_llm
        self.remote_llm = remote_llm
    
    def route(self, query: str) -> Dict:
        """
        Routes query to appropriate LLM based on difficulty.
        
        Routing policy:
        - difficulty < 0.3: local GGUF model
        - 0.3 <= difficulty < 0.6: local model, check confidence, escalate if needed
        - difficulty >= 0.6: large API model directly
        
        Args:
            query: The input query/prompt string
            
        Returns:
            Dictionary containing:
            {
                "text": str,              # Generated text
                "input_tokens": int,       # Input token count
                "output_tokens": int,      # Output token count
                "latency_ms": float,       # Generation latency
                "model": str,              # Model used
                "device": str,             # Device used
                "difficulty": float,       # Query difficulty score
                "routing_decision": str,   # "local", "escalated", or "remote"
                "cost_saved": float        # Estimated cost savings (if applicable)
            }
        """
        # 1. Estimate difficulty
        difficulty = self.difficulty_estimator.estimate(query)
        
        # Cost units (relative)
        LOCAL_COST = 1
        REMOTE_COST = 20
        
        # Helper: confidence check
        def is_low_confidence(response_text: str, output_tokens: int) -> bool:
            """Check if response indicates low confidence."""
            low_conf_phrases = [
                "i'm not sure",
                "i am not sure",
                "cannot determine",
                "not enough information",
                "unclear",
                "it depends"
            ]
            
            text_lower = response_text.lower()
            
            # Low confidence if output is too short
            if output_tokens < 20:
                return True
            
            # Low confidence if contains uncertainty phrases
            if any(p in text_lower for p in low_conf_phrases):
                return True
            
            # Low confidence if response doesn't end with punctuation
            if not response_text.strip().endswith((".", "!", "?")):
                return True
            
            return False
        
        # 2. Easy queries → local model (shorter responses)
        if difficulty < 0.3:
            result = self.local_llm.generate(query, max_tokens=64)
            # Estimate what remote call would cost (for cost_saved calculation)
            estimated_remote_cost = (
                (result["input_tokens"] / 1000) * 0.005 +
                (result["output_tokens"] / 1000) * 0.015
            )
            result.update({
                "difficulty": difficulty,
                "routing_decision": "local",
                "cost_usd": 0.0,  # Local model cost is effectively $0
                "cost_saved_usd": round(estimated_remote_cost, 6),
                "cost_saved": REMOTE_COST - LOCAL_COST  # Keep relative units too
            })
            return result
        
        # 3. Medium queries → local first, maybe escalate (medium responses)
        if difficulty < 0.6:
            local_result = self.local_llm.generate(query, max_tokens=128)
            
            low_conf = is_low_confidence(
                local_result["text"],
                local_result["output_tokens"]
            )
            
            if low_conf and self.remote_llm is not None:
                remote_result = self.remote_llm.generate(query)
                # Escalation negates savings (we paid for both local + remote)
                estimated_remote_cost = remote_result.get("cost_usd", 0.0)
                remote_result.update({
                    "difficulty": difficulty,
                    "routing_decision": "escalated",
                    "cost_saved_usd": 0.0,  # Escalation negates savings
                    "cost_saved": 0  # escalation negates savings
                })
                return remote_result
            
            # Local was sufficient
            estimated_remote_cost = (
                (local_result["input_tokens"] / 1000) * 0.005 +
                (local_result["output_tokens"] / 1000) * 0.015
            )
            local_result.update({
                "difficulty": difficulty,
                "routing_decision": "local",
                "cost_usd": 0.0,  # Local model cost is effectively $0
                "cost_saved_usd": round(estimated_remote_cost, 6),
                "cost_saved": REMOTE_COST - LOCAL_COST
            })
            return local_result
        
        # 4. Hard queries → remote model directly
        if self.remote_llm is None:
            raise ValueError("Remote LLM not provided for hard queries")
        
        remote_result = self.remote_llm.generate(query)
        remote_result.update({
            "difficulty": difficulty,
            "routing_decision": "remote",
            "cost_saved_usd": 0.0,  # No savings, we used the expensive model
            "cost_saved": 0
        })
        return remote_result

