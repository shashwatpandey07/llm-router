"""
LLM Router - Cost-aware routing and cascading system.

This module implements the core routing logic that selects which model to call
based on query difficulty, with escalation and cost optimization.
"""

from typing import Dict, Optional
from llm.base import BaseLLM
from .difficulty import QueryDifficultyEstimator
from .verifier import ResponseVerifier


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
        self.verifier = ResponseVerifier()
        self.max_retries = 1  # Allow one regeneration attempt before escalating
    
    def _max_tokens_for_difficulty(self, difficulty: float) -> int:
        """
        Determine adaptive token budget based on query difficulty.
        
        This implements adaptive token budgeting:
        - Easy queries (< 0.3): 128 tokens (cheap & fast)
        - Medium queries (0.3-0.6): 256 tokens (enough for explanations)
        - Hard queries (≥ 0.6): 512 tokens (proofs, analysis, multi-part answers)
        
        Args:
            difficulty: Query difficulty score (0.0 to 1.0)
            
        Returns:
            Maximum tokens to generate
        """
        if difficulty < 0.3:
            return 128
        elif difficulty < 0.6:
            return 256
        else:
            return 512
    
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
        
        # 2. Determine adaptive token budget based on difficulty
        max_tokens = self._max_tokens_for_difficulty(difficulty)
        
        # Cost units (relative)
        LOCAL_COST = 1
        REMOTE_COST = 20
        
        # Helper: build continuation prompt for truncated responses
        def build_continuation_prompt(original_query: str, partial_answer: str) -> str:
            """Build a prompt to continue a truncated answer."""
            return (
                f"Question:\n{original_query}\n\n"
                f"Partial answer:\n{partial_answer}\n\n"
                "Continue the answer. Finish the current sentence and conclude clearly in 2–3 sentences."
            )
        
        # 2. Easy queries → local model with adaptive tokens, verify and regenerate if needed
        if difficulty < 0.3:
            retry_count = 0
            current_max_tokens = max_tokens
            
            while True:
                result = self.local_llm.generate(query, max_tokens=current_max_tokens)
                
                # Verify the response
                verdict = self.verifier.verify(
                    answer=result["text"],
                    output_tokens=result["output_tokens"],
                    max_tokens=current_max_tokens,
                    query=query,  # For relevance checking
                    difficulty=difficulty  # For relevance gating
                )
                
                # If verification passes, return
                if verdict.passed:
                    result["verification"] = "passed"
                    break
                
                # If truncated and we have retry budget, regenerate with more tokens
                if verdict.truncated and retry_count < self.max_retries:
                    retry_count += 1
                    current_max_tokens *= 2  # Double the token budget
                    continue
                
                # Verification failed and no retries left
                result["verification"] = f"failed: {', '.join(verdict.reasons)}"
                break
            
            # Estimate what remote call would cost (for cost_saved calculation)
            estimated_remote_cost = (
                (result["input_tokens"] / 1000) * 0.005 +
                (result["output_tokens"] / 1000) * 0.015
            )
            
            routing_decision = "repaired" if retry_count > 0 and verdict.passed else "local"
            result.update({
                "difficulty": difficulty,
                "routing_decision": routing_decision,
                "cost_usd": 0.0,  # Local model cost is effectively $0
                "cost_saved_usd": round(estimated_remote_cost, 6),
                "cost_saved": REMOTE_COST - LOCAL_COST  # Keep relative units too
            })
            return result
        
        # 3. Medium queries → local first, verify, regenerate if needed, escalate if still fails
        if difficulty < 0.6:
            retry_count = 0
            current_max_tokens = max_tokens
            
            while True:
                local_result = self.local_llm.generate(query, max_tokens=current_max_tokens)
                
                # Verify the response
                verdict = self.verifier.verify(
                    answer=local_result["text"],
                    output_tokens=local_result["output_tokens"],
                    max_tokens=current_max_tokens,
                    query=query,  # For relevance checking
                    difficulty=difficulty  # For relevance gating
                )
                
                # If verification passes, return local result
                if verdict.passed:
                    local_result["verification"] = "passed"
                    break
                
                # If truncated and we have retry budget, regenerate with more tokens
                if verdict.truncated and retry_count < self.max_retries:
                    retry_count += 1
                    current_max_tokens *= 2  # Double the token budget
                    continue
                
                # Verification failed and no retries left
                local_result["verification"] = f"failed: {', '.join(verdict.reasons)}"
                break
            
            # If verification passed (after regeneration if needed), return local result
            if verdict.passed:
                estimated_remote_cost = (
                    (local_result["input_tokens"] / 1000) * 0.005 +
                    (local_result["output_tokens"] / 1000) * 0.015
                )
                routing_decision = "repaired" if retry_count > 0 else "local"
                local_result.update({
                    "difficulty": difficulty,
                    "routing_decision": routing_decision,
                    "cost_usd": 0.0,
                    "cost_saved_usd": round(estimated_remote_cost, 6),
                    "cost_saved": REMOTE_COST - LOCAL_COST
                })
                return local_result
            
            # If verification failed (uncertainty, low relevance, or regeneration failed), escalate
            if self.remote_llm is not None:
                remote_result = self.remote_llm.generate(query)
                remote_result.update({
                    "difficulty": difficulty,
                    "routing_decision": "escalated",
                    "cost_saved_usd": 0.0,
                    "cost_saved": 0,
                    "verification": f"failed: {', '.join(verdict.reasons)}"
                })
                return remote_result
            else:
                # No remote LLM available, return local result with warning
                local_result.update({
                    "difficulty": difficulty,
                    "routing_decision": "local",
                    "cost_usd": 0.0,
                    "cost_saved_usd": 0.0,
                    "cost_saved": 0
                })
                return local_result
        
        # 4. Hard queries → remote model directly
        if self.remote_llm is None:
            raise ValueError("Remote LLM not provided for hard queries")
        
        remote_result = self.remote_llm.generate(query, max_tokens=max_tokens)
        # Verify remote result too (for consistency, though we trust GPT-4o more)
        verdict = self.verifier.verify(
            answer=remote_result["text"],
            output_tokens=remote_result["output_tokens"],
            max_tokens=max_tokens,
            query=query,  # For relevance checking
            difficulty=difficulty  # For relevance gating
        )
        remote_result.update({
            "difficulty": difficulty,
            "routing_decision": "remote",
            "cost_saved_usd": 0.0,  # No savings, we used the expensive model
            "cost_saved": 0,
            "verification": "passed" if verdict.passed else f"failed: {', '.join(verdict.reasons)}"
        })
        return remote_result

