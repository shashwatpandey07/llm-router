"""
Query Difficulty Estimation.

This module provides zero-cost signals for estimating query difficulty
without making additional LLM calls.

Phase 2.1: Zero-cost signals
- Prompt length (tokens)
- Sentence structure (question type)
- Keyword heuristics (why/how/prove vs define/list)
- Embedding norm / similarity (later)
"""

import re


class QueryDifficultyEstimator:
    """
    Estimates query difficulty using zero-cost signals.
    
    Returns a difficulty score between 0 (easy) and 1 (hard).
    Higher scores indicate more complex queries that may require
    larger or more capable models.
    """
    
    EASY_KEYWORDS = {
        "what", "define", "definition", "list", "name", "who", "when", "where"
    }
    
    MEDIUM_KEYWORDS = {
        "explain", "describe", "summarize", "compare", "difference", "overview"
    }
    
    HARD_KEYWORDS = {
        "why", "how", "prove", "derive", "analyze", "reason", "justify",
        "evaluate", "critique", "implications"
    }
    
    def __init__(self):
        """Initialize the difficulty estimator."""
        pass
    
    def _length_score(self, query: str) -> float:
        """
        Score based on token/word length.
        
        Capped to avoid over-weighting long but simple queries.
        
        Args:
            query: The input query string
            
        Returns:
            Length-based difficulty score (0.0 to 1.0)
        """
        word_count = len(query.split())
        
        # Normalize: <=5 words = easy, >=30 words = hard
        if word_count <= 5:
            return 0.1
        elif word_count >= 30:
            return 1.0
        else:
            return (word_count - 5) / 25
    
    def _keyword_score(self, query: str) -> float:
        """
        Score based on intent keywords.
        
        Args:
            query: The input query string
            
        Returns:
            Keyword-based difficulty score (0.0 to 1.0)
        """
        query_lower = query.lower()
        
        if any(k in query_lower for k in self.HARD_KEYWORDS):
            return 1.0
        elif any(k in query_lower for k in self.MEDIUM_KEYWORDS):
            return 0.5
        elif any(k in query_lower for k in self.EASY_KEYWORDS):
            return 0.1
        else:
            return 0.3  # neutral / unknown intent
    
    def _structure_score(self, query: str) -> float:
        """
        Score based on structural complexity.
        
        Args:
            query: The input query string
            
        Returns:
            Structure-based difficulty score (0.0 to 1.0)
        """
        score = 0.0
        
        # Multiple sentences
        if query.count(".") + query.count("?") > 1:
            score += 0.4
        
        # Conjunctions indicating multi-part reasoning
        if re.search(r"\b(and|or|vs|versus|while)\b", query.lower()):
            score += 0.3
        
        # Conditional or causal phrasing
        if re.search(r"\b(if|because|therefore|however)\b", query.lower()):
            score += 0.3
        
        return min(score, 1.0)
    
    def estimate(self, query: str) -> float:
        """
        Estimate difficulty score between 0 and 1.
        
        Combines three zero-cost signals with weighted average:
        - 50% weight on keyword intent (most important)
        - 25% weight on query length
        - 25% weight on structural complexity
        
        Applies force multipliers for:
        - Hard reasoning intent (prove/analyze/why) → minimum 0.6
        - Multi-part evaluative phrasing → minimum 0.5
        
        Args:
            query: The input query/prompt string
            
        Returns:
            Difficulty score between 0.0 (easy) and 1.0 (hard)
            - 0.0-0.3: Easy queries (simple facts, definitions)
            - 0.3-0.6: Medium queries (explanations, comparisons)
            - 0.6-1.0: Hard queries (reasoning, proofs, complex analysis)
        """
        length = self._length_score(query)
        keyword = self._keyword_score(query)
        structure = self._structure_score(query)
        
        # Base weighted score
        # Rebalanced: intent matters most (50%), length/structure are modifiers (25% each)
        difficulty = (
            0.25 * length +
            0.5 * keyword +
            0.25 * structure
        )
        
        q = query.lower()
        
        # Force harder classification for strong reasoning intent
        # "Prove X" is hard even if it's short
        if any(k in q for k in self.HARD_KEYWORDS):
            difficulty = max(difficulty, 0.6)
        
        # Explicit multi-part evaluative phrasing
        # "Advantages and disadvantages" is harder than "what is"
        if any(p in q for p in [
            "advantages and disadvantages",
            "pros and cons",
            "trade-offs",
            "implications",
            "limitations"
        ]):
            difficulty = max(difficulty, 0.5)
        
        return round(min(difficulty, 1.0), 3)

