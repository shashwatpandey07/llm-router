"""
Response Verification Module.

Verifies whether an LLM response is acceptable or needs repair/escalation.
Implements answer-aware routing by checking response quality.
"""

from dataclasses import dataclass
from typing import List, Optional
import os

try:
    import numpy as np
    from openai import OpenAI
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class VerificationResult:
    """
    Result of response verification.
    
    Attributes:
        passed: Whether the response passed verification
        reasons: List of failure reasons (if any)
        truncated: Whether the response was truncated
        uncertainty: Whether the response shows uncertainty
        low_relevance: Whether the answer is semantically off-topic
    """
    passed: bool
    reasons: List[str]
    truncated: bool
    uncertainty: bool
    low_relevance: bool


class ResponseVerifier:
    """
    Verifies whether an LLM response is acceptable or needs repair/escalation.
    
    Checks for:
    - Truncation (incomplete sentences when max_tokens reached)
    - Uncertainty (low confidence phrases)
    - Low relevance (semantic similarity between query and answer)
    """
    
    UNCERTAINTY_PHRASES = [
        "i'm not sure",
        "i am not sure",
        "cannot determine",
        "unclear",
        "it depends",
        "might be",
        "may be"
    ]
    
    # Relevance thresholds (for answer validation, not retrieval)
    # These are lower than retrieval thresholds because answers can be more abstract
    RELEVANCE_FAIL = 0.60  # Below this: clearly off-topic
    RELEVANCE_WARN = 0.70  # Suspicious but acceptable for explanations
    
    def __init__(self):
        """Initialize the verifier."""
        self._embedding_client = None
        if EMBEDDINGS_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self._embedding_client = OpenAI(api_key=api_key)
    
    def _embed(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text using OpenAI.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if embeddings unavailable
        """
        if not EMBEDDINGS_AVAILABLE or not self._embedding_client:
            return None
        
        try:
            response = self._embedding_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception:
            return None
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not EMBEDDINGS_AVAILABLE:
            return 1.0  # Default to high similarity if embeddings unavailable
        
        a_np = np.array(a)
        b_np = np.array(b)
        dot_product = np.dot(a_np, b_np)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _truncate_for_embedding(self, text: str, max_chars: int = 500) -> str:
        """
        Truncate text for embedding to preserve topic relevance.
        
        Embedding only the first N characters avoids dilution from long answers
        and focuses on the core semantic signal. This is an industry standard trick.
        
        Args:
            text: Text to truncate
            max_chars: Maximum characters to keep (default 500)
            
        Returns:
            Truncated text
        """
        return text[:max_chars]
    
    def _basic_coverage(self, query: str, answer: str) -> bool:
        """
        Cheap lexical sanity check before embeddings.
        
        Checks if answer contains key terms from the query. This catches:
        - Off-topic rambling
        - Partial answers (e.g., "Compare Python and Java" → only Python mentioned)
        
        Args:
            query: Original query
            answer: Generated answer
            
        Returns:
            True if answer has basic coverage of query keywords
        """
        # Extract keywords (words longer than 3 chars, excluding common stop words)
        stop_words = {"the", "and", "or", "but", "for", "with", "from", "that", "this", "what", "how", "why"}
        keywords = [w for w in query.lower().split() if len(w) > 3 and w not in stop_words]
        
        if not keywords:
            return True  # No keywords to check, assume coverage
        
        answer_lower = answer.lower()
        hits = sum(1 for w in keywords if w in answer_lower)
        
        # Require at least 1 hit, or at least 1/3 of keywords
        return hits >= max(1, len(keywords) // 3)
    
    def _is_list_query(self, query: str) -> bool:
        """
        Detect list-style queries.
        
        List queries naturally end without punctuation (e.g., "List three languages: Python, Java, C++").
        These should not be flagged as truncated.
        
        Args:
            query: Original query
            
        Returns:
            True if query is a list-style query
        """
        if not query:
            return False
        
        q = query.lower()
        return any(q.startswith(w) for w in ["list", "name", "give", "mention"])
    
    def _check_relevance(self, query: str, answer: str, difficulty: float = 1.0) -> tuple[bool, float]:
        """
        Check if answer is semantically relevant to query.
        
        Uses dual-threshold logic:
        - < 0.60: Clearly off-topic (fail)
        - 0.60-0.70: Suspicious but acceptable for explanations (warn)
        - ≥ 0.70: Good relevance
        
        For hard queries (difficulty ≥ 0.6), relevance checking is more lenient
        as answers are expected to be more abstract/analytical.
        
        Args:
            query: Original query
            answer: Generated answer
            difficulty: Query difficulty score (0.0 to 1.0)
            
        Returns:
            Tuple of (is_relevant, similarity_score)
        """
        if not self._embedding_client:
            return True, 1.0  # Skip relevance check if embeddings unavailable
        
        # Embed only the answer summary (first 500 chars) to preserve topic relevance
        query_embedding = self._embed(query)
        answer_summary = self._truncate_for_embedding(answer)
        answer_embedding = self._embed(answer_summary)
        
        if query_embedding is None or answer_embedding is None:
            return True, 1.0  # Skip if embedding failed
        
        similarity = self._cosine_similarity(query_embedding, answer_embedding)
        
        # Dual-threshold logic: only fail if clearly off-topic
        # For hard queries, be more lenient as answers are expected to drift
        if difficulty >= 0.6:
            # Hard queries: only fail if very low similarity
            is_relevant = similarity >= self.RELEVANCE_FAIL
        else:
            # Easy/Medium queries: use standard threshold
            is_relevant = similarity >= self.RELEVANCE_FAIL
        
        return is_relevant, similarity
    
    def _is_semantically_incomplete(self, answer: str) -> bool:
        """
        Check if an answer is semantically incomplete (cut off mid-thought).
        
        Uses semantic heuristics to detect when a response was cut off mid-sentence,
        even if it hit the token limit. This prevents false positives where a response
        is complete despite hitting max_tokens.
        
        Args:
            answer: The response text to check
            
        Returns:
            True if the answer appears semantically incomplete
        """
        text = answer.strip().lower()
        
        # If it ends with proper punctuation, it's semantically complete
        # Even if it hit max_tokens, a sentence ending with punctuation is acceptable
        if text.endswith((".", "!", "?")):
            return False
        
        # Check for bad endings that indicate mid-sentence truncation
        # These are common ways sentences get cut off (prepositions, conjunctions, etc.)
        BAD_ENDINGS = (
            " by", " which", " that", " because", " such as",
            " including", " like", " for example", " and", " or", " but",
            " with", " from", " to", " in", " on", " at", " of", " the",
            " a", " an", " is", " are", " was", " were", " has", " have",
            " can", " could", " should", " would", " will", " may", " might"
        )
        
        # If it ends with a bad ending, it's semantically incomplete
        if any(text.endswith(be) for be in BAD_ENDINGS):
            return True
        
        # If it doesn't end with punctuation and doesn't match bad endings,
        # but ends abruptly (short last word, no space before end), it might be incomplete
        # However, we're conservative: only flag if it matches a known bad pattern
        return False
    
    def verify(
        self,
        answer: str,
        output_tokens: int,
        max_tokens: int,
        query: Optional[str] = None,
        difficulty: float = 1.0
    ) -> VerificationResult:
        """
        Verify if a response is acceptable.
        
        Args:
            answer: The generated response text
            output_tokens: Number of tokens in the response
            max_tokens: Maximum tokens that were requested
            query: Original query (optional, for relevance checking)
            difficulty: Query difficulty score (0.0 to 1.0, for relevance gating)
            
        Returns:
            VerificationResult with pass/fail status and reasons
        """
        reasons = []
        
        # 1. Truncation / incompleteness check
        # Only flag as truncated if we hit max_tokens AND the answer is semantically incomplete
        truncated = False
        if output_tokens >= max_tokens:
            if self._is_semantically_incomplete(answer):
                truncated = True
                reasons.append("truncated")
                
                # Minimal fix: List queries naturally end without punctuation
                # Allow truncation for list-style queries
                if query and self._is_list_query(query):
                    truncated = False
                    reasons.remove("truncated")
        
        # 2. Uncertainty check
        # Look for phrases that indicate the model is uncertain
        uncertainty = False
        lower = answer.lower()
        if any(p in lower for p in self.UNCERTAINTY_PHRASES):
            uncertainty = True
            reasons.append("uncertainty")
        
        # 3. Relevance check (embedding-based semantic similarity)
        # Fix 1: Skip relevance checks for easy queries (definitions, lists, factual)
        # Easy queries need completeness, not topic drift detection
        low_relevance = False
        if query and difficulty >= 0.3:  # Only check relevance for medium/hard queries
            # Fix 3: Cheap lexical coverage check first (catches obvious off-topic answers)
            if not self._basic_coverage(query, answer):
                low_relevance = True
                reasons.append("low_relevance (basic coverage failed)")
            else:
                # Only do expensive embedding check if basic coverage passes
                if difficulty < 0.6:  # Medium queries
                    is_relevant, similarity = self._check_relevance(query, answer, difficulty)
                    if not is_relevant:
                        low_relevance = True
                        reasons.append(f"low_relevance (similarity: {similarity:.3f})")
                else:  # Hard queries (difficulty >= 0.6)
                    # For hard queries, check but don't fail - just log if suspicious
                    is_relevant, similarity = self._check_relevance(query, answer, difficulty)
                    if similarity < self.RELEVANCE_WARN:
                        # Log but don't fail for hard queries
                        reasons.append(f"low_relevance (similarity: {similarity:.3f}, but hard query - allowed)")
        
        # Fix 2: Lower relevance enforcement for medium queries
        # Truncation and uncertainty are fatal
        # Low relevance is advisory for medium queries (0.3-0.6), not fatal
        # Hard queries (>= 0.6) also allow low relevance (already gated above)
        if truncated or uncertainty:
            passed = False
        elif low_relevance:
            # For medium queries (0.3-0.6), low relevance is advisory, should pass
            # For hard queries (>= 0.6), low relevance is also allowed
            # Only fail if it's an easy query with low relevance (but easy queries skip relevance check)
            passed = difficulty >= 0.3  # Medium and hard queries pass despite low relevance
        else:
            # Pass if no issues
            passed = True
        
        return VerificationResult(
            passed=passed,
            reasons=reasons,
            truncated=truncated,
            uncertainty=uncertainty,
            low_relevance=low_relevance
        )

