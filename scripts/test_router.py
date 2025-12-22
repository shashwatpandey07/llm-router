"""
Integration test for LLMRouter.

Tests the complete routing system with difficulty estimation,
model selection, and cost tracking.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from routing.router import LLMRouter
from routing.difficulty import QueryDifficultyEstimator
from llm.local import LocalLLM
from llm.openai_llm import OpenAILLM
from llm.base import BaseLLM
from utils.metrics import MetricsLogger
from typing import Dict
import os


class MockRemoteLLM(BaseLLM):
    """
    Mock remote LLM for testing.
    
    Simulates a high-quality API model (e.g., GPT-4) without making actual API calls.
    """
    
    def __init__(self):
        """Initialize mock remote LLM."""
        self.model_name = "mock-remote-llm"
    
    def generate(self, prompt: str, max_tokens: int = 256) -> Dict:
        """
        Generate mock response.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with mock response data
        """
        # Simulate a high-quality response
        mock_responses = {
            "what is python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms and has a large standard library.",
            "explain": "This is a detailed explanation that demonstrates the capabilities of a high-quality language model. It provides comprehensive information with proper structure and clarity.",
            "prove": "Here is a rigorous mathematical proof that demonstrates the theoretical foundation and logical reasoning capabilities required for complex problem-solving."
        }
        
        prompt_lower = prompt.lower()
        if "python" in prompt_lower:
            text = mock_responses["what is python"]
        elif any(word in prompt_lower for word in ["explain", "describe", "difference"]):
            text = mock_responses["explain"]
        elif any(word in prompt_lower for word in ["prove", "analyze", "why"]):
            text = mock_responses["prove"]
        else:
            text = "This is a comprehensive response from a high-quality language model that addresses the query with depth and accuracy."
        
        # Simulate realistic metrics
        return {
            "text": text,
            "input_tokens": len(prompt.split()),
            "output_tokens": len(text.split()),
            "latency_ms": 1500.0,  # Simulate API latency
            "model": self.model_name,
            "device": "api"
        }


def main():
    """Test the router with various queries."""
    # Setup
    print("=" * 80)
    print("LLM Router Integration Test")
    print("=" * 80)
    print()
    
    # Initialize components
    print("Initializing components...")
    difficulty_estimator = QueryDifficultyEstimator()
    
    # Load local LLM (GGUF model)
    model_path = Path(__file__).parent.parent / "models" / "phi-2.Q4_K_M.gguf"
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please download the GGUF model first.")
        return
    
    print(f"Loading local model: {model_path}")
    local_llm = LocalLLM(str(model_path))
    
    # Create real OpenAI remote LLM
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not found in environment.")
        print("Falling back to mock remote LLM for testing.")
        print("To use real GPT-4o, set: export OPENAI_API_KEY=your_key_here")
        remote_llm = MockRemoteLLM()
    else:
        print("Initializing OpenAI GPT-4o...")
        remote_llm = OpenAILLM(api_key=api_key, model="gpt-4o")
    
    # Initialize router
    router = LLMRouter(
        difficulty_estimator=difficulty_estimator,
        local_llm=local_llm,
        remote_llm=remote_llm
    )
    print("Router initialized!")
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(log_dir="../logs")
    print("Metrics logger initialized!\n")
    
    # Test queries
    test_queries = [
        ("What is Python?", "Easy - should route to local"),
        ("List three programming languages", "Easy - should route to local"),
        ("Explain the difference between Python and Java", "Medium - local first, may escalate"),
        ("What are the advantages and disadvantages of deep learning?", "Medium - local first, may escalate"),
        ("Prove that the halting problem is undecidable", "Hard - should route to remote"),
        ("Analyze the implications of quantum computing on cryptography", "Hard - should route to remote"),
    ]
    
    print("=" * 80)
    print("Testing Routing Decisions")
    print("=" * 80)
    print()
    
    total_cost_saved = 0
    total_cost_saved_usd = 0.0
    total_cost_usd = 0.0
    routing_stats = {"local": 0, "escalated": 0, "remote": 0}
    
    for i, (query, description) in enumerate(test_queries, 1):
        print(f"{i}. Query: {query}")
        print(f"   Expected: {description}")
        print()
        
        try:
            result = router.route(query)
            
            # Log metrics
            metrics_logger.log(result, query)
            
            decision = result["routing_decision"]
            routing_stats[decision] += 1
            total_cost_saved += result.get("cost_saved", 0)
            total_cost_saved_usd += result.get("cost_saved_usd", 0.0)
            total_cost_usd += result.get("cost_usd", 0.0)
            
            print(f"   ✅ Routing Decision: {decision.upper()}")
            print(f"   📊 Difficulty: {result['difficulty']:.3f}")
            print(f"   💰 Cost: ${result.get('cost_usd', 0.0):.6f}")
            print(f"   💵 Cost Saved: ${result.get('cost_saved_usd', 0.0):.6f} ({result.get('cost_saved', 0)} units)")
            print(f"   ⏱️  Latency: {result['latency_ms']:.2f} ms")
            print(f"   📝 Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
            print(f"   🤖 Model: {result['model']}")
            print(f"   📄 Response: {result['text'][:100]}...")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total queries tested: {len(test_queries)}")
    print(f"Routing decisions:")
    print(f"  🟢 Local:     {routing_stats['local']}")
    print(f"  🟡 Escalated: {routing_stats['escalated']}")
    print(f"  🔴 Remote:    {routing_stats['remote']}")
    print(f"Total cost: ${total_cost_usd:.6f}")
    print(f"Total cost saved: ${total_cost_saved_usd:.6f} ({total_cost_saved} units)")
    
    # Export metrics
    metrics_logger.export_json()
    summary = metrics_logger.get_summary()
    print(f"\n📊 Metrics logged to:")
    print(f"   CSV: {summary.get('csv_file', 'N/A')}")
    print(f"   JSON: {summary.get('json_file', 'N/A')}")
    print("=" * 80)


if __name__ == "__main__":
    main()

