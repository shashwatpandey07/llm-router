"""
Test script for QueryDifficultyEstimator.

Tests the zero-cost difficulty estimation on 10-15 diverse queries
to validate the scoring system.
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from routing.difficulty import QueryDifficultyEstimator


def main():
    """Test difficulty estimation on 10-15 diverse queries."""
    estimator = QueryDifficultyEstimator()
    
    test_queries = [
        # Easy queries (should score < 0.3)
        ("What is Python?", "Simple definition"),
        ("List three programming languages", "Simple list"),
        ("Who invented the computer?", "Factual question"),
        ("Define machine learning", "Definition request"),
        ("When was Python created?", "Simple factual"),
        
        # Medium queries (should score 0.3-0.6)
        ("Explain the difference between Python and Java", "Comparison"),
        ("Describe how neural networks work", "Explanation"),
        ("Summarize the main concepts in machine learning", "Summary"),
        ("Compare supervised and unsupervised learning", "Comparison"),
        ("What are the advantages and disadvantages of deep learning?", "Multi-part question"),
        
        # Hard queries (should score > 0.6)
        ("Why does gradient descent converge and how can we prove it mathematically?", "Reasoning + proof"),
        ("Analyze the implications of quantum computing on cryptography", "Analysis"),
        ("Prove that the halting problem is undecidable", "Proof request"),
        ("How can we justify the use of regularization in preventing overfitting?", "Justification + reasoning"),
        ("What are the theoretical foundations and practical implications of transformer architecture in modern NLP?", "Complex multi-part"),
    ]
    
    print("=" * 80)
    print("Query Difficulty Estimation Test")
    print("=" * 80)
    print(f"Testing {len(test_queries)} queries\n")
    
    easy_count = 0
    medium_count = 0
    hard_count = 0
    
    for i, (query, description) in enumerate(test_queries, 1):
        score = estimator.estimate(query)
        length = estimator._length_score(query)
        keyword = estimator._keyword_score(query)
        structure = estimator._structure_score(query)
        
        # Categorize
        if score < 0.3:
            category = "🟢 Easy"
            easy_count += 1
        elif score < 0.6:
            category = "🟡 Medium"
            medium_count += 1
        else:
            category = "🔴 Hard"
            hard_count += 1
        
        print(f"{i:2d}. {category} (Score: {score:.3f})")
        print(f"    Query: {query}")
        print(f"    Type: {description}")
        print(f"    Breakdown: length={length:.3f}, keyword={keyword:.3f}, structure={structure:.3f}")
        print()
    
    print("=" * 80)
    print("Summary:")
    print(f"  🟢 Easy queries:   {easy_count}")
    print(f"  🟡 Medium queries: {medium_count}")
    print(f"  🔴 Hard queries:   {hard_count}")
    print(f"  Total:            {len(test_queries)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

