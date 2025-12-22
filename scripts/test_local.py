"""
Sanity test for LocalLLM implementation.

This script tests the basic functionality of the LocalLLM class
with a GGUF quantized model (phi-2).

IMPORTANT: Make sure to activate your virtual environment first:
    source ../../.venv/bin/activate
    python3 test_local.py
"""

import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.local import LocalLLM


def main():
    """Run sanity test for LocalLLM."""
    # Use GGUF model path
    model_path = Path(__file__).parent.parent / "models" / "phi-2.Q4_K_M.gguf"
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("\nPlease download the GGUF model first:")
        print("1. Visit: https://huggingface.co/TheBloke/phi-2-GGUF")
        print("2. Download: phi-2.Q4_K_M.gguf")
        print(f"3. Place it in: {model_path.parent}")
        return
    
    print(f"Using GGUF model: {model_path}")
    llm = LocalLLM(str(model_path))
    
    prompt = "Explain gradient descent in simple terms."
    print(f"\nPrompt: {prompt}\n")
    
    print("Generating response...")
    result = llm.generate(prompt, max_tokens=50)  # Limit tokens to prevent runaway generation
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Model: {result['model']}")
    print(f"Device: {result['device']}")
    print(f"Response: {result['text']}")
    print(f"Latency (ms): {result['latency_ms']:.2f}")
    print(f"Input tokens: {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")
    print("="*60)


if __name__ == "__main__":
    main()

