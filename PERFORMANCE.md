# Performance Guide for Apple Silicon (M4)

## What to Expect

### First Run (Model Loading + Compilation)
- **Model Loading**: 10-20 seconds (loading 2.7B parameters)
- **torch.compile()**: 30-60 seconds (one-time compilation)
- **Warm-up Generation**: 30-60 seconds (MPS kernel compilation)
- **Total First Run**: ~2-3 minutes

### Subsequent Generations
- **After compilation**: 1-5 seconds for 20 tokens
- **After warm-up**: 0.5-2 seconds for 20 tokens

## Optimizations Applied

1. **torch.compile()**: Compiles the model for faster inference (one-time cost)
2. **float16**: Uses half precision when possible (2x faster, 2x less memory)
3. **Model Warm-up**: Pre-compiles MPS kernels during initialization
4. **Optimized Generation**: Uses greedy decoding with KV cache

## Why First Generation is Slow

Apple Silicon uses Metal Performance Shaders (MPS) which compiles kernels on first use:
- First generation compiles all operations
- Subsequent generations reuse compiled kernels
- This is **normal behavior** and expected

## Tips for Faster Development

1. **Run warm-up once**: The model will be compiled after first generation
2. **Keep model in memory**: Don't reload between tests
3. **Use smaller max_tokens for testing**: Test with 10-20 tokens, not 256
4. **Consider quantized models**: For even faster inference, use 4-bit or 8-bit quantized models

## If Still Too Slow

If performance is still unacceptable after optimizations:

1. **Use a smaller model**: Try `microsoft/phi-1.5` (1.3B params) instead of phi-2 (2.7B)
2. **Use quantized version**: Look for `microsoft/phi-2-GGUF` or similar
3. **Consider cloud inference**: For production, consider using API-based models

## Expected Performance Metrics

| Metric | First Run | After Compilation |
|--------|-----------|-------------------|
| Model Load | 10-20s | 10-20s |
| First Generation | 60-120s | 1-5s |
| Subsequent | N/A | 0.5-2s |

These are realistic expectations for a 2.7B parameter model on M4 MacBook Pro.

