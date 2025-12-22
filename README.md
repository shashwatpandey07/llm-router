# LLM Router

A **cost-aware**, **accuracy-preserving** LLM routing and serving system that dynamically selects which model to call, how many times to call, and when to escalate or abstain.

## Problem

Large Language Models vary drastically in **cost**, **latency**, and **capability**, but most systems either:
- always call the largest model (expensive), or
- statically choose a smaller model (low quality)

## Goal

Build a **cost-aware**, **accuracy-preserving** LLM routing and serving system that dynamically selects:
- *which model to call*
- *how many times to call*
- *when to escalate or abstain*

This is a **systems + ML** problem, not a prompt problem.

## Project Structure

```
llm-router/
├── llm/
│   ├── __init__.py
│   ├── base.py          # Abstract LLM interface
│   ├── local.py         # Local llama.cpp implementation
├── routing/
│   ├── __init__.py
│   └── difficulty.py   # Query difficulty estimation
├── scripts/
│   └── test_local.py    # Quick sanity test
├── utils/               # Utility functions
├── main/                # Main application code
├── config/              # Configuration files
├── logs/                # Log files
├── models/              # GGUF model files
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm-router
```

2. Install dependencies:
```bash
# Install llama-cpp-python with Metal support (for Apple Silicon)
CMAKE_ARGS="-DLLAMA_METAL=on" pip3 install llama-cpp-python

# Install other dependencies
pip3 install -r requirements.txt
```

3. Download a GGUF model:
```bash
# Option 1: Use the download script
cd scripts
./download_model.sh

# Option 2: Download manually from Hugging Face
# Visit: https://huggingface.co/TheBloke/phi-2-GGUF
# Download: phi-2.Q4_K_M.gguf
# Place it in: llm-router/models/
```

**Note:** This project uses llama.cpp backend for 10-50x faster inference on Mac (Metal acceleration).

## Quick Start

### Option 1: Run test script (with venv activated)

```bash
cd llm-router
source ../.venv/bin/activate  # Activate your virtual environment
python3 scripts/test_local.py
```

### Option 2: Install package in development mode (recommended)

```bash
cd llm-router
source ../.venv/bin/activate
pip3 install -e .
python3 scripts/test_local.py
```

**Note:** Make sure your virtual environment is activated and has `llama-cpp-python` installed with Metal support.

This will:
1. Load the GGUF quantized model (phi-2.Q4_K_M.gguf)
2. Generate a response to a test prompt
3. Display the response, latency, and token counts
4. **Expected latency: 0.5-2 seconds** (10-50x faster than transformers!)

## Development

### Milestone 1: Local LLM Implementation ✅

- ✅ Base LLM abstraction (`llm/base.py`)
- ✅ Local LLM implementation with llama.cpp backend (`llm/local.py`)
- ✅ Sanity test (`scripts/test_local.py`)
- ✅ Metal-accelerated inference (10-50x faster than transformers)

### Milestone 1.5: Fast Local Inference (llama.cpp) ✅

- ✅ Switched to llama.cpp backend for Mac optimization
- ✅ GGUF quantized model support (4-bit/8-bit)
- ✅ Metal GPU acceleration enabled
- ✅ Sub-second inference times

### Milestone 2: Query Difficulty Estimation 🚧

- ✅ Routing module structure created
- ✅ QueryDifficultyEstimator class stub
- 🚧 Phase 2.1: Zero-cost signals (in progress)
  - Prompt length (tokens)
  - Sentence structure (question type)
  - Keyword heuristics (why/how/prove vs define/list)
  - Embedding norm / similarity (later)

## License

[To be determined]

