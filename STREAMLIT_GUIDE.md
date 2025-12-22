# Streamlit UI Guide

## Quick Start

### 1. Install Dependencies

```bash
cd llm-router
source ../.venv/bin/activate
pip install streamlit
```

### 2. Set OpenAI API Key (Optional)

```bash
export OPENAI_API_KEY=your_key_here
```

### 3. Run the App

```bash
streamlit run app.py
```

Or use the convenience script:
```bash
cd scripts
./run_app.sh
```

The app will open in your browser at `http://localhost:8501`

## Features

### Main Interface
- **Query Input**: Large text area for entering queries
- **Route Query Button**: Processes the query and shows results
- **Quick Stats**: Real-time tracking of total queries, cost, and savings

### Analysis Display
- **Difficulty Score**: Visual indicator (Easy/Medium/Hard)
- **Routing Decision**: Shows which model was used
- **Latency**: Response time in milliseconds and seconds
- **Detailed Breakdown**: 
  - Length, Keyword, and Structure scores
  - Cost analysis (actual cost and savings)
  - Token usage (input/output)
  - Model information

### Response Display
- Full or truncated response based on settings
- Copy-friendly text area

### Sidebar
- **System Status**: Shows initialization status
- **Settings**: Toggle detailed analysis and full response
- **About**: Quick reference for routing logic

## Usage Tips

1. **First Query**: May take longer as the model loads
2. **Local Queries**: Should be fast (~1-2 seconds)
3. **GPT-4o Queries**: Will take 5-8 seconds
4. **Cost Tracking**: Cumulative stats persist during session
5. **Clear Stats**: Use the "Clear" button to reset statistics

## Troubleshooting

- **Model not found**: Ensure `phi-2.Q4_K_M.gguf` is in `models/` directory
- **OpenAI errors**: Check that API key is set correctly
- **Slow loading**: First load takes time to initialize the local model
- **Port already in use**: Change port with `streamlit run app.py --server.port 8502`

