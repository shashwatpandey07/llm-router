#!/bin/bash

# Run Streamlit app for LLM Router

cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

# Set OpenAI API key if provided
if [ -n "$OPENAI_API_KEY" ]; then
    export OPENAI_API_KEY
fi

# Run Streamlit app
streamlit run app.py

