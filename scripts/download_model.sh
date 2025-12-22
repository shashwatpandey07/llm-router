#!/bin/bash

# Download script for Phi-2 GGUF model
# This downloads the Q4_K_M quantized version (best balance)

MODEL_DIR="../models"
MODEL_NAME="phi-2.Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"

echo "Downloading Phi-2 GGUF model..."
echo "Model: $MODEL_NAME"
echo "Destination: $MODEL_DIR"

mkdir -p "$MODEL_DIR"

cd "$MODEL_DIR"

if [ -f "$MODEL_NAME" ]; then
    echo "Model already exists. Skipping download."
else
    echo "Downloading from Hugging Face..."
    curl -L -o "$MODEL_NAME" "$MODEL_URL"
    
    if [ $? -eq 0 ]; then
        echo "✅ Download complete!"
        echo "Model saved to: $(pwd)/$MODEL_NAME"
    else
        echo "❌ Download failed. Please download manually from:"
        echo "https://huggingface.co/TheBloke/phi-2-GGUF"
    fi
fi

