#!/bin/bash

# Self-RAG Evaluation Setup Script for RunPod
# Run this once when you first connect to your RunPod instance

echo "🚀 Setting up Self-RAG Evaluation Environment on RunPod"
echo "=" * 60

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y git wget curl

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify CUDA setup
echo "🔥 Verifying CUDA setup..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Pre-cache the Self-RAG model (optional but recommended)
echo "📥 Pre-downloading Self-RAG model (this may take a while)..."
python -c "
from huggingface_hub import snapshot_download
try:
    snapshot_download('selfrag/selfrag_llama2_7b', cache_dir='/workspace/.cache')
    print('✅ Model pre-downloaded successfully')
except Exception as e:
    print(f'⚠️  Model pre-download failed: {e}')
    print('   Model will be downloaded during evaluation')
"

# Create results directory
mkdir -p results
mkdir -p logs

echo "✅ Setup complete! Ready to run Self-RAG evaluation."
echo "🎯 Now you can run: python selfrag_evaluation.py"
