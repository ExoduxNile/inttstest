#!/bin/bash

# setup.sh - Index-TTS setup with Python 3.10

set -e  # Exit on error

# Configuration
REQUIRED_PYTHON_VERSION="3.10"
MODEL_FILES=(
    "bigvgan_discriminator.pth"
    "bigvgan_generator.pth"
    "bpe.model"
    "dvae.pth"
    "gpt.pth"
    "unigram_12000.vocab"
)
REPO_URL="https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main"

echo "=== Setting up Index-TTS with Python $REQUIRED_PYTHON_VERSION ==="

# Install Python 3.10 if not available
if ! command -v python$REQUIRED_PYTHON_VERSION &> /dev/null; then
    echo "Installing Python $REQUIRED_PYTHON_VERSION..."
    sudo apt-get update
    sudo apt-get install -y \
        software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y \
        python$REQUIRED_PYTHON_VERSION \
        python$REQUIRED_PYTHON_VERSION-venv \
        python$REQUIRED_PYTHON_VERSION-dev
fi

# Create and activate virtual environment
echo "Creating Python $REQUIRED_PYTHON_VERSION virtual environment..."
python$REQUIRED_PYTHON_VERSION -m venv venv
source venv/bin/activate

# Install system dependencies
echo "Installing system packages..."
sudo apt-get install -y \
    wget \
    ffmpeg

# Ensure pip is up to date
python -m pip install --upgrade pip

# Install PyTorch for CPU
echo "Installing PyTorch..."
pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
echo "Installing project requirements..."
pip install -r requirements.txt
pip install -e ".[webui]"

# Download model files
echo "Downloading model files..."
mkdir -p checkpoints
for file in "${MODEL_FILES[@]}"; do
    if [ ! -f "checkpoints/${file}" ]; then
        echo "Downloading ${file}..."
        wget "${REPO_URL}/${file}" -P checkpoints || echo "Failed to download ${file}, continuing..."
    else
        echo "Found checkpoints/${file} - skipping download"
    fi
done

# Install production server
echo "Installing production server..."
pip install gunicorn uvicorn

# Display version information
echo -e "\n=== Setup complete ==="
echo -e "\nEnvironment Information:"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Virtual environment: $(which python)"
echo -e "\nKey packages:"
pip list --format=columns | grep -E "torch|gradio|gunicorn|uvicorn"

echo -e "\nTo run the web UI:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo "2. Run the development server:"
echo "   python webui.py"
echo -e "\nFor production (Cloud Run compatible):"
echo "   gunicorn -b :\$PORT -k uvicorn.workers.UvicornWorker -t 300 webui:app"
