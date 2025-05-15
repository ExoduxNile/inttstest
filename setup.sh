#!/bin/bash

# setup.sh - Index-TTS setup with Python 3.10

set -e  # Exit on error

# Configuration
PYTHON_VERSION="3.10"
MODEL_FILES=(
    "bigvgan_discriminator.pth"
    "bigvgan_generator.pth"
    "bpe.model"
    "dvae.pth"
    "gpt.pth"
    "unigram_12000.vocab"
)
REPO_URL="https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main"

echo "=== Setting up Index-TTS with Python $PYTHON_VERSION ==="

# Install Python 3.10 if not available
if ! command -v python$PYTHON_VERSION &> /dev/null; then
    echo "Installing Python $PYTHON_VERSION..."
    sudo apt-get update
    sudo apt-get install -y \
        software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y \
        python$PYTHON_VERSION \
        python$PYTHON_VERSION-venv \
        python$PYTHON_VERSION-dev
fi

# Install system dependencies
echo "Installing system packages..."
sudo apt-get install -y \
    wget \
    ffmpeg \
    python3-pip

# Create and activate virtual environment
echo "Creating Python $PYTHON_VERSION virtual environment..."
python$PYTHON_VERSION -m venv venv
source venv/bin/activate

# Ensure pip is up to date
python -m pip install --upgrade pip

# Install PyTorch for CPU (adjust if you need GPU support)
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
        wget "${REPO_URL}/${file}" -P checkpoints
    else
        echo "Found checkpoints/${file} - skipping download"
    fi
done

# Install gunicorn for production serving
echo "Installing production server..."
pip install gunicorn

# Display version information at the end
echo -e "\n=== Setup complete ==="
echo -e "\nVersion Information:"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Installed packages:"
pip list --format=columns | grep -E "torch|gradio|gunicorn"

echo -e "\nTo run the web UI:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo "2. Run the development server:"
echo "   python webui.py"
echo -e "\nOr for production use:"
echo "   gunicorn -b 0.0.0.0:\${PORT:-8080} -w 4 -k uvicorn.workers.UvicornWorker webui:app"
echo -e "\nNote: You may need to modify webui.py to expose a proper ASGI application"
echo "if you want to use uvicorn directly."
