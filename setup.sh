#!/bin/bash

# setup.sh - Index-TTS setup with Python 3.10 for Google Cloud

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

# Check system Python version
CURRENT_PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "Detected Python version: $CURRENT_PYTHON_VERSION"

# Install Python 3.10 if not available or if default is different
if [ "$CURRENT_PYTHON_VERSION" != "$REQUIRED_PYTHON_VERSION" ]; then
    echo "Installing Python $REQUIRED_PYTHON_VERSION..."
    sudo apt-get update
    sudo apt-get install -y \
        software-properties-common \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        wget
    
    # Install Python 3.10 from source
    PYTHON_SRC_URL="https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz"
    wget $PYTHON_SRC_URL
    tar -xf Python-3.10.13.tgz
    cd Python-3.10.13
    ./configure --enable-optimizations
    make -j $(nproc)
    sudo make altinstall
    cd ..
    rm -rf Python-3.10.13*
fi

# Verify Python 3.10 installation
echo "Verifying Python installation..."
python3.10 --version || { echo "Python 3.10 installation failed"; exit 1; }

# Install system dependencies
echo "Installing system packages..."
sudo apt-get install -y \
    wget \
    ffmpeg \
    python3-pip

# Create and activate virtual environment
echo "Creating Python $REQUIRED_PYTHON_VERSION virtual environment..."
python3.10 -m venv venv
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
echo -e "\nNote: Ensure webui.py exposes a proper ASGI application for production use."
