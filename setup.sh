#!/bin/bash

# setup.sh - Index-TTS setup with Python 3.12

set -e  # Exit on error

# Configuration
REQUIRED_PYTHON_VERSION="3.11"
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

# Install Python 3.12
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

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3.12-distutils

# Set Python 3.12 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --set python3 /usr/bin/python3.11
sudo ln -fs /usr/bin/python3.11 /usr/bin/python

# Install system dependencies
echo "Installing system packages..."
sudo apt-get install -y \
    wget \
    ffmpeg

# Create and activate virtual environment
echo "Creating Python $REQUIRED_PYTHON_VERSION virtual environment..."
python3.12 -m venv venv
source venv/bin/activate

# Install pip for Python 3.12
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with compatible versions for Python 3.12
echo "Installing PyTorch..."
pip install \
    torch==2.2.1 \
    torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies with version constraints
echo "Installing project requirements..."
pip install \
    "numba>=0.58,<0.59" \  # Version that supports Python 3.12
    "numpy>=1.26,<2.0" \   # Modern numpy version
    -r requirements.txt

# Install the package with webui extras
echo "Installing Index-TTS package..."
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
pip install "gunicorn==21.2.0" "uvicorn==0.29.0"

# Display version information
echo -e "\n=== Setup complete ==="
echo -e "\nEnvironment Information:"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo "Virtual environment: $(which python)"
echo -e "\nKey packages:"
pip list --format=columns | grep -E "torch|gradio|gunicorn|uvicorn|numba|numpy"

echo -e "\nTo run the web UI:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo "2. Run the development server:"
echo "   python webui.py"
echo -e "\nFor production (Cloud Run compatible):"
echo "   gunicorn -b :\$PORT -k uvicorn.workers.UvicornWorker -t 300 webui:app"
