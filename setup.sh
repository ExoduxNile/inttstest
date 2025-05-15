#!/bin/bash

# setup.sh - Simplified setup for Index-TTS web UI

set -e  # Exit on error

# Configuration
MODEL_FILES=(
    "bigvgan_discriminator.pth"
    "bigvgan_generator.pth"
    "bpe.model"
    "dvae.pth"
    "gpt.pth"
    "unigram_12000.vocab"
)
REPO_URL="https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main"

echo "=== Setting up Index-TTS ==="

# Install system dependencies
echo "Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    wget \
    ffmpeg \
    python3 \
    python3-pip \
    python3-venv

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e ".[webui]"

# Check for uvicorn and install if not present
if ! python -c "import uvicorn" &> /dev/null; then
    echo "Installing uvicorn for faster web server..."
    pip install uvicorn[standard]
fi

# Download model files if they don't exist
echo "Checking model files..."
mkdir -p checkpoints
for file in "${MODEL_FILES[@]}"; do
    if [ ! -f "checkpoints/${file}" ]; then
        echo "Downloading ${file}..."
        wget "${REPO_URL}/${file}" -P checkpoints
    else
        echo "Found checkpoints/${file} - skipping download"
    fi
done

# Determine how to run the web UI
echo "=== Setup complete ==="
echo -e "\nTo run the web UI, choose one of these methods:"

# Option 1: Uvicorn (if available)
if python -c "import uvicorn" &> /dev/null; then
    echo "1. Fast startup with uvicorn:"
    echo "   source venv/bin/activate && uvicorn webui:app --host 0.0.0.0 --port 8080"
    echo "   (Note: You might need to modify webui.py to expose a FastAPI/ASGI app)"
fi

# Option 2: Native Python
echo "2. Standard Python server:"
echo "   source venv/bin/activate && python webui.py"

# Option 3: Gunicorn (if Flask app)
echo "3. Production-style with gunicorn (if webui.py uses Flask):"
echo "   source venv/bin/activate && pip install gunicorn && gunicorn -b 0.0.0.0:8080 webui:app"

echo -e "\nNote: You may need to modify webui.py to properly expose the application object"
echo "for ASGI servers like uvicorn if it's not already compatible."
