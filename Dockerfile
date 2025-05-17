# Start from a minimal base OS
FROM debian:bullseye-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git ffmpeg build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev llvm libncursesw5-dev \
    xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.10 from source
RUN curl -O https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xzf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && rm -rf Python-3.10.14*

# Set python3.10 as default
RUN ln -s /usr/local/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip

# Create working directory
WORKDIR /app

# Create and activate venv
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy your code into the container
COPY . .

# Install dependencies inside the virtual environment
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e ".[webui]" && \
    rm -rf ~/.cache/pip /tmp/*

# Expose Cloud Run port
EXPOSE 8080

# Download checkpoints at container startup and launch app
CMD python -c "\
import os, subprocess; \
os.makedirs('checkpoints', exist_ok=True); \
urls = { \
  'bigvgan_discriminator.pth': 'https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_discriminator.pth', \
  'bigvgan_generator.pth': 'https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bigvgan_generator.pth', \
  'bpe.model': 'https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/bpe.model', \
  'dvae.pth': 'https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/dvae.pth', \
  'gpt.pth': 'https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/gpt.pth', \
  'unigram_12000.vocab': 'https://huggingface.co/IndexTeam/IndexTTS-1.5/resolve/main/unigram_12000.vocab' \
}; \
[subprocess.run(['wget', url, '-O', f'checkpoints/{fname}']) for fname, url in urls.items() if not os.path.exists(f'checkpoints/{fname}')]\
" && \
gunicorn webui:app --bind 0.0.0.0:${PORT:-7861} --workers 1 --threads 4
