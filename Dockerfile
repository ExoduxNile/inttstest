# Stage 1: Use slim Python image
FROM python:3.10-slim as base

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && pip install -e ".[webui]" \
 && rm -rf ~/.cache/pip /root/.cache /tmp/*

# Expose the port Cloud Run uses
EXPOSE 8080

# On container start, download checkpoints if missing
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
[ \
 subprocess.run(['wget', url, '-O', f'checkpoints/{fname}']) \
 for fname, url in urls.items() \
 if not os.path.exists(f'checkpoints/{fname}') \
]; \
" && \
gunicorn webui:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4
